#include "XPBD.h"
#include "CudaUtils.h"
#include "XPBD_Kernel.cuh"
#include <ncltech\NCLDebug.h>

void swap(CudaTextureWrapper& a, CudaTextureWrapper& b)
{
	CudaTextureWrapper t = a;
	a = b;
	b = t;
}






XPBD::XPBD()
	: m_NumIndicies(0)
	, m_NumConstraintIndicies(0)
	, m_NumParticles(0)
	, m_NumX(0)
	, m_NumY(0)
	, m_glArr(NULL)
	, m_glManageTexFront(false)
	, m_glManageTexBack(false)
	, m_glTexFront(NULL)
	, m_glTexBack(NULL)
	, m_cuVerticesRef(NULL)
	, m_cuNormalsRef(NULL)
#if XPBD_USE_BATCHED_CONSTRAINTS == FALSE
	, m_cuConstraintOutput(NULL)
	, m_cuConstraintLookups(NULL)
	, m_cuConstraintOutputLookups(NULL)
#endif

{
	m_BoundingRadius = FLT_MAX;
}

XPBD::~XPBD()
{
	Release();
}

void XPBD::Release()
{
	m_ConstraintLayers.clear();
	m_SphereConstraints.clear();



	if (m_cuParticlePos.mem != NULL)
	{
		gpuErrchk(cudaFreeArray(m_cuParticlePos.mem));
		gpuErrchk(cudaFreeArray(m_cuParticlePosTmp.mem));
		gpuErrchk(cudaFreeArray(m_cuParticlePosOld.mem));
		gpuErrchk(cudaFreeArray(m_cuParticleFaceNormals.mem));
		gpuErrchk(cudaFreeArray(m_cuParticleVertNormals.mem));
		m_cuParticlePos.mem = NULL;
	}



#if XPBD_USE_BATCHED_CONSTRAINTS == FALSE
	if (m_cuConstraintOutput != NULL)
	{
		gpuErrchk(cudaFree(m_cuConstraintOutput));
		gpuErrchk(cudaFree(m_cuConstraintLookups));
		gpuErrchk(cudaFree(m_cuConstraintOutputLookups));
		m_cuConstraintOutput = NULL;
	}
#endif

	if (m_glArr != NULL)
	{
		glDeleteVertexArrays(1, &m_glArr);
		glDeleteBuffers(5, m_glBuffers);
		m_glArr = NULL;
	}

	if (m_glTexFront != NULL)
	{
		if (m_glManageTexFront) glDeleteTextures(1, &m_glTexFront);
		m_glTexFront = NULL;
		m_glManageTexFront = false;
	}

	if (m_glTexBack != NULL)
	{
		if (m_glManageTexBack) glDeleteTextures(1, &m_glTexBack);
		m_glTexBack = NULL;
		m_glManageTexBack = false;
	}
}

void XPBD::InitializeCloth(int div_x, int div_y, const Matrix4& transform)
{
	Release();

	m_NumX = div_x;
	m_NumY = div_y;
	m_NumParticles = m_NumX * m_NumY;

	float factor_x = 1.0f / float(div_x - 1);
	float factor_y = 1.0f / float(div_y - 1);
	float half_x = -0.5f;
	float half_y = -0.5f;

	float imass = float(m_NumParticles) / 0.165f;
	Vector4* particles_initial = new Vector4[m_NumParticles];
	for (int y = 0; y < m_NumY; ++y)
	{
		for (int x = 0; x < m_NumX; ++x)
		{
			uint idx = y * m_NumX + x;
			Vector3 pos = transform * Vector3(x * factor_x + half_x, y * factor_y + half_y, 0.0f);
			particles_initial[idx].x = pos.x;
			particles_initial[idx].y = pos.y;
			particles_initial[idx].z = pos.z;
			particles_initial[idx].w = imass;

		}
	}
	//particles_initial[0].w = 0.0f;
	//particles_initial[m_NumX - 1].w = 0.0f;
	particles_initial[(m_NumY -1) * m_NumX].w = 0.0f;
	particles_initial[m_NumY * m_NumX - 1].w = 0.0f;


	int min_split = min(m_NumX, m_NumY);
	int n_layers = floor(log2(min_split));

	m_ConstraintLayers.resize(n_layers);
	for (int i = 0; i < n_layers; ++i)
	{
		m_ConstraintLayers[i].Initialize(this, 1 << (n_layers - i - 1));
		m_ConstraintLayers[i].GenerateData(particles_initial);
	}

	/*m_ConstraintLayers.resize(2);
	m_ConstraintLayers[0].Initialize(this, 1);
	m_ConstraintLayers[0].GenerateData(particles_initial);

	m_ConstraintLayers[1].Initialize(this, 2);
	m_ConstraintLayers[1].GenerateData(particles_initial);*/

	InitializeCuda(particles_initial);
	InitializeGL(particles_initial);


	delete[] particles_initial;
}

void XPBD::InitializeCuda(const Vector4* initial_positions)
{
	//Allocate Particle Arrays


	cudaChannelFormatDesc channelDesc1F = cudaCreateChannelDesc<float>();
	cudaChannelFormatDesc channelDesc4F = cudaCreateChannelDesc<float4>();

	m_cuParticlePos.allocate(channelDesc4F, m_NumX, m_NumY);
	m_cuParticlePosTmp.allocate(channelDesc4F, m_NumX, m_NumY);
	m_cuParticlePosOld.allocate(channelDesc4F, m_NumX, m_NumY);
	

	gpuErrchk(cudaMemcpyToArray(m_cuParticlePos.mem, 0, 0, initial_positions, m_NumParticles * sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyToArray(m_cuParticlePosTmp.mem, 0, 0, initial_positions, m_NumParticles * sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyToArray(m_cuParticlePosOld.mem, 0, 0, initial_positions, m_NumParticles * sizeof(float4), cudaMemcpyHostToDevice));

	
	m_cuParticleFaceNormals.allocate(channelDesc4F, m_NumX - 1, m_NumY - 1);
	m_cuParticleVertNormals.allocate(channelDesc4F, m_NumX, m_NumY);

	float4* initial_normals = new float4[m_NumParticles];
	memset(initial_normals, 0, m_NumParticles * sizeof(float4));
	gpuErrchk(cudaMemcpyToArray(m_cuParticleFaceNormals.mem, 0, 0, initial_normals, (m_NumX - 1) * (m_NumY - 1) * sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyToArray(m_cuParticleVertNormals.mem, 0, 0, initial_normals, m_NumParticles * sizeof(float4), cudaMemcpyHostToDevice));
	delete[] initial_normals;
}

void XPBD::InitializeGL(const Vector4* initial_positions)
{
	//Create temp data structures for uploading to device
	m_NumIndicies = (m_NumX - 1) * (m_NumY - 1) * 6;

	Vector4* vertices = new Vector4[m_NumParticles * 2];
	Vector2* texcoords = new Vector2[m_NumParticles * 2];
	uint* indicies = new uint[m_NumIndicies * 2];

	float factor_x = 1.0f / float(m_NumX - 1);
	float factor_y = 1.0f / float(m_NumY - 1);

	float n_tex_repitions = 10.0f;
	uint bf_offset = m_NumParticles;
	for (int y = 0; y < m_NumY; ++y)
	{
		for (int x = 0; x < m_NumX; ++x)
		{
			uint idx = y * m_NumX + x;
			vertices[idx] = initial_positions[idx];
			vertices[bf_offset + idx] = initial_positions[idx];
			texcoords[idx] = Vector2(x * factor_x * n_tex_repitions, y * factor_y * n_tex_repitions);
			texcoords[bf_offset + idx] = Vector2(x * factor_x * n_tex_repitions, y * factor_y * n_tex_repitions);
		}
	}
	
	for (int y = 1; y < m_NumY; ++y)
	{
		for (int x = 1; x < m_NumX; ++x)
		{
			unsigned int a = (y - 1) * m_NumX + x - 1;
			unsigned int b = (y - 1) * m_NumX + x;
			unsigned int c = (y) * m_NumX + x;
			unsigned int d = (y) * m_NumX + x - 1;

			uint idx = ((y-1) * (m_NumX - 1) + (x-1)) * 6;
			if (((y ^ x) & 0x1) == 0)
			{
				//Front Face
				indicies[idx+0] = a;
				indicies[idx+1] = b;
				indicies[idx+2] = c;

				indicies[idx+3] = a;
				indicies[idx+4] = c;
				indicies[idx+5] = d;

				//Back Face
				indicies[m_NumIndicies + idx + 0] = bf_offset + b;
				indicies[m_NumIndicies + idx + 1] = bf_offset + a;
				indicies[m_NumIndicies + idx + 2] = bf_offset + c;

				indicies[m_NumIndicies + idx + 3] = bf_offset + c;
				indicies[m_NumIndicies + idx + 4] = bf_offset + a;
				indicies[m_NumIndicies + idx + 5] = bf_offset + d;
			}
			else
			{
				//Front Face
				indicies[idx + 0] = b;
				indicies[idx + 1] = d;
				indicies[idx + 2] = a;

				indicies[idx + 3] = b;
				indicies[idx + 4] = c;
				indicies[idx + 5] = d;

				//Back Face
				indicies[m_NumIndicies + idx + 0] = bf_offset + d;
				indicies[m_NumIndicies + idx + 1] = bf_offset + b;
				indicies[m_NumIndicies + idx + 2] = bf_offset + a;

				indicies[m_NumIndicies + idx + 3] = bf_offset + c;
				indicies[m_NumIndicies + idx + 4] = bf_offset + b;
				indicies[m_NumIndicies + idx + 5] = bf_offset + d;
			}
		}
	}

	XPBD_ConstraintLayer& layer = m_ConstraintLayers[m_ConstraintLayers.size() - 1];
	m_NumConstraintIndicies = layer.m_DistanceConstraints.size() * 2 + layer.m_BendingConstraints.size() * 4;
	uint* cindicies = new uint[m_NumConstraintIndicies];
	int cidx = 0;
	for (size_t i = 0; i < layer.m_DistanceConstraints.size(); ++i)
	{
		unsigned int a = layer.m_DistanceConstraints[i].p1.y * m_NumX + layer.m_DistanceConstraints[i].p1.x;
		unsigned int b = layer.m_DistanceConstraints[i].p2.y * m_NumX + layer.m_DistanceConstraints[i].p2.x;

		cindicies[cidx++] = a;
		cindicies[cidx++] = b;
	}
	for (size_t i = 0; i < layer.m_BendingConstraints.size(); ++i)
	{
		unsigned int a = layer.m_BendingConstraints[i].p1.y * m_NumX + layer.m_BendingConstraints[i].p1.x;
		unsigned int b = layer.m_BendingConstraints[i].p2.y * m_NumX + layer.m_BendingConstraints[i].p2.x;
		unsigned int c = layer.m_BendingConstraints[i].p3.y * m_NumX + layer.m_BendingConstraints[i].p3.x;

		cindicies[cidx++] = a;
		cindicies[cidx++] = b;
		cindicies[cidx++] = b;
		cindicies[cidx++] = c;
	}



	glGenVertexArrays(1, &m_glArr);
	glGenBuffers(5, m_glBuffers);


	glBindVertexArray(m_glArr);

	//Buffer vertex data
	glBindBuffer(GL_ARRAY_BUFFER, m_glBuffers[0]);
	glBufferData(GL_ARRAY_BUFFER, m_NumParticles * 2 * sizeof(Vector4), NULL, GL_STATIC_DRAW);
	glVertexAttribPointer(VERTEX_BUFFER, 3, GL_FLOAT, GL_FALSE, 16, (void**)0);
	glEnableVertexAttribArray(VERTEX_BUFFER);

	//Buffer normal data
	glBindBuffer(GL_ARRAY_BUFFER, m_glBuffers[1]);
	glBufferData(GL_ARRAY_BUFFER, m_NumParticles * 2 * sizeof(Vector4), NULL, GL_STATIC_DRAW);
	glVertexAttribPointer(NORMAL_BUFFER, 3, GL_FLOAT, GL_FALSE, 16, (void**)0);
	glEnableVertexAttribArray(NORMAL_BUFFER);

	//Buffer texture data
	glBindBuffer(GL_ARRAY_BUFFER, m_glBuffers[2]);
	glBufferData(GL_ARRAY_BUFFER, m_NumParticles * 2 * sizeof(Vector2), texcoords, GL_STATIC_DRAW);
	glVertexAttribPointer(TEXTURE_BUFFER, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(TEXTURE_BUFFER);

	//buffer index data
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_glBuffers[3]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_NumIndicies * 2 * sizeof(GLuint), indicies, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_glBuffers[4]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_NumConstraintIndicies*sizeof(GLuint), cindicies, GL_STATIC_DRAW);

	glBindVertexArray(0);


	delete[] texcoords;
	delete[] indicies;
	delete[] cindicies;


	cudaGraphicsGLRegisterBuffer(&m_cuVerticesRef, m_glBuffers[0], cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&m_cuNormalsRef, m_glBuffers[1], cudaGraphicsMapFlagsWriteDiscard);
}

bool render_constraints = true;

void XPBD::OnUpdateObject(float dt)
{
	if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_B))
		render_constraints = !render_constraints;
}

void XPBD::UpdateGLBuffers()
{
	float4* cuda_vbo_arr;
	uint num_bytes;

	gpuErrchk(cudaGraphicsMapResources(1, &m_cuNormalsRef, 0));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&cuda_vbo_arr, &num_bytes, m_cuNormalsRef));
	//FRONT FACE
	GenerateNormals(false);
	gpuErrchk(cudaMemcpyFromArray(cuda_vbo_arr, m_cuParticleVertNormals.mem, 0, 0, m_NumParticles * sizeof(Vector4), cudaMemcpyDeviceToDevice));

	//BACK FACE
	GenerateNormals(true);
	gpuErrchk(cudaMemcpyFromArray(&cuda_vbo_arr[m_NumParticles], m_cuParticleVertNormals.mem, 0, 0, m_NumParticles * sizeof(Vector4), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaGraphicsUnmapResources(1, &m_cuNormalsRef, 0));


	gpuErrchk(cudaGraphicsMapResources(1, &m_cuVerticesRef, 0));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&cuda_vbo_arr, &num_bytes, m_cuVerticesRef));
	gpuErrchk(cudaMemcpyFromArray(&cuda_vbo_arr[0], m_cuParticlePos.mem, 0, 0, m_NumParticles * sizeof(Vector4), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpyFromArray(&cuda_vbo_arr[m_NumParticles], m_cuParticlePos.mem, 0, 0, m_NumParticles * sizeof(Vector4), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaGraphicsUnmapResources(1, &m_cuVerticesRef, 0));
}

void XPBD::Update(float dt)
{
		xpbd_kernel::CudaParams params;
		params.dt = dt;
		params.dampFactor = 1.0f - m_MaterialDamp;
		params.gravity = make_float3(0.0f, -9.81f, 0.0f);
		params.dims.x = m_NumX;
		params.dims.y = m_NumY;

		xpbd_kernel::set_parameters(&params);

		xpbd_kernel::updateforce(m_NumX, m_NumY, m_cuParticlePos.tex, m_cuParticlePosOld.tex, NULL);
		swap(m_cuParticlePosOld, m_cuParticlePos);

		int max_itr = 40;
		int min_itr = 4;

		for (size_t litr = 0; litr < m_ConstraintLayers.size(); ++litr)
		{
			XPBD_ConstraintLayer& l = m_ConstraintLayers[litr];
			//l.ResetLambda();

			float factor =  litr / float(m_ConstraintLayers.size() - 1);
			//factor = factor * factor;
			int iterations = 40;// floor(min_itr * (1.0f - factor) + max_itr * (factor));

			if (l.Solve(iterations, m_cuParticlePos.tex, m_cuParticlePosTmp.tex, NULL, NULL))
				swap(m_cuParticlePos, m_cuParticlePosTmp);
		}
		
		for (size_t j = 0; j < m_SphereConstraints.size(); ++j)
		{
			xpbd_kernel::solvesphereconstraint(m_NumX, m_NumY, m_SphereConstraints[j], m_cuParticlePos.tex);
		}
	

}

void XPBD::GenerateNormals(bool reverseOrder)
{
	//Generate Quad Normals
	xpbd_kernel::genquadnormals(m_NumX - 1, m_NumY - 1, reverseOrder, m_cuParticlePos.tex, m_cuParticleFaceNormals.surface);

	//Build Vertex Normals from Quads
	xpbd_kernel::genvertnormals(m_NumX, m_NumY, m_cuParticleFaceNormals.tex, m_cuParticleVertNormals.surface);


	//Bluring??
}



void XPBD::OnRenderObject()
{


	
	glBindVertexArray(m_glArr);
	if (!render_constraints)
	{
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_glBuffers[3]);

		if (Window::GetKeyboard()->KeyDown(KEYBOARD_F))
		{
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		}
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, m_glTexFront);
		glDrawElements(GL_TRIANGLES, m_NumIndicies, GL_UNSIGNED_INT, 0);

		glBindTexture(GL_TEXTURE_2D, m_glTexBack);
		glDrawElements(GL_TRIANGLES, m_NumIndicies, GL_UNSIGNED_INT, (void**)(m_NumIndicies * sizeof(uint)));
		if (Window::GetKeyboard()->KeyDown(KEYBOARD_F))
		{
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		}
	}
	else
	{
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_glBuffers[4]);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

#if XPBD_USE_BATCHED_CONSTRAINTS
		int batch = -1;
		if (Window::GetKeyboard()->KeyDown(KEYBOARD_1))
			batch = 0;
		if (Window::GetKeyboard()->KeyDown(KEYBOARD_2))
			batch = 1;
		if (Window::GetKeyboard()->KeyDown(KEYBOARD_3))
			batch = 2;
		if (Window::GetKeyboard()->KeyDown(KEYBOARD_4))
			batch = 3;
		if (Window::GetKeyboard()->KeyDown(KEYBOARD_5))
			batch = 4;
		if (Window::GetKeyboard()->KeyDown(KEYBOARD_6))
			batch = 5;
		if (Window::GetKeyboard()->KeyDown(KEYBOARD_7))
			batch = 6;
		if (Window::GetKeyboard()->KeyDown(KEYBOARD_8))
			batch = 7;
		if (Window::GetKeyboard()->KeyDown(KEYBOARD_H) && batch != -1)
		{
			batch += 8;
		}
		if (batch == -1)
			glDrawElements(GL_LINES, m_NumConstraintIndicies, GL_UNSIGNED_INT, 0);
		else
		{
			uint start, end;
			XPBD_ConstraintLayer& layer = m_ConstraintLayers[m_ConstraintLayers.size() - 1];
			if (batch < 8)
			{
				start = layer.m_DistanceConstraintBatches[batch].x * 2;
				end = layer.m_DistanceConstraintBatches[batch].y * 2;
			}
			else
			{
				uint offset = layer.m_DistanceConstraints.size() * 2;
				start = layer.m_BendingConstraintBatches[batch - 8].x * 4 + offset;
				end = layer.m_BendingConstraintBatches[batch - 8].y * 4 + offset;
			}
			glDrawElements(GL_LINES, (end - start), GL_UNSIGNED_INT, (GLvoid**)(start * sizeof(uint)));
		}
#else
		glDrawElements(GL_LINES, m_NumConstraintIndicies, GL_UNSIGNED_INT, 0);
#endif
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
}


