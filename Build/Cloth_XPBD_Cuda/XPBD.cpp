#include "XPBD.h"
#include "CudaUtils.h"


XPBD::XPBD()
	: m_NumIndicies(0)
	, m_NumParticles(0)
	, m_NumX(0)
	, m_NumY(0)
	, m_cuParticlePosTex(NULL)
	, m_cuParticlePosTmpTex(NULL)
	, m_cuParticlePosArr(NULL)
	, m_cuParticlePosTmpArr(NULL)
	, m_cuDistanceConstraints(NULL)
	, m_cuBendingConstraints(NULL)
	, m_cuSphereConstraints(NULL)
	, m_glArr(NULL)
	, m_glManageTex(false)
	, m_glTex(NULL)
{
	m_BoundingRadius = FLT_MAX;
}

XPBD::~XPBD()
{
	Release();
}

void XPBD::Release()
{
	m_DistanceConstraints.clear();
	m_BendingConstraints.clear();
	m_SphereConstraints.clear();

	if (m_cuParticlePosArr != NULL)
	{
		gpuErrchk(cudaFreeArray(m_cuParticlePosArr));
		gpuErrchk(cudaFreeArray(m_cuParticlePosTmpArr));
		m_cuParticlePosArr = NULL;
	}

	if (m_cuDistanceConstraints != NULL)
	{
		gpuErrchk(cudaFree(m_cuDistanceConstraints));
		m_cuDistanceConstraints = NULL;
	}

	if (m_cuBendingConstraints != NULL)
	{
		gpuErrchk(cudaFree(m_cuBendingConstraints));
		m_cuBendingConstraints = NULL;
	}

	if (m_cuSphereConstraints != NULL)
	{
		gpuErrchk(cudaFree(m_cuSphereConstraints));
		m_cuSphereConstraints = NULL;
	}

	if (m_glArr != NULL)
	{
		glDeleteVertexArrays(1, &m_glArr);
		glDeleteBuffers(4, m_glBuffers);
		m_glArr = NULL;
	}

	if (m_glTex != NULL)
	{
		if (m_glManageTex) glDeleteTextures(1, &m_glTex);
		m_glTex = NULL;
		m_glManageTex = false;
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

	Vector3* particles_initial = new Vector3[m_NumParticles];
	for (int y = 0; y < m_NumY; ++y)
	{
		for (int x = 0; x < m_NumX; ++x)
		{
			uint idx = y * m_NumX + x;
			particles_initial[idx] = transform * Vector3(x * factor_x, y * factor_y, 0.0f);
		}
	}

	InitializeConstraints(particles_initial);
	InitializeGL(particles_initial);
	InitializeCuda(particles_initial);

	delete[] particles_initial;
}

void XPBD::InitializeConstraints(const Vector3* initial_positions)
{
	// Horizontal
	int l1, l2;
	for (l1 = 0; l1 < m_NumY; l1++)	// v
	for (l2 = 0; l2 < (m_NumX - 1); l2++) {
		m_DistanceConstraints.push_back(BuildDistanceConstraint(initial_positions, (l1 * m_NumX) + l2, (l1 * m_NumX) + l2 + 1, m_MaterialShear));
	}

	// Vertical
	for (l1 = 0; l1 < (m_NumX); l1++)
	for (l2 = 0; l2 < (m_NumY - 1); l2++) {
		m_DistanceConstraints.push_back(BuildDistanceConstraint(initial_positions, (l2 * m_NumX) + l1, ((l2 + 1) * m_NumX) + l1, m_MaterialShear));
	}

	// Shearing distance constraint
	for (l1 = 0; l1 < (m_NumY - 1); l1++)
	for (l2 = 0; l2 < (m_NumX - 1); l2++) {
		m_DistanceConstraints.push_back(BuildDistanceConstraint(initial_positions, (l1 * m_NumX) + l2, ((l1 + 1) * m_NumX) + l2 + 1, m_MaterialShear));
		m_DistanceConstraints.push_back(BuildDistanceConstraint(initial_positions, ((l1 + 1) * m_NumX) + l2, (l1 * m_NumX) + l2 + 1, m_MaterialShear));
	}

	auto get_idx = [&](int i, int j) {
		return i * m_NumX + j;
	};

	//add vertical constraints
	for (int i = 0; i < m_NumX; i++) {
		for (int j = 0; j<m_NumY - 2; j++) {
			m_BendingConstraints.push_back(BuildBendingConstraint(initial_positions, get_idx(i, j), get_idx(i, (j + 1)), get_idx(i, j + 2), m_MaterialBend));
		}
	}
	//add horizontal constraints
	for (int i = 0; i < m_NumX - 2; i++) {
		for (int j = 0; j < m_NumY; j++) {
			m_BendingConstraints.push_back(BuildBendingConstraint(initial_positions, get_idx(i, j), get_idx(i + 1, j), get_idx(i + 2, j), m_MaterialBend));
		}
	}
}

void InitializeCuda(const Vector3* initial_positions)
{





	//Create Host Constraints
	
}

void XPBD::InitializeGL(const Vector3* initial_positions)
{
	//Create temp data structures for uploading to device
	m_NumIndicies = (m_NumX - 1) * (m_NumY - 1) * 12;

	Vector2* texcoords = new Vector2[m_NumParticles];
	Vector3* normals = new Vector3[m_NumParticles];
	uint* indicies = new uint[m_NumIndicies];

	float factor_x = 1.0f / float(m_NumX - 1);
	float factor_y = 1.0f / float(m_NumY - 1);

	
	for (int y = 0; y < m_NumY; ++y)
	{
		for (int x = 0; x < m_NumX; ++x)
		{
			uint idx = y * m_NumX + x;
			texcoords[idx] = Vector2(x * factor_x, y * factor_y);
			normals[idx] = Vector3(0.0f, 1.0f, 0.0f);
		}
	}

	int cidx = 0;
	for (int y = 1; y < m_NumY; ++y)
	{
		for (int x = 1; x < m_NumX; ++x)
		{
			unsigned int a = (y - 1) * m_NumX + x - 1;
			unsigned int b = (y - 1) * m_NumX + x;
			unsigned int c = (y) * m_NumX + x;
			unsigned int d = (y) * m_NumX + x - 1;

			if (((y ^ x) & 0x1) == 0)
			{
				//Front Face
				indicies[cidx++] = a;
				indicies[cidx++] = b;
				indicies[cidx++] = c;

				indicies[cidx++] = a;
				indicies[cidx++] = c;
				indicies[cidx++] = d;

				//Back Face
				indicies[cidx++] = b;
				indicies[cidx++] = a;
				indicies[cidx++] = c;

				indicies[cidx++] = c;
				indicies[cidx++] = a;
				indicies[cidx++] = d;
			}
			else
			{
				//Front Face
				indicies[cidx++] = b;
				indicies[cidx++] = d;
				indicies[cidx++] = a;

				indicies[cidx++] = b;
				indicies[cidx++] = c;
				indicies[cidx++] = d;

				//Back Face
				indicies[cidx++] = d;
				indicies[cidx++] = b;
				indicies[cidx++] = a;

				indicies[cidx++] = c;
				indicies[cidx++] = b;
				indicies[cidx++] = d;
			}
		}
	}

	glGenVertexArrays(1, &m_glArr);
	glGenBuffers(4, m_glBuffers);


	glBindVertexArray(m_glArr);

	//Buffer vertex data
	glBindBuffer(GL_ARRAY_BUFFER, m_glBuffers[0]);
	glBufferData(GL_ARRAY_BUFFER, m_NumParticles*sizeof(Vector3), initial_positions, GL_STATIC_DRAW);
	glVertexAttribPointer(VERTEX_BUFFER, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(VERTEX_BUFFER);

	//Buffer normal data
	glBindBuffer(GL_ARRAY_BUFFER, m_glBuffers[1]);
	glBufferData(GL_ARRAY_BUFFER, m_NumParticles*sizeof(Vector3), normals, GL_STATIC_DRAW);
	glVertexAttribPointer(NORMAL_BUFFER, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(NORMAL_BUFFER);

	//Buffer texture data
	glBindBuffer(GL_ARRAY_BUFFER, m_glBuffers[2]);
	glBufferData(GL_ARRAY_BUFFER, m_NumParticles*sizeof(Vector2), texcoords, GL_STATIC_DRAW);
	glVertexAttribPointer(TEXTURE_BUFFER, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(TEXTURE_BUFFER);

	//buffer index data
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_glBuffers[3]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_NumIndicies*sizeof(GLuint), indicies, GL_STATIC_DRAW);

	glBindVertexArray(0);


	delete[] normals;
	delete[] texcoords;
	delete[] indicies;
}

void XPBD::OnUpdateObject(float dt)
{

}


void XPBD::OnRenderObject()
{
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_glTex);


	if (Window::GetKeyboard()->KeyDown(KEYBOARD_F))
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	}

	glBindVertexArray(m_glArr);
	glDrawElements(GL_TRIANGLES, m_NumIndicies, GL_UNSIGNED_INT, 0);

	if (Window::GetKeyboard()->KeyDown(KEYBOARD_F))
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
}



XPBDDistanceConstraint XPBD::BuildDistanceConstraint(const Vector3* positions, uint idxA, uint idxB, float k)
{
	XPBDDistanceConstraint out_constraint;

	out_constraint.p1 = idxA;
	out_constraint.p2 = idxB;
	out_constraint.k = k;

	Vector3 deltaP = positions[idxA] - positions[idxB];
	out_constraint.rest_length = deltaP.Length();

	return out_constraint;
}

XPBDBendingConstraint XPBD::BuildBendingConstraint(const Vector3* positions, uint idxA, uint idxB, uint idxC, float k)
{
	XPBDBendingConstraint out_constraint;

	out_constraint.p1 = idxA;
	out_constraint.p2 = idxB;
	out_constraint.p3 = idxC;

	out_constraint.k = k;

	Vector3 center = (positions[idxA] + positions[idxB] + positions[idxC]) / 3.0f;
	out_constraint.rest_length = (positions[idxC] - center).Length();

	return out_constraint;
}