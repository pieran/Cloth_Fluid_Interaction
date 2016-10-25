

#include "PicFlip.cuh"
#include "Cuda_Utils.cuh"
#include "Fluid_Kernel_Utils.cuh"

#include "PicFlip_Kernel.cuh"
#include "PicFlip_Kernel.cu"


#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using namespace cudautils;
using namespace utils;
using namespace picflip;

#include <ncltech\NCLDebug.h>

FluidPicFlip::FluidPicFlip()
: m_glPos(0)
, m_glVel(0)
, m_glDens(0)
, m_glPres(0)
, m_gridSize(0)
, m_NumParticles(0)
, m_ParticlePos(NULL)
, m_ParticlePosTemp(NULL)
, m_ParticleVel(NULL)
, m_ParticleVelTemp(NULL)
, m_ParticleRnd(NULL)	//contains a random normalized direction for each particle
, m_GridVel(NULL)
, m_GridVelTemp(NULL)
, m_GridVelOrig(NULL)
, m_GridMarker(NULL)//marks fluid/air, 1 if fluid, 0 if air
, m_GridDivergence(NULL)
, m_GridPressure(NULL)
, m_GridPressureTemp(NULL)
, m_GridMapperStart(NULL)
, m_GridMapperEnd(NULL)
, vbo_pos(NULL)
{
	m_Params.dt = 1.0f / 60.0f;
	m_Params.grid_size.x = 128 - 1;

	const float grid_resolution_factor = 1.0f;
	m_Params.grid_resolution.x = m_Params.grid_size.x * grid_resolution_factor;

	m_Params.flipness = 0.99f;

	//set_vox_dims(10, 22); //Cube Splash
	set_vox_dims(12, 22); //Dambreak
	m_Params.particles_per_cell *= 3.0f;
}

void FluidPicFlip::set_vox_dims(int vox_dens, int vox_width)
{
	const double split = double(vox_width * vox_dens) / double(m_Params.grid_resolution.x);
	m_Params.particles_per_cell = split * split * split;// 120.0f;// 14.0f;
}

FluidPicFlip::~FluidPicFlip()
{
	if (m_ParticlePos != NULL)
	{
		free_array(m_ParticlePos);
		free_array(m_ParticlePosTemp);
		free_array(m_ParticleVel);
		free_array(m_ParticleVelTemp);
		free_array(m_ParticleRnd);
		free_array(m_KeyValues);
		free_array(m_KeyValuesTemp);
		m_ParticlePos = NULL;
	}

	if (m_GridVel != NULL)
	{
		gpuErrchk(cudaFreeArray(m_GridVel));
		gpuErrchk(cudaFreeArray(m_GridVelTemp));
		gpuErrchk(cudaFreeArray(m_GridVelOrig));
		gpuErrchk(cudaFreeArray(m_GridMarker));
		gpuErrchk(cudaFreeArray(m_GridDivergence));
		gpuErrchk(cudaFreeArray(m_GridPressure));
		gpuErrchk(cudaFreeArray(m_GridPressureTemp));
		gpuErrchk(cudaFreeArray(m_GridMapperStart));
		gpuErrchk(cudaFreeArray(m_GridMapperEnd));
		m_GridVel = NULL;
	}
}

uint frameidx = 0;
void FluidPicFlip::update()
{
	NCLDebug::AddStatusEntry(Vector4(1.f, 1.f, 1.f, 1.f), "\tNum Particles: %d", m_NumParticles);
	NCLDebug::AddStatusEntry(Vector4(1.f, 1.f, 1.f, 1.f), "\tGrid Size: %d, %d, %d", m_Params.grid_resolution.x, m_Params.grid_resolution.y, m_Params.grid_resolution.z);
	NCLDebug::AddStatusEntry(Vector4(1.f, 1.f, 1.f, 1.f), "\tParticles Per Cell: %f", m_Params.particles_per_cell);
#if PICFLIP_PROFILE_EACH_KERNEL
	printf("STARTING FRAME: %d\n", frameidx++);

#endif
	if (BuildGridBuffers())
	{
		set_parameters(&m_Params);

		uint3 grid_Resolution_vel = m_Params.grid_resolution;
		grid_Resolution_vel.x++; grid_Resolution_vel.y++; grid_Resolution_vel.z++;

		float4 empty4f; memset(&empty4f, 0, sizeof(float4));

		//Zero grid offsets

		MemSetSurface<uint>(NULL, m_GridMapperStartTex, 0, m_Params.grid_resolution);
		MemSetSurface<uint>(NULL, m_GridMapperEndTex, 0, m_Params.grid_resolution);

		//Sort Particles + Build Grid Offsets
		sortByGridIndex(NULL, m_NumParticles, m_GridMapperStartTex, m_GridMapperEndTex, m_KeyValues, m_KeyValuesTemp, m_ParticlePos, m_ParticlePosTemp, m_ParticleVel, m_ParticleVelTemp);
		std::swap(m_ParticlePos, m_ParticlePosTemp);
		std::swap(m_ParticleVel, m_ParticleVelTemp);
		std::swap(m_KeyValues, m_KeyValuesTemp);
	}

	struct TexParams
	{
		TexParams(cudaArray_t a,
		cudaSurfaceObject_t s,
		cudaTextureObject_t t)
		: arr(a), sur(s), tex(t) {};

		cudaArray_t arr;
		cudaSurfaceObject_t sur;
		cudaTextureObject_t tex;
	};

	TexParams vel_ping(m_GridVel, m_GridVelTex, m_GridVelSur);
	TexParams vel_pong(m_GridVelTemp, m_GridVelTempTex, m_GridVelTempSur);
	TexParams* vel_cur = &vel_ping;
	TexParams* vel_old = &vel_pong;

	TexParams pres_ping(m_GridPressure, m_GridPressureTex, m_GridPressureSur);
	TexParams pres_pong(m_GridPressureTemp, m_GridPressureTempTex, m_GridPressureTempSur);
	TexParams* pres_cur = &pres_ping;
	TexParams* pres_old = &pres_pong;

	//Transfer to Velocity Grid + Weight Grid

	//gpuErrchk(cudaMemset(vel_cur->arr, 0, vel_grid_size * sizeof(float4)));
	//MemSetSurface<float4>(NULL, vel_cur->tex, empty4f, grid_Resolution_vel);
	//MemSetSurface<float4>(NULL, m_GridVelOrigTex, empty4f, grid_Resolution_vel);

	transferToGridProgram(NULL, m_Params.grid_resolution, m_NumParticles, m_GridMapperStartTex, m_GridMapperEndTex, vel_cur->tex, m_GridVelOrigTex, m_ParticlePos, m_ParticleVel);
	std::swap(vel_cur, vel_old);

	//Mark Fluid Cells
	//gpuErrchk(cudaMemset(m_GridMarker, 0, m_gridSize * sizeof(char)));
	MemSetSurface<unsigned char>(NULL, m_GridMarkerSur, 0, m_Params.grid_resolution);
	markProgram(NULL, m_NumParticles, m_ParticlePos, m_GridMarkerSur);
	/*marksolidcells(
		NULL,
		make_uint3(55, 0, 0),
		make_uint3(5, m_Params.grid_resolution.y, m_Params.grid_resolution.z),
		m_GridMarkerSur);*/

	//Add Force + Enforce Boundary Conditions
	addForceProgram(NULL, m_Params.grid_resolution, vel_old->tex, vel_cur->tex);
	std::swap(vel_cur, vel_old);

	//Compute Divergence
	MemSetSurface<float>(NULL, m_GridDivergenceTex, 0.0f, m_Params.grid_resolution);
	divergenceProgram(NULL, m_Params.grid_resolution, vel_old->tex, m_GridMarkerTex, m_GridDivergenceTex);

	//Jacobi Pressure Solver
	//gpuErrchk(cudaMemset(pres_old->arr, 0, m_gridSize * sizeof(float4)));
	MemSetSurface<float>(NULL, pres_old->sur, 0.0f, m_Params.grid_resolution);
	MemSetSurface<float>(NULL, pres_cur->sur, 0.0f, m_Params.grid_resolution);
	//initPressureGrid(NULL, m_Params.grid_resolution, pres_cur->sur, pres_old->sur);

	const uint JACOBI_ITERATIONS = 500;
	jacobiProgram(NULL, JACOBI_ITERATIONS, m_Params.grid_resolution, m_GridMarkerTex, m_GridDivergenceTex,
		pres_cur->tex, pres_cur->tex, pres_old->tex, pres_old->tex);

	if (JACOBI_ITERATIONS % 2 == 1) std::swap(pres_cur, pres_old);

	//Subtract Pressure Gradient from grid velocities
	subtractProgram(NULL, m_Params.grid_resolution, m_GridMarkerTex, vel_old->tex, pres_old->tex, vel_cur->tex);
	std::swap(vel_old, vel_cur);


	//Transfer Velocities back to Particles
	transferToParticlesProgram(NULL, m_NumParticles, m_ParticlePos, m_ParticleVel, m_ParticleVelTemp, vel_old->tex, m_GridVelOrigTex);
	std::swap(m_ParticleVel, m_ParticleVelTemp);


	//Update Positions (RK2)
	advectProgram(NULL, m_NumParticles, m_ParticlePos, m_ParticleVel, m_ParticlePosTemp, vel_old->tex);
	std::swap(m_ParticlePos, m_ParticlePosTemp);


	MemSetSurface<uint>(NULL, m_GridMapperStartTex, 0, m_Params.grid_resolution);
	MemSetSurface<uint>(NULL, m_GridMapperEndTex, 0, m_Params.grid_resolution);

	//Sort Particles + Build Grid Offsets
	sortByGridIndex(NULL, m_NumParticles, m_GridMapperStartTex, m_GridMapperEndTex, m_KeyValues, m_KeyValuesTemp, m_ParticlePos, m_ParticlePosTemp, m_ParticleVel, m_ParticleVelTemp);
	std::swap(m_ParticlePos, m_ParticlePosTemp);
	std::swap(m_ParticleVel, m_ParticleVelTemp);
	std::swap(m_KeyValues, m_KeyValuesTemp);
#if PICFLIP_PROFILE_EACH_KERNEL
	printf("\n\n");
#endif
}

void FluidPicFlip::enforceBoundaries()
{
	picflip::enforceBoundaries(NULL, m_NumParticles, m_ParticlePos);
}

void FluidPicFlip::allocate_buffers(const std::vector<Real>& fluid_positions, const std::vector<Real>& boundary_positions)
{
	Vector3 bbmin = Vector3(FLT_MAX, FLT_MAX, FLT_MAX);
	Vector3 bbmax = Vector3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	//Compute Bounding Box
	auto add_to_bb = [&](const std::vector<Real>& arr)
	{
		for (int i = 0; i < arr.size(); i += 3)
		{
			bbmin.x = min(bbmin.x, arr[i]);
			bbmin.y = min(bbmin.y, arr[i + 1]);
			bbmin.z = min(bbmin.z, arr[i + 2]);
			bbmax.x = max(bbmax.x, arr[i]);
			bbmax.y = max(bbmax.y, arr[i + 1]);
			bbmax.z = max(bbmax.z, arr[i + 2]);
		}
	};
	add_to_bb(fluid_positions);
	add_to_bb(boundary_positions);

	//bbmax.x = bbmax.y = bbmax.z = max(max(bbmax.x, bbmax.y), bbmax.z);
	//bbmin.x = bbmin.y = bbmin.z = min(min(bbmin.x, bbmin.y), bbmin.z);

	bbmax = bbmax + 0.001f;
	bbmin = bbmin - 0.001f;
	//bbmin.y -= 1.0f;

	Vector3 bbsize = bbmax - bbmin;

	uint max_grid_dim = m_Params.grid_resolution.x;
	m_Params.grid_resolution.y = round(max_grid_dim * (bbsize.y / bbsize.x));
	m_Params.grid_resolution.z = round(max_grid_dim * (bbsize.z / bbsize.x));




	Vector3 wrld2Grid_off = -bbmin;
	Vector3 wrld2Grid = Vector3(m_Params.grid_resolution.x, m_Params.grid_resolution.y, m_Params.grid_resolution.z) / bbsize;


	memcpy(&m_Params.world_to_grid, &wrld2Grid, sizeof(float3));
	memcpy(&m_Params.world_to_grid_offset, &wrld2Grid_off, sizeof(float3));

	m_NumParticles = fluid_positions.size() / 3;
	alloc_array((void**)&m_ParticlePos, m_NumParticles * sizeof(float3));
	alloc_array((void**)&m_ParticlePosTemp, m_NumParticles * sizeof(float3));
	alloc_array((void**)&m_ParticleVel, m_NumParticles * sizeof(float3));
	alloc_array((void**)&m_ParticleVelTemp, m_NumParticles * sizeof(float3));
	//alloc_array((void**)&m_ParticleRnd, m_NumParticles * sizeof(float3));


	copy_array(m_ParticlePos, &fluid_positions[0], m_NumParticles * sizeof(float3), 1);
	gpuErrchk(cudaMemset(m_ParticlePosTemp, 0, m_NumParticles * sizeof(float3)));
	gpuErrchk(cudaMemset(m_ParticleVel, 0, m_NumParticles * sizeof(float3)));
	gpuErrchk(cudaMemset(m_ParticleVelTemp, 0, m_NumParticles * sizeof(float3)));
	//gpuErrchk(cudaMemset(m_ParticleRnd, 0, m_NumParticles * sizeof(float3)));


	/*for (int i = 0; i < fluid_positions.size(); i += 3)
	{
	Vector3 pos;
	pos.x = fluid_positions[i];
	pos.y = fluid_positions[i+1];
	pos.z = fluid_positions[i+2];

	Vector3 grid_pos = (pos + wrld2Grid_off) * wrld2Grid;
	printf("Grid: %5.2f %5.2f %5.2f\n", grid_pos.x, grid_pos.y, grid_pos.z);
	}*/

	alloc_array((void**)&m_KeyValues, m_NumParticles * sizeof(KeyValuePair));
	alloc_array((void**)&m_KeyValuesTemp, m_NumParticles * sizeof(KeyValuePair));
}

void FluidPicFlip::update_deduced_attributes()
{

}

void FluidPicFlip::set_ogl_vbo(GLuint pos, GLuint vel, GLuint dens, GLuint pres)
{
	cudaGraphicsGLRegisterBuffer(&vbo_pos, pos, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&vbo_vel, vel, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&vbo_dens, dens, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&vbo_pres, pres, cudaGraphicsMapFlagsWriteDiscard);
}
void FluidPicFlip::copy_arrays_to_ogl_vbos()
{
	copy_to_ogl_buffer(&vbo_pos, m_ParticlePos, m_NumParticles * sizeof(float3));
	copy_to_ogl_buffer(&vbo_vel, m_ParticleVel, m_NumParticles * sizeof(float3));

	if (m_GridVel != NULL)
	{
		void* cuda_vbo_arr;
		uint num_bytes;

		gpuErrchk(cudaGraphicsMapResources(1, &vbo_dens, 0));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&cuda_vbo_arr, &num_bytes, vbo_dens));
		copyPressures(NULL, m_NumParticles, m_ParticlePos, m_GridVelTex, (float*)cuda_vbo_arr);
		gpuErrchk(cudaGraphicsUnmapResources(1, &vbo_dens, 0));
	}
}

bool FluidPicFlip::BuildGridBuffers()
{
	uint new_grid_size = m_Params.grid_resolution.x * m_Params.grid_resolution.y * m_Params.grid_resolution.z;
	if (new_grid_size != m_gridSize)
	{
		m_gridSize = new_grid_size;

		if (m_GridVel != NULL)
		{
			gpuErrchk(cudaFreeArray(m_GridVel));
			gpuErrchk(cudaFreeArray(m_GridVelTemp));
			gpuErrchk(cudaFreeArray(m_GridVelOrig));
			gpuErrchk(cudaFreeArray(m_GridMarker));
			gpuErrchk(cudaFreeArray(m_GridDivergence));
			gpuErrchk(cudaFreeArray(m_GridPressure));
			gpuErrchk(cudaFreeArray(m_GridPressureTemp));
			gpuErrchk(cudaFreeArray(m_GridMapperStart));
			gpuErrchk(cudaFreeArray(m_GridMapperEnd));
			m_GridVel = NULL;
		}


		cudaChannelFormatDesc channelDesc1F = cudaCreateChannelDesc<float>();			// cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
		cudaChannelFormatDesc channelDesc4F = cudaCreateChannelDesc<float4>();			// cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
		cudaChannelFormatDesc channelDesc1UC = cudaCreateChannelDesc<unsigned char>();	// cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
		cudaChannelFormatDesc channelDesc1UI = cudaCreateChannelDesc<uint>();			// cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

		cudaExtent ce_grid;
		ce_grid.width = m_Params.grid_resolution.x;
		ce_grid.height = m_Params.grid_resolution.y;
		ce_grid.depth = m_Params.grid_resolution.z;

		cudaExtent ce_vel;
		ce_vel.width = m_Params.grid_resolution.x + 1;
		ce_vel.height = m_Params.grid_resolution.y + 1;
		ce_vel.depth = m_Params.grid_resolution.z + 1;


		auto init_tex = [&](
			cudaSurfaceObject_t* surref,
			cudaTextureObject_t* texref,
			cudaArray_t* arr,
			cudaChannelFormatDesc& desc,
			cudaExtent ce,
			cudaTextureFilterMode filter)
		{
			gpuErrchk(cudaMalloc3DArray(arr, &desc, ce, cudaArraySurfaceLoadStore));

			cudaResourceDesc sur_desc;
			memset(&sur_desc, 0, sizeof(cudaResourceDesc));
			sur_desc.resType = cudaResourceTypeArray;
			sur_desc.res.array.array = *arr;

			cudaTextureDesc             texDescr;
			memset(&texDescr, 0, sizeof(cudaTextureDesc));
			texDescr.normalizedCoords = 0;
			texDescr.filterMode = cudaFilterModePoint;
			texDescr.mipmapFilterMode = cudaFilterModePoint;
			texDescr.addressMode[0] = cudaAddressModeClamp;
			texDescr.addressMode[1] = cudaAddressModeClamp;
			texDescr.addressMode[2] = cudaAddressModeClamp;
			texDescr.readMode = cudaReadModeElementType;

			gpuErrchk(cudaCreateTextureObject(texref, &sur_desc, &texDescr, NULL));
			gpuErrchk(cudaCreateSurfaceObject(surref, &sur_desc));
		};

		init_tex(&m_GridVelSur, &m_GridVelTex, &m_GridVel, channelDesc4F, ce_vel, cudaFilterModeLinear);
		init_tex(&m_GridVelTempSur, &m_GridVelTempTex, &m_GridVelTemp, channelDesc4F, ce_vel, cudaFilterModeLinear);
		init_tex(&m_GridVelOrigSur, &m_GridVelOrigTex, &m_GridVelOrig, channelDesc4F, ce_vel, cudaFilterModeLinear);

		init_tex(&m_GridMarkerSur, &m_GridMarkerTex, &m_GridMarker, channelDesc1UC, ce_grid, cudaFilterModePoint);
		init_tex(&m_GridDivergenceSur, &m_GridDivergenceTex, &m_GridDivergence, channelDesc1F, ce_grid, cudaFilterModeLinear);
		init_tex(&m_GridPressureSur, &m_GridPressureTex, &m_GridPressure, channelDesc1F, ce_grid, cudaFilterModeLinear);
		init_tex(&m_GridPressureTempSur, &m_GridPressureTempTex, &m_GridPressureTemp, channelDesc1F, ce_grid, cudaFilterModeLinear);

		init_tex(&m_GridMapperStartSur, &m_GridMapperStartTex, &m_GridMapperStart, channelDesc1UI, ce_grid, cudaFilterModeLinear);
		init_tex(&m_GridMapperEndSur, &m_GridMapperEndTex, &m_GridMapperEnd, channelDesc1UI, ce_grid, cudaFilterModeLinear);
	
		return true;
	}

	return false;
}