#include "FluidClothCoupler.cuh"
#include <cuda_runtime.h>
#include <libsim\Cuda_Utils.cuh>
#include <libsim\Fluid_Kernel_Utils.cuh>
#include <libsim\radixsort.cuh>

#define COUPLER_FORCE_SYNC_AFTER_EACH_KERNEL FALSE
#define COUPLER_PROFILE_EACH_KERNEL TRUE

#if COUPLER_PROFILE_EACH_KERNEL
cudaEvent_t pfprofile_start = NULL, pfprofile_stop = NULL;
#define COUPLER_PROFILE_BEGIN_KERNEL cudaEventRecord(pfprofile_start);
#define COUPLER_PROFILE_END_KERNEL(description, identifier) { cudaEventRecord(pfprofile_stop); \
	cudaEventSynchronize(pfprofile_stop); \
	float milliseconds = 0; \
	cudaEventElapsedTime(&milliseconds, pfprofile_start, pfprofile_stop); \
	printf("\tKernel Timing: %5.2fms (%s -> %d)\n", milliseconds, description, identifier); }
#else
#define COUPLER_PROFILE_BEGIN_KERNEL
#define COUPLER_PROFILE_END_KERNEL(description, identifier)
#endif



__constant__ FluidClothCoupler::Params pfParams;

template <class T>
__device__
void writeTex(cudaSurfaceObject_t surface, const T& data, int x, int y, int z)
{
	surf3Dwrite(data, surface, (x)* sizeof(T), y, z);
}

template <class T>
__device__
T readTexNearest(cudaTextureObject_t texture, float xs, float ys, float zs)
{
	return tex3D<T>(texture,
		(xs + 0.5f),
		(ys + 0.5f),
		(zs + 0.5f));
}

template <class T>
__global__ void Kernel_MemSetSurface(cudaSurfaceObject_t out_grid, T val, uint step, uint3 grid_size)
{
	/*int dimxy = grid_size.x * grid_size.y;
	dim3 idx = thread3d(grid_size.x, dimxy);

	if (idx.x >= grid_size.x || idx.y >= grid_size.y || idx.z >= grid_size.z)
	return;*/

	uint index_x = blockIdx.x*blockDim.x + threadIdx.x;
	if (index_x >= grid_size.x)
		return;
	uint index_y = blockIdx.y*blockDim.y + threadIdx.y;
	if (index_y >= grid_size.y)
		return;
	uint index_z = blockIdx.z*blockDim.z + threadIdx.z;
	if (index_z >= grid_size.z)
		return;

	surf3Dwrite(val, out_grid, index_x * step, index_y, index_z);
}

template <class T>
void MemSetSurface(cudaStream_t stream, cudaSurfaceObject_t out_grid, T val, uint3 grid_size)
{
	dim3 num_threads;
	dim3 num_blocks;
	utils::compute_grid_size(grid_size.x, CUDA_BLOCK_SIZE_3D, num_blocks.x, num_threads.x);
	utils::compute_grid_size(grid_size.y, CUDA_BLOCK_SIZE_3D, num_blocks.y, num_threads.y);
	utils::compute_grid_size(grid_size.z, CUDA_BLOCK_SIZE_3D, num_blocks.z, num_threads.z);

	COUPLER_PROFILE_BEGIN_KERNEL
		Kernel_MemSetSurface<T> << <num_blocks, num_threads, 0, stream >> >(
		out_grid,
		val,
		sizeof(val),
		grid_size);
	COUPLER_PROFILE_END_KERNEL("Kernel_MemSetSurface<T>", grid_size.x * grid_size.y * grid_size.z)

#if COUPLER_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}


__global__
void pfkernel_sort_initialize_keyvalues(uint particle_count, KeyValuePair* particle_keyvalues, float4* particle_positions)
{
	uint index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index >= particle_count)
	{
		return;
	}

	float4 cloth_pos = particle_positions[index];

	int3 cp;
	cp.x = floor((cloth_pos.x + pfParams.world_to_grid_offset.x) * pfParams.world_to_grid.x);
	cp.y = floor((cloth_pos.y + pfParams.world_to_grid_offset.y) * pfParams.world_to_grid.y);
	cp.z = floor((cloth_pos.z + pfParams.world_to_grid_offset.z) * pfParams.world_to_grid.z);

	uint hash;
	if (cp.x < 0 || cp.x >= pfParams.grid_resolution.x ||
		cp.y < 0 || cp.y >= pfParams.grid_resolution.y ||
		cp.z < 0 || cp.z >= pfParams.grid_resolution.z)
	{
		hash = 0xFFFFFFFF;
	}
	else
	{
		hash = (cp.z * pfParams.grid_resolution.y + cp.y) * pfParams.grid_resolution.x + cp.x;
	}

	particle_keyvalues[index].key = hash;
	particle_keyvalues[index].value = index;
}

__global__
void pfkernel_sort_reorder_and_insert_boundary_offsets(
	uint particle_count,
	cudaSurfaceObject_t particles_start, cudaSurfaceObject_t particles_end,
	KeyValuePair* boundary_sort_pair,
	float4* in_positions, float4* out_positions,
	float4* in_positionsold, float4* out_positionsold,
	uint* lookups)
{
	uint index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index >= particle_count)
	{
		return;
	}

	KeyValuePair sort_pair = boundary_sort_pair[index];

	//Load src position/velocity
	lookups[sort_pair.value] = index;
	out_positions[index] = in_positions[sort_pair.value];
	out_positionsold[index] = in_positionsold[sort_pair.value];


	//Calculate Offset
	if (sort_pair.key < 0xFFFFFFFF)
	{
		uint grid_xy = pfParams.grid_resolution.x * pfParams.grid_resolution.y;
		uint3 cell_pos = make_uint3(
			sort_pair.key % pfParams.grid_resolution.x,
			(sort_pair.key % grid_xy) / pfParams.grid_resolution.x,
			sort_pair.key / grid_xy
			);


		// -> key != prev_key => cell_start
		if (index == 0 || sort_pair.key != boundary_sort_pair[index - 1].key)
		{
			//cell_offsets[sort_pair.key].x = index;
			writeTex<uint>(particles_start, index, cell_pos.x, cell_pos.y, cell_pos.z);
		}

		// -> key != next_key => cell_end
		if (index == particle_count - 1 || sort_pair.key != boundary_sort_pair[index + 1].key)
		{
			//cell_offsets[sort_pair.key].y = index + 1;
			writeTex<uint>(particles_end, index + 1, cell_pos.x, cell_pos.y, cell_pos.z);
		}
	}
}


__global__
void kernel_unsort_cloth_particles(
	uint particle_count,
	uint* lookups,
	float4* in_positions, float4* out_positions,
	float4* in_positionsold, float4* out_positionsold)
{
	uint index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index >= particle_count)
	{
		return;
	}

	uint oidx = lookups[index];

	//Load src position/velocity
	out_positions[index] = in_positions[oidx];
	out_positionsold[index] = in_positionsold[oidx];
}

void unsortByGridIndex(cudaStream_t stream,
	uint particle_count,
	uint* lookups,
	float4* positions,
	float4* positions_tmp,
	float4* positions_old,
	float4* positions_old_tmp)
{
	if (particle_count == 0)
		return;

	uint num_threads;
	uint num_blocks;

	utils::compute_grid_size(particle_count, CUDA_BLOCK_SIZE, num_blocks, num_threads);

	//Reorder and insert boundary offsets
	COUPLER_PROFILE_BEGIN_KERNEL
		kernel_unsort_cloth_particles << <num_blocks, num_threads, 0, stream >> >(
		particle_count,
		lookups,
		positions, positions_tmp,
		positions_old, positions_old_tmp);
	COUPLER_PROFILE_END_KERNEL("kernel_unsort_cloth_particles", particle_count)

#if COUPLER_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

}

void sortByGridIndex(cudaStream_t stream,
	uint particle_count,
	cudaSurfaceObject_t particles_start, cudaSurfaceObject_t particles_end,
	KeyValuePair* keyvalues,
	KeyValuePair* keyvalues_tmp,
	float4* positions,
	float4* positions_tmp,
	float4* positions_old,
	float4* positions_old_tmp,
	uint* lookups)
{
	if (particle_count == 0)
		return;

	uint num_threads;
	uint num_blocks;

	utils::compute_grid_size(particle_count, CUDA_BLOCK_SIZE, num_blocks, num_threads);

	//Create Cell Indexes
	COUPLER_PROFILE_BEGIN_KERNEL
		pfkernel_sort_initialize_keyvalues << <num_blocks, num_threads, 0, stream >> >(particle_count, keyvalues, positions);
	COUPLER_PROFILE_END_KERNEL("kernel_sort_initialize_keyvalues", particle_count)

#if COUPLER_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	//Sort CellIndexes
	COUPLER_PROFILE_BEGIN_KERNEL
		RadixSort(keyvalues, keyvalues_tmp, particle_count, 32, stream);
	COUPLER_PROFILE_END_KERNEL("RadixSort", particle_count)

#if COUPLER_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	//Reorder and insert boundary offsets
	COUPLER_PROFILE_BEGIN_KERNEL
		pfkernel_sort_reorder_and_insert_boundary_offsets << <num_blocks, num_threads, 0, stream >> >(particle_count,
		particles_start, particles_end,
		keyvalues_tmp,
		positions, positions_tmp,
		positions_old, positions_old_tmp,
		lookups);
	COUPLER_PROFILE_END_KERNEL("kernel_sort_reorder_and_insert_boundary_offsets", particle_count)

#if COUPLER_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

}

__device__
float3 tof3(const float4& f4)
{
	return make_float3(f4.x, f4.y, f4.z);
}

__device__
void CollideParticles(
	uint fluid_idx, float3* fluid_positions, float3* fluid_velocities,
	uint cloth_idx, float4* cloth_positions, float4* cloth_positions_old, float4* cloth_normals)
{
	
	float4 cp = cloth_positions[cloth_idx];
	float3 fp = fluid_positions[fluid_idx];

	float3 dir = make_float3(cp.x - fp.x, cp.y - fp.y, cp.z - fp.z);
	//float3 dir = make_float3(fp.x - cp.x, fp.y - cp.y, fp.z - cp.z);
	float dist = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;

	const float rsq = pfParams.particle_size * pfParams.particle_size * 4.0f;

	if (dist < rsq)
	{
		//DO COLLISION RESPONSE
		//dist = sqrtf(dist);
		//dir = dir / dist;


		float3 normal = tof3(cloth_normals[cloth_idx]);
		if (float3_dot(dir, normal) < 0.0f)
			normal *= -1.0f;


		/*float3 dv = 

		dist -= pfParams.particle_size;

		float w = cp.w + cp.w;// pfParams.particle_iweight;
		dir *= (dist / w);


		cp.x += dir.x * cp.w;
		cp.y += dir.y * cp.w;
		cp.z += dir.z * cp.w;

		fp -= dir * cp.w;// pfParams.particle_iweight;*/

		float4 cpo = cloth_positions_old[cloth_idx];
		float3 v0 = (tof3(cp) - tof3(cpo)) * pfParams.idt;
		float3 v1 = fluid_velocities[fluid_idx];

		float w0 = cp.w;
		float w1 = pfParams.particle_iweight;
		float iconstraintMass = 1.0f / (w0 + w1);

		float3 dv = v0 - v1;
		if (float3_dot(dv, normal) < 0.0f)
		{
			//Collision Resolution
			float jn = -(1.0f * float3_dot(dv, normal)) * iconstraintMass;

			v0 += normal * (jn * w0);
			v1 -= normal * (jn * w1);
						
			fluid_velocities[fluid_idx] = v1;
			cloth_positions_old[cloth_idx] = cpo;
		}

		float pen = pfParams.particle_size * 2.0f - (float3_dot(make_float3(cp.x, cp.y, cp.z), normal) - float3_dot(fp, normal));
		pen *= iconstraintMass;

		cp.x += normal.x * pen * w0;
		cp.y += normal.y * pen * w0;
		cp.z += normal.z * pen * w0;

		cpo.x = cp.x - v0.x * pfParams.dt;
		cpo.y = cp.y - v0.y * pfParams.dt;
		cpo.z = cp.z - v0.z * pfParams.dt;

		fluid_positions[fluid_idx] = fp - normal * (pen * w1);
		cloth_positions[cloth_idx] = cp;
		cloth_positions_old[cloth_idx] = cpo;
	}
}

//Each Cell searches left and itself (forming a safe concurrent collision handling in only 8 batches)
__global__
void kernel_collideParticles(
	int3 offset,
	cudaSurfaceObject_t fluid_start, cudaSurfaceObject_t fluid_end,
	cudaSurfaceObject_t cloth_start, cudaSurfaceObject_t cloth_end,
	float3* fluid_positions, float3* fluid_velocities,
	float4* cloth_positions,
	float4* cloth_positions_old,
	float4* cloth_normals)
{
	int3 cell_pos = make_int3(
		(blockIdx.x*blockDim.x + threadIdx.x),
		(blockIdx.y*blockDim.y + threadIdx.y),
		(blockIdx.z*blockDim.z + threadIdx.z)
		);

	if (cell_pos.x >= pfParams.grid_resolution.x
		&& cell_pos.y >= pfParams.grid_resolution.y
		&& cell_pos.z >= pfParams.grid_resolution.z)
	return;


	//Get Current Cell
	uint cstart = readTexNearest<uint>(cloth_start, cell_pos.x, cell_pos.y, cell_pos.z);
	uint cend = readTexNearest<uint>(cloth_end, cell_pos.x, cell_pos.y, cell_pos.z);

	//Get Search Cell
	int3 fcell_pos = offset + cell_pos;
	uint fstart = readTexNearest<uint>(fluid_start, fcell_pos.x, fcell_pos.y, fcell_pos.z);
	uint fend = readTexNearest<uint>(fluid_end, fcell_pos.x, fcell_pos.y, fcell_pos.z);

	//Collide with all fluid particles in cell marked with offset
	//if (fcell_pos.x == cell_pos.x && fcell_pos.y == cell_pos.y && fcell_pos.z == cell_pos.z)
	//	printf("Cloth Cell Pos: %d, %d, %d - %d / %d\n", cell_pos.x, cell_pos.y, cell_pos.z, cend - cstart, fend - fstart);

	for (uint j = cstart; j < cend; j++)
	{
		for (uint i = fstart; i < fend; i++)
		{
			CollideParticles(i, fluid_positions, fluid_velocities, j, cloth_positions, cloth_positions_old, cloth_normals);
		}
	}
}

//Collides cloth particles against all fluid particles in that (or neighbouring) cells
void collideParticles(cudaStream_t stream,
	dim3 grid_dims,
	cudaSurfaceObject_t fluid_start, cudaSurfaceObject_t fluid_end,
	cudaSurfaceObject_t cloth_start, cudaSurfaceObject_t cloth_end,
	float3* fluid_positions, float3* fluid_velocities,
	float4* cloth_positions,
	float4* cloth_positions_old,
	float4* cloth_normals)
{
	if (grid_dims.x == 0 || grid_dims.y == 0 || grid_dims.z == 0)
		return;

	dim3 num_threads;
	dim3 num_blocks;

	utils::compute_grid_size(grid_dims.x, CUDA_BLOCK_SIZE_3D, num_blocks.x, num_threads.x);
	utils::compute_grid_size(grid_dims.y, CUDA_BLOCK_SIZE_3D, num_blocks.y, num_threads.y);
	utils::compute_grid_size(grid_dims.z, CUDA_BLOCK_SIZE_3D, num_blocks.z, num_threads.z);

	//Create Cell Indexes
	COUPLER_PROFILE_BEGIN_KERNEL
	int3 offset;
	for (offset.x = -1; offset.x <= 1; offset.x++)
	{
		for (offset.y = -1; offset.y <= 1; offset.y++)
		{
			for (offset.z = -1; offset.z <= 1; offset.z++)
			{
				kernel_collideParticles<< <num_blocks, num_threads, 0, stream >> >(
					offset,
					fluid_start, fluid_end,
					cloth_start, cloth_end,
					fluid_positions, fluid_velocities,
					cloth_positions,
					cloth_positions_old,
					cloth_normals);
			}
		}
	}
	COUPLER_PROFILE_END_KERNEL("kernel_collideParticles", grid_dims.x * grid_dims.y * grid_dims.z)

#if COUPLER_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}





FluidClothCoupler::FluidClothCoupler(XPBD* cloth, FluidPicFlip* fluid)
	: m_Cloth(cloth)
	, m_Fluid(fluid)
	, m_cuClothParticlesNew(NULL)
	, m_cuClothParticlesNewTmp(NULL)
	, m_cuClothParticlesOld(NULL)
	, m_cuClothParticlesOldTmp(NULL)
	, m_cuKeyValuePairs(NULL)
	, m_cuKeyValuePairsTmp(NULL)
	, m_cuClothLookups(NULL)
{
	m_Params.grid_resolution = m_Fluid->m_Params.grid_resolution;
	m_Params.world_to_grid = m_Fluid->m_Params.world_to_grid;
	m_Params.world_to_grid_offset = m_Fluid->m_Params.world_to_grid_offset;
	m_Params.particle_size = 0.0225f * 2.0f;
	m_Params.particle_iweight = 1.0f / 0.0001f;


	AllocateMemory();
}


FluidClothCoupler::~FluidClothCoupler()
{
	m_Cloth = NULL;
	m_Fluid = NULL;

	if (m_cuClothParticlesNew != NULL)
	{
		gpuErrchk(cudaFree(m_cuClothParticlesNew));
		gpuErrchk(cudaFree(m_cuClothParticlesNewTmp));
		gpuErrchk(cudaFree(m_cuClothParticlesOld));
		gpuErrchk(cudaFree(m_cuClothParticlesOldTmp));
		gpuErrchk(cudaFree(m_cuClothNormals));
		gpuErrchk(cudaFree(m_cuKeyValuePairs));
		gpuErrchk(cudaFree(m_cuKeyValuePairsTmp));
		gpuErrchk(cudaFree(m_cuClothLookups));
		m_cuClothParticlesNew = NULL;
	}

	if (m_cuClothGridStart.mem != NULL)
	{
		gpuErrchk(cudaFreeArray(m_cuClothGridStart.mem));
		gpuErrchk(cudaFreeArray(m_cuClothGridEnd.mem));
		m_cuClothGridStart.mem = NULL;
	}
}

void FluidClothCoupler::HandleCollisions(float dt)
{
	m_Params.dt = dt;
	m_Params.idt = 1.0f / dt;

	//Update Constant Memory
	Params* dParamsArr;
	gpuErrchk(cudaGetSymbolAddress((void **)&dParamsArr, pfParams));
	gpuErrchk(cudaMemcpy(dParamsArr, &m_Params, sizeof(Params), cudaMemcpyHostToDevice));

	//Copy Cloth Particles to linear Array
	gpuErrchk(cudaMemcpyFromArray(m_cuClothParticlesNew, m_Cloth->m_cuParticlePos.mem, 0, 0, m_Cloth->m_NumParticles * sizeof(float4), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpyFromArray(m_cuClothParticlesOld, m_Cloth->m_cuParticlePosOld.mem, 0, 0, m_Cloth->m_NumParticles * sizeof(float4), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpyFromArray(m_cuClothNormals, m_Cloth->m_cuParticleVertNormals.mem, 0, 0, m_Cloth->m_NumParticles * sizeof(float4), cudaMemcpyDeviceToDevice));

	MemSetSurface<uint>(NULL, m_cuClothGridStart.tex, 0, m_Params.grid_resolution);
	MemSetSurface<uint>(NULL, m_cuClothGridEnd.tex, 0, m_Params.grid_resolution);

	//Sort Cloth Particles into fluid 'Grid Cells' 
	sortByGridIndex(NULL,
		m_Cloth->m_NumParticles,
		m_cuClothGridStart.tex,
		m_cuClothGridEnd.tex,
		m_cuKeyValuePairs,
		m_cuKeyValuePairsTmp,
		m_cuClothParticlesNew,
		m_cuClothParticlesNewTmp,
		m_cuClothParticlesOld,
		m_cuClothParticlesOldTmp,
		m_cuClothLookups);
	std::swap(m_cuClothParticlesNew, m_cuClothParticlesNewTmp);
	std::swap(m_cuClothParticlesOld, m_cuClothParticlesOldTmp);
	std::swap(m_cuKeyValuePairs, m_cuKeyValuePairsTmp);


	//Collide Cloth Particles against fluid particles
	collideParticles(
		NULL,
		m_Params.grid_resolution,
		m_Fluid->m_GridMapperStartTex,
		m_Fluid->m_GridMapperEndTex,
		m_cuClothGridStart.tex,
		m_cuClothGridEnd.tex,
		m_Fluid->m_ParticlePos,
		m_Fluid->m_ParticleVel,
		m_cuClothParticlesNew,
		m_cuClothParticlesOld,
		m_cuClothNormals);


	//Copy Cloth Positions Back To Cloth
	unsortByGridIndex(NULL, m_Cloth->m_NumParticles, m_cuClothLookups, m_cuClothParticlesNew, m_cuClothParticlesNewTmp, m_cuClothParticlesOld, m_cuClothParticlesOldTmp);

	gpuErrchk(cudaMemcpyToArray(m_Cloth->m_cuParticlePos.mem, 0, 0, m_cuClothParticlesNewTmp, m_Cloth->m_NumParticles * sizeof(float4), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpyToArray(m_Cloth->m_cuParticlePosOld.mem, 0, 0, m_cuClothParticlesOldTmp, m_Cloth->m_NumParticles * sizeof(float4), cudaMemcpyDeviceToDevice));


	//Be Happy
}

void FluidClothCoupler::AllocateMemory()
{
	size_t size = m_Cloth->m_NumParticles * sizeof(float4);
	size_t size_kv = m_Cloth->m_NumParticles * sizeof(KeyValuePair);

	gpuErrchk(cudaMalloc((void**)&m_cuClothParticlesNew, size));
	gpuErrchk(cudaMalloc((void**)&m_cuClothParticlesNewTmp, size));
	gpuErrchk(cudaMalloc((void**)&m_cuClothParticlesOld, size));
	gpuErrchk(cudaMalloc((void**)&m_cuClothParticlesOldTmp, size));
	gpuErrchk(cudaMalloc((void**)&m_cuClothNormals, size));
	gpuErrchk(cudaMalloc((void**)&m_cuKeyValuePairs, size_kv));
	gpuErrchk(cudaMalloc((void**)&m_cuKeyValuePairsTmp, size_kv));
	gpuErrchk(cudaMalloc((void**)&m_cuClothLookups, m_Cloth->m_NumParticles * sizeof(uint)));

	gpuErrchk(cudaMemset(m_cuClothParticlesNew, 0, size));
	gpuErrchk(cudaMemset(m_cuClothParticlesNewTmp, 0, size));
	gpuErrchk(cudaMemset(m_cuClothParticlesOld, 0, size));
	gpuErrchk(cudaMemset(m_cuClothParticlesOldTmp, 0, size));
	gpuErrchk(cudaMemset(m_cuClothNormals, 0, size));
	gpuErrchk(cudaMemset(m_cuKeyValuePairs, 0, size_kv));
	gpuErrchk(cudaMemset(m_cuKeyValuePairsTmp, 0, size_kv));
	gpuErrchk(cudaMemset(m_cuClothLookups, 0, m_Cloth->m_NumParticles * sizeof(uint)));

	cudaChannelFormatDesc channelDesc1UI = cudaCreateChannelDesc<uint>();
	cudaExtent ce;
	ce.width = m_Fluid->m_Params.grid_resolution.x;
	ce.height = m_Fluid->m_Params.grid_resolution.y;
	ce.depth = m_Fluid->m_Params.grid_resolution.z;

	m_cuClothGridStart.allocate3D(channelDesc1UI, ce);
	m_cuClothGridEnd.allocate3D(channelDesc1UI, ce);

#if COUPLER_PROFILE_EACH_KERNEL
	cudaEventCreate(&pfprofile_start);
	cudaEventCreate(&pfprofile_stop);
#endif
}