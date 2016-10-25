#include "PicFlip_Kernel.cuh"
#include "Fluid_Kernel_Utils.cuh"
#include "radixsort.cuh"

using namespace utils;

#define PICFLIP_PROFILE_EACH_KERNEL		FALSE
#define XFER_USE_TRPLE_CUDA_DIM			FALSE
#define PICFLIP_FORCE_SYNC_AFTER_EACH_KERNEL FALSE

__constant__ PicFlip_Params dParams;


#if PICFLIP_PROFILE_EACH_KERNEL
cudaEvent_t pfprofile_start = NULL, pfprofile_stop = NULL;
#define PICFLIP_PROFILE_BEGIN_KERNEL cudaEventRecord(pfprofile_start);
#define PICFLIP_PROFILE_END_KERNEL(description, identifier) { cudaEventRecord(pfprofile_stop); \
	cudaEventSynchronize(pfprofile_stop); \
	float milliseconds = 0; \
	cudaEventElapsedTime(&milliseconds, pfprofile_start, pfprofile_stop); \
	printf("\tKernel Timing: %5.2fms (%s -> %d)\n", milliseconds, description, identifier); }
#else
#define PICFLIP_PROFILE_BEGIN_KERNEL
#define PICFLIP_PROFILE_END_KERNEL(description, identifier)
#endif



#if XFER_USE_TRPLE_CUDA_DIM 
__device__
bool GetCellPos(int3& cell_pos)
{
	cell_pos = make_int3(
		blockIdx.x*blockDim.x + threadIdx.x,
		blockIdx.y*blockDim.y + threadIdx.y,
		blockIdx.z*blockDim.z + threadIdx.z
		);

	return (cell_pos.x < dParams.grid_resolution.x
		&& cell_pos.y < dParams.grid_resolution.y
		&& cell_pos.z < dParams.grid_resolution.z);
}
__device__
bool GetCellPosVel(int3& cell_pos)
{
	cell_pos = make_int3(
		blockIdx.x*blockDim.x + threadIdx.x,
		blockIdx.y*blockDim.y + threadIdx.y,
		blockIdx.z*blockDim.z + threadIdx.z
		);

	return (cell_pos.x <= dParams.grid_resolution.x
		&& cell_pos.y <= dParams.grid_resolution.y
		&& cell_pos.z <= dParams.grid_resolution.z);
}
#else
__device__
bool GetCellPos(int3& cell_pos)
{
	int idx = blockIdx.y*blockDim.y + threadIdx.y;
	cell_pos = make_int3(
		blockIdx.x*blockDim.x + threadIdx.x,
		idx % dParams.grid_resolution.y,
		idx / dParams.grid_resolution.y
		);

	return (cell_pos.x < dParams.grid_resolution.x
		&& cell_pos.y < dParams.grid_resolution.y
		&& cell_pos.z < dParams.grid_resolution.z);
}

__device__
bool GetCellPosVel(int3& cell_pos)
{
	int idx = blockIdx.y*blockDim.y + threadIdx.y;
	cell_pos = make_int3(
		blockIdx.x*blockDim.x + threadIdx.x,
		idx % (dParams.grid_resolution.y + 1),
		idx / (dParams.grid_resolution.y + 1)
		);

	return (cell_pos.x <= dParams.grid_resolution.x
		&& cell_pos.y <= dParams.grid_resolution.y
		&& cell_pos.z <= dParams.grid_resolution.z);
}
#endif




void picflip::set_parameters(PicFlip_Params *hParam)
{
#if PICFLIP_PROFILE_EACH_KERNEL
	if (pfprofile_start == NULL)
	{
		cudaEventCreate(&pfprofile_start);
		cudaEventCreate(&pfprofile_stop);
	}
#endif

	PicFlip_Params* dParamsArr;
	//Copy Paramaters to device
	gpuErrchk(cudaGetSymbolAddress((void **)&dParamsArr, dParams));
	gpuErrchk(cudaMemcpy(dParamsArr, hParam, sizeof(PicFlip_Params), cudaMemcpyHostToDevice));
	//gpuErrchk(cudaMemcpyToSymbol(dParamsArr, hParam, sizeof(SimulationParams)));


#if PICFLIP_FORCE_SYNC_AFTER_EACH_KERNEL
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}


template <class T>
__device__
void writeTex(cudaSurfaceObject_t surface, const T& data, int x, int y, int z)
{
	surf3Dwrite(data, surface, (x)* sizeof(T), y, z);
}

template <class T>
__device__
void writeTexVel(cudaSurfaceObject_t surface, const T& data, int x, int y, int z)
{
	surf3Dwrite(data, surface, (x)* sizeof(T), y, z);
}

template <class T>
__device__
T readTexInterpolate(cudaTextureObject_t texture, float xs, float ys, float zs)
{
	/*return tex3D<T>(texture,
	(xs + 0.5f),
	(ys + 0.5f),
	(zs + 0.5f));*/

	//ys = dParams.grid_resolution.y - 1 - ys;

	float x = floor(xs);
	float y = floor(ys);
	float z = floor(zs);

	float fx = xs - x;
	float fy = ys - y;
	float fz = zs - z;

	T ftl = tex3D<T>(texture, x + 0.5f, y + 0.5f, z + 0.5f);
	T ftr = tex3D<T>(texture, x + 0.5f + 1.0f, y + 0.5f, z + 0.5f);
	T fbl = tex3D<T>(texture, x + 0.5f, y + 0.5f + 1.0f, z + 0.5f);
	T fbr = tex3D<T>(texture, x + 0.5f + 1.0f, y + 0.5f + 1.0f, z + 0.5f);

	T btl = tex3D<T>(texture, x + 0.5f, y + 0.5f, z + 0.5f + 1.0f);
	T btr = tex3D<T>(texture, x + 0.5f + 1.0f, y + 0.5f, z + 0.5f + 1.0f);
	T bbl = tex3D<T>(texture, x + 0.5f, y + 0.5f + 1.0f, z + 0.5f + 1.0f);
	T bbr = tex3D<T>(texture, x + 0.5f + 1.0f, y + 0.5f + 1.0f, z + 0.5f + 1.0f);


	ftl = ftl * (1.0f - fx) + ftr * fx;
	fbl = fbl * (1.0f - fx) + fbr * fx;
	btl = btl * (1.0f - fx) + btr * fx;
	bbl = btl * (1.0f - fx) + bbl * fx;

	ftl = ftl * (1.0f - fy) + fbl * fy;
	btl = btl * (1.0f - fy) + bbl * fy;

	return ftl * (1.0f - fz) + btl * fz;
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

__device__
float h(const float& r) {
	return fmaxf(1.0 - fabsf(r), 0.0);
}

__device__
float k(const float3& v) {
	return h(v.x) * h(v.y) * h(v.z);
}

__device__
float kx(const float3& v) {
	volatile float half = 0.5f;
	return h(v.x) * h(v.y - half) * h(v.z - half);
}

__device__
float ky(const float3& v) {
	volatile float half = 0.5f;
	return h(v.x - 0.5f) * h(v.y) * h(v.z - 0.5f);
}
__device__
float kz(const float3& v) {
	volatile float half = 0.5f;
	return h(v.x - half) * h(v.y - half) * h(v.z);
}
__device__
float kw(const float3& v) {
	volatile float half = 0.5f;
	return h(v.x - half) * h(v.y - half) * h(v.z - half);
}


__device__ void clamp_float3(float3& v, float minv, float maxv)
{
	v.x = min(max(v.x, minv), maxv);
	v.y = min(max(v.y, minv), maxv);
	v.z = min(max(v.z, minv), maxv);
}

__device__ float3 get_wrld_posf(const float3& pos)
{
	float3 wp;
	wp.x = pos.x / dParams.world_to_grid.x - dParams.world_to_grid_offset.x;
	wp.y = pos.y / dParams.world_to_grid.y - dParams.world_to_grid_offset.y;
	wp.z = pos.z / dParams.world_to_grid.z - dParams.world_to_grid_offset.z;

	return wp;
}

__device__ float3 get_cell_posf(const float3& pos)
{
	float3 cp;
	cp.x = (pos.x + dParams.world_to_grid_offset.x) * dParams.world_to_grid.x;
	cp.y = (pos.y + dParams.world_to_grid_offset.y) * dParams.world_to_grid.y;
	cp.z = (pos.z + dParams.world_to_grid_offset.z) * dParams.world_to_grid.z;

	return cp;
}

__device__ int3 get_cell_pos(const float3& pos)
{
	int3 cp;
	cp.x = floor((pos.x + dParams.world_to_grid_offset.x) * dParams.world_to_grid.x);
	cp.y = floor((pos.y + dParams.world_to_grid_offset.y) * dParams.world_to_grid.y);
	cp.z = floor((pos.z + dParams.world_to_grid_offset.z) * dParams.world_to_grid.z);

	return cp;
}

__device__ uint get_cell_hash(const int3& cell_pos)
{
	return (cell_pos.z * dParams.grid_resolution.y + cell_pos.y) * dParams.grid_resolution.x + cell_pos.x;
}


__global__
void pfkernel_sort_initialize_keyvalues(uint particle_count, KeyValuePair* particle_keyvalues, float3* particle_positions)
{
	uint index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index >= particle_count)
	{
		return;
	}

	int3 grid_pos = get_cell_pos(particle_positions[index]);
	uint hash = get_cell_hash(grid_pos);

	particle_keyvalues[index].key = hash;
	particle_keyvalues[index].value = index;
}

__global__
void pfkernel_sort_reorder_and_insert_boundary_offsets(uint particle_count,
cudaSurfaceObject_t particles_start, cudaSurfaceObject_t particles_end,
KeyValuePair* boundary_sort_pair,
float3* in_positions, float3* out_positions, float3* in_velocities, float3* out_velocities)
{
	uint index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index >= particle_count)
	{
		return;
	}

	KeyValuePair sort_pair = boundary_sort_pair[index];

	//Load src position/velocity
	out_positions[index] = in_positions[sort_pair.value];
	out_velocities[index] = in_velocities[sort_pair.value];


	//Calculate Offset
	uint grid_xy = dParams.grid_resolution.x * dParams.grid_resolution.y;
	uint3 cell_pos = make_uint3(
		sort_pair.key % dParams.grid_resolution.x,
		(sort_pair.key % grid_xy) / dParams.grid_resolution.x,
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



void picflip::sortByGridIndex(cudaStream_t stream,
	uint particle_count,
	cudaSurfaceObject_t particles_start, cudaSurfaceObject_t particles_end,
	KeyValuePair* keyvalues,
	KeyValuePair* keyvalues_tmp,
	float3* positions,
	float3* positions_tmp,
	float3* velocities,
	float3* velocities_tmp)
{
	if (particle_count == 0)
		return;

	uint num_threads;
	uint num_blocks;


	utils::compute_grid_size(particle_count, CUDA_BLOCK_SIZE, num_blocks, num_threads);

	//Create Cell Indexes
	PICFLIP_PROFILE_BEGIN_KERNEL
		pfkernel_sort_initialize_keyvalues << <num_blocks, num_threads, 0, stream >> >(particle_count, keyvalues, positions);
	PICFLIP_PROFILE_END_KERNEL("kernel_sort_initialize_keyvalues", particle_count)

#if PICFLIP_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif



	//Sort CellIndexes
	PICFLIP_PROFILE_BEGIN_KERNEL
		RadixSort(keyvalues, keyvalues_tmp, particle_count, 32, stream);
	PICFLIP_PROFILE_END_KERNEL("RadixSort", particle_count)

#if PICFLIP_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif


	//Reorder and insert boundary offsets
	PICFLIP_PROFILE_BEGIN_KERNEL
		pfkernel_sort_reorder_and_insert_boundary_offsets << <num_blocks, num_threads, 0, stream >> >(particle_count,
		particles_start, particles_end,
		keyvalues_tmp,
		positions, positions_tmp, velocities, velocities_tmp);
	PICFLIP_PROFILE_END_KERNEL("kernel_sort_reorder_and_insert_boundary_offsets", particle_count)

#if PICFLIP_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

}




__global__
void kernel_transferToGridProgram(
cudaTextureObject_t particles_start,
cudaTextureObject_t particles_end,
float3* positions, float3* velocities,
cudaSurfaceObject_t out_velgrid,
cudaSurfaceObject_t out_veloriggrid)
{
	int3 cell_pos;
	if (!GetCellPosVel(cell_pos))
		return;

	/*const float3 xPosition = make_float3(cell_pos.x, cell_pos.y + 0.5, cell_pos.z + 0.5);
	const float3 yPosition = make_float3(cell_pos.x + 0.5, cell_pos.y, cell_pos.z + 0.5);
	const float3 zPosition = make_float3(cell_pos.x + 0.5, cell_pos.y + 0.5, cell_pos.z);
	const float3 scalarPosition = make_float3(cell_pos.x + 0.5, cell_pos.y + 0.5, cell_pos.z + 0.5);*/

	float4 out_weight = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float3 out_vel = make_float3(0.0f, 0.0f, 0.0f);

	/*uint3 search_min;
	search_min.x = max(cell_pos.x - 1, 0);
	search_min.y = max(cell_pos.y - 1, 0);
	search_min.z = max(cell_pos.z - 1, 0);

	uint3 search_range;
	search_range.x = min(cell_pos.x + 1, dParams.grid_resolution.x - 1) - search_min.x + 1;
	search_range.y = min(cell_pos.y + 1, dParams.grid_resolution.y - 1) - search_min.y + 1;
	search_range.z = min(cell_pos.z + 1, dParams.grid_resolution.z - 1) - search_min.z + 1;

	search_range.z = search_range.x * search_range.y * search_range.z;
	search_range.y = search_range.x * search_range.y;

	uint i, hash;
	float3 fs_range = make_float3(search_range.x, search_range.y, search_range.z);
	int3 cell_offset;
	for (i = 0; i < search_range.z; i++)
	{
	//Get Cell Particle List
	int3 cell_offset = make_int3(
	search_min.x + (i % search_range.x),
	search_min.y + ((i % search_range.y) / search_range.x),
	search_min.z + (i / search_range.y)
	);

	//const float eps = 0.001f;
	//float fi = float(i);
	//float ix = fi / fs_range.x;
	//float iz = fi / fs_range.y;
	//cell_offset.x = search_min.x + (int)(ix - floorf(ix + eps) + eps);
	//cell_offset.z = search_min.y + (int)(iz + eps);
	//cell_offset.y = search_min.z + (int)((iz - floorf(iz + eps)) / fs_range.x + eps);

	//hash = ((search_min.z + (i / search_range.y)) * dParams.grid_resolution.y + (search_min.y + ((i % search_range.y) / search_range.x))) * dParams.grid_resolution.x + (search_min.x + (i % search_range.x));

	uint hash = get_cell_hash(cell_offset);

	cell_desc = grid_offsets[hash];

	//Iterate over each particle
	for (; cell_desc.x < cell_desc.y; cell_desc.x++)
	{
	v_velocity = velocities[cell_desc.x];
	g_position = get_cell_posf(positions[cell_desc.x]);
	g_position.x -= float(cell_pos.x);
	g_position.y -= float(cell_pos.y);
	g_position.z -= float(cell_pos.z);

	cur_weight.x = kx(g_position);
	cur_weight.y = ky(g_position);
	cur_weight.z = kz(g_position);
	cur_weight.w = kw(g_position);

	out_vel.x += cur_weight.x * v_velocity.x;
	out_vel.y += cur_weight.y * v_velocity.y;
	out_vel.z += cur_weight.z * v_velocity.z;

	out_weight += cur_weight;
	}
	}*/

	//Search all neighbours -1, +1 (x, y ,z)
	int3 cell_max = make_int3(min(cell_pos.x + 1, dParams.grid_resolution.x - 1),
		min(cell_pos.y + 1, dParams.grid_resolution.y - 1),
		min(cell_pos.z + 1, dParams.grid_resolution.z - 1));
	int3 cell_offset;
	for (cell_offset.z = max(cell_pos.z - 1, 0); cell_offset.z <= cell_max.z; cell_offset.z++)
	{
		for (cell_offset.y = max(cell_pos.y - 1, 0); cell_offset.y <= cell_max.y; cell_offset.y++)
		{
			for (cell_offset.x = max(cell_pos.x - 1, 0); cell_offset.x <= cell_max.x; cell_offset.x++)
			{
				//Get Cell Particle List
				//uint2 cell_desc = grid_offsets[get_cell_hash(cell_offset)];
				uint cell_itr = readTexNearest<uint>(particles_start, cell_offset.x, cell_offset.y, cell_offset.z);
				uint cell_end = readTexNearest<uint>(particles_end, cell_offset.x, cell_offset.y, cell_offset.z);

				//Iterate over each particle
				for (; cell_itr < cell_end; cell_itr++)
				{
					float3 v_velocity = velocities[cell_itr];
					float3 g_position = get_cell_posf(positions[cell_itr]);
					g_position.x -= float(cell_pos.x);
					g_position.y -= float(cell_pos.y);
					g_position.z -= float(cell_pos.z);

					float4 cur_weight = make_float4(
						kx(g_position),
						ky(g_position),
						kz(g_position),
						kw(g_position));

					out_vel.x += cur_weight.x * v_velocity.x;
					out_vel.y += cur_weight.y * v_velocity.y;
					out_vel.z += cur_weight.z * v_velocity.z;

					out_weight.x += cur_weight.x;
					out_weight.y += cur_weight.y;
					out_weight.z += cur_weight.z;
					out_weight.w += cur_weight.w;
				}
			}
		}
	}


	//Store Output (out_weight = [normalized vel].xyz + out_weight.w)
	out_weight.x = (out_weight.x > 0) ? out_vel.x / out_weight.x : 0.0;
	out_weight.y = (out_weight.y > 0) ? out_vel.y / out_weight.y : 0.0;
	out_weight.z = (out_weight.z > 0) ? out_vel.z / out_weight.z : 0.0;

	writeTexVel<float4>(out_velgrid, out_weight, cell_pos.x, cell_pos.y, cell_pos.z);
	writeTexVel<float4>(out_veloriggrid, out_weight, cell_pos.x, cell_pos.y, cell_pos.z);
}

void picflip::transferToGridProgram(
	cudaStream_t stream,
	uint3 grid_resolution,
	uint particle_count,
	cudaTextureObject_t particles_start,
	cudaTextureObject_t particles_end,
	cudaSurfaceObject_t out_velgrid,
	cudaSurfaceObject_t out_veloriggrid,
	float3* positions,
	float3* velocities)
{
	//Optimisations:
	//  - Reduce offset's list to ignore any empty cells
	//  - Run one warp per cell instead of one thread


	if (particle_count == 0)
		return;

	dim3 num_threads;
	dim3 num_blocks;
#if XFER_USE_TRPLE_CUDA_DIM
	utils::compute_grid_size(grid_resolution.x + 1, CUDA_BLOCK_SIZE_3D, num_blocks.x, num_threads.x);
	utils::compute_grid_size(grid_resolution.y + 1, CUDA_BLOCK_SIZE_3D, num_blocks.y, num_threads.y);
	utils::compute_grid_size(grid_resolution.z + 1, CUDA_BLOCK_SIZE_3D, num_blocks.z, num_threads.z);
#else
	utils::compute_grid_size(grid_resolution.x + 1, CUDA_BLOCK_SIZE_3D, num_blocks.x, num_threads.x);
	utils::compute_grid_size((grid_resolution.y + 1) * (grid_resolution.z + 1), CUDA_BLOCK_SIZE_2D, num_blocks.y, num_threads.y);
	num_threads.z = 1; num_blocks.z = 1;
#endif


	//Create Cell Indexes
	PICFLIP_PROFILE_BEGIN_KERNEL
		kernel_transferToGridProgram << <num_blocks, num_threads, 0, stream >> >(particles_start, particles_end,
		positions, velocities, out_velgrid, out_veloriggrid);
	PICFLIP_PROFILE_END_KERNEL("kernel_transferToGridProgram", (grid_resolution.x + 1) * (grid_resolution.y + 1) * (grid_resolution.z + 1))

#if PICFLIP_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}


__global__
void kernel_markProgram(uint particle_count, float3* positions, cudaSurfaceObject_t out_markergrid)
{
	uint index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= particle_count)
	{
		return;
	}

	float3 pos = positions[index];
	int3 cell_pos = get_cell_pos(pos);

	//Thread write safety
	if (index > 0)
	{
		float3 pos2 = positions[index - 1];
		int3 cell_pos2 = get_cell_pos(pos2);
		if (cell_pos2.x == cell_pos.x
			&& cell_pos2.y == cell_pos.y
			&& cell_pos2.z == cell_pos.z)
		{
			return;
		}
	}

	writeTex<unsigned char>(out_markergrid, 1, cell_pos.x, cell_pos.y, cell_pos.z);
}



void picflip::markProgram(cudaStream_t stream,
	uint particle_count,
	float3* positions,
	cudaSurfaceObject_t out_markergrid)
{
	if (particle_count == 0)
		return;

	uint num_threads;
	uint num_blocks;


	utils::compute_grid_size(particle_count, CUDA_BLOCK_SIZE, num_blocks, num_threads);

	//Create Cell Indexes
	PICFLIP_PROFILE_BEGIN_KERNEL
		kernel_markProgram << <num_blocks, num_threads, 0, stream >> >(particle_count, positions, out_markergrid);
	PICFLIP_PROFILE_END_KERNEL("kernel_markProgram", particle_count)

#if PICFLIP_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}


__global__
void kernel_addForceProgram(cudaTextureObject_t in_velgrid, cudaSurfaceObject_t out_velgrid)
{
	int3 cell_pos;
	if (!GetCellPosVel(cell_pos))
		return;


	float4 vel = readTexNearest<float4>(in_velgrid, cell_pos.x, cell_pos.y, cell_pos.z);

	//Apply Gravity
	vel.y -= 9.81f * dParams.dt;

	//Enforce Tank Boundary Conditions
	if (cell_pos.x == 0) {
		vel.x = 0.0;
	}
	if (cell_pos.x == dParams.grid_resolution.x) {
		vel.x = 0.0;
	}

	if (cell_pos.y == 0) {
		vel.y = 0.0f;
	}
	if (cell_pos.y == dParams.grid_resolution.y) {
		vel.y = min(vel.y, 0.0);
	}

	if (cell_pos.z == 0) {
		vel.z = 0.0;
	}
	if (cell_pos.z == dParams.grid_resolution.z) {
		vel.z = 0.0;
	}

	writeTexVel<float4>(out_velgrid, vel, cell_pos.x, cell_pos.y, cell_pos.z);
}

void picflip::addForceProgram(cudaStream_t stream,
	uint3 grid_resolution,
	cudaTextureObject_t in_velgrid,
	cudaSurfaceObject_t out_velgrid)
{
	dim3 num_threads;
	dim3 num_blocks;
#if XFER_USE_TRPLE_CUDA_DIM
	utils::compute_grid_size(grid_resolution.x + 1, CUDA_BLOCK_SIZE_3D, num_blocks.x, num_threads.x);
	utils::compute_grid_size(grid_resolution.y + 1, CUDA_BLOCK_SIZE_3D, num_blocks.y, num_threads.y);
	utils::compute_grid_size(grid_resolution.z + 1, CUDA_BLOCK_SIZE_3D, num_blocks.z, num_threads.z);
#else
	utils::compute_grid_size(grid_resolution.x + 1, CUDA_BLOCK_SIZE_3D, num_blocks.x, num_threads.x);
	utils::compute_grid_size((grid_resolution.y + 1) * (grid_resolution.z + 1), CUDA_BLOCK_SIZE_2D, num_blocks.y, num_threads.y);
	num_threads.z = 1; num_blocks.z = 1;
#endif

	//Create Cell Indexes
	PICFLIP_PROFILE_BEGIN_KERNEL
		kernel_addForceProgram << <num_blocks, num_threads, 0, stream >> >(in_velgrid, out_velgrid);
	PICFLIP_PROFILE_END_KERNEL("kernel_addForceProgram", grid_resolution.x * grid_resolution.y * grid_resolution.z)

#if PICFLIP_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}



__global__
void kernel_divergenceProgram(
cudaTextureObject_t in_velgrid,
cudaTextureObject_t in_markergrid,
cudaSurfaceObject_t out_divgrid)
{
	int3 cell_pos;
	if (!GetCellPos(cell_pos))
		return;

	float out_div = 0.0;

	char marker = readTexNearest<char>(in_markergrid, cell_pos.x, cell_pos.y, cell_pos.z);

	//Only compute divergence for fluid
	if (marker & 1)
	{
		float3 idx = make_float3(cell_pos.x, cell_pos.y, cell_pos.z);

		float4 vel_min = readTexInterpolate<float4>(in_velgrid, idx.x, idx.y, idx.z);
		float3 vel_max;
		vel_max.x = readTexInterpolate<float4>(in_velgrid, idx.x + 1, idx.y, idx.z).x;
		vel_max.y = readTexInterpolate<float4>(in_velgrid, idx.x, idx.y + 1, idx.z).y;
		vel_max.z = readTexInterpolate<float4>(in_velgrid, idx.x, idx.y, idx.z + 1).z;

		out_div = ((vel_max.x - vel_min.x) + (vel_max.y - vel_min.y) + (vel_max.z - vel_min.z));

		//float density = readTexNearest<float4>(in_weightgrid, index_x, index_y, index_z).w;
		out_div -= max((vel_min.w - dParams.particles_per_cell), 0.0f); //volume conservation

		//out_div *= 2.0f;
	}

	writeTex<float>(out_divgrid, out_div, cell_pos.x, cell_pos.y, cell_pos.z);
}

void picflip::divergenceProgram(cudaStream_t stream,
	uint3 grid_resolution,
	cudaTextureObject_t in_velgrid,
	cudaTextureObject_t in_markergrid,
	cudaSurfaceObject_t out_divgrid)
{
	dim3 num_threads;
	dim3 num_blocks;
#if XFER_USE_TRPLE_CUDA_DIM
	utils::compute_grid_size(grid_resolution.x, CUDA_BLOCK_SIZE_3D, num_blocks.x, num_threads.x);
	utils::compute_grid_size(grid_resolution.y, CUDA_BLOCK_SIZE_3D, num_blocks.y, num_threads.y);
	utils::compute_grid_size(grid_resolution.z, CUDA_BLOCK_SIZE_3D, num_blocks.z, num_threads.z);
#else
	utils::compute_grid_size(grid_resolution.x, CUDA_BLOCK_SIZE_3D, num_blocks.x, num_threads.x);
	utils::compute_grid_size((grid_resolution.y) * (grid_resolution.z), CUDA_BLOCK_SIZE_2D, num_blocks.y, num_threads.y);
	num_threads.z = 1; num_blocks.z = 1;
#endif

	//Create Cell Indexes
	PICFLIP_PROFILE_BEGIN_KERNEL
		kernel_divergenceProgram << <num_blocks, num_threads, 0, stream >> >(in_velgrid, in_markergrid, out_divgrid);
	PICFLIP_PROFILE_END_KERNEL("kernel_divergenceProgram", grid_resolution.x * grid_resolution.y * grid_resolution.z)

#if PICFLIP_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

__global__
void kernel_marksolidcells(
	uint3 grid_start,
	uint3 grid_size,
	cudaTextureObject_t markergrid)
{
	int idx = blockIdx.y*blockDim.y + threadIdx.y;
	int3 cell_pos = make_int3(
				blockIdx.x*blockDim.x + threadIdx.x,
				idx % grid_size.y,
				idx / grid_size.y
				);

	if (cell_pos.x >= grid_size.x
		|| cell_pos.y >= grid_size.y
		|| cell_pos.z >= grid_size.z)
	{
		return;
	}
		
	cell_pos.x += grid_start.x;
	cell_pos.y += grid_start.y;
	cell_pos.z += grid_start.z;

	unsigned char marker = readTexNearest<unsigned int>(markergrid, cell_pos.x, cell_pos.y, cell_pos.z);
	marker |= 2;
	writeTex<unsigned char>(markergrid, marker, cell_pos.x, cell_pos.y, cell_pos.z);
}

void picflip::marksolidcells(cudaStream_t stream,
	uint3 grid_start,
	uint3 grid_size,
	cudaTextureObject_t markergrid)
{
	dim3 num_threads;
	dim3 num_blocks;

	utils::compute_grid_size(grid_size.x, CUDA_BLOCK_SIZE_3D, num_blocks.x, num_threads.x);
	utils::compute_grid_size((grid_size.y) * (grid_size.z), CUDA_BLOCK_SIZE_2D, num_blocks.y, num_threads.y);
	num_threads.z = 1; num_blocks.z = 1;


	//Create Cell Indexes
	PICFLIP_PROFILE_BEGIN_KERNEL
		kernel_marksolidcells << <num_blocks, num_threads, 0, stream >> >(grid_start, grid_size, markergrid);
	PICFLIP_PROFILE_END_KERNEL("kernel_marksolidcells", grid_size.x * grid_size.y * grid_size.z)

#if PICFLIP_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}




__global__
void kernel_initPressureGrid(
	cudaSurfaceObject_t presgrid,
	cudaSurfaceObject_t presgrid_old)
{
	int3 cell_pos;
	if (!GetCellPos(cell_pos))
		return;

	float op = 0.0;
	
	//if (cell_pos.x >= 55 && cell_pos.x < 60)
	//{
	//	op = 32.0f;
	//}

	surf3Dwrite(op, presgrid, cell_pos.x * sizeof(float), cell_pos.y, cell_pos.z);
	surf3Dwrite(op, presgrid_old, cell_pos.x * sizeof(float), cell_pos.y, cell_pos.z);
}

void picflip::initPressureGrid(cudaStream_t stream, uint3 grid_resolution, cudaSurfaceObject_t presgrid, cudaSurfaceObject_t presgrid_old)
{
	dim3 num_threads;
	dim3 num_blocks;
#if XFER_USE_TRPLE_CUDA_DIM
	utils::compute_grid_size(grid_resolution.x, CUDA_BLOCK_SIZE_3D, num_blocks.x, num_threads.x);
	utils::compute_grid_size(grid_resolution.y, CUDA_BLOCK_SIZE_3D, num_blocks.y, num_threads.y);
	utils::compute_grid_size(grid_resolution.z, CUDA_BLOCK_SIZE_3D, num_blocks.z, num_threads.z);
#else
	utils::compute_grid_size(grid_resolution.x, CUDA_BLOCK_SIZE_3D, num_blocks.x, num_threads.x);
	utils::compute_grid_size((grid_resolution.y) * (grid_resolution.z), CUDA_BLOCK_SIZE_2D, num_blocks.y, num_threads.y);
	num_threads.z = 1; num_blocks.z = 1;
#endif

	//Create Cell Indexes
	PICFLIP_PROFILE_BEGIN_KERNEL
		kernel_initPressureGrid << <num_blocks, num_threads, 0, stream >> >(presgrid, presgrid_old);
	PICFLIP_PROFILE_END_KERNEL("kernel_divergenceProgram", grid_resolution.x * grid_resolution.y * grid_resolution.z)

#if PICFLIP_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}


__device__
float GetAdjPressure(const float& mypres, cudaTextureObject_t in_markergrid, cudaTextureObject_t in_presgrid, const int& cx, const int& cy, const int& cz)
{
#if 1
	return readTexNearest<float>(in_presgrid, cx, cy, cz);
#else
	unsigned char marker = readTexNearest<unsigned char>(in_markergrid, cx, cy, cz);
	float pres = readTexNearest<float>(in_presgrid, cx, cy, cz);

	if (marker & 2)
	{
		pres = mypres;
		//printf("moo");
	}

	return pres;
#endif
}


__global__
void kernel_jacobiProgram(
cudaTextureObject_t in_markergrid,
cudaTextureObject_t in_divgrid,
cudaTextureObject_t in_presgrid,
cudaSurfaceObject_t out_presgrid)
{
	int3 cell_pos;
	if (!GetCellPos(cell_pos))
		return;

	char marker = readTexNearest<char>(in_markergrid, cell_pos.x, cell_pos.y, cell_pos.z);
	if (marker & 1 == 0)
		return;

	//Only compute pressure for fluid cells
	float out_pressure = 0.0;
	//if (marker == 1)
	{
		float divergenceCenter = readTexNearest<float>(in_divgrid, cell_pos.x, cell_pos.y, cell_pos.z);

		float mypres = readTexNearest<float>(in_presgrid, cell_pos.x, cell_pos.y, cell_pos.z);

		float left = GetAdjPressure(mypres, in_markergrid, in_presgrid, cell_pos.x - 1, cell_pos.y, cell_pos.z);
		float right = GetAdjPressure(mypres, in_markergrid, in_presgrid, cell_pos.x + 1, cell_pos.y, cell_pos.z);
		float bottom = GetAdjPressure(mypres, in_markergrid, in_presgrid, cell_pos.x, cell_pos.y - 1, cell_pos.z);
		float top = GetAdjPressure(mypres, in_markergrid, in_presgrid, cell_pos.x, cell_pos.y + 1, cell_pos.z);
		float back = GetAdjPressure(mypres, in_markergrid, in_presgrid, cell_pos.x, cell_pos.y, cell_pos.z - 1);
		float front = GetAdjPressure(mypres, in_markergrid, in_presgrid, cell_pos.x, cell_pos.y, cell_pos.z + 1);

		out_pressure = (left + right + bottom + top + back + front - divergenceCenter) / 6.0;
	}

	writeTex<float>(out_presgrid, out_pressure, cell_pos.x, cell_pos.y, cell_pos.z);
}

void picflip::jacobiProgram(cudaStream_t stream,
	uint jacobi_iterations,
	uint3 grid_resolution,
	cudaTextureObject_t in_markergrid,
	cudaTextureObject_t in_divgrid,
	cudaTextureObject_t presgridtex_ping,
	cudaSurfaceObject_t presgridsur_ping,
	cudaTextureObject_t presgridtex_pong,
	cudaSurfaceObject_t presgridsur_pong)
{
	dim3 num_threads;
	dim3 num_blocks;
#if XFER_USE_TRPLE_CUDA_DIM
	utils::compute_grid_size(grid_resolution.x, CUDA_BLOCK_SIZE_3D, num_blocks.x, num_threads.x);
	utils::compute_grid_size(grid_resolution.y, CUDA_BLOCK_SIZE_3D, num_blocks.y, num_threads.y);
	utils::compute_grid_size(grid_resolution.z, CUDA_BLOCK_SIZE_3D, num_blocks.z, num_threads.z);
#else
	utils::compute_grid_size(grid_resolution.x, CUDA_BLOCK_SIZE_3D, num_blocks.x, num_threads.x);
	utils::compute_grid_size((grid_resolution.y) * (grid_resolution.z), CUDA_BLOCK_SIZE_2D, num_blocks.y, num_threads.y);
	num_threads.z = 1; num_blocks.z = 1;
#endif

	PICFLIP_PROFILE_BEGIN_KERNEL
	for (uint i = 0; i < jacobi_iterations; ++i)
	{
		bool swap = i % 2 == 0;

		kernel_jacobiProgram << <num_blocks, num_threads, 0, stream >> >(
			in_markergrid,
			in_divgrid,
			swap ? presgridtex_ping : presgridtex_pong,
			swap ? presgridsur_pong : presgridsur_ping);

#if PICFLIP_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
	}
	PICFLIP_PROFILE_END_KERNEL("kernel_jacobiProgram", grid_resolution.x * grid_resolution.y * grid_resolution.z)

}



__device__
bool readInterpolated(float& out_float, cudaTextureObject_t markergrid, cudaTextureObject_t pressuregrid, float xs1, float ys1, float zs1, float xs2, float ys2, float zs2, float factor)
{
	unsigned char  uftl = tex3D<unsigned char >(markergrid, xs1 + 0.5f, ys1 + 0.5f, zs1 + 0.5f);
	unsigned char  uftr = tex3D<unsigned char >(markergrid, xs2 + 0.5f, ys2 + 0.5f, zs2 + 0.5f);

	float ftl = tex3D<float>(pressuregrid, xs1 + 0.5f, ys1 + 0.5f, zs1 + 0.5f);
	float ftr = tex3D<float>(pressuregrid, xs2 + 0.5f, ys2 + 0.5f, zs2 + 0.5f);

	if (uftl & uftr & 2)
	{
		out_float = 0.0f;
		return false;
	}		
	else if (uftl & 2)
		ftl = ftr;
	else if (uftr & 2)
	{
		ftr = ftl;
	}

	out_float = ftl * (1.0f - factor) + ftr * factor;
	return true;
}

__device__
float readInterpolatedPressures(cudaTextureObject_t markergrid, cudaTextureObject_t pressuregrid, float xs, float ys, float zs)
{
	float x = floor(xs);
	float y = floor(ys);
	float z = floor(zs);

	float fx = xs - x;
	float fy = ys - y;
	float fz = zs - z;

	float ftl, fbl, btl, bbl;

	bool bftl = readInterpolated(ftl, markergrid, pressuregrid,
		x , y, z,
		x + 1.0f, y, z, fx);

	bool bfbl = readInterpolated(fbl, markergrid, pressuregrid,
		x , y + 1.0f, z ,
		x + 1.0f, y + 1.0f, z , fx);

	bool bbtl = readInterpolated(btl, markergrid, pressuregrid,
		x , y, z + 1.0f,
		x + 1.0f, y, z  + 1.0f, fx);

	bool bbbl = readInterpolated(bbl, markergrid, pressuregrid,
		x, y + 1.0f, z + 1.0f,
		x + 1.0f, y + 1.0f, z + 1.0f, fx);

	bool by1 = true, by2 = true;

	if (!bftl && !bfbl)
		by1 = false;
	else if (!bftl)
		ftl = fbl;
	else if (!bfbl)
	{
		fbl = ftl;
	}

	if (!bbtl && !bbbl)
		by2 = false;
	else if (!bbtl)
		btl = bbl;
	else if (!bbbl)
	{
		bbl = btl;
	}

	ftl = ftl * (1.0f - fy) + fbl * fy;
	btl = btl * (1.0f - fy) + bbl * fy;


	if (!by1 && !by2)
		return 0.0f;
	else if (!by1)
		ftl = btl;
	else if (!by2)
	{
		btl = ftl;
	}

	return ftl * (1.0f - fz) + btl * fz;
}

__global__
void kernel_subtractProgram(
cudaTextureObject_t in_markergrid,
cudaTextureObject_t in_velgrid,
cudaTextureObject_t in_presgrid,
cudaSurfaceObject_t out_velgrid)
{
	int3 cell_pos;
	if (!GetCellPosVel(cell_pos))
		return;

	float3 idx = make_float3(cell_pos.x, cell_pos.y, cell_pos.z);

#if 1
	float pres_max = readTexInterpolate<float>(in_presgrid, idx.x, idx.y, idx.z);

	float3 pres_min;
	pres_min.x = readTexInterpolate<float>(in_presgrid, idx.x - 1, idx.y, idx.z);
	pres_min.y = readTexInterpolate<float>(in_presgrid, idx.x, idx.y - 1, idx.z);
	pres_min.z = readTexInterpolate<float>(in_presgrid, idx.x, idx.y, idx.z - 1);
#else

	float pres_max = readInterpolatedPressures(in_markergrid, in_presgrid, idx.x, idx.y, idx.z);

	float3 pres_min;
	pres_min.x = readInterpolatedPressures(in_markergrid, in_presgrid, idx.x - 1, idx.y, idx.z);
	pres_min.y = readInterpolatedPressures(in_markergrid, in_presgrid, idx.x, idx.y - 1, idx.z);
	pres_min.z = readInterpolatedPressures(in_markergrid, in_presgrid, idx.x, idx.y, idx.z - 1);

#endif
	//compute gradient of pressure
	float4 gradient;
	gradient.x = (pres_max - pres_min.x);
	gradient.y = (pres_max - pres_min.y);
	gradient.z = (pres_max - pres_min.z);
	gradient.w = 0.0f;

	float4 velocity = readTexNearest<float4>(in_velgrid, cell_pos.x, cell_pos.y, cell_pos.z);

	float4 newVelocity = velocity - gradient;

	writeTexVel<float4>(out_velgrid, newVelocity, cell_pos.x, cell_pos.y, cell_pos.z);
}

void picflip::subtractProgram(cudaStream_t stream,
	uint3 grid_resolution,
	cudaTextureObject_t in_markergrid,
	cudaTextureObject_t in_velgrid,
	cudaTextureObject_t in_presgrid,
	cudaSurfaceObject_t out_velgrid)
{
	dim3 num_threads;
	dim3 num_blocks;
#if XFER_USE_TRPLE_CUDA_DIM
	utils::compute_grid_size(grid_resolution.x + 1, CUDA_BLOCK_SIZE_3D, num_blocks.x, num_threads.x);
	utils::compute_grid_size(grid_resolution.y + 1, CUDA_BLOCK_SIZE_3D, num_blocks.y, num_threads.y);
	utils::compute_grid_size(grid_resolution.z + 1, CUDA_BLOCK_SIZE_3D, num_blocks.z, num_threads.z);
#else
	utils::compute_grid_size(grid_resolution.x + 1, CUDA_BLOCK_SIZE_3D, num_blocks.x, num_threads.x);
	utils::compute_grid_size((grid_resolution.y + 1) * (grid_resolution.z + 1), CUDA_BLOCK_SIZE_2D, num_blocks.y, num_threads.y);
	num_threads.z = 1; num_blocks.z = 1;
#endif

	PICFLIP_PROFILE_BEGIN_KERNEL
		kernel_subtractProgram << <num_blocks, num_threads, 0, stream >> >(
		in_markergrid,
		in_velgrid,
		in_presgrid,
		out_velgrid);
	PICFLIP_PROFILE_END_KERNEL("kernel_subtractProgram", grid_resolution.x * grid_resolution.y * grid_resolution.z)

#if PICFLIP_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

}


















__device__
float3 sample_velocity(float3 cell_pos, cudaTextureObject_t in_velgrid)
{
	float3 vel;
	vel.x = readTexInterpolate<float4>(in_velgrid, cell_pos.x, cell_pos.y - 0.5f, cell_pos.z - 0.5f).x;
	vel.y = readTexInterpolate<float4>(in_velgrid, cell_pos.x - 0.5f, cell_pos.y, cell_pos.z - 0.5f).y;
	vel.z = readTexInterpolate<float4>(in_velgrid, cell_pos.x - 0.5f, cell_pos.y - 0.5f, cell_pos.z).z;

	return vel;
}

__global__
void kernel_transferToParticlesProgram(uint particle_count, float3* positions, float3* velocities, float3* out_velocities,
cudaTextureObject_t in_velgrid, cudaTextureObject_t in_veloriggrid)
{
	uint index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= particle_count)
		return;

	float3 pos = positions[index];
	float3 cell_pos = get_cell_posf(pos);

	float3 particle_vel = velocities[index];
	float3 grid_vel = sample_velocity(cell_pos, in_velgrid);
	float3 grid_vel_orig = sample_velocity(cell_pos, in_veloriggrid);

	float3 grid_change = grid_vel - grid_vel_orig;

	float3 flip_vel = particle_vel + grid_change;
	float3 pic_vel = grid_vel;

	//pic_vel.y += ((index % 5) / 5.0f) * 0.05f;

	float3 new_vel = pic_vel * (1.0 - dParams.flipness) + flip_vel * dParams.flipness;


	//CFL Condition
	float3 cfl = make_float3(1.0, 1.0, 1.0) / (dParams.world_to_grid * dParams.dt);
	new_vel.x = max(min(new_vel.x, cfl.x), -cfl.x);
	new_vel.y = max(min(new_vel.y, cfl.y), -cfl.y);
	new_vel.z = max(min(new_vel.z, cfl.z), -cfl.z);


	out_velocities[index] = new_vel;
}

void picflip::transferToParticlesProgram(cudaStream_t stream, uint particle_count, float3* positions, float3* velocities, float3* out_velocities,
	cudaTextureObject_t in_velgrid, cudaTextureObject_t in_veloriggrid)
{
	uint num_threads;
	uint num_blocks;

	utils::compute_grid_size(particle_count, CUDA_BLOCK_SIZE, num_blocks, num_threads);

	PICFLIP_PROFILE_BEGIN_KERNEL
		kernel_transferToParticlesProgram << <num_blocks, num_threads, 0, stream >> >(particle_count, positions, velocities, out_velocities,
		in_velgrid, in_veloriggrid);
	PICFLIP_PROFILE_END_KERNEL("kernel_transferToParticlesProgram", particle_count)

#if PICFLIP_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}


__global__
void kernel_advectProgram(uint particle_count, float3* positions, float3* velocities, float3* out_positions, cudaTextureObject_t in_velgrid)
{
	uint index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= particle_count)
		return;

	float3 pos = positions[index];
	float3 cell_pos = get_cell_posf(pos);

	float3 grid_vel = sample_velocity(cell_pos, in_velgrid);

	float3 halfway_pos = pos + grid_vel * dParams.dt * 0.5f;
	cell_pos = get_cell_posf(halfway_pos);
	float3 halfway_vel = sample_velocity(cell_pos, in_velgrid);



	//CFL Condition
	float3 step = (halfway_vel * dParams.dt);

	/*	float3 cfl = make_float3(1.0f, 1.0f, 1.0f) / dParams.world_to_grid;
	step.x = max(min(step.x, cfl.x), -cfl.x);
	step.y = max(min(step.y, cfl.y), -cfl.y);
	step.z = max(min(step.z, cfl.z), -cfl.z);*/

	float3 new_pos = pos + step;

	//Clamp positions inside grid
	cell_pos = get_cell_posf(new_pos);
	const float WALL_OFFSET = 0.01f;

	cell_pos.x = min(max(cell_pos.x, WALL_OFFSET), dParams.grid_resolution.x - WALL_OFFSET);
	cell_pos.y = min(max(cell_pos.y, WALL_OFFSET), dParams.grid_resolution.y - WALL_OFFSET);
	cell_pos.z = min(max(cell_pos.z, WALL_OFFSET), dParams.grid_resolution.z - WALL_OFFSET);

	new_pos = get_wrld_posf(cell_pos);

	out_positions[index] = new_pos;
}


void picflip::advectProgram(cudaStream_t stream, uint particle_count, float3* positions, float3* velocities, float3* out_positions, cudaTextureObject_t in_velgrid)
{
	uint num_threads;
	uint num_blocks;

	utils::compute_grid_size(particle_count, CUDA_BLOCK_SIZE, num_blocks, num_threads);

	PICFLIP_PROFILE_BEGIN_KERNEL
		kernel_advectProgram << <num_blocks, num_threads, 0, stream >> >(particle_count, positions, velocities, out_positions, in_velgrid);
	PICFLIP_PROFILE_END_KERNEL("kernel_advectProgram", particle_count)

#if PICFLIP_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}



__global__
void kernel_enforceBoundaries(uint particle_count, float3* positions)
{
	uint index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= particle_count)
		return;

	float3 pos = positions[index];
	float3 cell_pos = get_cell_posf(pos);

	
	const float WALL_OFFSET = 0.01f;
	cell_pos.x = min(max(cell_pos.x, WALL_OFFSET), dParams.grid_resolution.x - WALL_OFFSET);
	cell_pos.y = min(max(cell_pos.y, WALL_OFFSET), dParams.grid_resolution.y - WALL_OFFSET);
	cell_pos.z = min(max(cell_pos.z, WALL_OFFSET), dParams.grid_resolution.z - WALL_OFFSET);

	cell_pos = get_wrld_posf(cell_pos);

	positions[index] = cell_pos;
}


void picflip::enforceBoundaries(cudaStream_t stream, uint particle_count, float3* positions)
{
	uint num_threads;
	uint num_blocks;

	utils::compute_grid_size(particle_count, CUDA_BLOCK_SIZE, num_blocks, num_threads);

	PICFLIP_PROFILE_BEGIN_KERNEL
		kernel_enforceBoundaries << <num_blocks, num_threads, 0, stream >> >(particle_count, positions);
	PICFLIP_PROFILE_END_KERNEL("kernel_enforceBoundaries", particle_count)

#if PICFLIP_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
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


	PICFLIP_PROFILE_BEGIN_KERNEL
		Kernel_MemSetSurface<T> << <num_blocks, num_threads, 0, stream >> >(
		out_grid,
		val,
		sizeof(val),
		grid_size);
	PICFLIP_PROFILE_END_KERNEL("Kernel_MemSetSurface<T>", grid_size.x * grid_size.y * grid_size.z)

#if PICFLIP_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}







__global__
void kernel_copyPressures(uint particle_count, float3* positions, cudaTextureObject_t in_weightgrid, float* out_pressures)
{
	uint index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= particle_count)
		return;

	float3 pos = positions[index];
	float3 cell_pos = get_cell_posf(pos);

	float4 weight = tex3D<float4>(in_weightgrid, cell_pos.x, cell_pos.y, cell_pos.z);

	out_pressures[index] = weight.w;
}


void copyPressures(cudaStream_t stream, uint particle_count, float3* positions, cudaTextureObject_t in_weightgrid, float* out_pressures)
{
	uint num_threads;
	uint num_blocks;

	utils::compute_grid_size(particle_count, CUDA_BLOCK_SIZE, num_blocks, num_threads);

	PICFLIP_PROFILE_BEGIN_KERNEL
		kernel_copyPressures << <num_blocks, num_threads, 0, stream >> >(particle_count, positions, in_weightgrid, out_pressures);
	PICFLIP_PROFILE_END_KERNEL("kernel_copyPressures", particle_count)

#if PICFLIP_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}