#include "Cuda_Utils.cuh"
#include "Fluid_Kernel_Utils.cuh"
#include "radixsort.cuh"
#include <exception>

using namespace utils;



__constant__ cudautils::SortParams sortParams;

void cudautils::set_parameters(SortParams *hParam)
{
#if PROFILE_EACH_KERNEL
	cudaEventCreate(&profile_start);
	cudaEventCreate(&profile_stop);
#endif

	SortParams* dParamsArr;
	//Copy Paramaters to device
	gpuErrchk(cudaGetSymbolAddress((void **)&dParamsArr, sortParams));
	gpuErrchk(cudaMemcpy(dParamsArr, hParam, sizeof(SortParams), cudaMemcpyHostToDevice));
	//gpuErrchk(cudaMemcpyToSymbol(dParamsArr, &params, sizeof(SortParams)));


#if CUDA_FORCE_SYNC_AFTER_EACH_KERNEL
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

void cudautils::copy_to_ogl_buffer(cudaGraphicsResource_t* cuda_vbo, const void* cuda_arr, uint expected_size)
{
	void* cuda_vbo_arr;
	uint num_bytes;

	PROFILE_BEGIN_KERNEL
		gpuErrchk(cudaGraphicsMapResources(1, cuda_vbo, 0));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&cuda_vbo_arr, &num_bytes, *cuda_vbo));

	if (num_bytes != expected_size)
	{
		throw std::exception("Error: OGL VBO is not the same size as the cuda simulation");
	}

	copy_array(cuda_vbo_arr, cuda_arr, num_bytes, CUDA_DEV_TO_DEV);

	gpuErrchk(cudaGraphicsUnmapResources(1, cuda_vbo, 0));
	PROFILE_END_KERNEL("Copy To OpenGL Buffer", cuda_arr)

#if CUDA_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}



__global__
void kernel_convert_double_float_async(uint count, float *ptr_dst, const Real* ptr_src)
{
	uint index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index >= count)
	{
		return;
	}

	ptr_dst[index] = float(ptr_src[index]);
}

void cudautils::convert_double_ogl_float_async(cudaStream_t stream, cudaGraphicsResource_t* cuda_vbo, size_t count, const Real*ptr_src)
{
	if (count == 0)
	{
		return;
	}

	void* cuda_vbo_arr;
	uint num_bytes;

	PROFILE_BEGIN_KERNEL
		gpuErrchk(cudaGraphicsMapResources(1, cuda_vbo, 0));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&cuda_vbo_arr, &num_bytes, *cuda_vbo));

	if (num_bytes != count * sizeof(float))
	{
		throw std::exception("Error: OGL VBO is not the same size as the cuda simulation");
	}

	uint num_threads;
	uint num_blocks;
	compute_grid_size(count, CUDA_BLOCK_SIZE, num_blocks, num_threads);

	//calc_hashK<<< num_blocks, num_threads >>>(dHash, dIndex, dMem, num_particle);

	PROFILE_BEGIN_KERNEL
		kernel_convert_double_float_async << <num_blocks, num_threads, 0, stream >> >(count, (float*)cuda_vbo_arr, ptr_src);

	gpuErrchk(cudaGraphicsUnmapResources(1, cuda_vbo, 0));
	PROFILE_END_KERNEL("kernel_sort_initialize_keyvalues", count)

#if CUDA_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

void cudautils::alloc_array(void **dev_ptr, size_t size)
{
	gpuErrchk(cudaMalloc(dev_ptr, size));
}

void cudautils::free_array(void *dev_ptr)
{
	gpuErrchk(cudaFree(dev_ptr));
}


void cudautils::copy_array_async(cudaStream_t stream, void *ptr_dst, const void *ptr_src, size_t size, int type)
{
	if (type == 1)
	{
		gpuErrchk(cudaMemcpyAsync(ptr_dst, ptr_src, size, cudaMemcpyHostToDevice, stream));
		return;
	}

	if (type == 2)
	{
		gpuErrchk(cudaMemcpyAsync(ptr_dst, ptr_src, size, cudaMemcpyDeviceToHost, stream));
		return;
	}

	if (type == 3)
	{
		gpuErrchk(cudaMemcpyAsync(ptr_dst, ptr_src, size, cudaMemcpyDeviceToDevice, stream));
		return;
	}


#if CUDA_FORCE_SYNC_AFTER_EACH_KERNEL
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	return;
}

void cudautils::copy_array(void *ptr_dst, const void *ptr_src, size_t size, int type)
{
	if (type == 1)
	{
		gpuErrchk(cudaMemcpy(ptr_dst, ptr_src, size, cudaMemcpyHostToDevice));
		return;
	}

	if (type == 2)
	{
		gpuErrchk(cudaMemcpy(ptr_dst, ptr_src, size, cudaMemcpyDeviceToHost));
		return;
	}

	if (type == 3)
	{
		gpuErrchk(cudaMemcpy(ptr_dst, ptr_src, size, cudaMemcpyDeviceToDevice));
		return;
	}


#if CUDA_FORCE_SYNC_AFTER_EACH_KERNEL
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	return;
}

void cudautils::swap_array_ptrs(void** dev_ptrA, void** dev_ptrB)
{
	void* tmp = *dev_ptrA;
	*dev_ptrA = *dev_ptrB;
	*dev_ptrB = tmp;
}

void cudautils::zero_array(void* dev_ptr, size_t size)
{
	gpuErrchk(cudaMemset(dev_ptr, 0, size));
}




__global__
void kernel_sort_initialize_keyvalues(uint particle_count, KeyValuePair* particle_keyvalues, Real3* particle_positions)
{
	uint index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index >= particle_count)
	{
		return;
	}

	int3 grid_pos = utils::kernel_sort_get_cell_pos(particle_positions[index], sortParams.cell_size);
	uint hash = utils::kernel_sort_get_hash_key(grid_pos, sortParams.bucket_count);

	particle_keyvalues[index].key = hash;
	particle_keyvalues[index].value = index;
}

void cudautils::sort_initialize_keyvalues(cudaStream_t stream, uint particle_count, KeyValuePair* particle_keyvalues, Real3* particle_positions)
{
	if (particle_count == 0)
	{
		return;
	}

	uint num_threads;
	uint num_blocks;
	compute_grid_size(particle_count, CUDA_BLOCK_SIZE, num_blocks, num_threads);

	//calc_hashK<<< num_blocks, num_threads >>>(dHash, dIndex, dMem, num_particle);

	PROFILE_BEGIN_KERNEL
		kernel_sort_initialize_keyvalues << <num_blocks, num_threads, 0, stream >> >(particle_count, particle_keyvalues, particle_positions);
	PROFILE_END_KERNEL("kernel_sort_initialize_keyvalues", particle_count)

#if CUDA_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

void cudautils::sort_radixsort(cudaStream_t stream, uint particle_count, KeyValuePair* src_unsorted, KeyValuePair* dst_sorted)
{
	if (particle_count == 0)
	{
		return;
	}

	PROFILE_BEGIN_KERNEL
		RadixSort(src_unsorted, dst_sorted, particle_count, 32, stream);
	PROFILE_END_KERNEL("RadixSort", particle_count)

#if CUDA_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

__global__
void kernel_sort_reorder_and_insert_boundary_offsets(uint boundary_count, uint* cell_offsets, KeyValuePair* boundary_sort_pair,
Real3* in_positions, Real3* out_positions)
{
	uint index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index >= boundary_count)
	{
		return;
	}

	KeyValuePair sort_pair = boundary_sort_pair[index];

	//Load src position
	out_positions[index] = in_positions[sort_pair.value];

	//Calculate Offset
	// -> key != prev_key => cell_start
	if (index == 0 || sort_pair.key != boundary_sort_pair[index - 1].key)
	{
		cell_offsets[2 * sort_pair.key + 0] = index;
	}

	// -> key != next_key => cell_end
	if (index == boundary_count - 1 || sort_pair.key != boundary_sort_pair[index + 1].key)
	{
		cell_offsets[2 * sort_pair.key + 1] = index + 1;
	}
}

void cudautils::sort_reorder_and_insert_boundary_offsets(cudaStream_t stream, uint boundary_count, uint* cell_offsets, KeyValuePair* boundary_sort_pair,
	Real3* in_positions, Real3* out_positions)
{
	if (boundary_count == 0)
	{
		return;
	}

	uint num_threads;
	uint num_blocks;
	compute_grid_size(boundary_count, CUDA_BLOCK_SIZE, num_blocks, num_threads);

	PROFILE_BEGIN_KERNEL
		kernel_sort_reorder_and_insert_boundary_offsets << <num_blocks, num_threads, 0, stream >> >(boundary_count, cell_offsets, boundary_sort_pair, in_positions, out_positions);
	PROFILE_END_KERNEL("kernel_sort_reorder_and_insert_boundary_offsets", boundary_count)

#if CUDA_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

__global__
void kernel_sort_reorder_and_insert_fluid_offsets(uint fluid_count, uint* cell_offsets, KeyValuePair* fluid_sort_pair,
Real3* in_positions, Real3* in_velocities, Real3* out_positions, Real3* out_velocities)
{
	uint index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index >= fluid_count)
	{
		return;
	}

	KeyValuePair sort_pair = fluid_sort_pair[index];

	//Load src position/velocity
	out_positions[index] = in_positions[sort_pair.value];
	out_velocities[index] = in_velocities[sort_pair.value];

	//Calculate Offset
	// -> key != prev_key => cell_start
	if (index == 0 || sort_pair.key != fluid_sort_pair[index - 1].key)
	{
		cell_offsets[2 * sort_pair.key + 0] = index;
	}

	// -> key != next_key => cell_end
	if (index == fluid_count - 1 || sort_pair.key != fluid_sort_pair[index + 1].key)
	{
		cell_offsets[2 * sort_pair.key + 1] = index + 1;
	}
}

void cudautils::sort_reorder_and_insert_fluid_offsets(cudaStream_t stream, uint fluid_count, uint* cell_offsets, KeyValuePair* fluid_sort_pair,
	Real3* in_positions, Real3* in_velocities, Real3* out_positions, Real3* out_velocities)
{
	if (fluid_count == 0)
	{
		return;
	}

	uint num_threads;
	uint num_blocks;
	compute_grid_size(fluid_count, CUDA_BLOCK_SIZE, num_blocks, num_threads);

	PROFILE_BEGIN_KERNEL
		kernel_sort_reorder_and_insert_fluid_offsets << <num_blocks, num_threads, 0, stream >> >(fluid_count, cell_offsets, fluid_sort_pair, in_positions, in_velocities, out_positions, out_velocities);
	PROFILE_END_KERNEL("sort_reorder_and_insert_fluid_offsets", fluid_count)

#if CUDA_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}



__global__
void kernel_sort_reorder_and_insert_fluid_offsets_wpressure(uint fluid_count, uint* cell_offsets, KeyValuePair* fluid_sort_pair,
Real3* in_positions, Real3* in_velocities, Real* in_pressures, Real3* out_positions, Real3* out_velocities, Real* out_pressures)
{
	uint index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index >= fluid_count)
	{
		return;
	}

	KeyValuePair sort_pair = fluid_sort_pair[index];

	//Load src position/velocity
	out_positions[index] = in_positions[sort_pair.value];
	out_velocities[index] = in_velocities[sort_pair.value];
	out_pressures[index] = in_pressures[sort_pair.value];

	//Calculate Offset
	// -> key != prev_key => cell_start
	if (index == 0 || sort_pair.key != fluid_sort_pair[index - 1].key)
	{
		cell_offsets[2 * sort_pair.key + 0] = index;
	}

	// -> key != next_key => cell_end
	if (index == fluid_count - 1 || sort_pair.key != fluid_sort_pair[index + 1].key)
	{
		cell_offsets[2 * sort_pair.key + 1] = index + 1;
	}
}

void cudautils::sort_reorder_and_insert_fluid_offsets(cudaStream_t stream, uint fluid_count, uint* cell_offsets, KeyValuePair* fluid_sort_pair,
	Real3* in_positions, Real3* in_velocities, Real* in_pressures, Real3* out_positions, Real3* out_velocities, Real* out_pressures)
{
	if (fluid_count == 0)
	{
		return;
	}

	uint num_threads;
	uint num_blocks;
	compute_grid_size(fluid_count, CUDA_BLOCK_SIZE, num_blocks, num_threads);

	PROFILE_BEGIN_KERNEL
		kernel_sort_reorder_and_insert_fluid_offsets_wpressure << <num_blocks, num_threads, 0, stream >> >(fluid_count, cell_offsets, fluid_sort_pair, in_positions, in_velocities, in_pressures, out_positions, out_velocities, out_pressures);
	PROFILE_END_KERNEL("sort_reorder_and_insert_fluid_offsets_wpressures", fluid_count)

#if CUDA_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

