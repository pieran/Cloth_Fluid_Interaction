#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include "KeyValuePair.cuh"
#include "Real.h"

#define CUDA_HOST_TO_DEV	1
#define CUDA_DEV_TO_HOST	2
#define CUDA_DEV_TO_DEV		3

#define CUDA_FORCE_SYNC_AFTER_EACH_KERNEL FALSE
#define PROFILE_EACH_KERNEL FALSE
#define CUDA_BLOCK_SIZE 256

#define NEIGHBOR_SEARCH_METHOD 1


#ifndef min
	#define max(a,b)    (((a) > (b)) ? (a) : (b))
	#define min(a,b)    (((a) < (b)) ? (a) : (b))
#endif

namespace cudautils
{
	struct SortParams
	{
		Real cell_size;
		uint bucket_count;
	};

	void set_parameters(SortParams* hParam);
	void copy_to_ogl_buffer(cudaGraphicsResource_t* cuda_vbo, const void* cuda_arr, uint expected_size);
	void convert_double_ogl_float_async(cudaStream_t stream, cudaGraphicsResource_t* cuda_vbo, size_t count, const Real* ptr_src);


	void alloc_array(void **dev_ptr, size_t size);
	void free_array(void *dev_ptr);
	void copy_array(void *ptr_dst, const void *ptr_src, size_t size, int type);
	void copy_array_async(cudaStream_t stream, void *ptr_dst, const void *ptr_src, size_t size, int type);
	void swap_array_ptrs(void** dev_ptrA, void** dev_ptrB); //WARNING: NO SAFETY CHECKS!!! 
	void zero_array(void* dev_ptr, size_t size);



	void sort_initialize_keyvalues(cudaStream_t stream, uint particle_count, KeyValuePair* particle_keyvalues, Real3* particle_positions);
	void sort_radixsort(cudaStream_t stream, uint particle_count, KeyValuePair* src_unsorted, KeyValuePair* dst_sorted);
	void sort_reorder_and_insert_boundary_offsets(cudaStream_t stream, uint boundary_count, uint* cell_offsets, KeyValuePair* boundary_sort_pair, Real3* in_positions, Real3* out_positions);
	void sort_reorder_and_insert_fluid_offsets(cudaStream_t stream, uint boundary_count, uint* cell_offsets, KeyValuePair* boundary_sort_pair, Real3* in_positions, Real3* in_velocities, Real3* out_positions, Real3* out_velocities);
	void sort_reorder_and_insert_fluid_offsets(cudaStream_t stream, uint boundary_count, uint* cell_offsets, KeyValuePair* boundary_sort_pair, Real3* in_positions, Real3* in_velocities, Real* in_pressures, Real3* out_positions, Real3* out_velocities, Real* out_pressures);


	inline void compute_grid_size(uint num_particle, uint block_size, uint &num_blocks, uint &num_threads)
	{
		num_threads = min(block_size, num_particle);
		num_blocks = (uint)(ceil(float(num_particle) / float(num_threads)));
	}

};