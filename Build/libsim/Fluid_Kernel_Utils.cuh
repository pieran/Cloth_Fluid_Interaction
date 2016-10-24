#pragma once

#include <cuda_runtime.h>
#include "KeyValuePair.cuh"
#include "Real.h"
#include <stdio.h>

#define CUDA_HOST_TO_DEV	1
#define CUDA_DEV_TO_HOST	2
#define CUDA_DEV_TO_DEV		3

#define CUDA_FORCE_SYNC_AFTER_EACH_KERNEL FALSE
#define PROFILE_EACH_KERNEL FALSE //ONLY FOR IISPH!!
#define CUDA_BLOCK_SIZE 256
#define CUDA_BLOCK_SIZE_2D 16
#define CUDA_BLOCK_SIZE_3D 8

#define NEIGHBOR_SEARCH_METHOD 1


namespace utils
{

#define gpuErrchk(ans) { utils::gpuAssert((ans), __FILE__, __LINE__); }
	inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
	{
		if (code != cudaSuccess)
		{
			printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
			if (abort)
			{
				system("pause");
				exit(code);
			}
		}
	}


#if PROFILE_EACH_KERNEL
	cudaEvent_t profile_start, profile_stop;
#define PROFILE_BEGIN_KERNEL cudaEventRecord(profile_start);
#define PROFILE_END_KERNEL(description, identifier) { cudaEventRecord(profile_stop); \
	cudaEventSynchronize(profile_stop); \
	float milliseconds = 0; \
	cudaEventElapsedTime(&milliseconds, profile_start, profile_stop); \
	printf("\tKernel Timing: %5.2fms (%s -> %d)\n", milliseconds, description, identifier); }
#else
#define PROFILE_BEGIN_KERNEL
#define PROFILE_END_KERNEL(description, identifier)
#endif



	inline void compute_grid_size(uint num_particle, uint block_size, uint &num_blocks, uint &num_threads)
	{
		num_threads = min(block_size, num_particle);
		num_blocks = (uint)(ceil(float(num_particle) / float(num_threads)));
	}




	__device__ inline int3 kernel_sort_get_cell_pos(Real3 pos, Real cell_size) {
		int3 grid_pos;
		grid_pos.x = floor(pos.x / cell_size);
		grid_pos.y = floor(pos.y / cell_size);
		grid_pos.z = floor(pos.z / cell_size);
		return grid_pos;
	}

	__device__ inline uint kernel_sort_get_hash_key(int3 cell_pos, uint bucket_count) {
		//Get a better dist with the following primes. Previous primes (below) were too big and such ended up clashing when mod'ing with bucket count


		return (cell_pos.x * 18397U + cell_pos.y * 20483U + cell_pos.z * 29303U) % bucket_count;

		/*return (
		// global hash
		(((cell_pos.x * 19349663 + cell_pos.y * 73856093 + cell_pos.z * 83492791) % 0x02000000) << 6) & 0xFFFFFFC0
		// locally unique part
		| (((cell_pos.z & 0x3) << 4) & 0x30) | (((cell_pos.y & 0x3) << 2) & 0xC) | (cell_pos.x & 0x3)
		) % bucket_count;*/

		/*
		return (
		// global hash
		((cell_pos.x & 0xFFFFFFFC) * 18397U + (cell_pos.y & 0xFFFFFFFC) * 20483U + (cell_pos.z & 0xFFFFFFFC) * 29303U) & 0x7FFFFFC0
		// locally unique part
		+ (cell_pos.x & 0x3) + (cell_pos.y & 0x3) << 2U + (cell_pos.z & 0x3) << 4U
		) % bucket_count;*/

		/*#if 0
		return (
		// global hash
		(((cell_pos.x * 19349663 ^ cell_pos.y * 73856093 ^ cell_pos.z * 83492791) << 6)
		// locally unique part
		+ (cell_pos.x & 0x3) + 4 * (cell_pos.y & 0x3) + 16 * (cell_pos.z & 0x3)
		) & 0x7FFFFFFF) % bucket_count;
		#else
		return (
		// global hash
		((cell_pos.x * 19349663U ^ cell_pos.y * 73856093U ^ cell_pos.z * 83492791U) & 0x7FFFFFC0)
		// locally unique part
		| ((cell_pos.x & 3U) + ((cell_pos.y & 3U) << 2U) + ((cell_pos.z & 3U) << 4U))
		) % bucket_count;
		#endif*/
	}

	// smoothing kernels
	__device__
		inline Real kernel_pcisph_poly6(Real r2, Real h2) {
			if (r2 > h2) {
				return 0.0;
			}
			else {
				const float dr = h2 - r2;
				return dr * dr * dr;
			}
		}

	__device__
		inline Real3 kernel_pcisph_poly6_d1(Real3 d, Real h2) {
			Real r2 = d.x * d.x + d.y * d.y + d.z * d.z;
			if (r2 > h2) {
				return real3_make(0.0);
			}
			else {
				return d * ((r2 - h2) * (r2 - h2));
			}
		}

	__device__
		inline Real3 kernel_pcisph_spiky_d1(Real3 d, Real h) {
			Real r = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
			if (r > h || r < 0.0001) {
				return real3_make(0.0);
			}
			else {
				const Real dr = h - r;
				return d * (dr * dr / r);
			}
		}

	__device__
		inline Real kernel_pcisph_viscosity_d2(Real r, Real h) {
			if (r > h) {
				return 0.0;
			}
			else {
				return h - r;
			}
		}

	__device__
		inline Real kernel_pcisph_surface_tension(Real r, Real h, Real st_term) {
			if (r <= 0.5 * h) {
				Real t1 = (h - r);
				Real t2 = (t1 * t1 * t1) * (r * r * r);
				return 2.0 * t2 - st_term;
			}
			else if (r <= h) {
				Real t1 = (h - r);
				return (t1 * t1 * t1) * (r * r * r);
			}
			else {
				return 0.0;
			}
		}

	__device__
		inline void kernel_get_cell_start_end_offset(uint* cell_offsets, uint hash_key, uint bucket_count, uint* out_start, uint* out_end) {
			if (hash_key < bucket_count)
			{
				*out_start = cell_offsets[hash_key * 2];
				*out_end = cell_offsets[hash_key * 2 + 1];
			}
			else
			{
				*out_end = *out_start = 0;
			}
		}

#if NEIGHBOR_SEARCH_METHOD == 0
#define FOREACH_NEIGHBOR_VARDEFINES(pos) \
	int3 cur_cell_pos, cell_pos = kernel_sort_get_cell_pos(pos, dParams.cell_size); \
	uint hash_key, start, end, other_id, repeated_cell_itr, processed_hash_key_count, processed_hash_keys[3 * 3 * 3]; \
	bool skip;

#define FOREACH_NEIGHBOR(params, cell_offsets, pos, FOREACH_NEIGHBOR_BODY) \
	{ \
	processed_hash_key_count = 0; \
	for (cur_cell_pos.x = cell_pos.x - 1; cur_cell_pos.x <= cell_pos.x + 1; cur_cell_pos.x++) \
	{ \
	for (cur_cell_pos.y = cell_pos.y - 1; cur_cell_pos.y <= cell_pos.y + 1; cur_cell_pos.y++) \
	{ \
	for (cur_cell_pos.z = cell_pos.z - 1; cur_cell_pos.z <= cell_pos.z + 1; cur_cell_pos.z++) \
	{ \
	hash_key = kernel_sort_get_hash_key(cur_cell_pos, params.bucket_count); \
	skip = false; \
	for (repeated_cell_itr = 0; repeated_cell_itr < processed_hash_key_count; repeated_cell_itr++) \
	{ \
	if (hash_key == processed_hash_keys[repeated_cell_itr]) \
	{ \
	skip = true; \
	break; \
		} \
		} \
	if (skip) \
	continue; \
	processed_hash_keys[processed_hash_key_count++] = hash_key; \
	kernel_get_cell_start_end_offset(cell_offsets, hash_key, params.bucket_count, &start, &end); \
	for (other_id = start; other_id < end; other_id++) \
	{ \
	FOREACH_NEIGHBOR_BODY; \
		} \
		} \
		} \
		} \
		}

#elif NEIGHBOR_SEARCH_METHOD == 1
#define FOREACH_NEIGHBOR_VARDEFINES(pos) \
	int3 cur_cell_pos, cell_pos = kernel_sort_get_cell_pos(pos, dParams.cell_size); \
	uint hash_key, start, end, other_id;


#define FOREACH_NEIGHBOR(params, cell_offsets, pos, FOREACH_NEIGHBOR_BODY) \
	{ \
	for (cur_cell_pos.x = cell_pos.x - 1; cur_cell_pos.x <= cell_pos.x + 1; cur_cell_pos.x++) \
	{ \
	for (cur_cell_pos.y = cell_pos.y - 1; cur_cell_pos.y <= cell_pos.y + 1; cur_cell_pos.y++) \
	{ \
	for (cur_cell_pos.z = cell_pos.z - 1; cur_cell_pos.z <= cell_pos.z + 1; cur_cell_pos.z++) \
	{ \
	hash_key = kernel_sort_get_hash_key(cur_cell_pos, params.bucket_count); \
	kernel_get_cell_start_end_offset(cell_offsets, hash_key, params.bucket_count, &start, &end); \
	for (other_id = start; other_id < end; other_id++) \
	{ \
	FOREACH_NEIGHBOR_BODY; \
		} \
		} \
		} \
		} \
		}
#endif
};