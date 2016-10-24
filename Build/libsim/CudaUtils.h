#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <Windows.h>
#include <math.h>


#define CUDA_BLOCK_SIZE 8

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
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

inline void compute_grid_size(uint num_particle, uint block_size, uint &num_blocks, uint &num_threads)
{
	num_threads = min(block_size, num_particle);
	num_blocks = (uint)(ceil(float(num_particle) / float(num_threads)));
}