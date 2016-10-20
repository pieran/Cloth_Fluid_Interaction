#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <Windows.h>

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