#include "XPBD_Kernel.cuh"

#ifndef uint
typedef unsigned int uint;
#endif


__global__
void test_kernel(int num, float3* data)
{
	uint self_id = blockIdx.x*blockDim.x + threadIdx.x;

	if (self_id >= num)
	{
		return;
	}

	data[self_id].x = 1.0f;
}