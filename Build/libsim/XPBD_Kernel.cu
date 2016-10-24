#include "XPBD_Kernel.cuh"
#include "CudaUtils.h"
#include "float3.cuh"

#define XPBD_PROFILE_EACH_KERNEL		FALSE
#define XPBD_FORCE_SYNC_AFTER_EACH_KERNEL FALSE

__constant__ xpbd_kernel::CudaParams dParams;

#if XPBD_PROFILE_EACH_KERNEL
cudaEvent_t pfprofile_start = NULL, pfprofile_stop = NULL;
#define XPBD_PROFILE_BEGIN_KERNEL cudaEventRecord(pfprofile_start);
#define XPBD_PROFILE_END_KERNEL(description, identifier) { cudaEventRecord(pfprofile_stop); \
	cudaEventSynchronize(pfprofile_stop); \
	float milliseconds = 0; \
	cudaEventElapsedTime(&milliseconds, pfprofile_start, pfprofile_stop); \
	printf("\tKernel Timing: %5.2fms (%s -> %d)\n", milliseconds, description, identifier); }
#else
#define XPBD_PROFILE_BEGIN_KERNEL
#define XPBD_PROFILE_END_KERNEL(description, identifier)
#endif

void xpbd_kernel::set_parameters(xpbd_kernel::CudaParams *hParam)
{
#if XPBD_PROFILE_EACH_KERNEL
	if (pfprofile_start == NULL)
	{
		cudaEventCreate(&pfprofile_start);
		cudaEventCreate(&pfprofile_stop);
	}
#endif

	xpbd_kernel::CudaParams* dParamsArr;
	//Copy Paramaters to device
	gpuErrchk(cudaGetSymbolAddress((void **)&dParamsArr, dParams));
	gpuErrchk(cudaMemcpy(dParamsArr, hParam, sizeof(xpbd_kernel::CudaParams), cudaMemcpyHostToDevice));


#if XPBD_FORCE_SYNC_AFTER_EACH_KERNEL
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

template <class T>
__device__
void writeTex(cudaSurfaceObject_t surface, const T& data, int x, int y)
{
	surf2Dwrite(data, surface, (x)* sizeof(T), y);
}

template <class T>
__device__
T readTexNearest(cudaTextureObject_t texture, float xs, float ys)
{
	return tex2D<T>(texture,
		(xs + 0.5f),
		(ys + 0.5f));
}










__global__
void kernel_updateforce(cudaTextureObject_t nparticles, cudaTextureObject_t oparticles, cudaTextureObject_t velocities)
{
	uint idx_x = blockIdx.x*blockDim.x + threadIdx.x;
	uint idx_y = blockIdx.y*blockDim.y + threadIdx.y;
	if (idx_x >= dParams.dims.x || idx_y >= dParams.dims.y)
	{
		return;
	}
#if 0
	float4 op = readTexNearest<float4>(oparticles, idx_x, idx_y);
	float4 np = readTexNearest<float4>(nparticles, idx_x, idx_y);

	float3 vel = make_float3(np.x - op.x, np.y - op.y, np.z - op.z) * dParams.dampFactor;
	if (np.w > 0.0f)
	{
		vel += dParams.gravity * (dParams.dt * dParams.dt);// *np.w;
	}

	
	np.x += vel.x;
	np.y += vel.y;
	np.z += vel.z;

	writeTex<float4>(oparticles, np, idx_x, idx_y);
#else
	float4 op = readTexNearest<float4>(oparticles, idx_x, idx_y);
	float4 np = readTexNearest<float4>(nparticles, idx_x, idx_y);
	//float4 vel = readTexNearest<float4>(velocities, idx_x, idx_y);

	float3 vel = make_float3(np.x - op.x, np.y - op.y, np.z - op.z);
	vel *= dParams.dampFactor;

	if (np.w > 0.0f)
	{
		vel += dParams.gravity * (0.5f * dParams.dt * dParams.dt);

		np.x += vel.x;
		np.y += vel.y;
		np.z += vel.z;

		vel += make_float3(np.x - op.x, np.y - op.y, np.z - op.z) / (2.0f);
		vel += dParams.gravity * (0.5f * dParams.dt * dParams.dt);

		np.x = op.x + vel.x;
		np.y = op.y + vel.y;
		np.z = op.z + vel.z;

	}
	
	writeTex<float4>(oparticles, np, idx_x, idx_y);
	//<float4>(velocities, vel, idx_x, idx_y);
#endif
}


void xpbd_kernel::updateforce(uint numx, uint numy, cudaTextureObject_t nparticles, cudaTextureObject_t oparticles, cudaTextureObject_t velocities)
{
	if (numx * numy == 0)
		return;

	dim3 num_threads;
	dim3 num_blocks;


	compute_grid_size(numx, CUDA_BLOCK_SIZE, num_blocks.x, num_threads.x);
	compute_grid_size(numy, CUDA_BLOCK_SIZE, num_blocks.y, num_threads.y);
	num_threads.z = 1; num_blocks.z = 1;

	//Create Cell Indexes
	XPBD_PROFILE_BEGIN_KERNEL
		kernel_updateforce << <num_blocks, num_threads, 0, NULL >> >(nparticles, oparticles, velocities);
	XPBD_PROFILE_END_KERNEL("kernel_updateforce", numx * numy)

#if XPBD_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}




__global__
void kernel_solvedistanceconstraints(uint num_constraints, float* constraints_lambdaij, XPBDDistanceConstraint* constraints, float k_stretch, cudaTextureObject_t particles, float3* out_updates)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= num_constraints)
	{
		return;
	}

	float lambdaij = constraints_lambdaij[idx];
	XPBDDistanceConstraint c = constraints[idx];
	float4 p1 = readTexNearest<float4>(particles, c.p1.x, c.p1.y);
	float4 p2 = readTexNearest<float4>(particles, c.p2.x, c.p2.y);

	float3 dir = make_float3(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
	float w = p1.w + p2.w;

	float3 dP = make_float3(0.0f, 0.0f, 0.0f);
	if (w > 0.0f)
	{
		float len = sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
		if (len >= 0.00001f)
		{
			//dP = dir / len * (c.k * (len - c.rest_length));
			//dP = dP / w;

			float lambda = (len - c.rest_length - k_stretch * lambdaij) / (w + k_stretch);
			dP = dir / len * lambda;

			lambdaij += lambda;
		}
	}

	constraints_lambdaij[idx] = lambdaij;
#if (XPBD_USE_BATCHED_CONSTRAINTS == FALSE)
	out_updates[c.out_offset] = dP * (-p1.w);
	out_updates[c.out_offset + 1] = dP * p2.w;
#endif
}


void xpbd_kernel::solvedistanceconstraints(uint num_constraints, float* constraints_lambdaij, XPBDDistanceConstraint* constraints, float k_stretch, cudaTextureObject_t particles, float3* out_updates)
{
	if (num_constraints == 0)
		return;

	uint num_threads;
	uint num_blocks;

	compute_grid_size(num_constraints, CUDA_BLOCK_SIZE, num_blocks, num_threads);

	//Create Cell Indexes
	XPBD_PROFILE_BEGIN_KERNEL
		kernel_solvedistanceconstraints << <num_blocks, num_threads, 0, NULL >> >(num_constraints, constraints_lambdaij, constraints, k_stretch, particles, out_updates);
	XPBD_PROFILE_END_KERNEL("kernel_solvedistanceconstraints", num_constraints)

#if XPBD_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

__global__
void kernel_solvebendingconstraints(uint num_constraints, float* constraints_lambdaij, XPBDBendingConstraint* constraints, float k_bend, cudaTextureObject_t particles, float3* out_updates)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= num_constraints)
	{
		return;
	}

	float lambdaij = constraints_lambdaij[idx];
	XPBDBendingConstraint c = constraints[idx];
	float4 p1 = readTexNearest<float4>(particles, c.p1.x, c.p1.y);
	float4 p2 = readTexNearest<float4>(particles, c.p2.x, c.p2.y);
	float4 p3 = readTexNearest<float4>(particles, c.p3.x, c.p3.y);

	float3 dP = make_float3(0.0f, 0.0f, 0.0f);


	float3 center = make_float3(
		(p1.x + p2.x + p3.x) / 3.0f,
		(p1.y + p2.y + p3.y) / 3.0f, 
		(p1.z + p2.z + p3.z) / 3.0f);

	float3 dir_center = make_float3(p3.x, p3.y, p3.z) - center;
	float dist_center = sqrt(dir_center.x * dir_center.x + dir_center.y * dir_center.y + dir_center.z * dir_center.z);

	float w = p1.w + p2.w + p3.w;

	float diff = (1.0f - (c.rest_length / dist_center) - k_bend * lambdaij) / (w + k_bend);
	lambdaij += diff;
	float3 dir_force = dir_center * diff;

	constraints_lambdaij[idx] = lambdaij;
#if (XPBD_USE_BATCHED_CONSTRAINTS == FALSE)

	out_updates[c.out_offset] = dir_force * (c.k * 2.0f* p1.w);
	out_updates[c.out_offset+1] = dir_force * (c.k * 2.0f* p2.w);
	out_updates[c.out_offset+2] = dir_force  * (-c.k * 4.0f* p3.w);
#endif
}


void xpbd_kernel::solvebendingconstraints(uint num_constraints, float* constraints_lambdaij, XPBDBendingConstraint* constraints, float k_bend, cudaTextureObject_t particles, float3* out_updates)
{
	if (num_constraints == 0)
		return;

	uint num_threads;
	uint num_blocks;

	compute_grid_size(num_constraints, CUDA_BLOCK_SIZE, num_blocks, num_threads);

	//Create Cell Indexes
	XPBD_PROFILE_BEGIN_KERNEL
		kernel_solvebendingconstraints << <num_blocks, num_threads, 0, NULL >> >(num_constraints, constraints_lambdaij, constraints, k_bend, particles, out_updates);
	XPBD_PROFILE_END_KERNEL("kernel_solvedistanceconstraints", num_constraints)

#if XPBD_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}



__global__
void kernel_mergeoutputs(float weighting, cudaTextureObject_t iparticles, cudaSurfaceObject_t oparticles, uint2* constraint_lookups, uint* constraint_output_lookups, float3* constraint_outputs)
{
	uint idx_x = blockIdx.x*blockDim.x + threadIdx.x;
	uint idx_y = blockIdx.y*blockDim.y + threadIdx.y;
	if (idx_x >= dParams.dims.x || idx_y >= dParams.dims.y)
	{
		return;
	}

	float4 p = readTexNearest<float4>(iparticles, idx_x, idx_y);
	uint2 cl = constraint_lookups[idx_y * dParams.dims.x + idx_x];

	float3 sum = make_float3(0.0f, 0.0f, 0.0f);
	for (uint i = cl.x; i < cl.y; ++i)
	{
		sum += constraint_outputs[constraint_output_lookups[i]];
	}

	float factor = 1.0f / float(cl.y - cl.x);
	sum *= factor * weighting;

	p.x += sum.x;
	p.y += sum.y;
	p.z += sum.z;

	writeTex<float4>(oparticles, p, idx_x, idx_y);
}


void xpbd_kernel::mergeoutputs(uint numx, uint numy, float weighting, cudaTextureObject_t iparticles, cudaSurfaceObject_t oparticles, uint2* constraint_lookups, uint* constraint_output_lookups, float3* constraint_outputs)
{
	if (numx * numy == 0)
		return;

	dim3 num_threads;
	dim3 num_blocks;

	compute_grid_size(numx, CUDA_BLOCK_SIZE, num_blocks.x, num_threads.x);
	compute_grid_size(numy, CUDA_BLOCK_SIZE, num_blocks.y, num_threads.y);
	num_threads.z = 1; num_blocks.z = 1;

	//Create Cell Indexes
	XPBD_PROFILE_BEGIN_KERNEL
		kernel_mergeoutputs << <num_blocks, num_threads, 0, NULL >> >(weighting, iparticles, oparticles, constraint_lookups, constraint_output_lookups, constraint_outputs);
	XPBD_PROFILE_END_KERNEL("kernel_mergeoutputs", numx * numy)

#if XPBD_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}





__device__
void kernel_solvedistance(float4& p1, float4& p2, float rest_length, float k)
{
	float3 dir = make_float3(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
	float w = p1.w + p2.w;

	float3 dP = make_float3(0.0f, 0.0f, 0.0f);
	if (w > 0.0f)
	{
		float len = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
		if (len > 0.00001f)
		{
			dP = dir / len * (k * (len - rest_length));
			dP = dP / w;

			/*float lambda = (len - c.rest_length - c.k * lambdaij) / (w + c.k);
			len = 1.0f / len;
			dP = dir * (len * lambda);
			lambdaij += lambda; */
		}
	}

	p1.x -= dP.x * p1.w;
	p1.y -= dP.y * p1.w;
	p1.z -= dP.z * p1.w;

	p2.x += dP.x * p2.w;
	p2.y += dP.y * p2.w;
	p2.z += dP.z * p2.w;
}


__device__
void kernel_solvebending(float4& p1, float4& p2, float4& p3, float rest_length, float k)
{
	float third = 1.0f / 3.0f;
	float3 center = make_float3(
		(p1.x + p2.x + p3.x) * third,
		(p1.y + p2.y + p3.y) * third,
		(p1.z + p2.z + p3.z) * third);

	float3 dir_center = make_float3(p3.x, p3.y, p3.z) - center;
	float dist_center = sqrtf(dir_center.x * dir_center.x + dir_center.y * dir_center.y + dir_center.z * dir_center.z);

	float w = p1.w + p2.w + p3.w;

	//float diff = (1.0f - (c.rest_length / dist_center) - c.k * lambdaij) / (w + c.k);	
	//lambdaij += diff;

	float3 dir_force = make_float3(0.0f, 0.0f, 0.0f);
	if (w > 0.0001f)
	{
		float diff = (1.0f - ((rest_length) / dist_center)) / w;
		dir_force = dir_center * diff;
	}

	float ck2 = k * 2.0f;
	p1.x += dir_force.x * (p1.w * ck2);
	p1.y += dir_force.y * (p1.w * ck2);
	p1.z += dir_force.z * (p1.w * ck2);

	p2.x += dir_force.x * (p2.w * ck2);
	p2.y += dir_force.y * (p2.w * ck2);
	p2.z += dir_force.z * (p2.w * ck2);

	p3.x -= dir_force.x * (p3.w * ck2 * 2.0f);
	p3.y -= dir_force.y * (p3.w * ck2 * 2.0f);
	p3.z -= dir_force.z * (p3.w * ck2 * 2.0f);
}






__global__
void kernel_batched_solvedistanceconstraints(uint c_num, float* constraint_lambdaij, XPBDDistanceConstraint* constraints, float k_stretch, cudaTextureObject_t particles, cudaSurfaceObject_t oparticles)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= c_num)
	{
		return;
	}

	XPBDDistanceConstraint c = constraints[idx];
	//float lambdaij = constraint_lambdaij[idx];

	float4 p1 = readTexNearest<float4>(particles, c.p1.x, c.p1.y);
	float4 p2 = readTexNearest<float4>(particles, c.p2.x, c.p2.y);

	kernel_solvedistance(p1, p2, c.rest_length, k_stretch);

	//constraint_lambdaij[idx] = lambdaij;
	writeTex<float4>(oparticles, p1, c.p1.x, c.p1.y);
	writeTex<float4>(oparticles, p2, c.p2.x, c.p2.y);
}


void xpbd_kernel::batched_solvedistanceconstraints(uint c_num, float* constraint_lambdaij, XPBDDistanceConstraint* constraints, float k_stretch, cudaTextureObject_t particles, cudaSurfaceObject_t oparticles)
{
	if (c_num == 0)
		return;

	uint num_threads;
	uint num_blocks;

	compute_grid_size(c_num, CUDA_BLOCK_SIZE, num_blocks, num_threads);

	//Create Cell Indexes
	XPBD_PROFILE_BEGIN_KERNEL
		kernel_batched_solvedistanceconstraints << <num_blocks, num_threads, 0, NULL >> >(c_num, constraint_lambdaij, constraints, k_stretch, particles, oparticles);
	XPBD_PROFILE_END_KERNEL("kernel_batched_solvedistanceconstraints", c_num)

#if XPBD_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}



__global__
void kernel_batched_solvebendingconstraints(uint c_num, float* constraint_lambdaij, XPBDBendingConstraint* constraints, float k_bend, cudaTextureObject_t particles, cudaSurfaceObject_t oparticles)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= c_num)
	{
		return;
	}

	XPBDBendingConstraint c = constraints[idx];
	//float lambdaij = constraint_lambdaij[idx];
	float4 p1 = readTexNearest<float4>(particles, c.p1.x, c.p1.y);
	float4 p2 = readTexNearest<float4>(particles, c.p2.x, c.p2.y);
	float4 p3 = readTexNearest<float4>(particles, c.p3.x, c.p3.y);

	kernel_solvebending(p1, p2, p3, c.rest_length, k_bend);

	//constraint_lambdaij[idx] = lambdaij;
	writeTex<float4>(oparticles, p1, c.p1.x, c.p1.y);
	writeTex<float4>(oparticles, p2, c.p2.x, c.p2.y);
	writeTex<float4>(oparticles, p3, c.p3.x, c.p3.y);
}


void xpbd_kernel::batched_solvebendingconstraints(uint c_num, float* constraint_lambdaij, XPBDBendingConstraint* constraints, float k_bend, cudaTextureObject_t particles, cudaSurfaceObject_t oparticles)
{
	if (c_num == 0)
		return;

	uint num_threads;
	uint num_blocks;

	compute_grid_size(c_num, CUDA_BLOCK_SIZE, num_blocks, num_threads);

	//Create Cell Indexes
	XPBD_PROFILE_BEGIN_KERNEL
		kernel_batched_solvebendingconstraints << <num_blocks, num_threads, 0, NULL >> >(c_num, constraint_lambdaij, constraints, k_bend, particles, oparticles);
	XPBD_PROFILE_END_KERNEL("kernel_batched_solvebendingconstraints", c_num)

#if XPBD_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}


#if (XPBD_USE_BATCHED_CONSTRAINTS == TRUE)
template <bool horizontal>
__global__
void kernel_batched_solvedistancebendingconstraints(uint c_num, XPBDBendDistConstraint* constraints, float k_bend, float k_stretch, cudaTextureObject_t particles)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= c_num)
	{
		return;
	}

	XPBDBendDistConstraint c = constraints[idx];
	float2 tp = make_float2(c.p1.x, c.p1.y);

	float4 p1, p2, p3;
	if (horizontal)
	{
		p1 = readTexNearest<float4>(particles, tp.x, tp.y);
		p2 = readTexNearest<float4>(particles, tp.x + 1.0f, tp.y);
		p3 = readTexNearest<float4>(particles, tp.x + 2.0f, tp.y);
	}
	else
	{
		p1 = readTexNearest<float4>(particles, tp.x, tp.y);
		p2 = readTexNearest<float4>(particles, tp.x, tp.y + 1.0f);
		p3 = readTexNearest<float4>(particles, tp.x, tp.y + 2.0f);
	}

	kernel_solvebending(p1, p2, p3, c.rest_length, k_bend);
	kernel_solvedistance(p1, p2, c.rest_length, k_stretch);
	kernel_solvedistance(p2, p3, c.rest_length, k_stretch);
	
	if (horizontal)
	{
		writeTex<float4>(particles, p1, tp.x, tp.y);
		writeTex<float4>(particles, p2, tp.x + 1.0f, tp.y);
		writeTex<float4>(particles, p3, tp.x + 2.0f, tp.y);
	}
	else
	{
		writeTex<float4>(particles, p1, tp.x, tp.y);
		writeTex<float4>(particles, p2, tp.x, tp.y + 1.0f);
		writeTex<float4>(particles, p3, tp.x, tp.y + 2.0f);
	}
}

void xpbd_kernel::batched_solvebendingdistanceconstraints(
	bool horizontal,
	uint c_num,
	XPBDBendDistConstraint* constraints,
	float k_bend, float k_stretch,
	cudaTextureObject_t particles)
{
	if (c_num == 0)
		return;

	uint num_threads;
	uint num_blocks;

	compute_grid_size(c_num, CUDA_BLOCK_SIZE, num_blocks, num_threads);

	//Create Cell Indexes
	XPBD_PROFILE_BEGIN_KERNEL
	if (horizontal)
		kernel_batched_solvedistancebendingconstraints<true> << <num_blocks, num_threads, 0, NULL >> >(c_num, constraints, k_bend, k_stretch, particles);
	else
		kernel_batched_solvedistancebendingconstraints<false> << <num_blocks, num_threads, 0, NULL >> >(c_num, constraints, k_bend, k_stretch, particles);
	XPBD_PROFILE_END_KERNEL("kernel_batched_solvedistancebendingconstraints", c_num)

#if XPBD_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

#endif



__global__
void kernel_solversphereconstraint(uint numx, uint numy, XPBDSphereConstraint c, cudaTextureObject_t particles)
{
	int idx_x = blockIdx.x*blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y*blockDim.y + threadIdx.y;
	if (idx_x >= numx || idx_y >= numy)
	{
		return;
	}

	float4 p = readTexNearest<float4>(particles, idx_x, idx_y);

	if (p.w < 0.0001f)
		return;

	float3 axis = make_float3(p.x - c.centre.x, p.y - c.centre.y, p.z - c.centre.z);
	float radiusSq = c.radius * c.radius;
	float distSquared = float3_dot(axis, axis);

	if (distSquared < radiusSq)
	{
		float dist = sqrtf(distSquared);

		float excess = 1.0f - dist / c.radius;
		p.x += axis.x * excess;
		p.y += axis.y * excess;
		p.z += axis.z * excess;

		writeTex<float4>(particles, p, idx_x, idx_y);
	}
}


void xpbd_kernel::solvesphereconstraint(uint numx, uint numy, const XPBDSphereConstraint& c, cudaTextureObject_t particles)
{
	if (numx * numy == 0)
		return;

	dim3 num_threads;
	dim3 num_blocks;

	compute_grid_size(numx, CUDA_BLOCK_SIZE, num_blocks.x, num_threads.x);
	compute_grid_size(numy, CUDA_BLOCK_SIZE, num_blocks.y, num_threads.y);
	num_threads.z = 1; num_blocks.z = 1;

	//Create Cell Indexes
	XPBD_PROFILE_BEGIN_KERNEL
		kernel_solversphereconstraint << <num_blocks, num_threads, 0, NULL >> >(numx, numy, c, particles);
	XPBD_PROFILE_END_KERNEL("kernel_genvertnormals", numx * numy)

#if XPBD_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}


__device__
float3 toF3(const float4& f4)
{
	return make_float3(f4.x, f4.y, f4.z);
}


__global__
void kernel_genquadnormals(int reverseOrder, cudaTextureObject_t particles, cudaSurfaceObject_t facenormals)
{
	int idx_x = blockIdx.x*blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y*blockDim.y + threadIdx.y;
	if (idx_x >= (dParams.dims.x - 1) || idx_y >= (dParams.dims.y - 1))
	{
		return;
	}

	float3 a = toF3(readTexNearest<float4>(particles, idx_x, idx_y));
	float3 b = toF3(readTexNearest<float4>(particles, idx_x + 1 - reverseOrder, idx_y + reverseOrder));
	float3 c = toF3(readTexNearest<float4>(particles, idx_x + reverseOrder, idx_y + 1 - reverseOrder));
	float3 d = toF3(readTexNearest<float4>(particles, idx_x + 1, idx_y + 1));

	float3 normal = float3_normalize(
			float3_normalize(float3_cross(a - c, b - c)) +
			float3_normalize(float3_cross(b - c, d - c))
			);
	
	float4 normal4 = make_float4(normal.x, normal.y, normal.z, 0.0f);
	writeTex<float4>(facenormals, normal4, idx_x, idx_y);
}


void xpbd_kernel::genquadnormals(uint numx, uint numy, bool reverseOrder, cudaTextureObject_t particles, cudaSurfaceObject_t facenormals)
{
	if (numx * numy == 0)
		return;

	dim3 num_threads;
	dim3 num_blocks;

	compute_grid_size(numx, CUDA_BLOCK_SIZE, num_blocks.x, num_threads.x);
	compute_grid_size(numy, CUDA_BLOCK_SIZE, num_blocks.y, num_threads.y);
	num_threads.z = 1; num_blocks.z = 1;

	//Create Cell Indexes
	XPBD_PROFILE_BEGIN_KERNEL
		kernel_genquadnormals << <num_blocks, num_threads, 0, NULL >> >(reverseOrder ?  1 : 0, particles, facenormals);
	XPBD_PROFILE_END_KERNEL("kernel_genquadnormals", numx * numy)

#if XPBD_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}


__global__
void kernel_genvertnormals(cudaTextureObject_t facenormals, cudaSurfaceObject_t vertnormals)
{
	int idx_x = blockIdx.x*blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y*blockDim.y + threadIdx.y;
	if (idx_x >= dParams.dims.x || idx_y >= dParams.dims.y)
	{
		return;
	}

	float3 a = toF3(readTexNearest<float4>(facenormals, idx_x, idx_y));
	float3 b = toF3(readTexNearest<float4>(facenormals, idx_x, idx_y - 1));
	float3 c = toF3(readTexNearest<float4>(facenormals, idx_x - 1, idx_y));
	float3 d = toF3(readTexNearest<float4>(facenormals, idx_x - 1, idx_y - 1));

	float3 normal = float3_normalize(
		a + b + c + d
		);

	float4 normal4 = make_float4(normal.x, normal.y, normal.z, 0.0f);
	writeTex<float4>(vertnormals, normal4, idx_x, idx_y);
}


void xpbd_kernel::genvertnormals(uint numx, uint numy, cudaTextureObject_t facenormals, cudaSurfaceObject_t vertnormals)
{
	if (numx * numy == 0)
		return;

	dim3 num_threads;
	dim3 num_blocks;

	compute_grid_size(numx, CUDA_BLOCK_SIZE, num_blocks.x, num_threads.x);
	compute_grid_size(numy, CUDA_BLOCK_SIZE, num_blocks.y, num_threads.y);
	num_threads.z = 1; num_blocks.z = 1;

	//Create Cell Indexes
	XPBD_PROFILE_BEGIN_KERNEL
		kernel_genvertnormals << <num_blocks, num_threads, 0, NULL >> >(facenormals, vertnormals);
	XPBD_PROFILE_END_KERNEL("kernel_genvertnormals", numx * numy)

#if XPBD_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}


__global__
void kernel_copytomipmap(int numx, int numy, int step, cudaTextureObject_t particles, cudaSurfaceObject_t mipmap)
{
	int idx_x = blockIdx.x*blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y*blockDim.y + threadIdx.y;
	if (idx_x >= numx || idx_y >= numy)
	{
		return;
	}

	float4 p = readTexNearest<float4>(particles, idx_x * step, idx_y * step);
	writeTex<float4>(mipmap, p, idx_x, idx_y);
}


void xpbd_kernel::copytomipmap(int numx, int numy, int step, cudaTextureObject_t particles, cudaSurfaceObject_t mipmap)
{
	if (numx * numy == 0)
		return;

	dim3 num_threads;
	dim3 num_blocks;

	compute_grid_size(numx, CUDA_BLOCK_SIZE, num_blocks.x, num_threads.x);
	compute_grid_size(numy, CUDA_BLOCK_SIZE, num_blocks.y, num_threads.y);
	num_threads.z = 1; num_blocks.z = 1;

	//Create Cell Indexes
	XPBD_PROFILE_BEGIN_KERNEL
		kernel_copytomipmap << <num_blocks, num_threads, 0, NULL >> >(numx, numy, step, particles, mipmap);
	XPBD_PROFILE_END_KERNEL("kernel_copytomipmap", numx * numy)

#if XPBD_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

__global__
void kernel_extrapolatemipmap(float step, cudaTextureObject_t nmipmap, cudaTextureObject_t omipmap, cudaSurfaceObject_t particles)
{
	int idx_x = blockIdx.x*blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y*blockDim.y + threadIdx.y;
	if (idx_x >= dParams.dims.x || idx_y >= dParams.dims.y)
	{
		return;
	}

	//Get Texture Coordinates
	float tex_x = float(idx_x) / step;
	float tex_y = float(idx_y) / step;

	float mix_x = tex_x - floor(tex_x);
	float mix_y = tex_y - floor(tex_y);

	float factor_x = 1.0f;// abs(mix_x - 0.5f) + 0.5f;
	float factor_y = 1.0f;// abs(mix_y - 0.5f) + 0.5f;

	tex_x = floor(tex_x);
	tex_y = floor(tex_y);

	float4 p = readTexNearest<float4>(particles, idx_x, idx_y);

	float3 a =
		toF3(readTexNearest<float4>(nmipmap, tex_x, tex_y)) -
		toF3(readTexNearest<float4>(omipmap, tex_x, tex_y));
	float3 b =
		toF3(readTexNearest<float4>(nmipmap, tex_x + 1, tex_y)) -
		toF3(readTexNearest<float4>(omipmap, tex_x + 1, tex_y));
	float3 c =
		toF3(readTexNearest<float4>(nmipmap, tex_x, tex_y + 1)) -
		toF3(readTexNearest<float4>(omipmap, tex_x, tex_y + 1));
	float3 d =
		toF3(readTexNearest<float4>(nmipmap, tex_x + 1, tex_y + 1)) -
		toF3(readTexNearest<float4>(omipmap, tex_x + 1, tex_y + 1));

	//Interpolate X
	a = (a * (1.0f - mix_x) + b * mix_x) * factor_x;
	c = (c * (1.0f - mix_x) + d * mix_x) * factor_x;

	//Iterpolate Y
	p.x += (a.x * (1.0f - mix_y) + c.x * mix_y) * factor_y;
	p.y += (a.y * (1.0f - mix_y) + c.y * mix_y) * factor_y;
	p.z += (a.z * (1.0f - mix_y) + c.z * mix_y) * factor_y;

	writeTex<float4>(particles, p, idx_x, idx_y);
}


void xpbd_kernel::extrapolatemipmap(int numx, int numy, float step, cudaTextureObject_t nmipmap, cudaTextureObject_t omipmap, cudaSurfaceObject_t particles)
{
	if (numx * numy == 0)
		return;

	dim3 num_threads;
	dim3 num_blocks;

	compute_grid_size(numx, CUDA_BLOCK_SIZE, num_blocks.x, num_threads.x);
	compute_grid_size(numy, CUDA_BLOCK_SIZE, num_blocks.y, num_threads.y);
	num_threads.z = 1; num_blocks.z = 1;

	//Create Cell Indexes
	XPBD_PROFILE_BEGIN_KERNEL
		kernel_extrapolatemipmap << <num_blocks, num_threads, 0, NULL >> >(step, nmipmap, omipmap, particles);
	XPBD_PROFILE_END_KERNEL("kernel_extrapolatemipmap", numx * numy)

#if XPBD_FORCE_SYNC_AFTER_EACH_KERNEL
		gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}