#pragma once

#include "XPBDConstraints.h"




namespace xpbd_kernel
{
	struct CudaParams
	{
		int2 dims;
		float dt;
		float3 gravity;
		float dampFactor;
	};

	void set_parameters(CudaParams *hParam);

	void updateforce(
		uint numx, uint numy,
		cudaTextureObject_t nparticles,
		cudaTextureObject_t oparticles,
		cudaTextureObject_t velocities);
	
	void solvedistanceconstraints(
		uint num_constraints,
		float* constraints_lambdaij,
		XPBDDistanceConstraint* constraints,
		float k_stretch,
		cudaTextureObject_t particles,
		float3* out_updates);
	
	void solvebendingconstraints(
		uint num_constraints,
		float* constraints_lambdaij,
		XPBDBendingConstraint* constraints,
		float k_bend,
		cudaTextureObject_t particles,
		float3* out_updates);
	
	void mergeoutputs(
		uint numx, uint numy,
		float weighting,
		cudaTextureObject_t iparticles,
		cudaSurfaceObject_t oparticles,
		uint2* constraint_lookups,
		uint* constraint_output_lookups,
		float3* constraint_outputs);




	void batched_solvedistanceconstraints(
		uint c_num,
		float* constraint_lambdaij,
		XPBDDistanceConstraint* constraints,
		float k_stretch,
		cudaTextureObject_t particles,
		cudaSurfaceObject_t oparticles);

	void batched_solvebendingconstraints(
		uint c_num,
		float* constraint_lambdaij,
		XPBDBendingConstraint* constraints,
		float k_bend,
		cudaTextureObject_t particles,
		cudaSurfaceObject_t oparticles);

#if (XPBD_USE_BATCHED_CONSTRAINTS == TRUE)
	void batched_solvebendingdistanceconstraints(
		bool horizontal,
		uint c_num,
		XPBDBendDistConstraint* constraints,
		float k_bend, float k_stretch,
		cudaTextureObject_t particles);
#endif

	void solvesphereconstraint(
		uint numx, uint numy,
		const XPBDSphereConstraint& c,
		cudaTextureObject_t particles);



	void copytomipmap(
		int numx, int numy, int step,
		cudaTextureObject_t particles,
		cudaSurfaceObject_t mipmap);

	void extrapolatemipmap(
		int numx, int numy, float step,
		cudaTextureObject_t nmipmap,
		cudaTextureObject_t omipmap,
		cudaSurfaceObject_t particles);



	void genquadnormals(
		uint numx, uint numy,
		bool reverseOrder,
		cudaTextureObject_t particles,
		cudaSurfaceObject_t facenormals);

	void genvertnormals(
		uint numx, uint numy,
		cudaTextureObject_t facenormals,
		cudaSurfaceObject_t vertnormals);

}