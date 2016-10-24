
#pragma once
#include "XPBDConstraints.h"
#include <cuda_runtime.h>
#include <nclgl\Vector4.h>
#include <vector>



struct CudaTextureWrapper
{
	CudaTextureWrapper() : mem(NULL), tex(NULL), surface(NULL) {}

	cudaArray* mem;
	cudaTextureObject_t tex;
	cudaSurfaceObject_t surface;

	void allocate(
		cudaChannelFormatDesc& desc,
		uint width, uint height);

	void allocate3D(
		cudaChannelFormatDesc& desc,
		cudaExtent ce);
};

class XPBD;

class XPBD_ConstraintLayer
{
public:
	XPBD_ConstraintLayer();
	~XPBD_ConstraintLayer();
	
	void Initialize(XPBD* parent, int step);
	void Release();

	void GenerateData(Vector4* initial_positions);
	void ResetLambda();
	bool Solve(int iterations, cudaTextureObject_t positions, cudaTextureObject_t positionstmp, cudaStream_t stream1, cudaStream_t stream2); //returns true if positions/positionstmp have swapped




protected:

	void GenerateConstraints(Vector4* initial_positions);
	void GenerateCudaArrays(Vector4* initial_positions);

	XPBDDistanceConstraint BuildDistanceConstraint(const Vector4* positions, uint2 idxA, uint2 idxB, float k);
	XPBDBendingConstraint BuildBendingConstraint(const Vector4* positions, uint2 idxA, uint2 idxB, uint2 idxC, float k);
	XPBDBendDistConstraint BuildBendingDistanceConstraint(const Vector4* positions, uint2 idxA, uint2 idxB, uint2 idxC, float k_bend, float k_stretch);


	uint GetGridIdx(const uint2& grid_ref);
	uint2 GenGridRef(uint x, uint y);

public:
	int m_NumX, m_NumY;

	float k_weft, k_warp, k_shear, k_bend;

	std::vector<XPBDDistanceConstraint> m_DistanceConstraints;
	std::vector<XPBDBendingConstraint> m_BendingConstraints;
	std::vector<XPBDBendDistConstraint> m_BendingDistanceConstraints;

	float*					m_cuDistanceLambdaij;
	float*					m_cuBendingLambdaij;
	XPBDDistanceConstraint* m_cuDistanceConstraints;
	XPBDBendingConstraint*  m_cuBendingConstraints;
	XPBDBendDistConstraint* m_cuBendingDistanceConstraints;

	CudaTextureWrapper m_cuMipMapParticles;
	CudaTextureWrapper m_cuMipMapParticlesOld;

#if XPBD_USE_BATCHED_CONSTRAINTS
	std::vector<uint2> m_DistanceConstraintBatches;
	std::vector<uint2> m_BendingConstraintBatches;
	std::vector<uint2> m_BendingDistanceConstraintBatches;
#else
	float3*			   m_cuConstraintOutput;
	uint2*			   m_cuConstraintLookups;		//Particle -> OutputLookups
	uint*			   m_cuConstraintOutputLookups; //Particle -> Constraint Output List
#endif

protected:
	XPBD* m_Parent;
	int m_Step;
};