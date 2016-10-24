#pragma once

#include <libsim\XPBD.h>
#include <libsim\PicFlip.cuh>
#include <cuda_runtime.h>

class FluidClothCoupler
{
public:
	struct Params
	{
		float dt, idt;
		uint3 grid_resolution;
		float3 world_to_grid_offset;
		float3 world_to_grid;
		float particle_size;	//Fluid
		float particle_iweight;	//Fluid
	};

	FluidClothCoupler(XPBD* cloth, FluidPicFlip* fluid);
	~FluidClothCoupler();

	void HandleCollisions(float dt);

	Params m_Params;
protected:

	void AllocateMemory();

protected:
	XPBD* m_Cloth;
	FluidPicFlip* m_Fluid;
	
	uint* m_cuClothLookups;
	KeyValuePair* m_cuKeyValuePairs;
	KeyValuePair* m_cuKeyValuePairsTmp;

	float4* m_cuClothParticlesNew;
	float4* m_cuClothParticlesNewTmp;
	float4* m_cuClothParticlesOld;
	float4* m_cuClothParticlesOldTmp;
	float4* m_cuClothNormals;

	CudaTextureWrapper m_cuClothGridStart;
	CudaTextureWrapper m_cuClothGridEnd;
};