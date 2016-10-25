
#pragma once
#include <nclgl\OGLRenderer.h>
#include "PicFlip_Params.h"
#include "KeyValuePair.cuh"
#include "Real.h"
#include <vector>

class FluidPicFlip {
	friend class FluidClothCoupler;
public:
	FluidPicFlip();
	~FluidPicFlip();


	void update();

	void set_vox_dims(int vox_dens, int vox_width);

	void allocate_buffers(const std::vector<Real>& fluid_positions, const std::vector<Real>& boundary_positions);

	void update_deduced_attributes();
	void set_ogl_vbo(uint pos, uint vel, uint dens, uint pres);
	void copy_arrays_to_ogl_vbos();

	void enforceBoundaries();

protected:
	bool BuildGridBuffers();

public:
	PicFlip_Params m_Params;
	uint m_glPos, m_glVel, m_glDens, m_glPres;
	uint	m_gridSize;

	uint m_NumParticles;
	float3* m_ParticlePos;
	float3* m_ParticlePosTemp;
	float3* m_ParticleVel;
	float3* m_ParticleVelTemp;
	float3* m_ParticleRnd;	//contains a random normalized direction for each particle

	KeyValuePair* m_KeyValues;
	KeyValuePair* m_KeyValuesTemp;




	//Data Arrays
	cudaArray* m_GridVel;
	cudaArray* m_GridVelTemp;
	cudaArray* m_GridVelOrig;
	cudaArray* m_GridMarker;//marks fluid/air, 1 if fluid, 0 if air
	cudaArray* m_GridDivergence;
	cudaArray* m_GridPressure;
	cudaArray* m_GridPressureTemp;
	cudaArray* m_GridMapperStart;
	cudaArray* m_GridMapperEnd;

	//Grid Texture Wrapper
	cudaTextureObject_t  m_GridVelTex;
	cudaTextureObject_t  m_GridVelTempTex;
	cudaTextureObject_t  m_GridVelOrigTex;
	cudaTextureObject_t  m_GridMarkerTex;//marks fluid/air, 1 if fluid, 0 if air
	cudaTextureObject_t  m_GridDivergenceTex;
	cudaTextureObject_t  m_GridPressureTex;
	cudaTextureObject_t  m_GridPressureTempTex;
	cudaTextureObject_t  m_GridMapperStartTex;
	cudaTextureObject_t  m_GridMapperEndTex;

	//Grid Surface Wrapper
	cudaSurfaceObject_t   m_GridVelSur;
	cudaSurfaceObject_t   m_GridVelTempSur;
	cudaSurfaceObject_t   m_GridVelOrigSur;
	cudaSurfaceObject_t   m_GridMarkerSur;
	cudaSurfaceObject_t   m_GridDivergenceSur;
	cudaSurfaceObject_t   m_GridPressureSur;
	cudaSurfaceObject_t   m_GridPressureTempSur;
	cudaTextureObject_t	  m_GridMapperStartSur;
	cudaTextureObject_t   m_GridMapperEndSur;

	// internal buffers
	struct cudaGraphicsResource* vbo_pos;
	struct cudaGraphicsResource* vbo_vel;
	struct cudaGraphicsResource* vbo_dens;
	struct cudaGraphicsResource* vbo_pres;
};