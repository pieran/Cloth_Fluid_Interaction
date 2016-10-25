#pragma once

#include <vector>
#include <list>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "PicFlip_Params.h"
#include "KeyValuePair.cuh"


namespace picflip
{
	void set_parameters(PicFlip_Params *hParam);


	void sortByGridIndex(cudaStream_t stream,
		uint particle_count,
		cudaSurfaceObject_t particles_start,
		cudaSurfaceObject_t particles_end,
		KeyValuePair* keyvalues,
		KeyValuePair* keyvalues_tmp,
		float3* positions,
		float3* positions_tmp,
		float3* velocities,
		float3* velocities_tmp);

	//Requires sorting positions by grid index then accumulating weight/velocity info
	void transferToGridProgram(
		cudaStream_t stream,
		uint3 grid_resolution,
		uint particle_count,
		cudaTextureObject_t particles_start,
		cudaTextureObject_t particles_end,
		cudaSurfaceObject_t out_velgrid,
		cudaSurfaceObject_t out_veloriggrid,
		float3* positions,
		float3* velocities);

	//Marks Cells which have a particle in them
	void markProgram(cudaStream_t stream,
		uint particle_count,
		float3* positions,
		cudaSurfaceObject_t out_markergrid);


	void addForceProgram(cudaStream_t stream,
		uint3 grid_resolution,
		cudaTextureObject_t in_velgrid,
		cudaSurfaceObject_t out_velgrid);

	void divergenceProgram(
		cudaStream_t stream,
		uint3 grid_resolution,
		cudaTextureObject_t in_velgrid,
		cudaTextureObject_t in_markergrid,
		cudaSurfaceObject_t out_divgrid);

	void jacobiProgram(cudaStream_t stream,
		uint jacobi_iterations,
		uint3 grid_resolution,
		cudaTextureObject_t in_markergrid,
		cudaTextureObject_t in_divgrid,
		cudaTextureObject_t presgridtex_ping,
		cudaSurfaceObject_t presgridsur_ping,
		cudaTextureObject_t presgridtex_pong,
		cudaSurfaceObject_t presgridsur_pong);

	void subtractProgram(cudaStream_t stream,
		uint3 grid_resolution,
		cudaTextureObject_t in_markergrid,
		cudaTextureObject_t in_velgrid,
		cudaTextureObject_t in_presgrid,
		cudaSurfaceObject_t out_velgrid);

	void transferToParticlesProgram(cudaStream_t stream, uint particle_count, float3* positions, float3* velocities, float3* out_velocities,
		cudaTextureObject_t in_velgrid, cudaTextureObject_t in_veloriggrid);

	void advectProgram(cudaStream_t stream, uint particle_count, float3* positions, float3* velocities, float3* out_positions, cudaTextureObject_t in_velgrid);

	void enforceBoundaries(cudaStream_t stream, uint particle_count, float3* positions);

	void initPressureGrid(cudaStream_t stream, uint3 grid_resolution, cudaSurfaceObject_t presgrid, cudaSurfaceObject_t presgrid_old);
	
	void marksolidcells(cudaStream_t stream,
		uint3 grid_start,
		uint3 grid_size,
		cudaTextureObject_t markergrid);
}