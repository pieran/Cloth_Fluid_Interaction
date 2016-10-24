#pragma once

#include <cuda_runtime.h>

struct PicFlip_Params
{
	float dt;
	float flipness;
	uint3 grid_size;
	uint3 grid_resolution;
	float particles_per_cell;
	float3 world_to_grid;
	float3 world_to_grid_offset;
};