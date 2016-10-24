#pragma once

//#define REAL_AS_DOUBLE

#ifdef REAL_AS_DOUBLE
#include "double3.cuh"

typedef double Real;
typedef double3 Real3;

#define real3_make double3_make
#define real3_dot double3_dot
#define real3_distance double3_distance
#define real3_distance_fast double3_distance_fast
#else
#include "float3.cuh"

typedef float Real;
typedef float3 Real3;

#define real3_make float3_make
#define real3_dot float3_dot
#define real3_distance float3_distance
#define real3_distance_fast float3_distance_fast
#endif