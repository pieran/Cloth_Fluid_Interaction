#pragma once


#include <cuda_runtime.h>

#ifndef uint
typedef unsigned int uint;
#endif

#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif

#define XPBD_USE_BATCHED_CONSTRAINTS TRUE

#if (XPBD_USE_BATCHED_CONSTRAINTS == TRUE)
struct XPBDDistanceConstraint { uint2 p1, p2; float rest_length; };
struct XPBDBendingConstraint { uint2 p1, p2, p3; float rest_length; };

struct XPBDBendDistConstraint { uint2 p1; float rest_length; }; //Can cheat due to rest_length being the same for all constraints on a uniform grid
#else
struct XPBDDistanceConstraint { uint out_offset; uint2 p1, p2;	float rest_length, k; };
struct XPBDBendingConstraint { uint out_offset; uint2 p1, p2, p3; float rest_length, k; };
#endif
struct XPBDSphereConstraint { float radius; float3 centre; };