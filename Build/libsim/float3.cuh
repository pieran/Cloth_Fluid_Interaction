
#pragma once
#include <cuda_runtime.h>
#include <math.h>

//TO SHUT UP GDAMN INTELLISENSE
const uint3 blockIdx;
const dim3 blockDim;
const uint3 threadIdx;

__host__ __device__
inline float3 operator+(const float3& a, const float3& b)
{
	float3 t;
	t.x = a.x + b.x;
	t.y = a.y + b.y;
	t.z = a.z + b.z;
	return t;
}

__host__ __device__
inline float3 operator-(const float3& a, const float3& b)
{
	float3 t;
	t.x = a.x - b.x;
	t.y = a.y - b.y;
	t.z = a.z - b.z;
	return t;
}

__host__ __device__
inline float3 operator*(float3 a, float b)
{
	float3 t;
	t.x = a.x * b;
	t.y = a.y * b;
	t.z = a.z * b;
	return t;
}

__host__ __device__
inline float3 operator/(float3 a, float b)
{
	float3 t;
	t.x = a.x / b;
	t.y = a.y / b;
	t.z = a.z / b;
	return t;
}

__host__ __device__
inline float3 operator/(float3 a, float3 b)
{
	float3 t;
	t.x = a.x / b.x;
	t.y = a.y / b.y;
	t.z = a.z / b.z;
	return t;
}

__host__ __device__
inline void operator+=(float3& a, const float3& b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

__host__ __device__
inline void operator+=(int3& a, const int3& b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

__host__ __device__
inline void operator-=(float3& a, const float3& b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}

__host__ __device__
inline void operator*=(float3& a, float b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

__host__ __device__
inline void operator/=(float3& a, float b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
}


__host__ __device__
inline int3 operator+(const int3& a, const int3& b)
{
	int3 t;
	t.x = a.x + b.x;
	t.y = a.y + b.y;
	t.z = a.z + b.z;
	return t;
}

__host__ __device__
inline int3 operator-(const int3& a, const int3& b)
{
	int3 t;
	t.x = a.x - b.x;
	t.y = a.y - b.y;
	t.z = a.z - b.z;
	return t;
}

__host__ __device__
inline int3 operator*(const int3& a, int b)
{
	int3 t;
	t.x = a.x * b;
	t.y = a.y * b;
	t.z = a.z * b;
	return t;
}

__host__ __device__
inline int3 operator/(const int3& a, int b)
{
	int3 t;
	t.x = a.x / b;
	t.y = a.y / b;
	t.z = a.z / b;
	return t;
}





__host__ __device__
inline float4 operator+(const float4& a, const float4& b)
{
	float4 t;
	t.x = a.x + b.x;
	t.y = a.y + b.y;
	t.z = a.z + b.z;
	t.w = a.w + b.w;
	return t;
}

__host__ __device__
inline float4 operator-(const float4& a, const float4& b)
{
	float4 t;
	t.x = a.x - b.x;
	t.y = a.y - b.y;
	t.z = a.z - b.z;
	t.w = a.w - b.w;
	return t;
}

__host__ __device__
inline float4 operator*(float4 a, float b)
{
	float4 t;
	t.x = a.x * b;
	t.y = a.y * b;
	t.z = a.z * b;
	t.w = a.w * b;
	return t;
}

__host__ __device__
inline float4 operator/(float4 a, float b)
{
	float4 t;
	t.x = a.x / b;
	t.y = a.y / b;
	t.z = a.z / b;
	t.w = a.w / b;
	return t;
}

__host__ __device__
inline void operator+=(float4& a, const float4& b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}

__host__ __device__
inline void operator-=(float4& a, const float4& b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}

__host__ __device__
inline void operator*=(float4& a, float b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}

__host__ __device__
inline void operator/=(float4& a, float b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
	a.w /= b;
}













__host__ __device__
inline float3 float3_make()
{
	float3 t;
	t.x = 0.0f;
	t.y = 0.0f;
	t.z = 0.0f;
	return t;
}

__host__ __device__
inline float3 float3_make(float v)
{
	float3 t;
	t.x = v;
	t.y = v;
	t.z = v;
	return t;
}

__host__ __device__
inline float3 float3_make(float x, float y, float z)
{
	float3 t;
	t.x = x;
	t.y = y;
	t.z = z;
	return t;
}

__host__ __device__
inline float float3_dot(const float3& a, const float3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__
inline float float3_length(const float3& a)
{
	return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

__host__ __device__
inline float float3_length_fast(const float3& a)
{
	return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

__host__ __device__
inline float float3_distance(const float3& a, const float3& b)
{
	return float3_length(b - a);
}

__host__ __device__
inline float float3_distance_fast(const float3& a, const float3& b)
{
	return float3_length_fast(b - a);
}

__host__ __device__
inline float3 float3_cross(const float3& a, const float3& b)
{
	return make_float3((a.y*b.z) - (a.z*b.y), (a.z*b.x) - (a.x*b.z), (a.x*b.y) - (a.y*b.x));
}

__host__ __device__
inline float3 float3_normalize(float3 a)
{
	float ilen = float3_length_fast(a);
	if (ilen > 0.0f)
	{
		ilen = 1.0f / ilen;
		a *= ilen;
	}
	return a;
}

