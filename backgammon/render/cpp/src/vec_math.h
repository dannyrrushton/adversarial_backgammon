// Small float3 helpers usable from host and device code (the OptiX headers ship none).
#pragma once

#include <cuda_runtime.h>

#include <cmath>

#if defined(__CUDACC__)
#define BGR_HD __host__ __device__ __forceinline__
#else
#define BGR_HD inline
#endif

BGR_HD float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
BGR_HD float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
BGR_HD float3 operator-(float3 a) { return make_float3(-a.x, -a.y, -a.z); }
BGR_HD float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
BGR_HD float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
BGR_HD float3 operator*(float s, float3 a) { return a * s; }
BGR_HD float3 operator/(float3 a, float s) { return a * (1.0f / s); }
BGR_HD float3& operator+=(float3& a, float3 b)
{
    a = a + b;
    return a;
}
BGR_HD float3& operator*=(float3& a, float3 b)
{
    a = a * b;
    return a;
}
BGR_HD float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
BGR_HD float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
BGR_HD float length(float3 a) { return sqrtf(dot(a, a)); }
BGR_HD float3 normalize(float3 a) { return a * (1.0f / sqrtf(dot(a, a))); }
BGR_HD float3 reflect(float3 d, float3 n) { return d - n * (2.0f * dot(d, n)); }
BGR_HD float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
BGR_HD float3 lerp(float3 a, float3 b, float t) { return a + (b - a) * t; }
BGR_HD float3 make3(const float* p) { return make_float3(p[0], p[1], p[2]); }
