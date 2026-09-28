// OptiX device programs: a Whitted-style ray tracer with soft shadows, glossy reflections and
// progressive anti-aliasing. Compiled to OptiX IR and loaded by renderer.cpp at runtime.
#include <optix.h>

#include "launch_params.h"

using namespace bgr;

extern "C" __constant__ LaunchParams params;

// --- random numbers -----------------------------------------------------------------------------

static __forceinline__ __device__ unsigned int tea(unsigned int v0, unsigned int v1)
{
    unsigned int s0 = 0;
    for (int n = 0; n < 4; ++n)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}

static __forceinline__ __device__ float rnd(unsigned int& state)
{
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(state & 0x00FFFFFF) / static_cast<float>(0x01000000);
}

// --- procedural patterns --------------------------------------------------------------------------

static __forceinline__ __device__ float hash3(float3 p)
{
    const float h = dot(p, make_float3(127.1f, 311.7f, 74.7f));
    const float s = sinf(h) * 43758.5453f;
    return s - floorf(s);
}

static __forceinline__ __device__ float value_noise(float3 p)
{
    const float3 i = make_float3(floorf(p.x), floorf(p.y), floorf(p.z));
    const float3 f = p - i;
    const float3 u = f * f * (make_float3(3.f, 3.f, 3.f) - f * 2.f);
    float c[8];
    for (int k = 0; k < 8; ++k)
        c[k] = hash3(i + make_float3(float(k & 1), float((k >> 1) & 1), float((k >> 2) & 1)));
    const float x00 = c[0] + (c[1] - c[0]) * u.x, x10 = c[2] + (c[3] - c[2]) * u.x;
    const float x01 = c[4] + (c[5] - c[4]) * u.x, x11 = c[6] + (c[7] - c[6]) * u.x;
    const float y0 = x00 + (x10 - x00) * u.y, y1 = x01 + (x11 - x01) * u.y;
    return y0 + (y1 - y0) * u.z;
}

static __forceinline__ __device__ float3 apply_pattern(const Material& m, float3 p)
{
    if (m.pattern == 1)
    {
        // Wood: stretched rings along x, perturbed by noise so the grain wanders.
        const float n = value_noise(make_float3(p.x * 0.6f, p.y * 4.f, p.z * 4.f));
        const float rings = 0.5f + 0.5f * sinf((p.z * 9.f + p.y * 3.f + n * 5.f) * 3.14159f);
        const float fine = value_noise(make_float3(p.x * 20.f, p.y * 60.f, p.z * 60.f));
        const float t = 0.75f + 0.2f * rings + 0.1f * fine;
        return m.albedo * t;
    }
    if (m.pattern == 2)
    {
        const float n = value_noise(p * 90.f);
        return m.albedo * (0.9f + 0.15f * n);
    }
    return m.albedo;
}

// --- payload helpers --------------------------------------------------------------------------------

struct RadiancePRD
{
    float3 radiance;
    float3 next_origin;
    float3 next_direction;
    float reflectivity;  // weight of the next bounce, 0 = stop
    unsigned int seed;
};

static __forceinline__ __device__ void* unpack_pointer(unsigned int i0, unsigned int i1)
{
    const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
    return reinterpret_cast<void*>(uptr);
}

static __forceinline__ __device__ void pack_pointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffffull;
}

static __forceinline__ __device__ RadiancePRD* get_prd()
{
    return reinterpret_cast<RadiancePRD*>(unpack_pointer(optixGetPayload_0(), optixGetPayload_1()));
}

static __forceinline__ __device__ bool visible(float3 origin, float3 direction, float distance)
{
    unsigned int seen = 0;
    optixTrace(params.handle, origin, direction, 0.f, distance, 0.f, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               RAY_SHADOW, RAY_TYPE_COUNT, RAY_SHADOW, seen);
    return seen != 0;
}

static __forceinline__ __device__ float3 sky(float3 dir)
{
    const float t = 0.5f * (dir.y + 1.f);
    return lerp(params.sky_bottom, params.sky_top, clampf(t, 0.f, 1.f));
}

// --- programs ---------------------------------------------------------------------------------------

extern "C" __global__ void __raygen__camera()
{
    const uint3 idx = optixGetLaunchIndex();
    const unsigned int pixel = idx.y * params.width + idx.x;
    unsigned int seed = tea(pixel, params.sample_index + 7919u);

    float3 sum = make_float3(0.f, 0.f, 0.f);
    for (unsigned int s = 0; s < params.samples; ++s)
    {
        // Jittered sub-pixel position; the first ever sample goes through the pixel centre so a
        // single-sample preview is stable while the camera moves.
        const bool centre = params.sample_index == 0 && s == 0;
        const float jx = centre ? 0.5f : rnd(seed);
        const float jy = centre ? 0.5f : rnd(seed);
        const float sx = 2.f * (idx.x + jx) / params.width - 1.f;
        const float sy = 1.f - 2.f * (idx.y + jy) / params.height;  // row 0 is the top of the image

        RadiancePRD prd;
        prd.seed = seed;
        float3 origin = params.eye;
        float3 direction = normalize(params.w + params.u * sx + params.v * sy);
        float3 throughput = make_float3(1.f, 1.f, 1.f);
        float3 colour = make_float3(0.f, 0.f, 0.f);

        for (unsigned int depth = 0; depth <= params.max_bounces; ++depth)
        {
            prd.reflectivity = 0.f;
            unsigned int p0, p1;
            pack_pointer(&prd, p0, p1);
            optixTrace(params.handle, origin, direction, 1e-4f, 1e16f, 0.f, OptixVisibilityMask(255),
                       OPTIX_RAY_FLAG_NONE, RAY_RADIANCE, RAY_TYPE_COUNT, RAY_RADIANCE, p0, p1);
            colour += throughput * prd.radiance;
            if (prd.reflectivity <= 0.01f)
                break;
            throughput *= make_float3(prd.reflectivity, prd.reflectivity, prd.reflectivity);
            origin = prd.next_origin;
            direction = prd.next_direction;
        }
        seed = prd.seed;
        sum += colour;
    }

    float4 acc = make_float4(sum.x, sum.y, sum.z, 0.f);
    if (params.sample_index > 0)
    {
        const float4 prev = params.accum[pixel];
        acc = make_float4(prev.x + acc.x, prev.y + acc.y, prev.z + acc.z, 0.f);
    }
    params.accum[pixel] = acc;

    const float inv = 1.f / static_cast<float>(params.sample_index + params.samples);
    float3 c = make_float3(acc.x * inv, acc.y * inv, acc.z * inv);
    // Reinhard-style tone map, then gamma 2.2.
    c = make_float3(c.x / (1.f + c.x * 0.15f), c.y / (1.f + c.y * 0.15f), c.z / (1.f + c.z * 0.15f));
    const float g = 1.f / 2.2f;
    params.image[pixel] = make_uchar4(static_cast<unsigned char>(255.f * clampf(powf(c.x, g), 0.f, 1.f)),
                                      static_cast<unsigned char>(255.f * clampf(powf(c.y, g), 0.f, 1.f)),
                                      static_cast<unsigned char>(255.f * clampf(powf(c.z, g), 0.f, 1.f)), 255);
}

extern "C" __global__ void __miss__radiance()
{
    RadiancePRD* prd = get_prd();
    prd->radiance = sky(optixGetWorldRayDirection());
    prd->reflectivity = 0.f;
}

extern "C" __global__ void __miss__shadow()
{
    optixSetPayload_0(1u);
}

extern "C" __global__ void __closesthit__radiance()
{
    RadiancePRD* prd = get_prd();
    const unsigned int prim = optixGetPrimitiveIndex();
    const uint3 tri = params.indices[prim];
    const float2 bary = optixGetTriangleBarycentrics();
    const float3 n0 = params.normals[tri.x], n1 = params.normals[tri.y], n2 = params.normals[tri.z];
    float3 n = normalize(n0 * (1.f - bary.x - bary.y) + n1 * bary.x + n2 * bary.y);

    const float3 dir = optixGetWorldRayDirection();
    if (dot(n, dir) > 0.f)
        n = -n;  // two-sided shading
    const float3 p = optixGetWorldRayOrigin() + dir * optixGetRayTmax();
    const float3 origin = p + n * 1e-3f;

    const Material& m = params.materials[params.material_ids[prim]];
    const float3 albedo = apply_pattern(m, p);

    float3 colour = params.ambient * albedo + m.emission;
    for (int i = 0; i < params.num_lights; ++i)
    {
        const Light& light = params.lights[i];
        // Sample a point on the spherical light for soft shadows.
        const float3 jitter = make_float3(rnd(prd->seed) - 0.5f, rnd(prd->seed) - 0.5f, rnd(prd->seed) - 0.5f);
        const float3 target = light.position + jitter * (2.f * light.radius);
        float3 to_light = target - p;
        const float dist = length(to_light);
        to_light = to_light / dist;
        const float ndotl = dot(n, to_light);
        if (ndotl <= 0.f || !visible(origin, to_light, dist - 1e-3f))
            continue;
        const float3 h = normalize(to_light - dir);
        const float spec = m.specular * powf(fmaxf(dot(n, h), 0.f), m.shininess);
        colour += light.color * (albedo * ndotl + make_float3(spec, spec, spec));
    }

    prd->radiance = colour * (1.f - m.reflectivity);
    prd->reflectivity = m.reflectivity;
    prd->next_origin = origin;
    prd->next_direction = reflect(dir, n);
}
