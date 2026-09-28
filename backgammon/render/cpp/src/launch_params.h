// Data shared between the host renderer and the OptiX device programs.
#pragma once

#include <optix.h>

#include "vec_math.h"

namespace bgr
{

enum RayType : unsigned int
{
    RAY_RADIANCE = 0,
    RAY_SHADOW   = 1,
    RAY_TYPE_COUNT
};

struct Material
{
    float3 albedo;
    float specular;
    float shininess;
    float reflectivity;
    float3 emission;
    int pattern;
};

struct Light
{
    float3 position;
    float3 color;
    float radius;
};

constexpr int MAX_LIGHTS = 4;

struct LaunchParams
{
    // Output
    uchar4* image;
    float4* accum;
    unsigned int width;
    unsigned int height;
    unsigned int samples;       // samples to take this launch
    unsigned int sample_index;  // samples already accumulated before this launch

    // Camera: primary ray direction = normalize(w + sx * u + sy * v) for sx, sy in [-1, 1]
    float3 eye;
    float3 u;
    float3 v;
    float3 w;

    // Lighting
    Light lights[MAX_LIGHTS];
    int num_lights;
    float3 ambient;
    float3 sky_top;
    float3 sky_bottom;
    unsigned int max_bounces;

    // Scene
    OptixTraversableHandle handle;
    const float3* normals;
    const uint3* indices;
    const unsigned int* material_ids;
    const Material* materials;
};

}  // namespace bgr
