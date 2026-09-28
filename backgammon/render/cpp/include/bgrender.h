// C API of the OptiX backgammon board renderer.
//
// Python drives this through ctypes (backgammon/render/optix.py), so the interface is plain C:
// opaque handle, POD structs, and int status codes (0 = success) with bgr_last_error() for detail.
// The scene is a triangle soup with per-vertex normals and a material index per triangle; Python
// builds the board geometry and re-uploads it whenever the position changes.
#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define BGR_API __declspec(dllexport)
#else
#define BGR_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum BgrPattern
{
    BGR_PATTERN_NONE = 0,
    BGR_PATTERN_WOOD = 1,  // procedural wood grain modulating the albedo
    BGR_PATTERN_FELT = 2,  // fine noise for cloth
};

typedef struct BgrMaterial
{
    float albedo[3];
    float specular;      // Blinn-Phong specular strength
    float shininess;     // Blinn-Phong exponent
    float reflectivity;  // mirror reflection weight, 0..1
    float emission[3];   // added unlit colour, used for highlights
    int32_t pattern;     // BgrPattern
} BgrMaterial;

typedef struct BgrCamera
{
    float eye[3];
    float look_at[3];
    float up[3];
    float fov_y_degrees;
} BgrCamera;

typedef struct BgrLight
{
    float position[3];
    float color[3];
    float radius;  // spherical area light radius, gives soft shadows
} BgrLight;

typedef struct BgrRenderer BgrRenderer;

// Creates a renderer on CUDA device `device`, loading the OptiX IR module at `module_path`.
// Returns NULL on failure and writes the reason into `error` (if non-NULL).
BGR_API BgrRenderer* bgr_create(const char* module_path, int device, char* error, size_t error_size);
BGR_API void bgr_destroy(BgrRenderer* renderer);
BGR_API const char* bgr_last_error(const BgrRenderer* renderer);
BGR_API const char* bgr_device_name(const BgrRenderer* renderer);

// Uploads geometry and rebuilds the acceleration structure. positions/normals hold 3 floats per
// vertex, indices 3 per triangle, material_ids 1 per triangle. Resets progressive accumulation.
BGR_API int bgr_set_scene(BgrRenderer* renderer,
                          const float* positions,
                          const float* normals,
                          size_t num_vertices,
                          const uint32_t* indices,
                          const uint32_t* material_ids,
                          size_t num_triangles,
                          const BgrMaterial* materials,
                          size_t num_materials);

// Up to BGR_MAX_LIGHTS lights plus an ambient term. Resets accumulation.
#define BGR_MAX_LIGHTS 4
BGR_API int bgr_set_lights(BgrRenderer* renderer, const BgrLight* lights, size_t num_lights, const float ambient[3]);

// Renders `samples` jittered samples per pixel into `out_rgba` (width*height*4 bytes, top row
// first). With `accumulate` set and nothing changed since the previous call, the new samples are
// averaged with the earlier ones, so repeated calls refine a still image.
BGR_API int bgr_render(BgrRenderer* renderer,
                       const BgrCamera* camera,
                       int width,
                       int height,
                       int samples,
                       int accumulate,
                       uint8_t* out_rgba);

// Total samples per pixel accumulated in the last rendered image.
BGR_API int bgr_accumulated_samples(const BgrRenderer* renderer);

#ifdef __cplusplus
}
#endif
