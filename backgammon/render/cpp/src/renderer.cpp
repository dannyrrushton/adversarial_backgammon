// Host side of the OptiX renderer: context, pipeline, shader binding table, acceleration structure
// and the C API declared in bgrender.h.
#include "bgrender.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "launch_params.h"

namespace
{

using bgr::LaunchParams;

#define BGR_CUDA(call)                                                                              \
    do                                                                                              \
    {                                                                                               \
        cudaError_t err_ = (call);                                                                  \
        if (err_ != cudaSuccess)                                                                    \
            throw std::runtime_error(std::string(#call) + " failed: " + cudaGetErrorString(err_)); \
    } while (0)

#define BGR_OPTIX(call)                                                                                      \
    do                                                                                                       \
    {                                                                                                        \
        OptixResult res_ = (call);                                                                           \
        if (res_ != OPTIX_SUCCESS)                                                                           \
            throw std::runtime_error(std::string(#call) + " failed: " + optixGetErrorName(res_) + " (" +    \
                                     optixGetErrorString(res_) + ")");                                       \
    } while (0)

// A device allocation that frees itself and only grows.
struct DeviceBuffer
{
    void* ptr = nullptr;
    size_t size = 0;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { cudaFree(ptr); }

    void reserve(size_t bytes)
    {
        if (bytes <= size)
            return;
        cudaFree(ptr);
        ptr = nullptr;
        size = 0;
        BGR_CUDA(cudaMalloc(&ptr, bytes));
        size = bytes;
    }

    void upload(const void* data, size_t bytes)
    {
        reserve(bytes);
        BGR_CUDA(cudaMemcpy(ptr, data, bytes, cudaMemcpyHostToDevice));
    }

    CUdeviceptr dptr() const { return reinterpret_cast<CUdeviceptr>(ptr); }
};

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
struct Empty
{
};

std::vector<char> read_file(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("cannot open OptiX module '" + path + "'");
    return std::vector<char>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

void log_callback(unsigned int level, const char* tag, const char* message, void*)
{
    if (level <= 2)
        fprintf(stderr, "[optix][%u][%s] %s\n", level, tag, message);
}

}  // namespace

struct BgrRenderer
{
    std::string error;
    std::string device_name;
    CUstream stream = nullptr;
    OptixDeviceContext context = nullptr;
    OptixModule module = nullptr;
    OptixPipeline pipeline = nullptr;
    std::vector<OptixProgramGroup> groups;
    OptixShaderBindingTable sbt = {};

    DeviceBuffer raygen_record, miss_records, hit_records;
    DeviceBuffer vertices, normals, indices, material_ids, materials;
    DeviceBuffer gas, gas_temp;
    DeviceBuffer image, accum, params_buffer;
    OptixTraversableHandle handle = 0;
    size_t num_triangles = 0;

    LaunchParams params = {};
    unsigned int accumulated = 0;
    BgrCamera last_camera = {};
    int last_width = 0, last_height = 0;

    BgrRenderer(const char* module_path, int device)
    {
        BGR_CUDA(cudaSetDevice(device));
        cudaDeviceProp prop;
        BGR_CUDA(cudaGetDeviceProperties(&prop, device));
        device_name = prop.name;
        BGR_CUDA(cudaFree(nullptr));  // create the primary context
        BGR_CUDA(cudaStreamCreate(&stream));
        BGR_OPTIX(optixInit());

        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &log_callback;
        options.logCallbackLevel = 2;
        BGR_OPTIX(optixDeviceContextCreate(nullptr, &options, &context));
        build_pipeline(read_file(module_path));
        set_default_lighting();
    }

    ~BgrRenderer()
    {
        if (pipeline)
            optixPipelineDestroy(pipeline);
        for (OptixProgramGroup g : groups)
            optixProgramGroupDestroy(g);
        if (module)
            optixModuleDestroy(module);
        if (context)
            optixDeviceContextDestroy(context);
        if (stream)
            cudaStreamDestroy(stream);
    }

    void build_pipeline(const std::vector<char>& code)
    {
        OptixModuleCompileOptions module_options = {};
        module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

        OptixPipelineCompileOptions pipeline_options = {};
        pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_options.numPayloadValues = 2;
        pipeline_options.numAttributeValues = 2;
        pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_options.pipelineLaunchParamsVariableName = "params";
        pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

        char log[4096];
        size_t log_size = sizeof(log);
        BGR_OPTIX(optixModuleCreate(context, &module_options, &pipeline_options, code.data(), code.size(), log,
                                    &log_size, &module));

        auto make_group = [&](OptixProgramGroupDesc desc) {
            OptixProgramGroupOptions group_options = {};
            OptixProgramGroup group = nullptr;
            size_t size = sizeof(log);
            BGR_OPTIX(optixProgramGroupCreate(context, &desc, 1, &group_options, log, &size, &group));
            groups.push_back(group);
            return group;
        };

        OptixProgramGroupDesc raygen = {};
        raygen.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen.raygen.module = module;
        raygen.raygen.entryFunctionName = "__raygen__camera";
        OptixProgramGroup raygen_group = make_group(raygen);

        OptixProgramGroupDesc miss = {};
        miss.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss.miss.module = module;
        miss.miss.entryFunctionName = "__miss__radiance";
        OptixProgramGroup miss_radiance = make_group(miss);
        miss.miss.entryFunctionName = "__miss__shadow";
        OptixProgramGroup miss_shadow = make_group(miss);

        OptixProgramGroupDesc hit = {};
        hit.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit.hitgroup.moduleCH = module;
        hit.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        OptixProgramGroup hit_radiance = make_group(hit);
        OptixProgramGroupDesc shadow_hit = {};
        shadow_hit.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;  // no programs: shadow rays skip closest hit
        OptixProgramGroup hit_shadow = make_group(shadow_hit);

        const unsigned int max_trace_depth = 2;  // radiance -> shadow; reflections loop in raygen
        OptixPipelineLinkOptions link_options = {};
        link_options.maxTraceDepth = max_trace_depth;
        size_t size = sizeof(log);
        BGR_OPTIX(optixPipelineCreate(context, &pipeline_options, &link_options, groups.data(),
                                      static_cast<unsigned int>(groups.size()), log, &size, &pipeline));

        OptixStackSizes stack_sizes = {};
        for (OptixProgramGroup g : groups)
            BGR_OPTIX(optixUtilAccumulateStackSizes(g, &stack_sizes, pipeline));
        unsigned int from_traversal, from_state, continuation;
        BGR_OPTIX(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth, 0, 0, &from_traversal, &from_state,
                                             &continuation));
        BGR_OPTIX(optixPipelineSetStackSize(pipeline, from_traversal, from_state, continuation, 1));

        SbtRecord<Empty> rg;
        BGR_OPTIX(optixSbtRecordPackHeader(raygen_group, &rg));
        raygen_record.upload(&rg, sizeof(rg));

        SbtRecord<Empty> ms[bgr::RAY_TYPE_COUNT];
        BGR_OPTIX(optixSbtRecordPackHeader(miss_radiance, &ms[bgr::RAY_RADIANCE]));
        BGR_OPTIX(optixSbtRecordPackHeader(miss_shadow, &ms[bgr::RAY_SHADOW]));
        miss_records.upload(ms, sizeof(ms));

        SbtRecord<Empty> hg[bgr::RAY_TYPE_COUNT];
        BGR_OPTIX(optixSbtRecordPackHeader(hit_radiance, &hg[bgr::RAY_RADIANCE]));
        BGR_OPTIX(optixSbtRecordPackHeader(hit_shadow, &hg[bgr::RAY_SHADOW]));
        hit_records.upload(hg, sizeof(hg));

        sbt.raygenRecord = raygen_record.dptr();
        sbt.missRecordBase = miss_records.dptr();
        sbt.missRecordStrideInBytes = sizeof(SbtRecord<Empty>);
        sbt.missRecordCount = bgr::RAY_TYPE_COUNT;
        sbt.hitgroupRecordBase = hit_records.dptr();
        sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<Empty>);
        sbt.hitgroupRecordCount = bgr::RAY_TYPE_COUNT;
    }

    void set_default_lighting()
    {
        const BgrLight lights[2] = {
            {{-4.f, 9.f, 5.f}, {0.95f, 0.9f, 0.82f}, 0.8f},
            {{5.f, 7.f, -4.f}, {0.35f, 0.38f, 0.45f}, 1.2f},
        };
        const float ambient[3] = {0.16f, 0.16f, 0.18f};
        set_lights(lights, 2, ambient);
        params.sky_top = make_float3(0.10f, 0.12f, 0.16f);
        params.sky_bottom = make_float3(0.03f, 0.03f, 0.04f);
        params.max_bounces = 2;
    }

    void set_lights(const BgrLight* lights, size_t count, const float ambient[3])
    {
        if (count > bgr::MAX_LIGHTS)
            throw std::runtime_error("too many lights (max " + std::to_string(bgr::MAX_LIGHTS) + ")");
        for (size_t i = 0; i < count; ++i)
        {
            params.lights[i].position = make3(lights[i].position);
            params.lights[i].color = make3(lights[i].color);
            params.lights[i].radius = lights[i].radius;
        }
        params.num_lights = static_cast<int>(count);
        params.ambient = make3(ambient);
        accumulated = 0;
    }

    void set_scene(const float* positions, const float* vertex_normals, size_t num_vertices, const uint32_t* tri_indices,
                   const uint32_t* tri_materials, size_t triangles, const BgrMaterial* mats, size_t num_materials)
    {
        if (num_vertices == 0 || triangles == 0)
            throw std::runtime_error("scene is empty");
        for (size_t i = 0; i < triangles * 3; ++i)
            if (tri_indices[i] >= num_vertices)
                throw std::runtime_error("triangle index out of range");
        for (size_t i = 0; i < triangles; ++i)
            if (tri_materials[i] >= num_materials)
                throw std::runtime_error("material id out of range");

        std::vector<bgr::Material> device_mats(num_materials);
        for (size_t i = 0; i < num_materials; ++i)
        {
            const BgrMaterial& m = mats[i];
            device_mats[i] = {make3(m.albedo), m.specular, m.shininess, m.reflectivity, make3(m.emission), m.pattern};
        }

        vertices.upload(positions, num_vertices * 3 * sizeof(float));
        normals.upload(vertex_normals, num_vertices * 3 * sizeof(float));
        indices.upload(tri_indices, triangles * 3 * sizeof(uint32_t));
        material_ids.upload(tri_materials, triangles * sizeof(uint32_t));
        materials.upload(device_mats.data(), device_mats.size() * sizeof(bgr::Material));
        num_triangles = triangles;

        OptixBuildInput input = {};
        input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        CUdeviceptr vertex_ptr = vertices.dptr();
        input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        input.triangleArray.vertexStrideInBytes = 3 * sizeof(float);
        input.triangleArray.numVertices = static_cast<unsigned int>(num_vertices);
        input.triangleArray.vertexBuffers = &vertex_ptr;
        input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        input.triangleArray.indexStrideInBytes = 3 * sizeof(uint32_t);
        input.triangleArray.numIndexTriplets = static_cast<unsigned int>(triangles);
        input.triangleArray.indexBuffer = indices.dptr();
        const unsigned int flags[1] = {OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};
        input.triangleArray.flags = flags;
        input.triangleArray.numSbtRecords = 1;

        OptixAccelBuildOptions accel = {};
        accel.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accel.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes sizes;
        BGR_OPTIX(optixAccelComputeMemoryUsage(context, &accel, &input, 1, &sizes));
        gas_temp.reserve(sizes.tempSizeInBytes);
        gas.reserve(sizes.outputSizeInBytes);
        BGR_OPTIX(optixAccelBuild(context, stream, &accel, &input, 1, gas_temp.dptr(), sizes.tempSizeInBytes,
                                  gas.dptr(), sizes.outputSizeInBytes, &handle, nullptr, 0));
        BGR_CUDA(cudaStreamSynchronize(stream));
        accumulated = 0;
    }

    void render(const BgrCamera& cam, int width, int height, int samples, bool accumulate, uint8_t* out)
    {
        if (!handle)
            throw std::runtime_error("no scene: call bgr_set_scene first");
        if (width <= 0 || height <= 0 || samples <= 0)
            throw std::runtime_error("width, height and samples must be positive");

        const bool same_view = width == last_width && height == last_height &&
                               std::memcmp(&cam, &last_camera, sizeof(cam)) == 0;
        if (!accumulate || !same_view)
            accumulated = 0;
        last_camera = cam;
        last_width = width;
        last_height = height;

        const size_t pixels = static_cast<size_t>(width) * height;
        image.reserve(pixels * sizeof(uchar4));
        accum.reserve(pixels * sizeof(float4));

        // Camera basis: w points at the target, u right, v up, scaled to the field of view.
        const float3 eye = make3(cam.eye);
        const float3 forward = normalize(make3(cam.look_at) - eye);
        const float3 right = normalize(cross(forward, make3(cam.up)));
        const float3 up = cross(right, forward);
        const float half_h = tanf(0.5f * cam.fov_y_degrees * 3.14159265f / 180.f);
        const float half_w = half_h * static_cast<float>(width) / static_cast<float>(height);

        params.image = static_cast<uchar4*>(image.ptr);
        params.accum = static_cast<float4*>(accum.ptr);
        params.width = width;
        params.height = height;
        params.samples = samples;
        params.sample_index = accumulated;
        params.eye = eye;
        params.u = right * half_w;
        params.v = up * half_h;
        params.w = forward;
        params.handle = handle;
        params.normals = static_cast<const float3*>(normals.ptr);
        params.indices = static_cast<const uint3*>(indices.ptr);
        params.material_ids = static_cast<const unsigned int*>(material_ids.ptr);
        params.materials = static_cast<const bgr::Material*>(materials.ptr);

        params_buffer.upload(&params, sizeof(params));
        BGR_OPTIX(optixLaunch(pipeline, stream, params_buffer.dptr(), sizeof(LaunchParams), &sbt, width, height, 1));
        BGR_CUDA(cudaStreamSynchronize(stream));
        BGR_CUDA(cudaMemcpy(out, image.ptr, pixels * sizeof(uchar4), cudaMemcpyDeviceToHost));
        accumulated += samples;
    }
};

namespace
{

template <typename F>
int guarded(BgrRenderer* r, F&& f)
{
    if (!r)
        return -1;
    try
    {
        f();
        r->error.clear();
        return 0;
    }
    catch (const std::exception& e)
    {
        r->error = e.what();
        return 1;
    }
}

}  // namespace

extern "C" {

BgrRenderer* bgr_create(const char* module_path, int device, char* error, size_t error_size)
{
    try
    {
        return new BgrRenderer(module_path, device);
    }
    catch (const std::exception& e)
    {
        if (error && error_size)
        {
            std::strncpy(error, e.what(), error_size - 1);
            error[error_size - 1] = '\0';
        }
        return nullptr;
    }
}

void bgr_destroy(BgrRenderer* renderer)
{
    delete renderer;
}

const char* bgr_last_error(const BgrRenderer* renderer)
{
    return renderer ? renderer->error.c_str() : "null renderer";
}

const char* bgr_device_name(const BgrRenderer* renderer)
{
    return renderer ? renderer->device_name.c_str() : "";
}

int bgr_set_scene(BgrRenderer* renderer, const float* positions, const float* normals, size_t num_vertices,
                  const uint32_t* indices, const uint32_t* material_ids, size_t num_triangles,
                  const BgrMaterial* materials, size_t num_materials)
{
    return guarded(renderer, [&] {
        renderer->set_scene(positions, normals, num_vertices, indices, material_ids, num_triangles, materials,
                            num_materials);
    });
}

int bgr_set_lights(BgrRenderer* renderer, const BgrLight* lights, size_t num_lights, const float ambient[3])
{
    return guarded(renderer, [&] { renderer->set_lights(lights, num_lights, ambient); });
}

int bgr_render(BgrRenderer* renderer, const BgrCamera* camera, int width, int height, int samples, int accumulate,
               uint8_t* out_rgba)
{
    return guarded(renderer, [&] { renderer->render(*camera, width, height, samples, accumulate != 0, out_rgba); });
}

int bgr_accumulated_samples(const BgrRenderer* renderer)
{
    return renderer ? static_cast<int>(renderer->accumulated) : 0;
}

}  // extern "C"
