cmake_minimum_required(VERSION 3.27)
project(bgrender LANGUAGES CXX CUDA)

# OptiX ray tracer for the backgammon board, exposed to Python as a C shared library.
#
#   cmake -S backgammon/render/cpp -B backgammon/render/cpp/build
#   cmake --build backgammon/render/cpp/build -j
#
# Produces build/lib/libbgrender.so plus build/lib/bgrender_device.optixir, which the Python
# module backgammon.render.optix loads.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(CUDAToolkit REQUIRED)

# OptiX is header-only on the SDK side (the implementation ships in the NVIDIA driver). Use an
# explicit path, an SDK install, or fetch NVIDIA's public header repository.
set(OPTIX_INCLUDE_DIR "" CACHE PATH "Directory containing optix.h")
if(NOT OPTIX_INCLUDE_DIR AND DEFINED ENV{OptiX_INSTALL_DIR})
    set(OPTIX_INCLUDE_DIR "$ENV{OptiX_INSTALL_DIR}/include")
endif()
if(NOT OPTIX_INCLUDE_DIR)
    include(FetchContent)
    FetchContent_Declare(optix_dev
        GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
        GIT_TAG v9.1.0
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR no-cmake-project)
    FetchContent_MakeAvailable(optix_dev)
    set(OPTIX_INCLUDE_DIR "${optix_dev_SOURCE_DIR}/include")
endif()
if(NOT EXISTS "${OPTIX_INCLUDE_DIR}/optix.h")
    message(FATAL_ERROR "optix.h not found in OPTIX_INCLUDE_DIR='${OPTIX_INCLUDE_DIR}'")
endif()
message(STATUS "OptiX headers: ${OPTIX_INCLUDE_DIR}")

# Device programs compiled to OptiX IR and loaded at runtime.
add_library(bgrender_device OBJECT src/device.cu)
set_target_properties(bgrender_device PROPERTIES CUDA_OPTIX_COMPILATION ON)
target_include_directories(bgrender_device PRIVATE ${OPTIX_INCLUDE_DIR} src)
target_compile_options(bgrender_device PRIVATE --use_fast_math -lineinfo -diag-suppress 20044)

add_library(bgrender SHARED src/renderer.cpp)
add_dependencies(bgrender bgrender_device)
target_include_directories(bgrender PUBLIC include PRIVATE src ${OPTIX_INCLUDE_DIR})
target_link_libraries(bgrender PRIVATE CUDA::cudart_static ${CMAKE_DL_LIBS})
add_custom_command(TARGET bgrender POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:bgrender_device>
            $<TARGET_FILE_DIR:bgrender>/bgrender_device.optixir
    COMMAND_EXPAND_LISTS
    COMMENT "Copying OptiX IR next to libbgrender")

enable_testing()
add_executable(bgrender_smoke tests/smoke.cpp)
target_link_libraries(bgrender_smoke PRIVATE bgrender)
target_compile_definitions(bgrender_smoke PRIVATE
    BGR_MODULE_PATH="$<TARGET_FILE_DIR:bgrender>/bgrender_device.optixir")
add_test(NAME bgrender_smoke COMMAND bgrender_smoke)
