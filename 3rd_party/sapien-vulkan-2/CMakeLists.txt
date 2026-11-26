cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(sapienvulkan2 LANGUAGES C CXX)

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "svulkan2 only supports Linux (got ${CMAKE_SYSTEM_NAME}).")
endif()

set(CMAKE_CXX_STANDARD 20)

option(SVULKAN2_SHARED "Force svulkan2 to be shared" ON)
option(SVULKAN2_PROFILE "Build with profiler" OFF)
option(SVULKAN2_BUILD_TEST "Build with unit test" OFF)
option(SVULKAN2_CUDA_INTEROP "Allow CUDA to use Vulkan buffer" ON)

add_compile_options("-Wall"
    "$<$<CONFIG:Debug>:-g3>" "$<$<CONFIG:Debug>:-O0>"
    "$<$<CONFIG:Release>:-O3>")

add_compile_definitions("$<$<CONFIG:DEBUG>:_DEBUG>")
add_compile_definitions("$<$<CONFIG:DEBUG>:VK_VALIDATION>")
add_compile_definitions("$<$<NOT:$<CONFIG:Debug>>:NDEBUG>")

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(glm)
include(assimp)
include(spdlog)
include(glfw)
include(ktx)
include(vulkan)
include(glslang)
include(spirv-cross)
include(imgui)
include(openexr)
include(openvr)

file(GLOB_RECURSE SVULKAN2_SRC "src/*.cpp")

if(SVULKAN2_SHARED)
    add_library(svulkan2 SHARED ${SVULKAN2_SRC})
else()
    add_library(svulkan2 ${SVULKAN2_SRC})
endif()

set_target_properties(svulkan2 PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

# The rt path tracer builds its environment/emissive alias tables in C++ (rt_integrator_tables.cpp)
# but the pure CPU table construction is a Zig module from the parent SAPIEN tree. Compile it in
# this directory's scope so the generated object is a source of the svulkan2 target here.
sapien_add_zig_module(svulkan2 ../../src/sapien_renderer/alias_table.zig)

target_compile_definitions(imgui PRIVATE VK_NO_PROTOTYPES)
target_compile_definitions(svulkan2 PUBLIC VK_NO_PROTOTYPES)

if (${VK_VALIDATION})
    target_compile_definitions(svulkan2 PUBLIC VK_VALIDATION)
endif()

# The PUBLIC compile definitions above and below, baked into a header that every public header
# including Vulkan includes, so programs built against the installed headers need no -D flags.
configure_file(cmake/build_options.h.in
    "${CMAKE_CURRENT_BINARY_DIR}/include/svulkan2/build_options.h")

target_include_directories(svulkan2 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(svulkan2 PRIVATE
    "3rd_party/stb" "3rd_party/OptiX7SDK/include"
    "3rd_party/tinygltf"
)
target_link_libraries(svulkan2 PUBLIC Vulkan::Headers glm)

find_package(Threads REQUIRED)
target_link_libraries(svulkan2 PRIVATE assimp::assimp glslang SPIRV spirv-cross-cpp glfw spdlog::spdlog ktx imgui ${CMAKE_DL_LIBS} Threads::Threads OpenEXR openvr_api)

if (${SVULKAN2_PROFILE})
    target_compile_definitions(svulkan2 PUBLIC SVULKAN2_PROFILE)
else()
    target_compile_definitions(svulkan2 PRIVATE NVTX_DISABLE=1)
endif()


if (${SVULKAN2_CUDA_INTEROP})
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(svulkan2 PUBLIC SVULKAN2_CUDA_INTEROP)
    target_link_libraries(svulkan2 PRIVATE CUDA::cudart_static)

    include(oidn)
    add_dependencies(svulkan2 OpenImageDenoise_device_cuda)
    ExternalProject_Get_property(OpenImageDenoise_device_cuda binary_dir)
    target_link_libraries(svulkan2 PRIVATE OpenImageDenoise)
endif()

# TODO: add option to control the following variables
# this library exports the storage api
target_compile_definitions(svulkan2 PRIVATE VULKAN_HPP_STORAGE_SHARED_EXPORT)
# other libraries use the exported storage api
target_compile_definitions(svulkan2 PUBLIC VULKAN_HPP_STORAGE_SHARED)

if (${SVULKAN2_CUDA_INTEROP})
    install(DIRECTORY ${binary_dir}/../preinstall/lib DESTINATION "." OPTIONAL)
    install(TARGETS svulkan2 Vulkan-Headers glm OpenImageDenoise OpenImageDenoise_core OpenImageDenoise_common
        EXPORT svulkan2Targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
else ()
    install(TARGETS svulkan2 Vulkan-Headers glm
        EXPORT svulkan2Targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif ()

install(EXPORT svulkan2Targets
    FILE svulkan2.cmake
    NAMESPACE svulkan2::
    DESTINATION lib/cmake/svulkan2
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/svulkan2 DESTINATION "include")
install(DIRECTORY "${vulkan_SOURCE_DIR}/include/vulkan" DESTINATION "include")
install(DIRECTORY "${vulkan_SOURCE_DIR}/include/vk_video" DESTINATION "include")
install(DIRECTORY "${glm_SOURCE_DIR}/glm" DESTINATION "include")

add_executable(svulkan2_main app/main.cpp)
target_link_libraries(svulkan2_main svulkan2 glfw imgui stdc++fs)

add_executable(svulkan2_vr app/vr.cpp)
target_link_libraries(svulkan2_vr svulkan2 glfw imgui stdc++fs)

add_executable(gencube app/gencube.cpp)
target_link_libraries(gencube svulkan2)

add_executable(glslc_embed app/glslc_embed.cpp)
target_link_libraries(glslc_embed svulkan2 stdc++fs)
