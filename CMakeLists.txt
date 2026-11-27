cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(sapien LANGUAGES C CXX CUDA)

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "SAPIEN only supports Linux (got ${CMAKE_SYSTEM_NAME}).")
endif()

set(CMAKE_CXX_STANDARD 20)

add_compile_options("-Wall"
    "$<$<CONFIG:Debug>:-g3>" "$<$<CONFIG:Debug>:-O0>"
    "$<$<CONFIG:Release>:-O3>")

add_compile_definitions("$<$<CONFIG:DEBUG>:_DEBUG>")
add_compile_definitions("$<$<CONFIG:DEBUG>:VK_VALIDATION>")
add_compile_definitions("$<$<CONFIG:DEBUG>:VK_CORE_LOGGING>")
add_compile_definitions("$<$<NOT:$<CONFIG:Debug>>:NDEBUG>")

set(CMAKE_DEBUG_POSTFIX "")

option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)
option(SAPIEN_PROFILE "Build SAPIEN with profiler" ON)
option(SAPIEN_DEBUG_VIEWER "Build debug viewer for debugging renderer in C++" OFF)
option(SAPIEN_CUDA "Enable SAPIEN CUDA functionalities, including dlpack, CUDA buffer, denoiser, and simsense" ON)

option(SAPIEN_BUILD_PYTHON "Build Python binding, only intended to generate compile_commands.json for IDE" OFF)
option(SAPIEN_BUILD_TEST "build unit tests")


set(SAPIEN_PHYSX5_DIR "" CACHE STRING "Directory for precompiled PhysX5 CPU SDK")
set(SAPIEN_PHYSX5_GPU_DIR "" CACHE STRING "Directory for precompiled PhysX5 GPU SDK")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif ()

# The install tree is SAPIEN's C++ SDK, which scripts/build.sh packs into an archive: every shared
# library goes to lib/ and finds the others through $ORIGIN, so the tree works wherever it is
# unpacked.
set(CMAKE_INSTALL_LIBDIR lib)
set(CMAKE_INSTALL_RPATH "$ORIGIN")

macro(FetchContent_MakeAvailableExclude)
    FetchContent_GetProperties(${ARGV})
    if(NOT ${ARGV}_POPULATED)
        FetchContent_Populate(${ARGV})
        add_subdirectory(${${ARGV}_SOURCE_DIR} ${${ARGV}_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
endmacro()

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(physx5)
include(zlib)
include(eigen)
include(vulkan)
# Zig is svulkan2's (its alias tables) and SAPIEN's (pack.json): one helper and version pin
include("${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/sapien-vulkan-2/cmake/zig.cmake")

if (${SAPIEN_DEBUG_VIEWER})
    add_definitions(_DEBUG_VIEWER)
endif ()

if (${SAPIEN_PROFILE})
    set(SVULKAN2_PROFILE ON CACHE BOOL "" FORCE)
else ()
    set(SVULKAN2_PROFILE OFF CACHE BOOL "" FORCE)
endif ()

if (${SAPIEN_CUDA})
    set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_PATH})
    set(CMAKE_CUDA_STANDARD 17)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lineinfo -ftemplate-depth=1024")
    file(GLOB_RECURSE SAPIEN_CUDA_SRC "src/*.cu")

    find_package(CUDA REQUIRED)
    set(SVULKAN2_CUDA_INTEROP ON CACHE BOOL "" FORCE)
else()
    set(SVULKAN2_CUDA_INTEROP OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory("3rd_party/sapien-vulkan-2" EXCLUDE_FROM_ALL)

if (${SAPIEN_CUDA})
    add_subdirectory("3rd_party/simsense")
    set_target_properties(simsense PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
    target_compile_definitions(simsense PUBLIC SAPIEN_SIMSENSE)
    target_include_directories(simsense PRIVATE "3rd_party/dlpack/include" "$ENV{CUDA_PATH}/include")
endif ()

file(GLOB_RECURSE SAPIEN_SRC "src/*.h" "src/*.cpp" "src/*.cc")
add_library(sapien SHARED ${SAPIEN_SRC} ${SAPIEN_CUDA_SRC})
sapien_add_zig_module(sapien src/sapien_renderer/pack_manifest.zig)
if (${SAPIEN_CUDA})
    # SAPIEN's own CUDA kernels. simsense has its own architecture list and is
    # currently used by the realistic depth sensor pipeline.
    set_target_properties(sapien PROPERTIES CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90;100;120")

    target_link_libraries(sapien PRIVATE "$ENV{CUDA_PATH}/lib64/libcudart_static.a")
    target_include_directories(sapien PRIVATE "3rd_party/dlpack/include" "$ENV{CUDA_PATH}/include")
endif ()

target_link_libraries(sapien PUBLIC eigen svulkan2)
# the public headers are C++20, also for programs using the installed CMake package
target_compile_features(sapien PUBLIC cxx_std_20)
target_link_libraries(sapien PRIVATE physx5 spdlog::spdlog ${CMAKE_DL_LIBS} assimp::assimp stdc++fs)

if (${SAPIEN_CUDA})
    target_compile_definitions(sapien PUBLIC SAPIEN_CUDA SAPIEN_CUDA)
    target_include_directories(sapien PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/dlpack/include>)
endif()

# The PUBLIC SAPIEN_CUDA definition above, baked into sapien/config.h, which every public header
# depending on it includes, so programs built against the installed headers need no -D flags.
configure_file(cmake/sapien_config.h.in "${CMAKE_CURRENT_BINARY_DIR}/include/sapien/config.h")
target_include_directories(sapien PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
)
target_include_directories(sapien PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

set_target_properties(sapien PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(sapien PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")

if (${SAPIEN_PROFILE})
    target_compile_definitions(sapien PUBLIC SAPIEN_PROFILE)
else ()
    target_compile_definitions(sapien PUBLIC NVTX_DISABLE=1)
endif()

target_include_directories(sapien INTERFACE
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/physx/include>
)

if (${SAPIEN_CUDA})
    add_dependencies(sapien OpenImageDenoise_device_cuda)
    ExternalProject_Get_property(OpenImageDenoise_device_cuda binary_dir)

    message(---- ${binary_dir}/../preinstall/lib)
    install(DIRECTORY ${binary_dir}/../preinstall/lib DESTINATION "." OPTIONAL)

    install(
        TARGETS sapien svulkan2 simsense physx5 OpenImageDenoise OpenImageDenoise_core OpenImageDenoise_common glm Vulkan-Headers
        EXPORT sapienTargets
    )
else()
    install(
        TARGETS sapien svulkan2 glm Vulkan-Headers
        EXPORT sapienTargets
    )
endif()

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/sapien" DESTINATION "include")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/sapien/config.h" DESTINATION "include/sapien")
install(DIRECTORY "${physx5_CPU_INCLUDE_DIR}/" DESTINATION "include/physx/include")
if (NOT physx5_GPU_INCLUDE_DIR STREQUAL physx5_CPU_INCLUDE_DIR)
    install(DIRECTORY "${physx5_GPU_INCLUDE_DIR}/" DESTINATION "include/physx/include")
endif()
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/dlpack/include/dlpack" DESTINATION "include")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/sapien-vulkan-2/include/svulkan2" DESTINATION "include")
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/3rd_party/sapien-vulkan-2/include/svulkan2/build_options.h"
    DESTINATION "include/svulkan2"
)
# public headers include <Eigen/...>; ship the Eigen headers SAPIEN was built with
if (DEFINED eigen_SOURCE_DIR)
    install(DIRECTORY "${eigen_SOURCE_DIR}/Eigen" DESTINATION "include")
endif()
install(DIRECTORY "${glm_SOURCE_DIR}/glm" DESTINATION "include")
install(DIRECTORY "${vulkan_SOURCE_DIR}/include/vulkan" DESTINATION "include")
install(DIRECTORY "${vulkan_SOURCE_DIR}/include/vk_video" DESTINATION "include")

if (${SAPIEN_CUDA})
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/simsense/include/simsense" DESTINATION "include")
endif()

# PhysxDefault::EnableGPU() looks for the PhysX GPU runtime next to libsapien first, so the SDK
# enables GPU PhysX without the download sapien.physx.enable_gpu() makes. The wheel leaves it out.
if (${SAPIEN_CUDA})
    set(_sapien_physx_gpu_runtime "${physx5_GPU_SOURCE_DIR}/bin/linux.x86_64/release/libPhysXGpu_64.so")
    if (EXISTS "${_sapien_physx_gpu_runtime}")
        install(FILES "${_sapien_physx_gpu_runtime}" DESTINATION "${CMAKE_INSTALL_LIBDIR}")
    else()
        message(WARNING "${_sapien_physx_gpu_runtime} does not exist: the installed SDK will not "
            "contain the PhysX GPU runtime")
    endif()
endif()

# Shaders for C++ programs that render: they pass <prefix>/share/sapien/vulkan_shader to
# SapienRendererDefault::internalSetShaderSearchPath, as the Python package does with its copy.
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_shader" DESTINATION "share/sapien")

include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/sapienConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/sapienConfig.cmake"
    INSTALL_DESTINATION "lib/cmake/sapien"
)
install(
    EXPORT sapienTargets
    FILE sapienTargets.cmake
    DESTINATION "lib/cmake/sapien"
    NAMESPACE sapien::
)
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/sapienConfig.cmake"
    DESTINATION "lib/cmake/sapien"
)

if (SAPIEN_BUILD_PYTHON)
    include(pybind11)
    file(GLOB_RECURSE PYBIND_SRC "python/pybind/*.cpp")
    pybind11_add_module(pysapien ${PYBIND_SRC} NO_EXTRAS)
    target_link_libraries(pysapien PRIVATE sapien eigen physx5 Vulkan::Headers glm)
    if (TARGET simsense)
        target_link_libraries(pysapien PRIVATE simsense)
    endif()
    target_compile_definitions(pysapien PRIVATE PYBIND11_USE_SMART_HOLDER_AS_DEFAULT)
endif()

if (${SAPIEN_BUILD_TEST})
    include(googletest)
    file(GLOB_RECURSE SAPIEN_TEST_SRC "test/*.cpp")
    # test/sdk is a separate project built against the packed SDK (scripts/test_sdk.sh)
    list(FILTER SAPIEN_TEST_SRC EXCLUDE REGEX "/test/sdk/")
    add_executable(sapien_test EXCLUDE_FROM_ALL ${SAPIEN_TEST_SRC})
    target_link_libraries(sapien_test sapien GTest::gtest_main physx5
        # -fsanitize=address
    )
    target_include_directories(sapien_test PRIVATE "test")

    add_executable(manual_test EXCLUDE_FROM_ALL "manualtest/main.cpp")
    target_link_libraries(manual_test sapien physx5)
endif()
