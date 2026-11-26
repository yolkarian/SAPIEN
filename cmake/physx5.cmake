if(TARGET physx5)
  return()
endif()

set(PHYSX_VERSION "107.3-physx-5.6.1" CACHE STRING "Precompiled PhysX package version")

function(_sapien_normalize_physx_root input output)
  foreach(candidate
      "${input}"
      "${input}/PhysX"
      "${input}/physxcpu-linux-clang/PhysX"
      "${input}/physxgpu-linux-clang/PhysX")
    if (IS_DIRECTORY "${candidate}/include")
      set(${output} "${candidate}" PARENT_SCOPE)
      return()
    endif()
  endforeach()

  set(${output} "" PARENT_SCOPE)
endfunction()

set(physx5_CPU_SOURCE_DIR "")
set(physx5_GPU_SOURCE_DIR "")

if (IS_DIRECTORY "${SAPIEN_PHYSX5_DIR}")
  _sapien_normalize_physx_root("${SAPIEN_PHYSX5_DIR}" physx5_CPU_SOURCE_DIR)
  if (IS_DIRECTORY "${SAPIEN_PHYSX5_GPU_DIR}")
    _sapien_normalize_physx_root("${SAPIEN_PHYSX5_GPU_DIR}" physx5_GPU_SOURCE_DIR)
  else()
    set(physx5_GPU_SOURCE_DIR "${physx5_CPU_SOURCE_DIR}")
  endif()
else()
  include(FetchContent)
  FetchContent_Declare(
    physx5cpu
    URL https://github.com/yolkarian/physx-release/releases/download/${PHYSX_VERSION}-Linux/physxcpu-linux-clang.zip
    URL_HASH SHA256=6982b76cbb53726f278e8f93d27af646129a7542125f84358fc3400d1c70ce94
  )
  FetchContent_Declare(
    physx5gpu
    URL https://github.com/yolkarian/physx-release/releases/download/${PHYSX_VERSION}-Linux/physxgpu-linux-clang.zip
    URL_HASH SHA256=5e063869d375cdd51184d822faf5cfe470a3cc768e6b0dfb511e9312ade31b48
  )
  FetchContent_MakeAvailable(physx5cpu physx5gpu)
  _sapien_normalize_physx_root("${physx5cpu_SOURCE_DIR}" physx5_CPU_SOURCE_DIR)
  _sapien_normalize_physx_root("${physx5gpu_SOURCE_DIR}" physx5_GPU_SOURCE_DIR)
endif()

if (NOT IS_DIRECTORY "${physx5_CPU_SOURCE_DIR}/include")
  message(FATAL_ERROR "Failed to locate PhysX CPU SDK headers under ${physx5_CPU_SOURCE_DIR}")
endif()

if (NOT physx5_GPU_SOURCE_DIR)
  set(physx5_GPU_SOURCE_DIR "${physx5_CPU_SOURCE_DIR}")
endif()

set(physx5_SOURCE_DIR "${physx5_CPU_SOURCE_DIR}")
set(physx5_CPU_INCLUDE_DIR "${physx5_CPU_SOURCE_DIR}/include")
set(physx5_GPU_INCLUDE_DIR "${physx5_GPU_SOURCE_DIR}/include")

if (${SAPIEN_CUDA} AND NOT IS_DIRECTORY "${physx5_GPU_INCLUDE_DIR}/cudamanager")
  message(FATAL_ERROR "PhysX GPU SDK headers not found. Set SAPIEN_PHYSX5_GPU_DIR or disable SAPIEN_CUDA.")
endif()

add_library(physx5 INTERFACE)

target_link_directories(physx5 INTERFACE $<BUILD_INTERFACE:${physx5_CPU_SOURCE_DIR}/bin/linux.x86_64/release>)
target_link_libraries(physx5 INTERFACE
  -Wl,--start-group
  libPhysXCharacterKinematic_static_64.a libPhysXCommon_static_64.a
  libPhysXCooking_static_64.a libPhysXExtensions_static_64.a
  libPhysXFoundation_static_64.a libPhysXPvdSDK_static_64.a
  libPhysX_static_64.a libPhysXVehicle2_static_64.a
  -Wl,--end-group)

target_include_directories(physx5 SYSTEM INTERFACE $<BUILD_INTERFACE:${physx5_CPU_INCLUDE_DIR}>)
if (NOT physx5_GPU_INCLUDE_DIR STREQUAL physx5_CPU_INCLUDE_DIR)
  target_include_directories(physx5 SYSTEM INTERFACE $<BUILD_INTERFACE:${physx5_GPU_INCLUDE_DIR}>)
endif()

target_compile_definitions(physx5 INTERFACE PX_PHYSX_STATIC_LIB)
target_compile_definitions(physx5 INTERFACE PHYSX_VERSION="${PHYSX_VERSION}")
