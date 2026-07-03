if(TARGET physx5)
  return()
endif()

set(PHYSX_VERSION "107.3-physx-5.6.1" CACHE STRING "Precompiled PhysX package version")

function(_sapien_normalize_physx_root input output)
  foreach(candidate
      "${input}"
      "${input}/PhysX"
      "${input}/physxcpu-linux-clang/PhysX"
      "${input}/physxgpu-linux-clang/PhysX"
      "${input}/physxcpu-windows-vc17win64/PhysX"
      "${input}/physxgpu-windows-vc17win64/PhysX")
    if (IS_DIRECTORY "${candidate}/include")
      set(${output} "${candidate}" PARENT_SCOPE)
      return()
    endif()
  endforeach()

  set(${output} "" PARENT_SCOPE)
endfunction()

function(_sapien_pick_physx_windows_lib output lib_dir)
  foreach(candidate IN LISTS ARGN)
    if (EXISTS "${lib_dir}/${candidate}")
      set(${output} "${candidate}" PARENT_SCOPE)
      return()
    endif()
  endforeach()

  list(JOIN ARGN ", " _sapien_physx_candidates)
  message(FATAL_ERROR
    "Failed to locate a compatible PhysX Windows library in ${lib_dir}. "
    "Tried: ${_sapien_physx_candidates}")
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
  if (UNIX)
    FetchContent_Declare(
      physx5cpu
      URL https://github.com/yolkarian/physx-release/releases/download/${PHYSX_VERSION}/physxcpu-linux-clang.zip
      URL_HASH SHA256=06d419123ef30dd8ad9396f2cc61af39a57d87ad4cdad110836d17294d937276
    )
    FetchContent_Declare(
      physx5gpu
      URL https://github.com/yolkarian/physx-release/releases/download/${PHYSX_VERSION}/physxgpu-linux-clang.zip
      URL_HASH SHA256=a390b9b11a63a28305c4cab871dc7f5a0dfd7380d2e540ce58011d558c87d68a
    )
    FetchContent_MakeAvailable(physx5cpu physx5gpu)
    _sapien_normalize_physx_root("${physx5cpu_SOURCE_DIR}" physx5_CPU_SOURCE_DIR)
    _sapien_normalize_physx_root("${physx5gpu_SOURCE_DIR}" physx5_GPU_SOURCE_DIR)
  elseif (WIN32)
    FetchContent_Declare(
      physx5cpu
      URL https://github.com/yolkarian/physx-release/releases/download/${PHYSX_VERSION}-windows/physxcpu-windows-vc17win64.zip
      URL_HASH SHA256=1ad3e8372fad2fde7955616c4921c8338a09b6660eed35e1bc253d14d55b5801
    )
    FetchContent_Declare(
      physx5gpu
      URL https://github.com/yolkarian/physx-release/releases/download/${PHYSX_VERSION}-windows/physxgpu-windows-vc17win64.zip
      URL_HASH SHA256=43f37f586caf8edb33de895267206bd5cb85d89a76896c81222e6c9b560954cf
    )
    FetchContent_MakeAvailable(physx5cpu physx5gpu)
    _sapien_normalize_physx_root("${physx5cpu_SOURCE_DIR}" physx5_CPU_SOURCE_DIR)
    _sapien_normalize_physx_root("${physx5gpu_SOURCE_DIR}" physx5_GPU_SOURCE_DIR)
  endif()
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

if(UNIX)
  target_link_directories(physx5 INTERFACE $<BUILD_INTERFACE:${physx5_CPU_SOURCE_DIR}/bin/linux.x86_64/release>)
  target_link_libraries(physx5 INTERFACE
    -Wl,--start-group
    libPhysXCharacterKinematic_static_64.a libPhysXCommon_static_64.a
    libPhysXCooking_static_64.a libPhysXExtensions_static_64.a
    libPhysXFoundation_static_64.a libPhysXPvdSDK_static_64.a
    libPhysX_static_64.a libPhysXVehicle2_static_64.a
    -Wl,--end-group)
endif()

if (WIN32)
  set(_sapien_physx_windows_lib_dir "${physx5_CPU_SOURCE_DIR}/bin/win.x86_64.vc143.mt/release")
  target_link_directories(physx5 INTERFACE $<BUILD_INTERFACE:${_sapien_physx_windows_lib_dir}>)

  _sapien_pick_physx_windows_lib(_sapien_physx_vehicle2_lib "${_sapien_physx_windows_lib_dir}"
    PhysXVehicle2_static_64.lib PhysXVehicle2_64.lib)
  _sapien_pick_physx_windows_lib(_sapien_physx_extensions_lib "${_sapien_physx_windows_lib_dir}"
    PhysXExtensions_static_64.lib PhysXExtensions_64.lib)
  _sapien_pick_physx_windows_lib(_sapien_physx_core_lib "${_sapien_physx_windows_lib_dir}"
    PhysX_static_64.lib PhysX_64.lib)
  _sapien_pick_physx_windows_lib(_sapien_physx_pvd_sdk_lib "${_sapien_physx_windows_lib_dir}"
    PhysXPvdSDK_static_64.lib PhysXPvdSDK_64.lib)
  _sapien_pick_physx_windows_lib(_sapien_physx_cooking_lib "${_sapien_physx_windows_lib_dir}"
    PhysXCooking_static_64.lib PhysXCooking_64.lib)
  _sapien_pick_physx_windows_lib(_sapien_physx_common_lib "${_sapien_physx_windows_lib_dir}"
    PhysXCommon_static_64.lib PhysXCommon_64.lib)
  _sapien_pick_physx_windows_lib(_sapien_physx_character_kinematic_lib "${_sapien_physx_windows_lib_dir}"
    PhysXCharacterKinematic_static_64.lib PhysXCharacterKinematic_64.lib)
  _sapien_pick_physx_windows_lib(_sapien_physx_foundation_lib "${_sapien_physx_windows_lib_dir}"
    PhysXFoundation_static_64.lib PhysXFoundation_64.lib)

  set(_sapien_physx_windows_libs
    ${_sapien_physx_vehicle2_lib}
    ${_sapien_physx_extensions_lib}
    ${_sapien_physx_core_lib}
    ${_sapien_physx_pvd_sdk_lib}
    ${_sapien_physx_cooking_lib}
    ${_sapien_physx_common_lib}
    ${_sapien_physx_character_kinematic_lib}
    ${_sapien_physx_foundation_lib})

  foreach(_sapien_physx_optional_lib
      PhysXTask_static_64.lib
      PhysXTask_64.lib
      PVDRuntime_64.lib)
    if (EXISTS "${_sapien_physx_windows_lib_dir}/${_sapien_physx_optional_lib}")
      list(APPEND _sapien_physx_windows_libs ${_sapien_physx_optional_lib})
    endif()
  endforeach()

  target_link_libraries(physx5 INTERFACE ${_sapien_physx_windows_libs})
endif()

target_include_directories(physx5 SYSTEM INTERFACE $<BUILD_INTERFACE:${physx5_CPU_INCLUDE_DIR}>)
if (NOT physx5_GPU_INCLUDE_DIR STREQUAL physx5_CPU_INCLUDE_DIR)
  target_include_directories(physx5 SYSTEM INTERFACE $<BUILD_INTERFACE:${physx5_GPU_INCLUDE_DIR}>)
endif()

target_compile_definitions(physx5 INTERFACE PX_PHYSX_STATIC_LIB)
target_compile_definitions(physx5 INTERFACE PHYSX_VERSION="${PHYSX_VERSION}")
