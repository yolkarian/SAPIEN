# Zig modules: one Zig source each, compiled to an object with a C interface and linked into a
# CMake target, so programs using SAPIEN never need Zig. The version is pinned to the one in the
# build image (docker/Dockerfile); Zig is pre-1.0 and breaks its standard library in every release.
# svulkan2 owns this file so that it builds on its own; SAPIEN includes it from here before adding
# svulkan2, whose own include is then skipped.
include_guard(GLOBAL)
set(SAPIEN_ZIG_VERSION 0.17.1)

find_program(ZIG_EXECUTABLE zig REQUIRED)
execute_process(
    COMMAND "${ZIG_EXECUTABLE}" version
    OUTPUT_VARIABLE _sapien_zig_found_version
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT _sapien_zig_found_version STREQUAL SAPIEN_ZIG_VERSION)
    message(FATAL_ERROR
        "SAPIEN needs Zig ${SAPIEN_ZIG_VERSION}, found ${_sapien_zig_found_version} at ${ZIG_EXECUTABLE}")
endif()

# glibc 2.28, the manylinux_2_28 floor of the wheels; libgcc supplies the 128-bit helpers
set(SAPIEN_ZIG_TARGET x86_64-linux-gnu.2.28)
set(SAPIEN_ZIG_FLAGS -fPIC)
set(_sapien_zig_object_suffix .o)

# Compile the Zig file `source` (relative to the calling directory) and link it into `target`.
# ReleaseSafe keeps bounds and overflow checks: these modules parse files, they are not hot.
# Stack probes would need Zig's compiler-rt, and the debug info of the standard library would
# dwarf the code, so both are off.
function(sapien_add_zig_module target source)
    get_filename_component(_name "${source}" NAME_WE)
    set(_object "${CMAKE_CURRENT_BINARY_DIR}/zig/${_name}${_sapien_zig_object_suffix}")
    file(GLOB_RECURSE _zig_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.zig")
    add_custom_command(
        OUTPUT "${_object}"
        COMMAND "${ZIG_EXECUTABLE}" build-obj "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
                -target ${SAPIEN_ZIG_TARGET}
                -O $<IF:$<CONFIG:Debug>,Debug,ReleaseSafe>
                -fno-stack-check -fstrip ${SAPIEN_ZIG_FLAGS}
                --cache-dir "${CMAKE_CURRENT_BINARY_DIR}/zig/cache"
                --global-cache-dir "${CMAKE_CURRENT_BINARY_DIR}/zig/global-cache"
                "-femit-bin=${_object}"
        DEPENDS ${_zig_sources}
        COMMENT "Building Zig module ${source}"
        VERBATIM)
    set_source_files_properties("${_object}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    target_sources(${target} PRIVATE "${_object}")
endfunction()
