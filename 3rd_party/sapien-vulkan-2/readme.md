# SAPIEN Vulkan 2

This repository is forked from [haosulab/sapien-vulkan-2](https://github.com/haosulab/sapien-vulkan-2).

Rendering library for [SAPIEN](https://github.com/haosulab/SAPIEN).

Report any issues to SAPIEN's [issues](https://github.com/haosulab/SAPIEN/issues) page.

## Building

svulkan2 configures and builds on its own (`cmake -S . -B build`) as well as inside
SAPIEN. Besides the C++ toolchain it needs Zig 0.17.1, pinned in `cmake/zig.cmake`:
the alias tables of the `rt` path tracer come from the Zig module
`src/renderer/alias_table.zig`, compiled to an object linked into the library, so
programs using svulkan2 need no Zig. SAPIEN includes the same `cmake/zig.cmake` for
its own Zig modules.

## Shader assets

`svulkan2` provides the generic shader-pack runtime. The shaders in
`shader_internal/` are private renderer assets used for tasks such as BRDF LUT
and environment-map generation. The top-level `shader/` directory supports the
standalone examples and is not the source of SAPIEN's packaged shader packs.

SAPIEN's canonical runtime packs live in the SAPIEN repository under
`vulkan_shader/` and are packaged into the `sapien` wheel. Product rendering
changes belong there; renderer capabilities and internal compute shaders belong
in this repository.
