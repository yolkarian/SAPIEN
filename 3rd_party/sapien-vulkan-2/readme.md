# SAPIEN Vulkan 2

This repository is forked from [haosulab/sapien-vulkan-2](https://github.com/haosulab/sapien-vulkan-2).

Rendering library for [SAPIEN](https://github.com/haosulab/SAPIEN).

Report any issues to SAPIEN's [issues](https://github.com/haosulab/SAPIEN/issues) page.

## Shader assets

`svulkan2` provides the generic shader-pack runtime. The shaders in
`shader_internal/` are private renderer assets used for tasks such as BRDF LUT
and environment-map generation. The top-level `shader/` directory supports the
standalone examples and is not the source of SAPIEN's packaged shader packs.

SAPIEN's canonical runtime packs live in the SAPIEN repository under
`vulkan_shader/` and are packaged into the `sapien` wheel. Product rendering
changes belong there; renderer capabilities and internal compute shaders belong
in this repository.
