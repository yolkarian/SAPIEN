(customize_shaders)=

# Customize Shaders

SAPIEN renderer shader packs are directories of GLSL files loaded at runtime.
The shader-pack interface is experimental and may change, but the current repo
uses the layout described here.

This page focuses on API names and file locations that match the current source
tree. Read the shader files in `vulkan_shader/` for the exact binding layout.

## Shader-pack directories

The packaged shader directories are copied into `sapien/vulkan_shader` when a
wheel is built. In the source tree they live at the repository root:

- `vulkan_shader/default`: default rasterization shader pack;
- `vulkan_shader/rt`: ray-tracing shader pack;
- additional packs such as `minimal`, `point`, `vertex_color`, and
  `shadow_catcher`.

To locate the installed shader directory:

```shell
python -c 'import importlib.resources as r; print(r.files("sapien").joinpath("vulkan_shader"))'
```

Select shader packs before creating cameras or viewers:

```python
import sapien

sapien.render.set_camera_shader_dir("default")
sapien.render.set_viewer_shader_dir("default")

# or use ray tracing
sapien.render.set_camera_shader_dir("rt")
sapien.render.set_viewer_shader_dir("rt")
```

A relative name such as `"rt"` is resolved under the installed shader search
path. A custom directory path can also be supplied.

## Shader ownership

The shader packs under SAPIEN's `vulkan_shader/` directory are the canonical
runtime assets shipped in the `sapien` wheel. They define SAPIEN-specific render
targets, material behavior, segmentation outputs, and visual defaults. The
`svulkan2` library owns the generic shader-pack loader, renderer implementation,
and its private `shader_internal/` compute shaders. Its top-level `shader/`
directory is for standalone examples and is not a second source for SAPIEN's
built-in packs.

Keep product shader changes in SAPIEN unless they require a renderer capability
or fix in `svulkan2`. This separation avoids two copies of the built-in packs
drifting while still keeping renderer-internal shaders next to the code that
uses them.

## Color management

Rasterization and ray tracing both shade in linear HDR and use the same display
transform for the `Color` target. The default raster, VR, ray-tracing, and shadow
catcher packs use ACES fitted tone mapping followed by the sRGB output transfer
function. Gamma 2.2 and plain sRGB transforms remain available for compatibility
and data inspection:

```python
camera.set_property("exposure", 1.0)
camera.set_property("toneMapper", 2)  # 0: gamma 2.2, 1: sRGB, 2: ACES + sRGB
```

The viewer's Render panel exposes the same controls for both rasterization and
ray tracing. Ray-tracing `HdrColor` and `Radiance` targets remain linear; use
`Color` when saving a display-ready image.

## Rasterization pipeline

A rasterization shader pack contains a required `gbuffer` pass and optional
passes. The default pack currently contains files such as:

- `gbuffer.vert` and `gbuffer.frag` for opaque geometry;
- `gbuffer1.vert` and `gbuffer1.frag` for transparent geometry;
- `shadow.vert` for shadow maps;
- `deferred.vert` and `deferred.frag` for deferred lighting;
- `composite.vert` and `composite0.frag` for post-processing and display
  targets;
- `point.*` and `line.*` for point and line rendering.

Render targets are named by the GLSL output name without the `out` prefix. For
example, `outColor` is read as `camera.get_picture("Color")`. Use
`camera.get_picture_names()` to inspect targets available from the current
shader pack.

```python
scene.update_render()
camera.take_picture()
print(camera.get_picture_names())
color = camera.get_picture("Color")
color_cuda = camera.get_picture_cuda("Color")
```

`get_picture` replaces the old `get_float_texture`, `get_uint32_texture`,
and `get_color_rgba` helpers. `get_picture_cuda` returns a
`sapien.CudaArray`; use its `torch()`, `cupy()`, `jax()`, or
`dlpack()` adapters to interoperate with GPU libraries.

## Common uniform sets

The default rasterization shader pack uses include files that define the current
uniform-buffer names and binding layout:

- `camera_set.glsl` defines `CameraBuffer` / `cameraBuffer` with
  `viewMatrix`, `projectionMatrix`, inverse matrices, width, and height.
- `object_set.glsl` defines `ObjectTransformBuffer` /
  `objectTransformBuffer` and `ObjectDataBuffer` / `objectDataBuffer`.
  Object data includes `segmentation`, `transparency`, and `shadeFlat`.
- `material_set.glsl` defines `MaterialBuffer` / `materialBuffer` and
  texture samplers for base color, roughness, normal, metallic, emission, and
  transmission maps.
- `scene_set.glsl` defines `SceneBuffer` / `sceneBuffer`,
  `ShadowBuffer` / `shadowBuffer`, shadow-map samplers, the environment
  sampler, and the BRDF lookup texture.

When writing a custom shader pack, keep these names and layouts compatible with
the renderer code or copy and modify the provided include files.

## Geometry input

G-buffer vertex shaders may consume built-in mesh attributes including
`position`, `normal`, `uv`, `tangent`, `bitangent`, and `color`.
`position` must be at `location = 0`. Other attributes are optional, but when
used their locations should be consecutive.

```glsl
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec3 tangent;
layout(location = 4) in vec3 bitangent;
```

All `gbuffer` passes in a shader pack should agree on the vertex layout.

## Ray-tracing pipeline

A ray-tracing shader pack is recognized by `camera.rgen`. The current `rt`
pack also includes hit/miss shaders, shared GLSL headers, and
`postprocessing.comp`. Configure ray tracing through `sapien.render`:

```python
sapien.render.set_camera_shader_dir("rt")
sapien.render.set_viewer_shader_dir("rt")
sapien.render.set_ray_tracing_samples_per_pixel(32)
sapien.render.set_ray_tracing_path_depth(8)
sapien.render.set_ray_tracing_denoiser("oidn")
```

The ray-generation shader declares storage images such as `outHdrColor`,
`outAlbedo`, `outNormal`, `outSegmentation`, `outRadiance`, and
`outPosition`. Read them from Python by target name after `take_picture`.

## Changing target formats

Use `sapien.render.set_picture_format` before creating cameras if a target
needs a non-default image format.

```python
sapien.render.set_picture_format("Color", "r16g16b16a16Sfloat")
```

Reset it before other examples if needed:

```python
sapien.render.set_picture_format("Color", "r32g32b32a32Sfloat")
```

## Tips for custom packs

1. Copy an existing pack from `vulkan_shader/default` or `vulkan_shader/rt`.
2. Point both camera and viewer shader dirs to your copy.
3. Keep the required buffer and sampler names unchanged unless you also update
   the renderer code.
4. Add new render targets with `out<Name>` in rasterization or `out<Name>`
   storage images in ray tracing, then read them with `camera.get_picture`.
5. Validate with a tiny scene and `camera.get_picture_names()` before using the
   shader pack in a larger workload.
