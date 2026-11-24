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

Select a shader pack per consumer with `sapien.render.RenderConfig`, or change
the process-wide defaults that new cameras and windows copy:

```python
import sapien

# one camera with a config of its own
config = sapien.render.RenderConfig("default")
camera = scene.add_camera(
    "camera", 640, 480, 1.0, 0.1, 100.0, render_config=config
)

# ray tracing, for every camera and window created next
rt_config = sapien.render.RenderConfig("rt")
rt_config["spp"] = 32
rt_config["maxDepth"] = 8
rt_config.denoiser = "oidn"
sapien.render.set_default_render_config("camera", rt_config)
sapien.render.set_default_render_config("viewer", rt_config)
```

A relative pack name such as `"rt"` is resolved under the installed shader
search path. A custom directory path can also be supplied, and
`config.shader_pack` returns the canonical directory. Consumers copy the config
they are given: editing it afterwards does not affect cameras or windows that
already exist.

## Shader ownership

The shader packs under SAPIEN's `vulkan_shader/` directory are the canonical
runtime assets shipped in the `sapien` wheel. They define SAPIEN-specific render
targets, material behavior, segmentation outputs, and visual defaults. The
`svulkan2` library, vendored in this repository at `3rd_party/sapien-vulkan-2`,
owns the generic shader-pack loader, renderer implementation, and its private
`shader_internal/` compute shaders. Its top-level `shader/`
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

These are the same properties a `RenderConfig` carries; set them on the config
to fix them before the camera exists, or through `camera.set_property` at
runtime.

The viewer's Render panel exposes the same controls for both rasterization and
ray tracing. Ray-tracing `HdrColor` and `Radiance` targets remain linear; use
`Color` when saving a display-ready image.

## Default visual style

A newly created render system has a neutral ambient fill of `[0.12, 0.12,
0.12]`. When no environment map is assigned, the default raster shader derives
a neutral studio environment from that fill. It provides image-based diffuse
lighting, visible dielectric reflections, and a sky/ground background instead
of sampling the renderer's black placeholder cubemap. Calling
`scene.set_environment_map(...)` replaces this fallback.

The default raster shader also applies screen-space ambient occlusion (SSAO) to
indirect light. Its camera properties can be tuned per camera, or the effect
can be disabled by setting its strength to zero:

```python
camera.set_property("ambientOcclusionStrength", 0.65)
camera.set_property("ambientOcclusionRadius", 0.3)  # scene units
# camera.set_property("ambientOcclusionStrength", 0.0)
```

SSAO supplies local contact shading, but cast shadows still require a light
created with `shadow=True`. A useful starting point for a meter-scale robotics
scene is:

```python
scene.set_ambient_light([0.12, 0.12, 0.12])
scene.add_directional_light(
   [1, 1, -1], [2.0, 1.9, 1.8], shadow=True
)
```

`RenderMaterial()` defaults to a neutral white dielectric with `specular=0.5`
and `roughness=0.45`, so adding a color or color texture retains a useful PBR
response. Primitive visuals that omit a material use a light blue-gray variant.
`Scene.add_ground` and `Scene.add_heightfield` instead use a subtle mipmapped
checker with `roughness=0.6` for scale cues and broader floor reflections.
Explicit render materials are never modified.

These raster effects approximate a polished real-time viewport. Use the `rt`
shader pack when physically traced object-to-object reflection, refraction, or
indirect lighting is required.

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
`postprocessing.comp`. Configure ray tracing with a `RenderConfig`:

```python
config = sapien.render.RenderConfig("rt")
config["spp"] = 32
config["maxDepth"] = 8
config.denoiser = "oidn"
camera = scene.add_camera(
    "camera", 640, 480, 1.0, 0.1, 100.0, render_config=config
)
```

When an RT pack is used and a key is absent, SAPIEN's defaults apply:
`russianRoulette=0`, `aperture=0.0`, `focusPlane=1.0`, and no denoiser.
`spp` and `maxDepth` come from the pack unless the config sets them.

The renderer fills these optional push-constant members by name when a pack
declares them: `frameCount` (frames accumulated since the last restart; `0`
on the first), `frameSeed` (the camera's render counter, which never resets;
seed random numbers with it so noise changes between frames even when
accumulation restarts every frame), the light counts, `envmap`, and
`ambientLight`. Other `int` and `float` members are set with
`camera.set_property`; changing one restarts accumulation unless it is
`exposure` or `toneMapper`. A name the pack does not declare raises
`RuntimeError` naming the key and the pack, instead of being silently dropped.

The ray-generation shader declares storage images such as `outHdrColor`,
`outAlbedo`, `outNormal`, `outSegmentation`, `outRadiance`, and
`outPosition`. Read them from Python by target name after `take_picture`.

## Batched ray-tracing pipeline

A pack containing `batch.comp` is a **batched** ray-tracing pack. `RenderConfig.kind`
reports it as `"batch_ray_tracing"`, and such a pack has no per-camera renderer: a
`RenderCameraGroup` builds one group renderer for all of its cameras. The built-in
example is `vulkan_shader/rt_batch`.

`batch.comp` is a compute shader with `local_size` 8x8x1, compiled at run time like
any other pack, so `config.defines` applies:

```
gl_GlobalInvocationID.xy   pixel
gl_GlobalInvocationID.z    camera row of the group
```

One dispatch covers every camera of the group. All cameras share one resolution;
`width`, `height` and `cameraCount` are push-constant members the renderer
writes. Every other `int`/`float` push-constant member is a property set
through the camera group's `RenderConfig` (the built-in pack declares `exposure`,
default 1.0, `toneMapper` and `spp`); a config property the pack does not declare
raises at `gpu_init()`. Each camera row carries the index of its environment; trace with
`GL_EXT_ray_query` against `tlas[nonuniformEXT(envIndex)]`, since neighbouring
workgroups may belong to different environments.

### Output blocks

Each requested picture is declared as a `std430` storage buffer whose block name is
`Out<Name>`. The renderer reflects the block and its element type selects the
picture format:

| Block element | Picture format |
|---|---|
| `vec4 c[]` | `r32g32b32a32Sfloat` |
| `uint c[]` | `r8g8b8a8Unorm` (packed 8-bit colour; default of `Color`) |
| `uvec4 c[]` | `r32g32b32a32Uint` |
| `float c[]` | `r32Sfloat` |

The built-in pack declares `Color`, `Albedo`, `Depth`, `Normal`, `Position` and
`Segmentation`. Only requested pictures are compiled in: the renderer defines
`OUTPUT_<NAME>` for each picture of the camera group, so guard every block:

```glsl
#ifdef OUTPUT_COLOR
layout(std430, set = 0, binding = 0) writeonly buffer OutColor {
#ifdef COLOR_FORMAT_R32G32B32A32SFLOAT
  vec4 c[];
#else
  uint c[];
#endif
} outColor;
#endif
```

Writing a picture nobody asked for is not possible; reading one nobody asked for
fails to compile. `picture_formats` maps to `<NAME>_FORMAT_<FORMAT>` (upper case,
for example `COLOR_FORMAT_R8G8B8A8UNORM`), and after compiling the reflected element
type must equal the requested format or construction raises.

### Shading tiers

Select a tier with a define; the tier is fixed for the life of the camera group:

- `SHADING_VISIBILITY` — primary ray only; albedo, depth, normal, position, segmentation.
- `SHADING_LAMBERT` (default) — emission + albedo·(ambient + Σ lights N·L).
- `SHADING_PBR` — adds the GGX specular lobe of `common/bsdf.glsl`.

`SHADOWS` casts a hard shadow ray for each of the first `MAX_SHADOW_LIGHTS` (default
4) lights facing the surface point; later lights are unshadowed. `SPP` > 1
supersamples the primary ray with a fixed, deterministic Halton pattern; there is no
accumulation and no denoiser. The pure BSDF functions live in
`vulkan_shader/common/bsdf.glsl`, shared with the `rt` pack, so a custom batched pack
and the `rt` pack shade identically when they use the same tier.

## Changing target formats

Set `picture_formats` on the config of a camera if a target needs a
non-default image format. The setting applies to that camera only.

```python
config = sapien.render.RenderConfig("default")
config.picture_formats = {"Color": "r8g8b8a8unorm"}
camera = scene.add_camera(
    "camera", 640, 480, 1.0, 0.1, 100.0, render_config=config
)
camera.get_picture("Color").dtype  # uint8
```

Ray-tracing packs write their targets as storage images, so `picture_formats`
is rasterization-only and raises on an RT config.

## Tips for custom packs

0. Shader files are compiled when a pack is first used and the pack is then
   cached. After editing a file or one of its includes, call
   `sapien.render.clear_cache(models=False, images=False, shaders=True)` and
   create a new camera; a compile error raises `RuntimeError` with the file
   and line. `config.defines` selects a preprocessor variant: each entry is
   inserted as `#define NAME value` (`#define NAME` when the value is empty).
1. Copy an existing pack from `vulkan_shader/default` or `vulkan_shader/rt`.
2. Point the camera and viewer configs to your copy:
   `sapien.render.RenderConfig("/path/to/pack")`.
3. Keep the required buffer and sampler names unchanged unless you also update
   the renderer code.
4. Add new render targets with `out<Name>` in rasterization or `out<Name>`
   storage images in ray tracing, then read them with `camera.get_picture`.
5. Validate with a tiny scene and `camera.get_picture_names()` before using the
   shader pack in a larger workload.
