(raytracing_renderer)=

# Ray Tracing Renderer

```{eval-rst}
.. highlight:: python
```

SAPIEN's render module supports both rasterization and ray tracing through shader
packs. There is no renderer object to pass into the scene in current SAPIEN;
configure global render options through `sapien.render` before creating
cameras or viewers.

In this tutorial, you will learn how to:

- switch cameras and viewers to the ray-tracing shader pack;
- configure samples per pixel, path depth, and denoising;
- use material parameters that are most visible in ray tracing.

## Ray tracing vs. rasterization

The default `"default"` shader pack is rasterization-based and is usually the
fastest choice for RL and data collection. The `"rt"` shader pack traces rays
and can model effects such as indirect lighting, reflection, refraction, and
soft shadows, at a higher cost.

:::{figure} assets/rst_vs_rt.png
:align: center
:width: 900px

The same SAPIEN material scene rendered with the current raster and ray-tracing
shader packs.
:::

## Enable ray tracing

Give the camera a ray-tracing `RenderConfig`. Windows copy the viewer default,
so set that too when the Viewer should trace rays.

```python
import sapien

rt_config = sapien.render.RenderConfig("rt")
rt_config["spp"] = 16
rt_config["maxDepth"] = 8
rt_config.denoiser = "oidn"  # "none", "oidn", or "optix"
sapien.render.set_default_render_config("viewer", rt_config)

scene = sapien.Scene()
camera = scene.add_camera(
    "camera", 640, 480, 1.0, 0.01, 100.0, render_config=rt_config
)
```

Rasterization and ray tracing live side by side in one process: a camera
created with the default config still rasterizes. To switch the viewer back,
set the viewer default to `sapien.render.RenderConfig("default")`.

## Sampling and denoising

Ray-traced images are noisy when the sample count is low. Increase samples per
pixel for quality, and use a denoiser when available.

```python
config = sapien.render.RenderConfig("rt")
config["spp"] = 64
config["maxDepth"] = 12
config.denoiser = "oidn"
camera = scene.add_camera(
    "camera", 640, 480, 1.0, 0.01, 100.0, render_config=config
)
```

`oidn` uses the packaged Open Image Denoise integration. `optix` requires a
compatible NVIDIA RTX driver stack. If either denoiser is unavailable on your
machine, use `"none"`.

## Light transport

The `rt` pack is a path tracer. At every surface it finds light in two ways:

- **light sampling** (next-event estimation) picks a point on a light and casts a shadow ray
  toward it;
- **BSDF sampling** follows the material's scattering and adds the emission of whatever the
  ray reaches, the environment included.

Point, directional, spot (also textured) and parallelogram lights have no geometry, so only
light sampling finds them, in every mode. The environment map and emissive meshes can be found
both ways, and `lightSampling` selects how:

| `lightSampling` | Environment and emissive meshes |
|---|---|
| `"mis"` (default) | both ways, combined by multiple importance sampling (power heuristic) |
| `"nee"` | light sampling only |
| `"bsdf"` | BSDF sampling only |

The three converge to the same image; `"nee"` and `"bsdf"` exist to debug a scene.
`lightSampling` selects shader code, so set it on the `RenderConfig` before creating the
camera.

- **Environment.** The map is sampled in proportion to its luminance over a grid of
  equal-solid-angle cells (16 to 256 columns, following the cube resolution), built when the
  map changes. Its radiance is no longer clamped to 100, so bright HDR maps light the scene
  at their real strength.
- **Emissive meshes.** Their triangles are sampled in proportion to area times emitted
  luminance, at the mesh's current pose, so moving an emitter needs no rebuild. Meshes with
  an emission texture are left to BSDF sampling.
- **Shadows.** A shadow ray crossing a surface keeps the part the surface lets through: the
  part its base-colour alpha and the object's `transparency` leave uncovered, plus the
  covered part's `transmission`, tinted by the base colour. A sheer or transmissive occluder
  casts a lighter, possibly coloured shadow; point clouds block light.
- **Russian roulette** (`russianRoulette`) is on by default: after
  `russianRouletteMinBounces` bounces, paths of low weight end at random and the survivors
  are weighted up, which keeps the image unbiased.
- **`clampIndirect`** (default `0`, off) limits the brightest channel of every contribution
  after the first bounce, trading some energy for fewer fireflies.

```python
config = sapien.render.RenderConfig("rt")
config["spp"] = 64
config["clampIndirect"] = 10.0   # firefly clamp, 0 disables
config["lightSampling"] = "mis"  # "mis" | "nee" | "bsdf"
```

## Materials for ray tracing

Render materials live in `sapien.render` and carry the first batch of the
OpenPBR Surface 1.1 parameters. The parameters `base_color`, `roughness`
(OpenPBR `specular_roughness`), `metallic` (`base_metalness`), `specular_weight`,
`specular_color`, `ior` (`specular_ior`), `transmission`, `transmission_color`,
`transmission_depth`, `coat_weight`, `coat_color`, `coat_roughness`, `coat_ior`,
`fuzz_weight`, `fuzz_color`, `fuzz_roughness`, `base_weight`, `thin_walled`, and
the emission strength affect ray-traced reflection, refraction and shading.

`base_color`'s alpha channel is the geometry coverage (a cut-out), not
transparency: glass is made with `transmission`, and `transmission_color` with
`transmission_depth` tints light over distance. `specular_weight` scales the
normal-incidence reflectance of `ior`; the removed `specular` parameter s is
`2*s` here at `ior=1.5`.

```python
glass = sapien.render.RenderMaterial(
   base_color=[0.8, 0.9, 1.0, 1.0],
   roughness=0.02,
   metallic=0.0,
   transmission=1.0,
   transmission_color=[0.8, 0.9, 1.0],
   ior=1.45,
)

metal = sapien.render.RenderMaterial(
   base_color=[0.9, 0.75, 0.5, 1.0],
   roughness=0.15,
   metallic=1.0,
)

builder = scene.create_actor_builder()
builder.add_sphere_collision(radius=0.3)
builder.add_sphere_visual(radius=0.3, material=glass)
sphere = builder.build(name="glass_sphere")

builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[0.2, 0.2, 0.2])
builder.add_box_visual(half_size=[0.2, 0.2, 0.2], material=metal)
box = builder.build(name="metal_box")
```

## Lighting and environment maps

Ray tracing benefits from realistic lighting. Use area lights for soft shadows
and environment maps for image-based lighting.

```python
scene.set_ambient_light([0.02, 0.02, 0.02])
scene.add_area_light_for_ray_tracing(
   pose=sapien.Pose([0, 0, 3]),
   color=[5, 5, 5],
   half_width=1.0,
   half_height=1.0,
)
# scene.set_environment_map("path/to/cubemap.ktx")
```

Render from the ray-tracing camera in the same way as the rasterized camera:

```python
scene.update_render()
camera.take_picture()
color = camera.get_picture("Color")
```

:::{figure} assets/mat_rt.png
:align: center
:width: 720px

Current SAPIEN ray-traced output for dielectric, metallic, and transmissive
materials.
:::

Attached lights propagate changes to shadow enable, shadow-map size (raster), and
texture before grouped rendering is initialized. These resource-changing setters
wait for prior GPU work and invalidate renderer resources for the next capture.
A textured light must have a texture when added to a scene; while attached,
`texture=None` is rejected without changing the current texture. Detached lights
can still clear their texture.

## Batched GPU rendering

The `"rt"` shader pack supports `sapien.render.RenderSystemGroup`, including
shared render scenes, rigid-body poses sourced directly from PhysX GPU, and
mounted cameras with GPU pose batch indices. Configure the RT shader before
creating cameras, then use the same batched workflow as rasterization:

```python
rt_config = sapien.render.RenderConfig("rt")

# After physx_system.gpu_init(), sibling PhysX GPU bodies/links are discovered
# automatically. Explicit pose indices remain available for custom pose buffers.

group = sapien.render.RenderSystemGroup(render_systems)
group.set_cuda_poses(physx_system.cuda_rigid_body_data)
camera_group = group.create_camera_group([camera], ["Color"])
group.gpu_init()  # prepares resources, seeds CPU snapshots, seals transform ownership

physx_system.gpu_fetch_rigid_dynamic_data()
group.update_render()  # updates RT instance transforms and the TLAS
camera_group.take_picture()
color = camera_group.get_picture_cuda("Color")
```

`RenderSystemGroup.gpu_init()` seals every member camera transform for GPU
ownership. Configure each free camera's pose mode with
`camera_group.set_pose_mode(camera, mode)` after `create_camera_group()` and
before `gpu_init()`. Free cameras default to `'static'` (a one-time CPU pose
snapshot; CPU pose setters raise afterwards). `'cpu'` keeps the CPU pose
authoritative and uploads at the next `group.update_render()` — use it for
host-driven follow/anchor cameras. `'cuda'` allocates a group-owned CUDA pose
row in `camera_group.cuda_poses`, seeded once from the CPU pose; move it by
writing that row (or `camera_group.set_cuda_pose(camera, pose)`) before
`group.update_render()`. Cameras mounted on PhysX GPU bodies are
auto-attached and reuse their PhysX parent pose row; they cannot be configured.
Dynamic bodies in implicitly discovered shared scenes are also bound to their
PhysX GPU rows and require `set_cuda_poses()`. CPU-owned static bodies and
point clouds are sealed snapshots; mutate them only after destroying the group.
Camera projection/intrinsics (fov, near/far, principal point, skew,
perspective/ortho) stay CPU real-time in every mode. Light color, spot
inner/outer FOV, parallelogram shape, shadow near/far, directional shadow
half-size, and ambient light stay CPU real-time after `gpu_init()`; for a light
that should follow a GPU body, put the light on a separate entity, set its pose
mode to `'cpu'` via `light.set_pose_mode("cpu")`, and write the pose from
downloaded state each frame (`gpu_init()` rejects a `'cpu'` light sharing its
entity with a PhysX GPU body).

For per-environment host-side lighting randomization, batch updates with
`sapien.render.set_light_poses`, `sapien.render.set_light_directions`, and
`sapien.render.set_light_colors`. Their array shapes, local-frame direction
convention, validate-then-apply guarantee, and sealed-light pose-mode
requirements are covered in {ref}`batched_light_randomization`.

The interactive Viewer uses the same direct pose source when configured after
PhysX GPU initialization:

```python
viewer.set_scene(scene)
viewer.configure_physx_gpu_rendering(physx_system, transport="direct")
viewer.update_render()
viewer.render()
```

Viewer and camera scene selection use the same base-plus-shared resolver and do
not apply scene offsets. Each rigid-pose update requires a TLAS update and
resets RT accumulation, so this path is more expensive than updating raster
vertex transforms. SAPIEN does
not currently provide deformable-body physics, although the separate
render-only `RenderCudaMeshComponent` API exposes mutable CUDA mesh vertices.
Supporting that component in batched RT would require synchronized BLAS
updates or rebuilds, aggregation across shared `SceneGroup` scenes, and an RT
accumulation reset after vertex changes; updating rigid TLAS instance
transforms alone is not sufficient.

## Batched ray tracing (`rt_batch`)

The `rt` pack renders one camera at a time: every camera of a `RenderCameraGroup`
owns a full renderer, and a capture submits one frame per camera. The `rt_batch`
pack instead renders the whole camera group with one compute dispatch, writing
the pictures straight into the buffers that `get_picture_cuda` exports.

```python
config = sapien.render.RenderConfig("rt_batch")
config["spp"] = 1
cameras = [scene.add_camera(f"camera{i}", 128, 128, 1.0, 0.05, 20.0,
                            render_config=config) for i in range(1024)]

group = sapien.render.RenderSystemGroup([scene.render_system])
camera_group = group.create_camera_group(cameras, ["Color", "Depth", "Segmentation"])
group.gpu_init()
for _ in range(60):
    group.update_render()
    camera_group.take_picture()
    color = camera_group.get_picture_cuda("Color")   # [1024, 128, 128, 4] uint8
```

A camera whose render config is a batched pack has **no per-camera renderer**:
`camera.take_picture()`, `camera.get_picture*()`, `RenderWindow`, `Viewer` and VR
all raise. Read pictures from the camera group instead.

### Rules of a camera group

- Every camera of one group must copy an **equal** `RenderConfig` and have the
  same width and height. `create_camera_group` raises naming the mismatch.
- A group mixes either batch or non-batch cameras, never both.
- All cameras of one group share one resolution, because one dispatch writes
  `[camera][height][width][channel]`. Create one camera group per resolution.
- Different camera groups of one `RenderSystemGroup` may use different kinds,
  for example one raster group and one `rt_batch` group.
- Uniform parameters come from the config and apply to the whole group:
  `config["exposure"]` (default 1.0), `config["toneMapper"]` (default 0, 0 gamma
  2.2, 1 sRGB, 2 ACES) and `config["spp"]` (default 1, 1..32 primary-ray
  samples). Variant parameters — `shading`, `shadows`, `maxShadowLights` —
  select shader source and are fixed when the group renderer is built.
  `camera.set_property` raises for a batch camera, and a uniform the shader does
  not declare raises at `gpu_init()` naming the parameter and the pack.
- The lifecycle is unchanged: construct → `set_cuda_poses` → `create_camera_group`
  → pose modes → `gpu_init()` → per frame `update_render()` + `take_picture()` +
  `get_picture_cuda(name)`. CUDA-sourced poses, `'cpu'` and `'cuda'` camera pose
  modes, and light changes after `gpu_init()` work exactly as with the `rt` pack.

### Environments

An *environment* is one distinct output render scene of the group, including a
`batched_render_shared` scene group. Several cameras may share an environment.
Each environment keeps its own top-level acceleration structure and is refit
every frame; a ray never leaves its own environment, so environments are
isolated exactly (not by spatial offsets).

### Pictures

`batch.comp` declares one storage buffer block per picture, named `Out<Name>`.
Only requested pictures are compiled in, and the element type selects the
format:

| Block element | Picture format | `get_picture_cuda` dtype |
|---|---|---|
| `uint c[]` | `r8g8b8a8Unorm` | `u1` (default `Color`) |
| `vec4 c[]` | `r32g32b32a32Sfloat` | `f4` |
| `float c[]` | `r32Sfloat` | `f4` with one channel |
| `uvec4 c[]` | `r32g32b32a32Uint` | `u4` |

`Color` is 8-bit by default. Request a float `Color` with
`config.picture_formats = {"Color": "r32g32b32a32Sfloat"}`; the shader must
declare `vec4` under the matching define, and the renderer raises when the
declared element type disagrees with the request.

`Depth` is the linear depth of the first hit in metres along the camera's
viewing axis (positive in front of the camera), and `0` where the ray hits
nothing. `Position` carries the camera-space xyz of the hit in its first three
channels and the rasterizer's normalized depth in its fourth; a miss leaves all
four at zero except the depth of 1.
`Segmentation` is the rasterizer's `uvec4` per-object segmentation id.

### Shading tiers and variants

The pack selects a tier with the `shading` **variant** parameter of its
`pack.json`; the tier is fixed for the life of the camera group because it selects
shader source.

| `shading` | Shader define | Content |
|---|---|---|
| `"visibility"` | `SHADING_VISIBILITY` | one primary ray; albedo, depth, normal, position, segmentation |
| `"lambert"` (default) | `SHADING_LAMBERT` | emission + albedo·(ambient + Σ light N·L) |
| `"pbr"` | `SHADING_PBR` | diffuse + GGX specular direct light, ambient term |

The `shadows` bool variant defines `SHADOWS`: one hard shadow ray per light for the
first `maxShadowLights` (default 4) lights facing the surface point, in the order
point, directional, spot, parallelogram; later lights are unshadowed. The `spp`
uniform (default 1, 1..32) supersamples the primary ray with a fixed Halton pattern:
deterministic, no random numbers, no accumulation or denoiser. Parallelogram lights
are sampled at their centre with a hard shadow.

Set the variants through the config, before creating the camera or the camera group:

```python
config = sapien.render.RenderConfig("rt_batch")
config["shading"] = "pbr"
config["shadows"] = True
config["maxShadowLights"] = 2
config["spp"] = 4

# the pack declares the same tiers as presets:
pbr = sapien.render.RenderConfig("rt_batch", preset="pbr")
```

Assigning a variant on a live camera through `camera.set_property` raises: a variant
selects shader source and cannot change after renderer creation. `cfg["spp"]` is a
uniform and stays settable at runtime.
