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

Set the camera and viewer shader directories before creating cameras or the
viewer.

```python
import sapien

sapien.render.set_camera_shader_dir("rt")
sapien.render.set_viewer_shader_dir("rt")
sapien.render.set_ray_tracing_samples_per_pixel(16)
sapien.render.set_ray_tracing_path_depth(8)
sapien.render.set_ray_tracing_denoiser("oidn")  # "none", "oidn", or "optix"

scene = sapien.Scene()
camera = scene.add_camera("camera", 640, 480, 1.0, 0.01, 100.0)
```

To return to rasterization later in the same process, reset the shader dirs:

```python
sapien.render.set_camera_shader_dir("default")
sapien.render.set_viewer_shader_dir("default")
```

## Sampling and denoising

Ray-traced images are noisy when the sample count is low. Increase samples per
pixel for quality, and use a denoiser when available.

```python
sapien.render.set_ray_tracing_samples_per_pixel(64)
sapien.render.set_ray_tracing_path_depth(12)
sapien.render.set_ray_tracing_denoiser("oidn")
```

`oidn` uses the packaged Open Image Denoise integration. `optix` requires a
compatible NVIDIA RTX driver stack. If either denoiser is unavailable on your
machine, use `"none"`.

## Materials for ray tracing

Render materials live in `sapien.render`. The parameters `base_color`,
`roughness`, `metallic`, `transmission`, `ior`, and
`transmission_roughness` affect ray-traced reflection/refraction.

```python
glass = sapien.render.RenderMaterial(
   base_color=[0.8, 0.9, 1.0, 0.4],
   roughness=0.02,
   metallic=0.0,
   transmission=0.9,
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

## Batched GPU rendering

The `"rt"` shader pack supports `sapien.render.RenderSystemGroup`, including
shared render scenes, rigid-body poses sourced directly from PhysX GPU, and
mounted cameras with GPU pose batch indices. Configure the RT shader before
creating cameras, then use the same batched workflow as rasterization:

```python
sapien.render.set_camera_shader_dir("rt")

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
