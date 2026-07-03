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
:width: 540px

From *A Shader-Based Ray Tracing Engine*, Park et al.
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

:::{figure} assets/rt_color.png
:align: center
:width: 540px
:::

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

Example result with the ray-tracing shader pack
:::
