(camera)=

# Camera

```{eval-rst}
.. highlight:: python
```

SAPIEN cameras are `sapien.render.RenderCameraComponent` objects attached to
entities. The old `CameraEntity` name and `get_float_texture` /
`get_uint32_texture` accessors are obsolete; use `camera.get_picture(name)`
or `camera.get_picture_cuda(name)`.

In this tutorial, you will learn how to:

- create free and mounted cameras;
- render RGB images offscreen;
- read depth/position and segmentation render targets;
- convert the position target to a world-space point cloud.

## Create a scene and camera

`sapien.Scene()` already contains a render system. `SapienRenderer` is no
longer required for normal Python usage.

```python
import numpy as np
import sapien

scene = sapien.Scene()
scene.add_ground(0)
scene.set_ambient_light([0.12, 0.12, 0.12])
scene.add_directional_light(
   [1, 1, -1], [2.0, 1.9, 1.8], shadow=True
)

width, height = 640, 480
camera = scene.add_camera(
   name="camera",
   width=width,
   height=height,
   fovy=np.deg2rad(60),
   near=0.01,
   far=100.0,
)
camera.local_pose = sapien.Pose([-3, 0, 1.0])
```

The camera component's local pose is relative to its owning entity. For a camera
created by `scene.add_camera`, the owning entity is created for you and starts
at identity, so setting `camera.local_pose` places the camera in the world.

## Mount a camera to an entity

To make a camera follow a body, attach it to that body's entity. The helper
`scene.add_mounted_camera` creates and attaches the camera component.

```python
mount = scene.create_actor_builder().build_kinematic(name="camera_mount")
mount.set_pose(sapien.Pose([-3, 0, 1.0]))

camera = scene.add_mounted_camera(
   name="mounted_camera",
   mount=mount,
   pose=sapien.Pose(),
   width=640,
   height=480,
   fovy=np.deg2rad(60),
   near=0.01,
   far=100.0,
)
```

If the mount entity moves, the camera's global pose changes with it. Camera axes
follow SAPIEN's robotics convention: x forward, y left, z up. Rendered position
images are in the renderer/OpenGL camera space, where -z is forward.

## Intrinsic parameters

Use field-of-view helpers or set full OpenCV-style intrinsics.

```python
camera.set_fovy(np.deg2rad(60), compute_x=True)
camera.set_focal_lengths(fx=600, fy=600)
camera.set_principal_point(cx=320, cy=240)
camera.skew = 0

camera.set_perspective_parameters(
   near=0.01,
   far=100.0,
   fx=600,
   fy=600,
   cx=320,
   cy=240,
   skew=0,
)
```

## Select multiple render scenes

A camera renders its owning scene plus render systems marked
`batched_render_shared=True` in the same render context by default. Use
`set_scenes` to replace its base scene selection:

```python
camera.set_scenes([scene0, scene1])
```

The selected scenes and associated shared scenes are aggregated once in stable
order. No scene-level transform is applied; objects retain their existing poses
and may overlap. This selection changes only render content, not the camera's
mount pose, properties, update flow, or `take_picture()` call.

## Render an RGB image

Update render poses, render the camera, then read the `"Color"` picture.

```python
scene.update_render()
camera.take_picture()
rgba = camera.get_picture("Color")  # float array, H x W x 4
rgb_u8 = (rgba[..., :3].clip(0, 1) * 255).astype(np.uint8)
```

`camera.get_picture_names()` lists the render targets produced by the current
shader pack. CUDA-capable builds can avoid a CPU copy with
`camera.get_picture_cuda("Color")`, which returns a `sapien.CudaArray` with
`torch()`, `cupy()`, `jax()`, and `dlpack()` adapters.

## Generate a point cloud

The default shader pack exposes a `"Position"` render target. Its first three
channels are camera-space coordinates and its fourth channel is the depth buffer
value; pixels at the far plane have depth close to 1.

```python
position = camera.get_picture("Position")
valid = position[..., 3] < 1
points_camera = position[..., :3][valid]

# camera.get_model_matrix() maps renderer camera coordinates to world space
model = camera.get_model_matrix()
points_h = np.concatenate(
   [points_camera, np.ones((points_camera.shape[0], 1))],
   axis=1,
)
points_world = (model @ points_h.T).T[:, :3]
```

## Segmentation

Shader packs define which segmentation targets are available. With the default
shader pack, inspect available names first, then read the desired target.

```python
print(camera.get_picture_names())
segmentation = camera.get_picture("Segmentation")
```

Some display-oriented targets, such as `"SegmentationView0"` and
`"SegmentationView1"`, are colorized for visualization. Raw segmentation
targets store integer identifiers.

## Screenshots from the viewer

The viewer exposes a `Screenshot` button in the `Control` window. For
programmatic screenshots, use the viewer window's picture methods after
`viewer.render()` or use an offscreen camera as shown above.
