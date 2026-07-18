(viewer)=

# Viewer

```{eval-rst}
.. highlight:: python
```

The Python viewer is available through `sapien.utils.Viewer` or the
convenience method `scene.create_viewer()`. It requires a display-capable
environment.

## Create a viewer

```python
import numpy as np
import sapien

scene = sapien.Scene()
viewer = scene.create_viewer()
viewer.set_camera_xyz(-4, 0, 2)
viewer.set_camera_rpy(0, -np.arctan2(2, 4), 0)

while not viewer.closed:
   scene.step()
   viewer.update_render()  # submit simulation state and render transforms
   viewer.render()         # draw the submitted state and Viewer UI
```

`Viewer.update_render()` and `Viewer.render()` are separate phases. Calling
`render()` repeatedly while paused redraws the last submitted state without
stepping render systems or fetching poses again.

## PhysX GPU rendering

After `PhysxGpuSystem.gpu_init()`, the Viewer can source dynamic transforms
directly from `cuda_rigid_body_data` when PhysX CUDA and Vulkan use the same
compatible device. This path does not call `sync_poses_gpu_to_cpu()`, so CPU
`Entity.pose` values intentionally remain stale.

```python
physx_system.gpu_init()
viewer = sapien.utils.Viewer()
viewer.set_scene(scene)
viewer.configure_physx_gpu_rendering(physx_system, transport="auto")

while not viewer.closed:
   physx_system.step()
   viewer.update_render()
   viewer.render()
```

The available transport requests are `"auto"`, `"direct"`, `"staged"`, and
`"cpu-debug"`. `"auto"` selects direct CUDA/Vulkan interop on a compatible
same physical device and otherwise selects staged transfer. Staged transfer
gathers one 7-float pose per rendered GPU body, copies it through reusable
pinned host memory, and composes raster or ray-tracing transforms with Vulkan
compute on the rendering device. `"cpu-debug"` explicitly performs the full
pose download and updates CPU entities.

The active choice is available as `viewer.pose_transport`. The cumulative pose
D2H bytes for the active transport are available as
`viewer.pose_transfer_bytes`; a staged submission transfers 28 bytes per
unique rendered GPU pose.

## Multiple render scenes

`viewer.set_scenes([scene0, scene1])` selects those base scenes. A camera can
make the same opt-in selection with `camera.set_scenes([scene0, scene1])`.
Render systems marked `batched_render_shared=True` in the same render context
are appended once. Scene selection never creates a grid or applies offsets, so
objects in different scenes retain their poses and may overlap.

## Free camera control

Use `w`, `a`, `s`, and `d` to move the free camera. Hold the right mouse
button and drag to rotate. Hold the middle mouse button and drag to translate.
The movement speed can be adjusted in the `Control` window.

## Selection and focus

Left-click an entity in the viewport to select it. Press `f` to focus the
camera on the selected entity. In focused mode, the right mouse button rotates
around the selected entity and the mouse wheel zooms in or out. Pressing any of
`w`, `a`, `s`, or `d` returns to free-camera mode.

With direct or staged PhysX GPU rendering, selection, focus, coordinate/joint
axes, mounted-camera overlays, and the Transform gizmo use the most recently
submitted GPU pose instead of stale `Entity.pose`. Direct mode downloads only
the selected 7-float pose once per submitted frame; staged mode reuses the pose
already present in its completed pinned-host slot.

## Control window

The `Control` window contains simulation, camera, display, selection, and
screenshot controls.

- `Pause` pauses simulation while keeping the viewer interactive.
- `Single Step` advances the simulation once while paused.
- `Movement Speed` controls camera move, rotate, and scroll speeds.
- `Camera` switches between the free camera and cameras in the scene and can
  copy camera parameters.
- `Display` selects the render target, resolution, and camera overlays.
- `Selection` controls joint-axis display, coordinate frame display, selected
  entity opacity, and selected frame size.
- `Screenshot` saves an image from the viewer window.

## Scene and entity windows

The current viewer uses SAPIEN 3's entity/component model.

- `Scene` shows the scene hierarchy and lets you select entities without
  clicking in the viewport.
- `Entity` shows the selected entity's name, per-scene id, pose, render body
  components, PhysX rigid components, collision shapes, and component-specific
  properties. The collision visualization buttons temporarily replace render
  bodies with visualizations of collision shapes.
- `Articulation` appears when the selected entity is an articulation link. It
  lists joints and exposes drive target, velocity target, damping, stiffness,
  force limit, friction, and drive mode controls.
- `Contacts` can display contacts reported by the current PhysX step.
- `Settings` exposes scene-level PhysX settings such as timestep and gravity.
- `Render` exposes renderer options including shader pack, denoiser, samples
  per pixel, ray depth, focal plane, and aperture.

## Move objects

When an entity is selected, press `g` to enter grab mode or `r` to enter
rotate mode. Press `x`, `y`, or `z` to constrain the operation to an axis;
press the axis key twice to use the local axis. Hold `Shift` while selecting
an axis to move in the orthogonal plane.

For articulated links, the transform window can optionally use IK controls when
a supported articulation is selected.
