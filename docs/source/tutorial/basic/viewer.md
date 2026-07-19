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

The Viewer auto-detects one initialized `PhysxGpuSystem` in its resolved base
plus shared render scenes. Explicit `configure_physx_gpu_rendering()` remains
available when several GPU systems make that choice ambiguous.

The active choice is available as `viewer.pose_transport` and is also shown in
the existing `Control` window. The cumulative pose D2H bytes for the active
transport are available as `viewer.pose_transfer_bytes`; a staged submission
transfers 28 bytes per unique rendered GPU pose.

Device identity and interop diagnostics are available on `sapien.Device` through
`uuid`, `pci_string`, `can_direct_cuda_vulkan_interop()`, `can_access_peer()`,
and the CUDA/Vulkan external-memory and external-semaphore capability flags.
Peer access is diagnostic only; `"auto"` still uses staged transport between
different physical devices.

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
- `GPU Interaction` shows the active pose transport and transferred bytes, and
  configures point-spring stiffness, damping, and maximum acceleration.
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
- `Contacts` can display contacts reported by a CPU PhysX step. Under PhysX GPU
  it shows an explicit notice instead; use GPU contact queries in application
  code.
- `Settings` exposes scene-level PhysX settings such as timestep and gravity.
- `Render` exposes renderer options including shader pack, denoiser, samples
  per pixel, ray depth, focal plane, and aperture.

Under PhysX GPU, the `Entity` pose and `Articulation` qpos/drive-target rows are
read on demand for only the selected object and cached for the submitted
Viewer frame. Pose, qpos, and drive-target edits are queued and applied by
`viewer.apply_interactions()` before physics. Collapsing these windows prevents
the corresponding GPU readback. Non-root articulation-link poses are read-only,
and the Transform window explicitly disables its CPU Pinocchio IK controls for
GPU articulations.

## Move objects

When an entity is selected, press `g` to enter grab mode or `r` to enter
rotate mode. Press `x`, `y`, or `z` to constrain the operation to an axis;
press the axis key twice to use the local axis. Hold `Shift` while selecting
an axis to move in the orthogonal plane.

For articulated links, the transform window can optionally use IK controls when
a supported articulation is selected.

With PhysX GPU rendering, hold `Ctrl` and left-drag a dynamic rigid body or
articulation link to apply a damped point spring at the clicked Position-buffer
hit point. Gizmo translation uses the same physical target by default. The
`Teleport` button queues an explicit GPU pose update; rigid-body teleports
preserve linear and angular velocity unless zeroing is explicitly requested.

Viewer plugins can drive the same deferred interaction path with
`begin_gpu_interaction()`, `update_gpu_interaction_target()`, and
`end_gpu_interaction()`. Programmatic teleports can be queued with
`queue_gpu_rigid_dynamic_pose()` or `queue_gpu_articulation_root_pose()`.
None of these commands reaches PhysX until `apply_interactions()` is called.

Apply Viewer commands before every physics substep:

```python
while not viewer.closed:
   viewer.apply_interactions()
   physx_system.step()
   viewer.update_render()
   viewer.render()
```

The Viewer composes its spring with the selected row of the application's
`cuda_*_force` and `cuda_*_torque` buffers in private scratch. It does not
modify those exposed buffers. While a Viewer spring is active,
`viewer.apply_interactions()` must be the final force/torque apply for that
selected body or link before `step()`; a later application apply would replace
the composed spring.
