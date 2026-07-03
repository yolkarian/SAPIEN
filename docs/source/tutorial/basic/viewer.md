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
   scene.update_render()
   viewer.render()
```

## Free camera control

Use `w`, `a`, `s`, and `d` to move the free camera. Hold the right mouse
button and drag to rotate. Hold the middle mouse button and drag to translate.
The movement speed can be adjusted in the `Control` window.

## Selection and focus

Left-click an entity in the viewport to select it. Press `f` to focus the
camera on the selected entity. In focused mode, the right mouse button rotates
around the selected entity and the mouse wheel zooms in or out. Pressing any of
`w`, `a`, `s`, or `d` returns to free-camera mode.

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
