(hello_world)=

# Hello World

```{eval-rst}
.. highlight:: python
```

SAPIEN 3 creates scenes directly. You no longer need to create an `Engine` or
bind a `SapienRenderer` before making a scene; both classes are kept only as
compatibility wrappers.

In this tutorial, you will learn how to:

- create a `sapien.Scene`;
- add a ground plane and a dynamic rigid body;
- add lights and a viewer;
- run a simulation/render loop.

:::{figure} assets/hello_world.png
:align: center
:figclass: align-center
:width: 640px
:::

The same example is installed with the package and can be run with:

```shell
python -m sapien.example.hello_world
```

## Create a scene

`sapien.Scene()` creates a simulation world with a CPU PhysX system and a
render system by default.

```python
import numpy as np
import sapien

scene = sapien.Scene()
scene.set_timestep(1 / 100.0)
```

For physics-only workloads, construct the scene with only the systems you need,
for example `sapien.Scene([sapien.physx.PhysxCpuSystem()])`.

## Add rigid bodies

Rigid bodies are represented as `sapien.Entity` objects with PhysX and, when
rendering is enabled, render components attached to them. The convenience
`ActorBuilder` still exists and returns such an entity.

```python
scene.add_ground(altitude=0)

builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[0.5, 0.5, 0.5])
builder.add_box_visual(half_size=[0.5, 0.5, 0.5], material=[1.0, 0.0, 0.0])
box = builder.build(name="box")
box.set_pose(sapien.Pose(p=[0, 0, 0.5]))
```

## Lighting and viewer

Use light helpers on `Scene` and create the viewer from the scene.

```python
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

viewer = scene.create_viewer()
viewer.set_camera_xyz(x=-4, y=0, z=2)
viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
```

The GUI is optional. For offscreen rendering, create a camera and call
`camera.take_picture()` as shown in {ref}`camera`.

## Simulation loop

`scene.step()` advances PhysX. `scene.update_render()` uploads CPU-side
entity poses to the renderer. The viewer's `render` call draws the frame.

```python
while not viewer.closed:
   scene.step()
   scene.update_render()
   viewer.render()
```

## Full packaged script

```{eval-rst}
.. literalinclude:: ../../../../python/py_package/example/hello_world.py
   :linenos:
```
