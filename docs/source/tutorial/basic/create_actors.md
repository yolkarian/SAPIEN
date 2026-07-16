(create_actors)=

# Create Rigid Bodies

```{eval-rst}
.. highlight:: python
```

SAPIEN's high-level actor API builds rigid-body `sapien.Entity` objects. An
entity can contain a PhysX rigid component for simulation and a render body
component for visualization.

In this tutorial, you will learn how to:

- create rigid bodies with primitive collision and visual shapes;
- create rigid bodies from mesh files;
- set body and shape poses with `sapien.Pose`;
- access the underlying PhysX component.

:::{figure} assets/create_actors.png
:align: center
:figclass: align-center
:width: 640px
:::

## Create a body from one primitive

Use `scene.create_actor_builder()` to create an `ActorBuilder`. Shape poses
are relative to the body frame, while `entity.set_pose(...)` sets the body pose
in the world frame.

```python
import sapien

scene = sapien.Scene()

builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[0.5, 0.5, 0.5])
builder.add_box_visual(half_size=[0.5, 0.5, 0.5], material=[1.0, 0.0, 0.0])
box = builder.build(name="box")
box.set_pose(sapien.Pose(p=[0, 0, 0.5]))
```

`Pose` stores a position `p` and a quaternion `q` in `wxyz` order. It can
also be constructed from a 4x4 transformation matrix. If a primitive visual
omits `material`, SAPIEN uses a light blue-gray dielectric with moderate
roughness and specular response instead of a flat white surface.

## Create a body from multiple primitives

A single rigid body may have multiple collision and render shapes. The following
creates a table from one top and four legs.

```python
builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[0.6, 0.4, 0.05])
builder.add_box_visual(half_size=[0.6, 0.4, 0.05], material=[0.7, 0.5, 0.3])

for x in [-0.45, 0.45]:
   for y in [-0.3, 0.3]:
      leg_pose = sapien.Pose([x, y, -0.35])
      builder.add_box_collision(pose=leg_pose, half_size=[0.05, 0.05, 0.35])
      builder.add_box_visual(
         pose=leg_pose,
         half_size=[0.05, 0.05, 0.35],
         material=[0.7, 0.5, 0.3],
      )

table = builder.build_static(name="table")
table.set_pose(sapien.Pose([0, 0, 0.75]))
```

## Create static and kinematic bodies

`builder.build()` creates a dynamic rigid body by default. Use
`build_static` for fixed geometry and `build_kinematic` for bodies whose
motion is controlled by user-provided poses or kinematic targets.

```python
static_builder = scene.create_actor_builder()
static_builder.add_box_collision(half_size=[1, 1, 0.05])
static_body = static_builder.build_static(name="static_body")

mover_builder = scene.create_actor_builder()
mover_builder.add_sphere_collision(radius=0.2)
mover_builder.add_sphere_visual(radius=0.2, material=[0.2, 0.4, 1.0])
mover = mover_builder.build_kinematic(name="kinematic_sphere")
```

## Create a height field

Use `scene.add_heightfield` for large static z-up terrain. Rows map to +x,
columns map to +y, and int16 sample values map to +z after `height_scale` is
applied. When `render=True`, SAPIEN also creates a triangle-mesh render shape
for visualization. In `PhysxGpuSystem` / Direct GPU API workflows, add the
height field before `gpu_init()`. Ground planes and height fields use a subtle,
mipmapped checker material by default; pass `render_material` to replace it.

```python
import numpy as np

height_field = np.zeros((128, 128), dtype=np.int16)
height_field[48:80, 48:80] = 20
terrain = scene.add_heightfield(
   height_field,
   row_scale=0.1,
   column_scale=0.1,
   height_scale=0.005,
   render=True,
)
```

## Create a body from mesh files

Visual meshes can be loaded with `add_visual_from_file`. For collisions, choose
a representation that matches the object and body type.

```python
builder = scene.create_actor_builder()
builder.add_visual_from_file("model.glb", scale=[1, 1, 1])
builder.add_convex_collision_from_file("collision.obj", scale=[1, 1, 1])
mesh_body = builder.build(name="mesh_body")
```

Available mesh collision helpers include:

- `add_convex_collision_from_file`: cook one convex mesh;
- `add_multiple_convex_collisions_from_file`: load multiple convex parts, or
  request `decomposition="coacd"`;
- `add_nonconvex_collision_from_file`: load a triangle mesh collision. For
  dynamic bodies and articulation links, SAPIEN builds an SDF-backed triangle
  mesh and accepts an optional `sapien.physx.PhysxSDFConfig`.

## Collision and render components

The builder returns an entity. Access PhysX or render details through its
components.

```python
body = box.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
print(body.mass, body.linear_velocity)

shapes = body.get_collision_shapes()
shapes[0].set_collision_groups([1, 1, 0, 0])
```

`collision_groups[3]` packs a scene ID in the upper 16 bits and an ignore ID in
lower 16 bits. ID `0xffff` is shared: as a scene ID it collides with all scene
IDs, and as an ignore ID it does not suppress collisions with matching ignore
IDs.

## Remove a body

After a body is built, remove it with `scene.remove_actor(entity)` or
`scene.remove_entity(entity)`. Do not use the removed entity or its components
after removal.
