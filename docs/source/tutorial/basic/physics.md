(physics)=

# Physics

```{eval-rst}
.. highlight:: python
```

This section describes the current PhysX-facing Python API for configuring a
scene and changing rigid-body properties.

In this tutorial, you will learn how to:

- configure global PhysX defaults before creating a scene;
- assign `PhysxMaterial` objects to collision shapes;
- create kinematic bodies;
- set damping and velocities on rigid components;
- read pose and velocity from entities and components.

:::{figure} assets/physics.gif
:align: center
:figclass: align-center
:width: 640px
:::

## Configure default PhysX properties

PhysX defaults are configured through `sapien.physx` before the scene/system is
created. `PhysxSceneConfig` controls scene-level settings such as gravity,
CCD, TGS, CPU worker count, and GPU broadphase environment-id bits. Default
contact material is configured separately.

```python
import sapien

scene_config = sapien.physx.PhysxSceneConfig()
scene_config.gravity = [0, 0, -9.81]
scene_config.enable_ccd = True
sapien.physx.set_scene_config(scene_config)

sapien.physx.set_default_material(
   static_friction=0.5,
   dynamic_friction=0.5,
   restitution=0.0,
)

scene = sapien.Scene()
scene.set_timestep(1 / 240)
```

`sapien.SceneConfig` is kept as an alias of `sapien.physx.PhysxSceneConfig`
for compatibility, but new code should use the `sapien.physx` namespace.

## Set physical materials

`sapien.physx.PhysxMaterial` stores contact friction and restitution. Pass it
to collision-shape builder methods.

```python
slippery = sapien.physx.PhysxMaterial(
   static_friction=0.05,
   dynamic_friction=0.03,
   restitution=0.0,
)

builder = scene.create_actor_builder()
builder.add_sphere_collision(radius=0.2, material=slippery, density=500)
builder.add_sphere_visual(radius=0.2, material=[0.2, 0.4, 1.0])
ball = builder.build(name="slippery_ball")
```

Density, patch radius, minimum patch radius, contact offset, rest offset, and
collision groups live on collision shapes, not on the render material.

## Combine contact materials

`PhysxMaterial.friction_combine_mode` and `restitution_combine_mode` accept
`"average"` (default), `"min"`, `"multiply"`, or `"max"`. Each also has
`get_*` and `set_*` methods. Other strings and non-string values raise `TypeError`.
For coefficients `a` and `b`, these compute `(a + b) / 2`, `min(a, b)`,
`a * b`, or `max(a, b)`. Friction applies separately to static and dynamic
coefficients; restitution chooses its mode independently.

A contact pair uses the higher-priority mode of its two materials:
`average < min < multiply < max`. Setting just the robot to `multiply` will
not override a ground material set to `max`.

```python
floor_material = sapien.physx.PhysxMaterial(1.0, 1.0, 0.0)
robot_material = sapien.physx.PhysxMaterial(0.7, 0.5, 0.2)
for material in (floor_material, robot_material):
   material.friction_combine_mode = "multiply"
   material.set_restitution_combine_mode("multiply")
```

Pass these materials to collision builders/shapes. This pair has effective
static/dynamic friction `0.7/0.5` and restitution `0.0`. The formulas above
assume ordinary nonnegative restitution; PhysX negative restitution represents
compliant contact stiffness and follows additional combination rules.

Materials are shared objects: changing one affects all shapes using it, including
shape clones. Use separate instances for independent settings. The default material
is **weakly cached**, not a persistent configuration object: keep a reference from
`get_default_material()` while building shapes. A temporary call such as
`get_default_material().friction_combine_mode = "multiply"` may immediately lose
the setting if nothing retains the instance. After all strong references disappear,
a newly created default material starts with `average` again. Calling
`set_default_material(...)` also resets that cache, without changing existing shapes.

Modify materials only between completed physics steps, with no concurrent users
in any scene sharing the material. For GPU PhysX, configure these modes **before
`gpu_init()`**. Updating them afterwards is not currently guaranteed; there is no
initialization-time freeze guard, and a CPU getter is not proof of GPU propagation.

## Body speed limits

`PhysxRigidBodyComponent.max_linear_velocity` and `max_angular_velocity`, also
available through `get_*`/`set_*`, apply to both rigid dynamic bodies and
articulation links. The linear limit is the magnitude of COM velocity in length
units/s (m/s for metre-based scenes); the angular limit is in rad/s. These are not
per-axis clamps or joint DOF limits.

```python
# Configure articulation links before GPU initialization.
for link in robot.links:
   link.max_linear_velocity = 1000.0
   link.set_max_angular_velocity(1000.0)
```

Values are converted to float32 and must be finite and in the inclusive range
`[0, 1e16f]`; invalid values raise `RuntimeError` without changing the parameter.
The upper endpoint is the float32 approximation to `1e16`, also the default
linear limit of rigid dynamic bodies, so reading and writing the default is valid.
Existing defaults are unchanged: rigid dynamics use approximately `1e16` linear
and `100` angular; articulation links use `100 * PxTolerancesScale::length` linear
and `50` angular. SAPIEN's default tolerance length is `0.1`, so its default
articulation linear limit is **10**, not 100 length units/s.

PhysX applies limiting before solving: contact/joint/drive solving can still
produce velocities above the limit at the end of a step. Articulation limiting
can change momentum unphysically; prefer joint damping/limits where appropriate.
These fields do not replace `max_joint_velocity`, nor do they cap kinematic targets.
Clone/reparent operations preserve both limits for all affected links.

CPU changes are supported between completed steps. On GPU, configure before
`gpu_init()`; later propagation is not guaranteed and is not guarded by a freeze
check. Never mutate these parameters between `step_start()` and `step_finish()`.

## Collision groups

Collision groups are stored as four 32-bit words `[g0, g1, g2, g3]` on each
collision shape.

- `g0` is the contact type bit mask.
- `g1` is the contact affinity bit mask.
- `g2` is the ignore-group bit mask.
- The upper 16 bits of `g3` are a scene ID. Different non-shared scene IDs do
  not collide. Scene ID `0xffff` is shared and collides with all scene IDs,
  matching PhysX GPU environment-ID shared-object behavior.
- The lower 16 bits of `g3` are an ignore ID. Shapes with matching non-shared
  ignore IDs and overlapping `g2` bits do not collide. Ignore ID `0xffff` is
  shared and does not match any ignore ID.

```python
shape.set_collision_groups([1, 1, 0, 0x00010000])  # scene ID 1
shared_shape.set_collision_groups([1, 1, 0, 0xFFFF0000])  # shared scene ID
```

## Create a kinematic body

Kinematic bodies are dynamic PhysX bodies whose motion is driven by the user
rather than by forces. Use `build_kinematic` or set the builder body type to
`"kinematic"`.

```python
builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[1.0, 0.5, 0.05])
builder.add_box_visual(half_size=[1.0, 0.5, 0.05], material=[0.6, 0.6, 0.6])
slope = builder.build_kinematic(name="slope")
slope.set_pose(sapien.Pose([0, 0, 0.5]))

slope_body = slope.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
slope_body.set_kinematic_target(sapien.Pose([0.1, 0, 0.5]))
```

For static world geometry that never moves, prefer `build_static`.

## Set damping and velocities

Actor-builder damping fields are applied when the PhysX component is created.
You can also edit the component after building.

```python
builder = scene.create_actor_builder()
builder.linear_damping = 0.1
builder.angular_damping = 0.05
builder.add_box_collision(half_size=[0.2, 0.2, 0.2])
body_entity = builder.build(name="damped_box")

body = body_entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
body.set_linear_velocity([0, 1, 0])
body.set_angular_velocity([0, 0, 2])
```

## Read kinematic quantities

The world pose belongs to the entity. Linear and angular velocity belong to the
rigid body component.

```python
pose = body_entity.get_pose()
linear_velocity = body.get_linear_velocity()
angular_velocity = body.get_angular_velocity()

print(pose.p, pose.q)  # quaternion order is wxyz
print(linear_velocity, angular_velocity)
```
