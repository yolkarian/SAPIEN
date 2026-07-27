(rl_index)=

# Reinforcement Learning

This tutorial focuses on how to use SAPIEN for reinforcement learning.

```{eval-rst}
.. toctree::

   gym
   manipulation
```

## GPU multi-environment isolation

When using `PhysxGpuSystem`, multiple SAPIEN scenes share one PhysX scene for
GPU Direct simulation. Declare the shape of the run on the scene config and
SAPIEN handles the environment IDs that keep those scenes apart:

```python
import sapien

sapien.physx.enable_gpu()

config = sapien.physx.PhysxSceneConfig()
config.num_scenes = 4096          # how many ordinary environments
config.with_shared_scene = True   # one scene holds the ground everyone stands on
sapien.physx.set_scene_config(config)

physx_system = sapien.physx.PhysxGpuSystem()   # reads and freezes the declaration
```

Construction is the only point where the declaration applies. PhysX bakes the
broadphase bit count into the scene at `createScene`, and refuses
`PxActor::setEnvironmentID` once an actor is in a scene — so by the time
environments are being built, everything is already frozen. Changing the config
afterwards affects the next system, never this one.

With `num_scenes` declared, SAPIEN hands each scene a unique ID on request:

```python
scene0 = sapien.Scene([physx_system])
scene1 = sapien.Scene([physx_system])

scene0.get_or_assign_environment_id()  # 0
scene1.get_or_assign_environment_id()  # 1
scene0.environment_id                  # 0, read-only
```

That is managed mode: SAPIEN is the sole assigner, so two scenes can never
collide over the same value. An ID is assigned lazily, at the latest when the
scene's first PhysX body is added, and `set_environment_id()` raises.

Leaving `num_scenes` unset selects manual mode: SAPIEN numbers nothing, an
unnumbered scene is environment 0, and you number scenes yourself with
`scene.set_environment_id(env_id)` before the scene gets any actor or
articulation (a body freezes its env ID at `PxScene::addActor`). `env_id` is
the raw, unshifted ID — `scene.environment_id` reads back exactly what you
passed — and `get_or_assign_environment_id()` raises. Duplicates are legal
and meaningful: giving several scenes the same ID puts them in one
environment so they collide with each other, which auto-assignment cannot
express. Without env IDs at all, separate scenes with
[`set_scene_offset`](#scene-offsets) instead.

The one scene holding objects common to every environment is marked with no ID
at all, and must be marked before it gets any body:

```python
shared_scene = sapien.Scene([physx_system])
shared_scene.set_shared_environment()
shared_scene.environment_id            # -1
```

### Tuning the broadphase bits

Merging the environment ID into an axis's broadphase bounds spreads environments apart so
sweep-and-prune has fewer candidate pairs to reject. `num_scenes` derives the counts for
you; set them by hand only when you have not declared a count:

```python
config.set_gpu_broadphase_env_id_bits(0, 0, 6)   # the only way to write them
config.gpu_broadphase_nb_bits_env_id_z           # 6, read-only
```

The default is `(0, 0, 4)`. Four matches PhysX's own "snap to grid" shift, so the banding
costs no coordinate precision at all; raising a count spends one bit of precision on that
axis. The axes are not independent — PhysX shifts the *same* environment ID on each, so
every axis keeps the low bits of one value and the distinct band count is
`2 ** max(x, y, z)`, never the product. Putting the bits on one axis is therefore as good
as spreading them, and cheaper; pick whichever axis can best afford the precision.

```{warning}
With `with_shared_scene`, every non-zero count must be equal. A narrower axis keeps fewer
of the shifted ID's low bits, so an environment that sits inside the widest axis's usable
band can wrap back out of it there and silently stop colliding with the shared object.
SAPIEN rejects mixed counts like `(12, 8, 0)` at construction, and widens a single-axis
layout such as the default to `(4, 4, 4)`.
```

#### Why a shared scene needs the shift

PhysX relocates environment `e` into the encoded broadphase band
`[e << (32 - b), (e + 1) << (32 - b))`, but gives objects shared by all environments a
*fixed* encoded interval that does not follow the banding. The bands at either end of the
range fall outside that interval, so bodies there silently stop colliding with shared
objects: a shared ground plane stops holding them up and they fall through it forever,
while everything else keeps working and the aggregate metrics still look healthy.

`with_shared_scene` shifts every environment ID into the bands that still reach the shared
object. That reserves about 1.2% of them, which is why declaring 4096 scenes picks 13 bits
rather than 12. The user-visible ID never moves:

```python
scene.environment_id                                # 0, what SAPIEN assigned you
physx_system.get_broadphase_environment_id(scene)   # 48, what PhysX receives
```

Declaring more scenes than the banding can give private bands to is fine — the surplus
wraps into the high bits of the ID, which PhysX drops when it places the box but compares
exactly when it filters the pair, so those environments stay correct and only share a band
in the spreading sense. With a shared scene the hard ceiling is 65535 ordinary
environments, because SAPIEN keeps the scene ID in a 16-bit collision-group field and
reserves `0xffff` for the shared scene itself.

Environment IDs affect PhysX broadphase membership only. Viewer and camera scene selection
never turns environment IDs into render offsets; place entities at different poses
explicitly when environments should appear separated. They are available only with
`PhysxGpuSystem`, and do not change SAPIEN GPU state-buffer indexing such as `gpu_index`
or `gpu_pose_index`.

For batched rendering and the Viewer, a render scene whose environment ID is shared
(`-1`/`0xFFFFFFFF`) is also marked as shared for
`sapien.render.RenderSystemGroup`. When that shared render system is included
in a render-system group, its objects and lights are rendered together with each
non-shared environment so every environment sees the shared world. You can also
control this explicitly with `scene.render_system.batched_render_shared`.
Shared content is included once in each resolved Viewer/camera output without a
scene-level transform.

## Reset-time domain randomization

`sapien.physx` provides batched setters for the physical properties most commonly
randomized at environment reset: link/body mass, inertia, center-of-mass pose, joint
friction, joint drive stiffness/damping/force limits, joint armature, and material
friction/restitution. Each function applies a whole batch in one call, validates every
entry before mutating anything, and works with both `PhysxCpuSystem` and
`PhysxGpuSystem`:

```python
import numpy as np
import sapien

# cache once after building the environments
links = [link for robot in robots for link in robot.links]
joints = [joint for robot in robots for joint in robot.active_joints]
base_masses = np.array([link.mass for link in links], np.float32)

# at reset time
sapien.physx.set_body_masses(links, base_masses * np.random.uniform(0.8, 1.2, len(links)))
sapien.physx.set_joint_frictions(joints, np.random.uniform(0.0, 0.2, len(joints)))
sapien.physx.set_joint_drive_properties(
    joints,
    stiffness=np.random.uniform(80.0, 120.0, len(joints)),
    damping=np.random.uniform(8.0, 12.0, len(joints)),
)
```

- `set_body_masses` accepts rigid dynamic bodies and articulation links. By default it
  scales each body's diagonal inertia by `new_mass / old_mass` and keeps the
  center-of-mass pose; pass `scale_inertia=False` to set only the mass.
- `set_joint_drive_properties` updates only the fields you pass (`stiffness`,
  `damping`, `force_limit`); omitted fields and the drive type keep their current
  values. Joints must have at least 1 DOF, so build the list from
  `articulation.active_joints`.
- `set_body_inertias` takes `[N, 3]` positive diagonal inertias in the center-of-mass
  frame; `set_body_cmass_local_poses` takes `[N, 7]` rows of
  `[x, y, z, qw, qx, qy, qz]` in the body frame (quaternions are normalized).
- `set_joint_armatures` applies one armature value per joint to all of its DOFs.
- `set_material_properties` batches `PhysxMaterial` static friction, dynamic friction,
  and restitution (each optional). Materials are shared objects: changing one affects
  every collision shape bound to it, so per-env randomization requires per-env
  materials. Bind a unique material per shape at build time and only change its values
  at reset; do not rebind shape materials at runtime.
- For partial resets, pass only the components of the environments being reset.

On `PhysxGpuSystem` these functions may be called any time after `gpu_init()` as long
as no step is in flight (never between `step_start()` and `step_finish()`). PhysX
uploads the new mass and joint properties to the GPU during the next step without
disturbing GPU-side poses, velocities, or joint states, so no state re-upload or
re-initialization is needed. Note that joint state randomization (qpos, qvel, drive
targets) is separate: on the GPU it goes through the CUDA buffers and
`gpu_apply_*` functions.

(batched_light_randomization)=

### Batched lighting randomization

`sapien.render` provides the same validate-then-apply behavior for reset-time
lighting randomization. The three setters require one array row per light,
validate the entire batch before changing any light, and therefore never leave
a failed batch partially applied:

```python
# One directional light per environment being reset.
lights = directional_lights
n = len(lights)

# Required before group.gpu_init() when a RenderSystemGroup will seal the lights.
for light in lights:
    light.set_pose_mode("cpu")

poses = np.tile(
    np.array([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0], np.float32),
    (n, 1),
)
directions = np.tile(np.array([1.0, 0.0, -1.0], np.float32), (n, 1))
colors = np.random.uniform(0.5, 1.5, (n, 3)).astype(np.float32)

sapien.render.set_light_poses(lights, poses)
sapien.render.set_light_directions(lights, directions)
sapien.render.set_light_colors(lights, colors)
```

- `set_light_poses` takes `[N, 7]` rows of
  `[x, y, z, qw, qx, qy, qz]` and normalizes each quaternion. These are local
  poses relative to the light's owning entity; they are world poses only when
  that entity has the identity pose.
- `set_light_directions` takes finite, non-zero `[N, 3]` directions for
  directional lights. A light shines along the local `+x` axis, so each
  direction is interpreted in the owning entity's local frame; the setter
  changes the local orientation while preserving the local position. Every
  other light type is rejected.
- `set_light_colors` takes finite, non-negative `[N, 3]` RGB rows. Values above
  `1` are valid HDR intensities.
- A light sealed by `RenderSystemGroup.gpu_init()` defaults to static pose
  ownership. To randomize its pose or direction later, call
  `light.set_pose_mode("cpu")` before `gpu_init()`; updates then become visible
  at the next `group.update_render()`. Colors remain CPU-mutable in every pose
  mode. For partial resets, pass only the lights belonging to the environments
  being reset.
