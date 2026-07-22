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
GPU Direct simulation. For batched reinforcement learning, configure PhysX GPU
broadphase environment ID bits before creating the GPU system, then let SAPIEN
assign one unique environment ID per scene automatically:

```python
import sapien

sapien.physx.enable_gpu()

config = sapien.physx.PhysxSceneConfig()
config.gpu_broadphase_env_id_bits = 8
sapien.physx.set_scene_config(config)

physx_system = sapien.physx.PhysxGpuSystem()
scene0 = sapien.Scene([physx_system])
scene1 = sapien.Scene([physx_system])

env_id0 = scene0.get_or_assign_environment_id()  # 0
env_id1 = scene1.get_or_assign_environment_id()  # 1
```

The environment ID must be assigned before adding actors or articulations to a
scene. If you do not assign it manually, SAPIEN lazily assigns a unique ID when
one is needed, including when the first PhysX body is added. Non-shared IDs are
unique by default; assigning the same non-shared ID to two live scenes raises an
error. Use `allow_duplicate=True` only when you intentionally want two SAPIEN
scenes to share one PhysX broadphase environment:

```python
scene0.set_environment_id(7)
scene1.set_environment_id(7, allow_duplicate=True)
```

Use `-1` or `0xFFFFFFFF` for a shared scene/object that collides with all
environments:

```python
shared_scene = sapien.Scene([physx_system])
shared_scene.set_environment_id(-1)
```

Environment IDs affect PhysX broadphase membership only. Viewer and camera
scene selection never turns environment IDs into render offsets; place entities
at different poses explicitly when environments should appear separated.

`scene.environment_id` and `scene.get_environment_id()` only inspect the
already assigned ID and return `None` if no ID exists yet; they do not allocate
a new ID. Use `scene.get_or_assign_environment_id()` when you want explicit
lazy allocation. Environment IDs are only available with `PhysxGpuSystem` and
do not change SAPIEN GPU state-buffer indexing such as `gpu_index` or
`gpu_pose_index`.

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
