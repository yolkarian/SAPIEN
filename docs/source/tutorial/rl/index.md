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
