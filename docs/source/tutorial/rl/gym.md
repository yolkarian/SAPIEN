(gym)=

# Build a Gym-style Interface

```{eval-rst}
.. highlight:: python
```

SAPIEN does not depend on Gym or Gymnasium, but it is straightforward to wrap a
`sapien.Scene` in a Gym-style class.

In this tutorial, you will learn how to:

- structure `reset` and `step` around SAPIEN's scene loop;
- save and restore CPU simulation state;
- keep rendering optional for headless training.

## Minimal environment skeleton

```python
import numpy as np
import sapien

class SapienEnv:
   def __init__(self, render: bool = False):
      self.render = render
      systems = [sapien.physx.PhysxCpuSystem()]
      if render:
         systems.append(sapien.render.RenderSystem())

      self.scene = sapien.Scene(systems)
      self.scene.set_timestep(1 / 100)
      self._build_world()

      self._initial_physx_state = self.scene.physx_system.pack()
      self._initial_pose_state = self.scene.pack_poses()

   def _build_world(self):
      self.scene.add_ground(0, render=self.render)
      builder = self.scene.create_actor_builder()
      builder.add_box_collision(half_size=[0.2, 0.2, 0.2])
      if self.render:
         builder.add_box_visual(half_size=[0.2, 0.2, 0.2], material=[1, 0, 0])
      self.box = builder.build(name="box")
      self.box.set_pose(sapien.Pose([0, 0, 0.2]))
      self.box_body = self.box.find_component_by_type(
         sapien.physx.PhysxRigidDynamicComponent
      )

   def reset(self):
      self.scene.unpack_poses(self._initial_pose_state)
      self.scene.physx_system.unpack(self._initial_physx_state)
      return self._get_obs()

   def step(self, action):
      force = np.asarray(action, dtype=np.float32)
      self.box_body.add_force_torque(force, [0, 0, 0])
      self.scene.step()
      obs = self._get_obs()
      reward = float(self.box.pose.p[2])
      terminated = False
      truncated = False
      info = {}
      return obs, reward, terminated, truncated, info

   def _get_obs(self):
      return np.concatenate([self.box.pose.p, self.box_body.linear_velocity])
```

## State save and restore

For CPU PhysX scenes, `scene.physx_system.pack()` and `unpack(...)` serialize
PhysX state. `scene.pack_poses()` and `unpack_poses(...)` serialize SAPIEN
entity poses. Store both if your reset must restore both PhysX and entity-side
state.

For GPU PhysX, state is managed through `PhysxGpuSystem.cuda_*` buffers. Write
reset states into the relevant buffers, call the matching `gpu_apply_*`
methods, then fetch the state needed for observations.

## Rendering during RL

Rendering is optional. For pure training, create scenes with only a PhysX system.
For evaluation videos or debugging, add a `sapien.render.RenderSystem` only to
the scenes you render.

With CPU PhysX, `scene.update_render()` is enough before viewer/camera render.
With GPU PhysX, avoid `sync_poses_gpu_to_cpu()` in training loops. For offscreen
capture, use `sapien.render.RenderSystemGroup` with CUDA pose buffers. For
interactive debugging, configure the Viewer and submit each displayed state
explicitly. `"auto"` uses direct CUDA/Vulkan interop on a compatible same device
and compact pinned-host staging when PhysX and Vulkan use different devices:

```python
viewer.configure_physx_gpu_rendering(physx_system, transport="auto")
viewer.apply_interactions()  # before every physics substep when interaction is enabled
physx_system.step()
viewer.update_render()
viewer.render()
```

## Random rollout

```python
env = SapienEnv(render=False)
obs = env.reset()
for _ in range(1000):
   action = np.random.uniform(-1, 1, size=3)
   obs, reward, terminated, truncated, info = env.step(action)
   if terminated or truncated:
      obs = env.reset()
```
