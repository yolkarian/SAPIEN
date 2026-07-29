(manipulation_index)=

# Basic Manipulation

```{eval-rst}
.. highlight:: python
```

:::{note}
Please complete {ref}`basic_index`, {ref}`basic_robot`, and {ref}`gym` before
building manipulation environments.
:::

A manipulation environment is usually a SAPIEN scene containing a robot
articulation, objects, task-specific observations, rewards, and reset logic.
The old repository-level `examples/rl/lift.py` file is not shipped in the
current tree, so this page shows the current API patterns directly.

## Scene setup

```python
import numpy as np
import sapien

scene = sapien.Scene()
scene.set_timestep(1 / 100)
scene.add_ground(0)

# Table
builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[0.5, 0.5, 0.05])
builder.add_box_visual(half_size=[0.5, 0.5, 0.05], material=[0.6, 0.4, 0.2])
table = builder.build_static(name="table")
table.set_pose(sapien.Pose([0.5, 0, 0.5]))

# Cube to lift
builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[0.03, 0.03, 0.03])
builder.add_box_visual(half_size=[0.03, 0.03, 0.03], material=[0.1, 0.5, 1.0])
cube = builder.build(name="cube")
cube.set_pose(sapien.Pose([0.5, 0, 0.58]))
cube_body = cube.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)

# Robot
loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot = loader.load("/path/to/panda.urdf", package_dir="/path/to/package")
```

## Control

For position or velocity control, configure PhysX drives on the robot's active
joints. Set targets per joint in the current API.

```python
arm_joints = robot.active_joints[:7]
finger_joints = robot.active_joints[7:]

for joint in arm_joints:
   joint.set_drive_properties(stiffness=0.0, damping=200.0, force_limit=100.0)

def apply_action(action):
   target_vel = action[: len(arm_joints)]
   finger_force = action[len(arm_joints):]

   for joint, velocity in zip(arm_joints, target_vel):
      joint.set_drive_velocity_target(float(velocity))

   qf = np.zeros(robot.dof, dtype=np.float32)
   qf[-len(finger_joints):] = finger_force
   robot.set_qf(qf)
```

## Task definition

A simple lift task can use observations from robot state, object pose, object
velocity, and task-relative vectors.

```python
def get_obs():
   ee_link = robot.links[-1]
   ee_pose = ee_link.entity.pose
   return np.concatenate([
      robot.qpos,
      robot.qvel,
      cube.pose.p,
      cube.pose.q,
      cube_body.linear_velocity,
      ee_pose.p,
      cube.pose.p - ee_pose.p,
   ])

def compute_reward():
   cube_height = cube.pose.p[2]
   lift_reward = max(cube_height - 0.58, 0.0)
   return float(lift_reward)
```

## Reset

Store initial state after building the world. During reset, restore SAPIEN poses
and PhysX state, then randomize task variables as needed.

```python
initial_pose_state = scene.pack_poses()
initial_physx_state = scene.physx_system.pack()

def reset():
   scene.unpack_poses(initial_pose_state)
   scene.physx_system.unpack(initial_physx_state)
   cube.set_pose(sapien.Pose([0.5, np.random.uniform(-0.05, 0.05), 0.58]))
   return get_obs()
```

For GPU PhysX manipulation, use the GPU reset and step workflows: write state to
`cuda_*` buffers, call only the matching `gpu_apply_*` methods, fetch the
state required by observations, and avoid CPU pose synchronization except for
explicit CPU debugging or the Viewer `cpu-debug` transport. Normal Viewer
rendering should use direct or staged GPU pose transport.
