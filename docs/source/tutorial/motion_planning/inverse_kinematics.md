(inverse_kinematics)=

# Inverse Kinematics

`mplib.Planner.IK` computes configurations that place `move_group` at a goal
pose. Unlike `plan_pose`, `IK` takes its goal directly in the robot base frame.
Use an explicit `mplib.Pose`:

```python
goal_pose = mplib.Pose(
   [0.4, 0.0, 0.4],
   [1.0, 0.0, 0.0, 0.0],
)
start_qpos = robot.qpos.copy()

status, ik_qpos = planner.IK(
   goal_pose,
   start_qpos,
   n_init_qpos=20,
   threshold=1e-3,
   return_closest=True,
)
```

Important arguments are:

- `goal_pose`: an `mplib.Pose` in the robot base frame;
- `start_qpos`: all active-joint positions in the `user_joint_names` order;
- `mask`: optional Boolean array where `True` disables a joint during IK;
- `n_init_qpos`: number of initial guesses to try;
- `threshold`: pose-error threshold;
- `return_closest`: return one closest solution instead of a list of sampled
  solutions.

With the default `return_closest=False`, the second return value is a list of
solutions or `None`. Set `return_closest=True` when one configuration will be
assigned to SAPIEN:

```python
if status == "Success" and ik_qpos is not None:
   robot.qpos = ik_qpos
```

Directly assigning `qpos` teleports the articulation state and is appropriate
for initialization. During simulation, set drive targets on the corresponding
active joints and step PhysX instead.

## GPU articulation data for custom batched IK

SAPIEN does not provide a high-level batched IK solver. For custom GPU IK,
build articulations in a `PhysxGpuSystem`, call `gpu_init()`, and use PhysX's
link-data and dense-Jacobian buffers.

`cuda_articulation_link_data` has shape
`(articulation_count, max_links, 13)`:

```text
row index: articulation.gpu_index
link index: low-level link.index
channels 0:3    world position xyz
channels 3:7    world quaternion wxyz
channels 7:10   world linear velocity
channels 10:13  world angular velocity
```

Call `gpu_compute_articulation_jacobian()` before reading
`cuda_articulation_jacobian`. Slice its padded rows with
`cuda_articulation_jacobian_shape` or `articulation.get_jacobian_shape()`.
Jacobian rows are ordered `[vx, vy, vz, wx, wy, wz]` per link.

For fixed-base articulations, the valid shape is
`((link_count - 1) * 6, dof)`. For floating-base articulations it is
`(6 + (link_count - 1) * 6, 6 + dof)`; the first six columns are root linear
and angular velocity. PhysX reports each linear Jacobian at the link center of
mass, so shift it in application code when the task frame is the link origin.
