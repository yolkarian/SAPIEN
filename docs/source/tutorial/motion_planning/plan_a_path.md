(plan_a_path)=

# Plan a Path

```{eval-rst}
.. highlight:: python
```

This page uses the `mplib` 0.2.1 API configured in {ref}`motion_planning_getting_started`.

## Plan to an end-effector pose

Use `Planner.plan_pose` for sampling-based pose planning. The goal is an
`mplib.Pose`; it is in the world frame unless `wrt_world=False` is passed.

```python
goal_pose = mplib.Pose(
   [0.4, 0.0, 0.4],
   [1.0, 0.0, 0.0, 0.0],
)
current_qpos = robot.qpos.copy()

result = planner.plan_pose(
   goal_pose,
   current_qpos,
   time_step=scene.timestep,
   planning_time=1.0,
)
```

Use `plan_qpos` instead when the goal is one or more joint configurations.

A successful result contains:

- `status`: exactly `"Success"` on success; otherwise an IK or RRT failure
  message;
- `position`: waypoint positions with shape `(n, move_group_dof)`;
- `velocity`: waypoint velocities with the same shape;
- `acceleration`: waypoint accelerations;
- `time`: sample times;
- `duration`: trajectory duration.

Only read trajectory arrays after checking `result["status"]`.

## Follow the path in SAPIEN

The result columns follow `planner.move_group_joint_indices`. Select the
matching SAPIEN joints instead of assuming that the controlled joints are a
prefix of `robot.active_joints`.

```python
controlled_joints = [
   robot.active_joints[index]
   for index in planner.move_group_joint_indices
]
for joint in controlled_joints:
   joint.set_drive_properties(
      stiffness=1000.0,
      damping=100.0,
      force_limit=1000.0,
   )

if result["status"] == "Success":
   for step_index, (qpos, qvel) in enumerate(
      zip(result["position"], result["velocity"])
   ):
      for joint, position, velocity in zip(controlled_joints, qpos, qvel):
         joint.set_drive_target(float(position))
         joint.set_drive_velocity_target(float(velocity))

      robot.qf = robot.compute_passive_force(
         gravity=True,
         coriolis_and_centrifugal=True,
      )
      scene.step()

      if step_index % 4 == 0:
         viewer.update_render()
         viewer.render()
```

Current Viewer rendering requires `viewer.update_render()` before
`viewer.render()`. Calling only `scene.update_render()` does not submit a new
Viewer frame.

If the robot does not follow a valid path, check drive stiffness, damping,
force limits, timestep, and passive-force compensation before changing planner
settings.

## Plan a screw motion

`Planner.plan_screw` attempts a direct Cartesian screw motion. It is faster and
straighter than sampling-based planning, but it cannot detour around an
obstacle.

```python
result = planner.plan_screw(
   goal_pose,
   robot.qpos.copy(),
   time_step=scene.timestep,
)

if result["status"] != "Success":
   result = planner.plan_pose(
      goal_pose,
      robot.qpos.copy(),
      time_step=scene.timestep,
   )
```

Collision objects previously added to the planning world are considered
automatically; see {ref}`collision_avoidance`.

## Gripper control

Gripper joints usually are not in `planner.move_group_joint_indices`, so control
them separately:

```python
finger_joints = robot.active_joints[-2:]

for joint in finger_joints:
   joint.set_drive_properties(stiffness=500.0, damping=50.0, force_limit=100.0)


def set_gripper(width: float) -> None:
   half_width = width / 2.0
   for joint in finger_joints:
      joint.set_drive_target(half_width)
```
