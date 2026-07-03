(collision_avoidance)=

# Collision Avoidance

Motion planners need an environment model to avoid collisions with objects that
are not part of the robot. With `mplib`, two common models are environment
point clouds and an attached collision object.

## Add environment point clouds

Point clouds may come from SAPIEN cameras, depth sensors, or sampled mesh
surfaces. Coordinates should be expressed in the robot root frame expected by the
planner.

```python
# Example: point cloud from a SAPIEN camera, already transformed as needed.
point_cloud = np.asarray(points_in_robot_root, dtype=np.float32)
planner.update_point_cloud(point_cloud, resolution=1e-3)
```

Enable the point-cloud collision model when planning:

```python
result = planner.plan(
   target_pose,
   robot.qpos.copy(),
   use_point_cloud=True,
)
```

If the point cloud comes from a sensor observation, remove robot points before
calling `update_point_cloud`; otherwise the planner may see the robot as
already colliding.

## Attach an object to the robot

When the robot grasps an object, add an attached collision shape so the planner
keeps the carried object away from the environment. The pose is relative to the
attached link, usually the `move_group` link.

```python
box_size = [0.06, 0.06, 0.06]
box_pose = [0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]  # xyz + wxyz
planner.update_attached_box(box_size, box_pose, link_id=-1)
```

Then plan with both point-cloud and attached-object collision enabled:

```python
result = planner.plan(
   target_pose,
   robot.qpos.copy(),
   use_point_cloud=True,
   use_attach=True,
)
```

Update the point cloud and attached object whenever the environment or grasped
object changes. Check your installed `mplib` version for exact argument names
and supported attached-shape types.
