(collision_avoidance)=

# Collision Avoidance

`mplib` plans against its own `PlanningWorld`. Add or update environment and
attached collision objects there before calling `plan_pose`, `plan_qpos`, or
`plan_screw`; current mplib considers those objects automatically.

## Add an environment point cloud

Point clouds may come from a SAPIEN camera, a depth sensor, or sampled mesh
surfaces. Coordinates must be in the mplib world frame. If the planner base was
moved with `planner.set_base_pose(...)`, world and robot-base coordinates are
not interchangeable.

```python
points_world = np.asarray(points_world, dtype=np.float32)
planner.update_point_cloud(
   points_world,
   resolution=1e-3,
   name="scene_pcd",
)
```

Updating a point cloud with the same name replaces it. Remove it when it is no
longer part of the environment:

```python
planner.remove_point_cloud("scene_pcd")
```

If the point cloud comes from a sensor, remove points belonging to the robot
before updating the planning world; otherwise every start state may appear to
be in collision.

## Attach a carried object

When the robot grasps an object, attach matching collision geometry to the
planner. `size` is the full box side length, and `pose` is relative to the
attached link. The default `link_id=-1` selects `move_group`.

```python
box_size = [0.06, 0.06, 0.06]
box_pose = mplib.Pose(
   [0.0, 0.0, 0.05],
   [1.0, 0.0, 0.0, 0.0],
)
names_before = set(planner.planning_world.get_object_names())
planner.update_attached_box(box_size, box_pose, link_id=-1)
attached_name = (
   set(planner.planning_world.get_object_names()) - names_before
).pop()
```

Current mplib also provides `update_attached_sphere`, `update_attached_mesh`,
and `update_attached_object`. The convenience shape methods generate an object
name internally, which is why the example records the new name. After releasing
the object, detach and remove it from the planning world:

```python
planner.detach_object(attached_name, also_remove=True)
```

## Plan with the collision world

No per-call collision flags are needed:

```python
goal_pose = mplib.Pose(
   [0.4, 0.0, 0.4],
   [1.0, 0.0, 0.0, 0.0],
)
result = planner.plan_pose(
   goal_pose,
   robot.qpos.copy(),
   time_step=scene.timestep,
)
```

Keep the planning world synchronized by updating named point clouds and
attached objects whenever the simulated environment or grasp changes.
