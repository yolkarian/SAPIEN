.. _plan_a_path:

Plan a Path
==================

.. highlight:: python

This page shows the SAPIEN-side workflow for executing a path produced by
``mplib``. It avoids references to old repository example files that are no
longer shipped in the current tree.

Plan with sampling-based algorithms
--------------------------------------

``mplib`` planners accept an end-effector target pose and the current active
joint positions. A target pose is commonly represented as
``[x, y, z, qw, qx, qy, qz]`` in the robot root frame.

.. code-block:: python

   target_pose = [0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
   current_qpos = robot.qpos.copy()

   result = planner.plan(
      target_pose,
      current_qpos,
      time_step=scene.timestep,
      planning_time=1.0,
   )

Typical result dictionaries include:

* ``status``: e.g. ``"Success"``, ``"IK Failed"``, or ``"RRT Failed"``;
* ``position``: waypoint joint positions, shape ``(n, m)``;
* ``velocity``: waypoint joint velocities, shape ``(n, m)``;
* ``acceleration``: waypoint joint accelerations, shape ``(n, m)``;
* ``time`` and ``duration``.

Consult the installed ``mplib`` version for the exact set of supported keyword
arguments.

Follow a path in SAPIEN
--------------------------------------

Configure PhysX drives on the active joints that the planner controls. The
current SAPIEN API sets targets per joint.

.. code-block:: python

   if result["status"] == "Success":
      controlled_joints = robot.active_joints[: result["position"].shape[1]]
      for joint in controlled_joints:
         joint.set_drive_property(stiffness=1000, damping=100, force_limit=1000)

      for qpos, qvel in zip(result["position"], result["velocity"]):
         for joint, p, v in zip(controlled_joints, qpos, qvel):
            joint.set_drive_target(float(p))
            joint.set_drive_velocity_target(float(v))

         qf = robot.compute_passive_force(True, True)
         robot.set_qf(qf)
         scene.step()
         scene.update_render()

If the robot does not follow the path, first check the drive stiffness, damping,
force limits, timestep, and whether passive forces are compensated.

Plan with screw motion
--------------------------------------

Some ``mplib`` versions provide ``plan_screw`` for direct Cartesian motion of the
``move_group`` link. It can be faster than sampling-based planning but usually
fails if the straight motion is in collision.

.. code-block:: python

   result = planner.plan_screw(
      target_pose,
      robot.qpos.copy(),
      time_step=scene.timestep,
      use_point_cloud=True,
      use_attach=True,
   )

   if result["status"] != "Success":
      result = planner.plan(target_pose, robot.qpos.copy(), time_step=scene.timestep)

Gripper control
--------------------------------------

For grippers with active finger joints, set their drive targets just like arm
joints.

.. code-block:: python

   finger_joints = robot.active_joints[-2:]

   def set_gripper(width):
      half_width = width / 2
      for joint in finger_joints:
         joint.set_drive_property(stiffness=500, damping=50, force_limit=100)
         joint.set_drive_target(half_width)

Collision-aware planning is covered in :ref:`collision_avoidance`.
