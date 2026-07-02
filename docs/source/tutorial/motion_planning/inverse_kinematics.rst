.. _inverse_kinematics:

Inverse Kinematics
==================

Inverse kinematics computes joint positions that place the planner's
``move_group`` link at a target pose. In ``mplib`` this is commonly exposed as
``planner.IK(...)`` (some versions may use a lowercase method name; check the
installed package).

.. code-block:: python

   target_pose = [0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]  # xyz + wxyz
   init_qpos = robot.qpos.copy()

   status, ik_qpos = planner.IK(
      target_pose,
      init_qpos,
      n_init_qpos=20,
      threshold=1e-3,
   )

Arguments:

* ``target_pose``: target pose of the ``move_group`` link, usually in the robot
  root frame and ordered as position plus quaternion ``wxyz``.
* ``init_qpos``: current active-joint positions from SAPIEN.
* ``n_init_qpos``: number of initial guesses to try.
* ``threshold``: pose-error threshold for success.

If IK succeeds, apply the solution through PhysX drives or by directly setting
state for initialization:

.. code-block:: python

   if status == "Success":
      robot.set_qpos(ik_qpos)

For controlled motion, prefer setting drive targets on each active joint rather
than teleporting with ``set_qpos`` during simulation.

GPU articulation data for custom batched IK
-------------------------------------------

SAPIEN does not provide a built-in batched IK policy. For custom GPU IK,
construct articulations in ``PhysxGpuSystem`` and use the low-level GPU buffers
and PhysX dense Jacobians.

``cuda_articulation_link_data`` stores link pose and velocity as a padded tensor:

.. code-block:: text

   shape: (articulation_count, max_links, 13)
   row index: articulation.gpu_index
   link index: low-level link.index

   channels 0:3    world position xyz
   channels 3:7    world quaternion wxyz
   channels 7:10   world linear velocity
   channels 10:13  world angular velocity

The root pose is ``cuda_articulation_link_data[:, 0, 0:7]`` and the root
velocity is ``cuda_articulation_link_data[:, 0, 7:13]``.

Dense Jacobians are available through ``cuda_articulation_jacobian`` after
calling ``gpu_compute_articulation_jacobian()``. The valid matrix size for each
articulation is available from ``cuda_articulation_jacobian_shape`` or
``articulation.get_jacobian_shape()``.

.. code-block:: text

   dense_jacobian row order: [vx, vy, vz, wx, wy, wz]

   fixed-base articulation:
      rows = (link_count - 1) * 6
      cols = dof

   floating-base articulation:
      rows = 6 + (link_count - 1) * 6
      cols = 6 + dof

For floating-base articulations, the first six columns correspond to root
linear and angular velocity in world coordinates. PhysX reports the linear
Jacobian component at the link center of mass. If your task frame is the link
frame origin, shift the linear rows in application code before solving IK.
