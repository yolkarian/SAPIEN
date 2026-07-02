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

Batched GPU IK without Pinocchio
--------------------------------

SAPIEN also provides a batched GPU Jacobian IK helper for articulations running
in ``PhysxGpuSystem``. It uses PhysX dense articulation Jacobians and PyTorch
CUDA tensors, and does not depend on Pinocchio.

.. code-block:: python

   import torch

   # After all articulations are built and physx_system.gpu_init() has run:
   robots = [robot0, robot1, robot2]
   ee_links = [robot.find_link_by_name("tool") for robot in robots]

   solver = sapien.physx.GpuInverseKinematicsSolver(
      physx_system,
      robots,
      ee_links,
   )

   # shape: (batch, 7), ordered as xyz + quaternion wxyz, in each scene frame
   target_poses = torch.tensor(
      [
         [0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
         [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
         [0.6, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
      ],
      device="cuda",
   )

   ik_qpos, success, error = solver.solve(target_poses, max_iterations=100)

``ik_qpos`` is a padded CUDA tensor with shape ``(batch, max_dof)`` in the same
joint order as SAPIEN active joints. By default, the solver applies the result
back to ``physx_system.cuda_articulation_qpos``, calls
``gpu_apply_articulation_qpos()``, updates articulation kinematics, and fetches
link poses. Use ``rotation_weight=0.0`` for position-only IK, or pass
``active_qmask`` to the solver constructor to freeze selected DOFs.
