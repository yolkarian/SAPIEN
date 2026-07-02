.. _pid:

Drive Robots with PD/PID Controllers
====================================

.. highlight:: python

.. note::

   Please complete :ref:`basic_robot` before continuing.

A common robotics task is to drive joints to target positions or velocities.
SAPIEN exposes PhysX's implicit PD drives on each active joint, and you can also
write external controllers that set generalized forces with ``robot.set_qf``.

PhysX internal PD drive
------------------------------------------------------

Configure each active joint's drive properties and targets.

.. code-block:: python

   for joint in robot.active_joints:
      joint.set_drive_property(
         stiffness=1000.0,
         damping=100.0,
         force_limit=1000.0,
         mode="force",  # or "acceleration"
      )
      joint.set_drive_target(0.0)
      joint.set_drive_velocity_target(0.0)

   target_qpos = [0.2] * robot.dof
   for joint, target in zip(robot.active_joints, target_qpos):
      joint.set_drive_target(target)

The drive behaves like a proportional-derivative controller:

.. centered:: *force = stiffness * (targetPosition - position) + damping * (targetVelocity - velocity)*

PhysX solves the drive implicitly during simulation, so it is usually more
stable than a naive explicit controller. The current API does not provide an
articulation-level ``robot.set_drive_target``; set targets on joints or use the
GPU target buffers.

Compensate passive forces
------------------------------------------------------

A PD drive may have steady-state error under gravity. Add passive-force
compensation when needed.

.. code-block:: python

   for _ in range(240):
      qf = robot.compute_passive_force(
         gravity=True,
         coriolis_and_centrifugal=True,
      )
      robot.set_qf(qf)
      scene.step()

For GPU simulation, use
``physx_system.gpu_compute_articulation_gravity_compensation()`` and
``physx_system.gpu_compute_articulation_coriolis_and_centrifugal_compensation()``
instead of CPU ``compute_passive_force``.

External PID controller
-----------------------

If you need an integral term, compute generalized forces yourself and assign
``qf``. Clamp forces to reasonable limits to avoid unstable simulation.

.. code-block:: python

   class PID:
      def __init__(self, kp, ki, kd, force_limit):
         self.kp = kp
         self.ki = ki
         self.kd = kd
         self.force_limit = force_limit
         self.integral = None

      def reset(self, dof):
         self.integral = np.zeros(dof, dtype=np.float32)

      def compute(self, target_qpos, qpos, qvel, dt):
         if self.integral is None:
            self.reset(len(qpos))

         error = target_qpos - qpos
         self.integral += error * dt
         qf = self.kp * error + self.ki * self.integral - self.kd * qvel
         return np.clip(qf, -self.force_limit, self.force_limit)

   controller = PID(kp=200.0, ki=10.0, kd=20.0, force_limit=100.0)
   target_qpos = np.zeros(robot.dof, dtype=np.float32)

   for _ in range(240):
      qf = controller.compute(target_qpos, robot.qpos, robot.qvel, scene.timestep)
      robot.set_qf(qf)
      scene.step()

In most cases, prefer PhysX's internal drive. Use an external PID only when you
need behavior that the built-in drive cannot express.
