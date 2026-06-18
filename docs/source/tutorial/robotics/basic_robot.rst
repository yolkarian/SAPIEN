.. _basic_robot:

Getting Started with Robot
===========================

.. highlight:: python

.. note::
   Please first complete :ref:`basic_index` before continuing this tutorial.
   The assets (robot) used in this tutorial can be found `here <https://github.com/haosulab/SAPIEN-Release/tree/master/examples/assets>`__.

In this tutorial, you will learn the following:

* Load a robot (URDF)
* Set joint positions
* Compute dense Jacobians
* Compensate passive forces
* Control the robot by torques

The full script can be downloaded here :download:`basic_robot.py <../../../../examples/robotics/basic_robot.py>`

Set up the engine, renderer and scene
-----------------------------------------

First of all, let's set up the simulation environment as illustrated in :ref:`hello_world`.

.. literalinclude:: ../../../../examples/robotics/basic_robot.py
   :dedent: 0
   :lines: 5-22

Load a robot URDF
-----------------------------------------

Now, you can create a ``URDFLoader`` to load the URDF XML of Kinova Jaco2 arm.
`URDF XML <http://wiki.ros.org/urdf/XML>`_ describes a robot.
Usually, URDF files are provided by manufacturers.
For example, the URDF XML of Kinova Jaco2 arm can be found `here <https://github.com/Kinovarobotics/kinova-ros>`__.

.. literalinclude:: ../../../../examples/robotics/basic_robot.py
   :dedent: 0
   :lines: 25-28

Note that there is a ``fix_root_link`` flag for the URDF loader.
If it is true (by default), then the root link of the robot will be fixed.
Otherwise, it is allowed to move freely.

SAPIEN also supports a custom ``<sdf>`` tag under mesh collisions to request
per-collision SDF cooking parameters. For example,

.. code-block:: xml

   <collision>
      <geometry>
         <mesh filename="../mesh/square_table_leg.obj" scale="1 1 1"/>
      </geometry>
      <sdf resolution="512"/>
   </collision>

This tag is SAPIEN-specific rather than standard URDF. When present on a mesh
collision, the loader will route that collision through the non-convex mesh
path automatically and apply the provided SDF config for that mesh only when
the collision is built with an SDF-backed triangle mesh.

The robot is loaded as ``Articulation``, which is a tree of links connected by joints.
We can set the pose of its root link through ``set_root_pose(...)``.

If you run the example with ``demo(fix_root_link=False, balance_passive_force=False)``, it is expected that you will observe the following "falling-down" robot arm.
We will see how to keep the robot at a certain pose later.

.. figure:: assets/robot_fall.gif
    :width: 640px
    :align: center
    :figclass: align-center

    The robot arm falls down.

.. note::
   When a robot is already loaded, changing the flag of the URDF loader will not take effect.

Set joint positions
--------------------------------------

.. literalinclude:: ../../../../examples/robotics/basic_robot.py
   :dedent: 0
   :lines: 31-34

We can also set initial joint positions through ``set_qpos(qpos=...)``.
The ``qpos`` should be a concatenation of the position of each joint.
Its length is the degree of freedom, and its order is the same as that returned by ``robot.get_joints()``.

.. note::
   If the articulation is loaded from a URDF file, its joints are in preorder (DFS preorder traversal over the articulation tree).
   If the articulation is built programmatically (refer to :ref:`create_articulations`), its joints are in the order when they are built.

Compute dense Jacobians
--------------------------------------

For robotics applications, it is often useful to map generalized velocities to the
spatial velocity of each link. SAPIEN provides ``compute_dense_jacobian()`` for CPU
articulations to compute this world-space dense Jacobian directly from PhysX.

The row order is ``[vx, vy, vz, wx, wy, wz]`` for each link. The valid matrix size is
available from ``robot.get_jacobian_shape()``. For a fixed-base robot, the shape is
``((link_count - 1) * 6, dof)``. For a floating-base robot, the shape is
``(6 + (link_count - 1) * 6, 6 + dof)`` and the first six columns correspond to the
root link's linear and angular velocity in the world frame.

For GPU simulation, the corresponding API is ``scene.physx_system.gpu_compute_articulation_jacobian()``.
You can also pass a CUDA int32 array of articulation ``gpu_index`` values to update only a
subset of robots. The result is written to ``scene.physx_system.cuda_articulation_jacobian`` as
a padded tensor of shape ``(articulation_count, max_rows, max_cols)``. Use ``robot.gpu_index``
to select a robot and ``robot.get_jacobian_shape()`` to extract the valid submatrix.

Direct GPU simulation can also apply world-space external forces and torques to individual robot
links. Write ``physx_system.cuda_articulation_link_force`` and/or
``physx_system.cuda_articulation_link_torque`` with shape ``(articulation_count, max_links, 4)``
(the first three channels are the vector and the fourth is padding), then call
``gpu_apply_articulation_link_force()`` and/or ``gpu_apply_articulation_link_torque()``.

Compensate passive forces (e.g. gravity)
-----------------------------------------

You may find that even if you run the example with ``fix_root_link=True``, the robot still can not maintain its initial joint positions.
It is due to gravitational force and other possible passive forces, like Coriolis and Centrifugal force.

.. figure:: assets/robot_fix.gif
    :width: 640px
    :align: center
    :figclass: align-center

    The root link (base) of the robot is fixed, but it still falls down due to passive forces.

For a real robot, gravity compensation is done by an internal controller hardware.
So it is usually desirable to skip this troublesome calculation of how to compensate gravity.
SAPIEN provides ``compute_passive_force`` to compute desired forces or torques on joints to compensate passive forces.
In this example, we only consider gravity as well as coriolis and centrifugal force.

.. literalinclude:: ../../../../examples/robotics/basic_robot.py
   :dedent: 0
   :lines: 36-46

We recompute the compensative torque every step and control the robot by ``set_qf(qf)``.
``qf`` should be a concatenation of the force or torque to apply on each joint.
Its length is the degree of freedom, and its order is the same as that returned by ``robot.get_joints()``.
Note that when ``qf`` is set, it will be applied every simulation step.
You can call ``robot.get_qf()`` to acquire its current value.

In Direct GPU simulation, avoid calling ``robot.compute_passive_force()`` in the step loop.
Instead, call ``physx_system.gpu_compute_articulation_gravity_compensation()`` and
``physx_system.gpu_compute_articulation_coriolis_and_centrifugal_compensation()``. Their output
CUDA arrays are padded to ``(articulation_count, max_dofs)`` and match
``cuda_articulation_qf``. Both methods accept a CUDA int32 array of articulation
``gpu_index`` values to update only a subset. After applying any GPU qpos/qvel updates, call
``gpu_update_articulation_kinematics()`` before computing compensation. In a control loop,
cache the ``.torch()`` views once after ``gpu_init()`` instead of recreating them every step.

.. code-block:: python

   physx_system.gpu_compute_articulation_gravity_compensation()
   physx_system.gpu_compute_articulation_coriolis_and_centrifugal_compensation()

   qf = physx_system.cuda_articulation_qf.torch()
   gravity = physx_system.cuda_articulation_gravity_compensation.torch()
   coriolis = physx_system.cuda_articulation_coriolis_and_centrifugal_compensation.torch()
   qf[:] = gravity + coriolis
   physx_system.gpu_apply_articulation_qf()

Now, if you run the example with ``demo(fix_root_link=True, balance_passive_force=True)``, it is observed that the robot can stay at the target pose for a short period.
However, it will then deviate from this pose gradually due to numerical error.

.. figure:: assets/robot_fix_balance.gif
    :width: 640px
    :align: center
    :figclass: align-center

    The robot arm is able to stay at the target pose, but might deviate gradually due to numerical error.
    The animation is accelerated.

.. note::
   To avoid deviating from the target pose gradually,
   either we specify the damping (resistence proportional to velocity) of each joint in the URDF XML, or a controller can be used to compute desired extra forces or torques to keep the robot around the target pose.
   :ref:`pid` will elaborate how to control the robot with a controller.
