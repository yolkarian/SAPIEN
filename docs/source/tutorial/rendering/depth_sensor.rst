.. _depth_sensor:

SAPIEN Realistic Depth
======================

.. highlight:: python

SAPIEN packages a CUDA ``simsense`` backend and exposes it through
``sapien.sensor.StereoDepthSensor``. The sensor simulates an active stereo depth
camera with one RGB camera, two infrared cameras, a textured active light, and a
semi-global matching pipeline.

.. note::

   ``StereoDepthSensor`` depends on the packaged ``simsense`` CUDA backend. It is
   separate from the rest of SAPIEN's CUDA/PhysX functionality and may have a
   narrower NVIDIA compute-capability support window than PhysX GPU simulation.

Create a stereo depth sensor
----------------------------

The current constructor takes a ``StereoDepthSensorConfig`` and a mount entity.
The old ``StereoDepthSensor(name, scene, config, ...)`` signature is obsolete.
The mount entity must be in the scene because the sensor adds camera, light, and
simsense components to it.

.. code-block:: python

   import sapien
   from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig

   # Optional: configure ray-traced camera rendering before cameras are created.
   sapien.render.set_camera_shader_dir("rt")
   sapien.render.set_ray_tracing_samples_per_pixel(8)

   scene = sapien.Scene()
   scene.set_ambient_light([0.1, 0.1, 0.1])
   scene.add_directional_light([0, 1, -1], [1, 1, 1])

   mount = sapien.Entity()
   mount.name = "sensor_mount"
   mount.set_pose(sapien.Pose([-1, 0, 1]))
   scene.add_entity(mount)

   config = StereoDepthSensorConfig(model="D435")
   sensor = StereoDepthSensor(config, mount, pose=sapien.Pose())

``sensor.set_local_pose(pose)`` changes the sensor pose relative to the mount.
To move the whole sensor in the world, move the mount entity.

Capture RGB, IR, depth, and point clouds
----------------------------------------

Call ``scene.update_render()`` before ``sensor.take_picture()``. The sensor
captures RGB and IR images, then ``compute_depth`` runs stereo matching.

.. code-block:: python

   scene.update_render()
   sensor.take_picture()
   sensor.compute_depth()

   rgb = sensor.get_rgb()          # H x W x 3 float array in RGB camera frame
   ir_left, ir_right = sensor.get_ir()
   depth = sensor.get_depth()      # H x W float array, same frame as RGB
   points = sensor.get_pointcloud(with_rgb=True)

The returned point cloud uses the RGB camera frame with x right, y down, z
forward. ``with_rgb=True`` appends RGB values to each point.

CUDA output
-----------

Use CUDA getters to keep data on the GPU. They return ``sapien.CudaArray``
objects, not old DLPack-only handles.

.. code-block:: python

   depth_cuda = sensor.get_depth_cuda()
   pointcloud_cuda = sensor.get_pointcloud_cuda(with_rgb=True)

   depth_torch = depth_cuda.torch()
   pointcloud_dlpack = pointcloud_cuda.dlpack()

``CudaArray`` also supports ``cupy()`` and ``jax()`` when those libraries are
available.

Configuration
-------------

``StereoDepthSensorConfig`` supports built-in ``"D415"`` and ``"D435"`` models.
Important fields include:

* ``rgb_resolution`` and ``ir_resolution``;
* ``rgb_intrinsic`` and ``ir_intrinsic``;
* ``trans_pose_l`` and ``trans_pose_r``;
* ``min_depth`` and ``max_depth``;
* ``max_disp``;
* ``census_width`` / ``census_height``;
* ``block_width`` / ``block_height``;
* ``ir_speckle_noise`` and ``ir_thermal_noise``;
* ``depth_dilation``.

Lowering ``ir_resolution`` or ``max_disp`` improves speed at the cost of output
quality. If RGB is not needed for a capture, skip the RGB render pass:

.. code-block:: python

   scene.update_render()
   sensor.take_picture(infrared_only=True)
   sensor.compute_depth()

Reference
---------

[1] Zhang, X., Chen, R., Li, A., Xiang, F., Qin, Y., Gu, J., Ling, Z., Liu, M.,
Zeng, P., Han, S., Huang, Z., Mu, T., Xu, J., & Su, H. (2023). Close the Optical
Sensing Domain Gap by Physics-Grounded Active Stereo Sensor Simulation. IEEE
Transactions on Robotics. https://doi.org/10.1109/TRO.2023.3235591
