(basic_robot)=

# Getting Started with Robots

```{eval-rst}
.. highlight:: python
```

Robots are represented as PhysX articulations. Most robots are loaded from URDF
files through `scene.create_urdf_loader()`.

In this tutorial, you will learn how to:

- load a robot URDF;
- set root and joint state;
- compute Jacobians and passive-force compensation;
- use GPU articulation buffers for batched simulation.

## Load a robot URDF

```python
import numpy as np
import sapien

scene = sapien.Scene()
scene.set_timestep(1 / 240)
scene.add_ground(0)
scene.set_ambient_light([0.12, 0.12, 0.12])
scene.add_directional_light(
   [1, -1, -1], [2.0, 1.9, 1.8], shadow=True
)

loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot = loader.load("/path/to/robot.urdf", package_dir="/path/to/package/root")
robot.name = "robot"
```

`loader.fix_root_link` is `True` by default. If it is `False`, the root
link is allowed to move. For URDFs that contain single rigid bodies or multiple
root objects, use `loader.load_multiple(...)`.

For kinematics-only or IK-only GPU workflows, geometry loading can be disabled:

```python
loader.load_visuals = False
loader.load_collisions = False
```

`load_visuals=False` avoids creating render shapes and does not require a
render system. `load_collisions=False` skips collision shape creation while
preserving the articulation topology, joints, and inertial data.

The packaged command-line smoke example demonstrates URDF loading:

```shell
python -m sapien.example.load_urdf /path/to/robot.urdf --package /path/to/package/root
```

SAPIEN also supports a custom, non-standard `<sdf>` tag under mesh collisions
to request per-collision SDF cooking parameters:

```xml
<collision>
   <geometry>
      <mesh filename="../mesh/link.obj" scale="1 1 1"/>
   </geometry>
   <sdf resolution="512"/>
</collision>
```

When present on a mesh collision, the loader routes that collision through the
non-convex mesh path and passes the parsed `PhysxSDFConfig` to the builder.

## Set robot state

The articulation stores a root pose and generalized coordinates. Quaternion
order is `wxyz`.

```python
robot.set_root_pose(sapien.Pose([0, 0, 0]))

qpos = np.zeros(robot.dof, dtype=np.float32)
qvel = np.zeros(robot.dof, dtype=np.float32)
robot.set_qpos(qpos)
robot.set_qvel(qvel)

print([joint.name for joint in robot.active_joints])
print(robot.qlimits)
```

For URDF-loaded robots, joint order follows the loaded articulation order. Use
joint names when possible instead of hard-coded indices.

Positive URDF `<limit velocity="...">` values are applied to PhysX as per-axis
maximum joint velocities. Revolute values use radians per second; prismatic
values use linear scene units per second and follow `loader.scale`. Non-positive
values are treated as unspecified because many URDFs use zero as a placeholder;
such joints retain the PhysX default limit. To intentionally use a zero limit,
call `joint.set_max_joint_velocity(0.0)` after loading.

```python
hip = robot.find_joint_by_name("hip")
print(hip.max_joint_velocity)
hip.set_max_joint_velocity(20.0)  # scalar applies to every DOF of this joint
```

On CPU, change joint limits only while simulation is not running. In GPU
workflows, set them before `gpu_init()`. PhysX 5.6.1 does not provide a maximum joint-acceleration
constraint API. `robot.qacc` reports or sets articulation state; it is not an
acceleration limit. Enforce acceleration bounds in the controller when needed.

## Compute dense Jacobians

`robot.compute_dense_jacobian()` computes a world-space dense Jacobian for CPU
articulations. The row order is `[vx, vy, vz, wx, wy, wz]` for each link. Use
`robot.get_jacobian_shape()` to slice the valid region.

```python
jacobian = robot.compute_dense_jacobian()
rows, cols = robot.get_jacobian_shape()
jacobian = jacobian[:rows, :cols]
```

For a fixed-base robot, the shape is `((link_count - 1) * 6, dof)`. For a
floating-base robot, it is `(6 + (link_count - 1) * 6, 6 + dof)`; the first
six columns correspond to the root link's linear and angular velocity in world
coordinates.

## Compensate passive forces

Use `compute_passive_force` to compute generalized forces that compensate
gravity and optionally Coriolis/centrifugal terms.

```python
for _ in range(240):
   qf = robot.compute_passive_force(
      gravity=True,
      coriolis_and_centrifugal=True,
   )
   robot.set_qf(qf)
   scene.step()
```

`qf` has length `robot.dof` and follows the same order as
`robot.active_joints`.

## Drive joints

PhysX drives live on active joints. The current API sets targets per joint.

```python
for joint in robot.active_joints:
   joint.set_drive_property(stiffness=1000, damping=100, force_limit=1000)
   joint.set_drive_target(0.0)
   joint.set_drive_velocity_target(0.0)
```

There is no articulation-level `robot.set_drive_target` helper in the current
API.

## GPU articulation workflow

For batched robotics, create a shared `PhysxGpuSystem` after enabling GPU
PhysX. Configure global PhysX before system creation, then initialize GPU after
all bodies are added.

```python
sapien.physx.enable_gpu()

config = sapien.physx.PhysxSceneConfig()
config.gpu_broadphase_env_id_bits = 8
sapien.physx.set_scene_config(config)

device = sapien.Device("cuda")
physx_system = sapien.physx.PhysxGpuSystem(device)
render_system = sapien.render.RenderSystem(device)
scene = sapien.Scene([physx_system, render_system])
scene.set_environment_id(0)

# load/build robots here, then:
physx_system.gpu_init()
```

Link poses and velocities are stored in
`physx_system.cuda_articulation_link_data` with shape
`(articulation_count, max_links, 13)`. Rows are indexed by
`robot.gpu_index` and low-level `link.index`. Channels `0:3` are world
position, `3:7` are quaternion `wxyz`, `7:10` are linear velocity, and
`10:13` are angular velocity.

GPU Jacobians are computed with
`physx_system.gpu_compute_articulation_jacobian()` and stored in the padded
`physx_system.cuda_articulation_jacobian` tensor. Use `robot.gpu_index` and
`physx_system.cuda_articulation_jacobian_shape` or
`robot.get_jacobian_shape()` to select the valid submatrix. PhysX reports the
linear Jacobian component at the link center of mass; shift it in application
code if your task frame is the link origin.

Indexed GPU APIs accept `sapien.CudaArray` and CUDA-array-interface objects as
1D contiguous CUDA `int32` arrays containing SAPIEN `articulation.gpu_index`
values. Keep the owner of an external index tensor alive until the SAPIEN CUDA
stream has finished using it.

For passive-force compensation in GPU simulation, avoid calling
`robot.compute_passive_force()` inside the step loop. Use the GPU buffers:

```python
physx_system.gpu_compute_articulation_gravity_compensation()
physx_system.gpu_compute_articulation_coriolis_and_centrifugal_compensation()

qf = physx_system.cuda_articulation_qf.torch()
gravity = physx_system.cuda_articulation_gravity_compensation.torch()
coriolis = physx_system.cuda_articulation_coriolis_and_centrifugal_compensation.torch()
qf[:] = gravity + coriolis
physx_system.gpu_apply_articulation_qf()
```

Cache CUDA views and GPU indices once after `gpu_init()`. If qpos/qvel were
modified through GPU buffers, apply them and call
`gpu_update_articulation_kinematics()` before computing compensation.
