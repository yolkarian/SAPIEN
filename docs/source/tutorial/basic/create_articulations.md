(create_articulations)=

# Create Articulations

```{eval-rst}
.. highlight:: python
```

An articulation is a tree of rigid links connected by joints. Robots loaded from
URDF are articulations, and you can also build small articulations directly in
Python.

In this tutorial, you will learn how to:

- create links with `ArticulationBuilder` and `LinkBuilder`;
- configure revolute, prismatic, and fixed joints;
- drive joints with PhysX drives;
- read articulation state, Jacobians, and GPU buffers.

:::{figure} assets/create_articulations.gif
:align: center
:figclass: align-center
:width: 640px
:::

## Create a root link

The first link builder is the root link. Link builders inherit the same shape
helpers as `ActorBuilder`.

```python
import numpy as np
import sapien

scene = sapien.Scene()
builder = scene.create_articulation_builder()

root = builder.create_link_builder()
root.set_name("base")
root.add_box_collision(half_size=[0.2, 0.1, 0.05])
root.add_box_visual(half_size=[0.2, 0.1, 0.05], material=[0.8, 0.2, 0.2])
```

## Create a child link and joint

`create_link_builder(parent)` creates a child of an existing link. The joint's
motion axis is the x-axis of the joint frame. `pose_in_parent` and
`pose_in_child` place that joint frame in the two link frames.

```python
arm = builder.create_link_builder(root)
arm.set_name("arm")
arm.set_joint_name("hinge")
arm.add_capsule_collision(
   pose=sapien.Pose(q=[0.7071068, 0, 0.7071068, 0]),
   radius=0.04,
   half_length=0.35,
)
arm.add_capsule_visual(
   pose=sapien.Pose(q=[0.7071068, 0, 0.7071068, 0]),
   radius=0.04,
   half_length=0.35,
   material=[0.2, 0.2, 0.8],
)
arm.set_joint_properties(
   "revolute",
   limits=[[-np.pi / 2, np.pi / 2]],
   pose_in_parent=sapien.Pose([0.2, 0, 0]),
   pose_in_child=sapien.Pose([-0.35, 0, 0]),
   friction=0.0,
   damping=0.1,
   velocity_limit=2.0,
)

articulation = builder.build(fix_root_link=True)
articulation.name = "single_hinge"
```

Supported joint types include `"fixed"`, `"revolute"`,
`"revolute_unwrapped"`, `"continuous"`, and `"prismatic"`. The root joint
is fixed when `fix_root_link=True`. `velocity_limit` configures PhysX's
per-axis maximum joint velocity; omit it to retain the PhysX default.

## Control an articulation with drives

Active joints are available as `articulation.active_joints` or
`articulation.get_active_joints()`. Set drive properties and targets on each
joint.

```python
for joint in articulation.active_joints:
   joint.set_drive_property(stiffness=50.0, damping=5.0, force_limit=100.0)
   joint.set_drive_target(0.3)
   joint.set_drive_velocity_target(0.0)

for _ in range(240):
   scene.step()
```

The drive implements a PhysX PD controller. There is no articulation-level
`set_drive_target` helper in the current API; set targets on the active joints
or use the GPU target buffers in `PhysxGpuSystem`.

## Read articulation state

The articulation stores root pose, generalized positions, velocities,
accelerations, and forces.

```python
print(articulation.root_pose)
print(articulation.qpos)
print(articulation.qvel)
print(articulation.qlimits)

articulation.set_qpos([0.1] * articulation.dof)
articulation.set_qvel([0.0] * articulation.dof)
articulation.set_qf([0.0] * articulation.dof)
```

## CPU Jacobians and passive forces

CPU articulations provide `compute_dense_jacobian()`. The returned matrix maps
generalized velocities to stacked link spatial velocities in row order
`[vx, vy, vz, wx, wy, wz]` for each link. `get_jacobian_shape()` returns the
valid matrix shape.

For a fixed-base articulation with `link_count` links and `dof` joint
degrees of freedom, the Jacobian shape is `((link_count - 1) * 6, dof)`. For a
floating-base articulation, the shape is `(6 + (link_count - 1) * 6, 6 + dof)`;
the first six columns correspond to the root link's linear and angular velocity
in world coordinates.

```python
jacobian = articulation.compute_dense_jacobian()
rows, cols = articulation.get_jacobian_shape()
jacobian = jacobian[:rows, :cols]

qf = articulation.compute_passive_force(
   gravity=True,
   coriolis_and_centrifugal=True,
)
articulation.set_qf(qf)
```

## GPU articulation buffers

With `sapien.physx.PhysxGpuSystem`, initialize GPU simulation after all bodies
are added, then use `cuda_*` buffers and `gpu_apply_*` / `gpu_fetch_*`
methods for batched control.

```python
sapien.physx.enable_gpu()
physx_system = sapien.physx.PhysxGpuSystem()
scene = sapien.Scene([physx_system, sapien.render.RenderSystem("cuda")])
# build actors/articulations here
physx_system.gpu_init()
```

For GPU Jacobians, call `scene.physx_system.gpu_compute_articulation_jacobian()`
to update all articulations, or pass a CUDA int32 array of articulation
`gpu_index` values to update a subset. The padded tensor
`scene.physx_system.cuda_articulation_jacobian` stores buffers for all
articulations. Use `articulation.gpu_index` to select a row and
`articulation.get_jacobian_shape()` to slice the valid submatrix.

GPU simulation also provides joint-only passive-force compensation buffers:
`cuda_articulation_gravity_compensation` and
`cuda_articulation_coriolis_and_centrifugal_compensation`. Cache their
`.torch()`/`.cupy()`/`.jax()` views once after `gpu_init()` instead of
recreating them in the control loop.

```python
system = scene.physx_system
system.gpu_compute_articulation_gravity_compensation()
system.gpu_compute_articulation_coriolis_and_centrifugal_compensation()

qf = system.cuda_articulation_qf.torch()
gravity = system.cuda_articulation_gravity_compensation.torch()
coriolis = system.cuda_articulation_coriolis_and_centrifugal_compensation.torch()
qf[:] = gravity + coriolis
system.gpu_apply_articulation_qf()
```

Direct GPU simulation can also apply world-space forces and torques to links via
`cuda_articulation_link_force` and `cuda_articulation_link_torque`. These
buffers have shape `(articulation_count, max_links, 4)`; the first three
channels are the vector and the fourth channel is padding. After writing them,
call `gpu_apply_articulation_link_force()` and/or
`gpu_apply_articulation_link_torque()`.

## Remove an articulation

`scene.remove_articulation(articulation)` removes all link entities from the
scene. Do not use the articulation, its links, or its joints after removal.
