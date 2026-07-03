(motion_planning_getting_started)=

# Getting Started

```{eval-rst}
.. highlight:: python
```

SAPIEN does not include a built-in motion planner. The tutorials use
[mplib](https://pypi.org/project/mplib/), a standalone Python package for
sampling-based planning, inverse kinematics, and simple environment collision
models.

Install `mplib` separately:

```bash
python -m pip install mplib
```

Check the `mplib` package metadata for its current Python and platform support.

## Load the robot in SAPIEN

```python
import numpy as np
import sapien

scene = sapien.Scene()
loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot = loader.load("/path/to/panda.urdf", package_dir="/path/to/package")
```

## Set up an mplib planner

`mplib` needs the same URDF/SRDF files and the link/joint names used by
SAPIEN. Use active joints for `user_joint_names`.

```python
import mplib

link_names = [link.name for link in robot.links]
joint_names = [joint.name for joint in robot.active_joints]

planner = mplib.Planner(
   urdf="/path/to/panda.urdf",
   srdf="/path/to/panda.srdf",
   user_link_names=link_names,
   user_joint_names=joint_names,
   move_group="panda_hand",
   joint_vel_limits=np.ones(len(joint_names)),
   joint_acc_limits=np.ones(len(joint_names)),
)
```

The exact `mplib.Planner` constructor may vary across `mplib` releases; keep
the SAPIEN side synchronized by deriving names from the loaded articulation.

## Configuration notes

- URDF describes the robot kinematics and geometry.
- SRDF complements URDF with planning-specific information, especially disabled
  self-collision pairs.
- `user_link_names` and `user_joint_names` align the planner's vectors with
  SAPIEN's articulation order.
- `move_group` is the target end-effector link.
- `joint_vel_limits` and `joint_acc_limits` constrain path parameterization.

After configuration, use {ref}`plan_a_path` for path planning,
{ref}`inverse_kinematics` for IK, and {ref}`collision_avoidance` for point-cloud
and attached-object collision models.
