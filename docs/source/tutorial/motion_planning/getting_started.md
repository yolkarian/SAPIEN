(motion_planning_getting_started)=

# Getting Started

```{eval-rst}
.. highlight:: python
```

SAPIEN does not include a motion planner. These tutorials use
[mplib](https://motion-planning-lib.readthedocs.io/), a standalone package for
sampling-based planning, inverse kinematics, and collision checking.

The examples below target the current stable `mplib` 0.2.1 API. Install it
separately:

```bash
python -m pip install "mplib==0.2.1"
```

`mplib` 0.2.1 requires NumPy `< 2`. Check the
[`mplib.Planner` reference](https://motion-planning-lib.readthedocs.io/stable/reference/Planner.html)
when using another release.

## Load the same robot in SAPIEN and mplib

Load the robot in SAPIEN first so the planner can use exactly the same active
joint and link order:

```python
from pathlib import Path

import mplib
import sapien

urdf_path = Path("/path/to/panda.urdf")
srdf_path = Path("/path/to/panda.srdf")

scene = sapien.Scene()
loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot = loader.load(str(urdf_path), package_dir="/path/to/package")

link_names = [link.name for link in robot.links]
joint_names = [joint.name for joint in robot.active_joints]

planner = mplib.Planner(
   urdf=urdf_path,
   srdf=srdf_path,
   move_group="panda_hand",
   user_link_names=link_names,
   user_joint_names=joint_names,
)
```

`move_group` is the link whose pose the planner controls, usually the
end-effector. `user_joint_names` contains active joints only; fixed joints must
not be included.

The planner defaults to velocity and acceleration limits of `1` for every joint
that affects `move_group`. If custom limits are supplied to the constructor,
their length must equal `len(planner.move_group_joint_indices)`, not necessarily
`robot.dof` (a Panda planner usually controls seven arm joints while the SAPIEN
articulation also contains two finger joints).

## Pose and frame conventions

SAPIEN and mplib both store quaternions in `wxyz` order, but they use different
`Pose` classes. Construct an mplib pose explicitly:

```python
goal_pose = mplib.Pose(
   [0.4, 0.0, 0.4],       # xyz
   [1.0, 0.0, 0.0, 0.0], # wxyz
)
```

`mplib.Pose(sapien_pose)` also accepts a `sapien.Pose` directly. Do not pass a
single seven-element list to `mplib.Pose`; that overload is interpreted as a
4-by-4 transformation matrix and raises.

`Planner.plan_pose` and `Planner.plan_screw` interpret goals in the world frame
by default. Pass `wrt_world=False` to interpret a goal in the robot base frame.
`Planner.IK` works directly in the robot base frame.

After configuration, continue with {ref}`plan_a_path`, {ref}`inverse_kinematics`,
and {ref}`collision_avoidance`.
