# PhysX API (`sapien.physx`)

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.physx`
- Source files: `python/py_package/pysapien/physx.pyi`, `python/pybind/physx.cpp`
- Notes: Common/CPU PhysX APIs. GPU-specific class rows are split into physx-gpu.md too.

## Functions

| API | Signature | Use | Notes |
|---|---|---|---|
| `sapien.physx.get_body_config` | `get_body_config() -> PhysxBodyConfig` | 读取 body config。 |  |
| `sapien.physx.get_default_material` | `get_default_material() -> PhysxMaterial` | 读取 default material。 |  |
| `sapien.physx.get_scene_config` | `get_scene_config() -> PhysxSceneConfig` | 读取 scene config。 |  |
| `sapien.physx.get_sdf_config` | `get_sdf_config() -> PhysxSDFConfig` | 读取 sdf config。 |  |
| `sapien.physx.get_shape_config` | `get_shape_config() -> PhysxShapeConfig` | 读取 shape config。 |  |
| `sapien.physx.is_gpu_enabled` | `is_gpu_enabled() -> bool` | 布尔状态：gpu enabled。 |  |
| `sapien.physx.set_body_config` | `set_body_config(solver_position_iterations: int=10, solver_velocity_iterations: int=1, sleep_threshold: float=0.004999999888241291) -> None<br>set_body_config(config: PhysxBodyConfig) -> None` | 设置 body config。 |  |
| `sapien.physx.set_default_material` | `set_default_material(static_friction: float, dynamic_friction: float, restitution: float) -> None` | 设置 default material。 |  |
| `sapien.physx.set_gpu_memory_config` | `set_gpu_memory_config(temp_buffer_capacity: int=16777216, max_rigid_contact_count: int=524288, max_rigid_patch_count: int=81920, heap_capacity: int=67108864, found_lost_pairs_capacity: int=262144, found_lost_aggregate_pairs_capacity: int=1024, total_aggregate_pairs_capacity: int=1024) -> None` | 设置 gpu memory config。 |  |
| `sapien.physx.set_scene_config` | `set_scene_config(gravity: np.ndarray[Literal[3], np.dtype[np.float32]]=..., bounce_threshold: float=2.0, enable_pcm: bool=True, enable_tgs: bool=True, enable_ccd: bool=False, enable_enhanced_determinism: bool=False, enable_friction_every_iteration: bool=True, friction_offset_threshold: float=0.04, friction_correlation_distance: float=0.025, cpu_workers: int=0) -> None<br>set_scene_config(config: PhysxSceneConfig) -> None` | 设置 scene config。 |  |
| `sapien.physx.set_sdf_config` | `set_sdf_config(spacing: float=0.009999999776482582, subgrid_size: int=6, num_threads_for_construction: int=4, resolution: int=0, bits_per_subgrid_pixel: int=16, narrow_band_thickness: float=0.009999999776482582, margin: float=0.0, enable_remeshing: bool=False, triangle_count_reduction_factor: float=1.0) -> None<br>set_sdf_config(config: PhysxSDFConfig) -> None` | 设置 sdf config。 |  |
| `sapien.physx.set_shape_config` | `set_shape_config(contact_offset: float=0.009999999776482582, rest_offset: float=0.0) -> None<br>set_shape_config(config: PhysxShapeConfig) -> None` | 设置 shape config。 |  |
| `sapien.physx.version` | `version() -> str` | 调用 version。 |  |

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.physx.PhysxArticulation` |  | 多关节 articulation/robot 句柄。 |  |
| `sapien.physx.PhysxArticulationJoint` |  | Articulation joint；drive、limit、friction、armature。 |  |
| `sapien.physx.PhysxArticulationLinkComponent` | `PhysxRigidBodyComponent` | Articulation link 的 PhysX 组件。 |  |
| `sapien.physx.PhysxBaseComponent` | `sapien.Component` | PhysX 物理 API 对象。 |  |
| `sapien.physx.PhysxBodyConfig` |  | 默认刚体 solver/sleep 配置。 |  |
| `sapien.physx.PhysxCollisionShape` |  | 碰撞形状基类；材质、pose、offset、collision groups。 |  |
| `sapien.physx.PhysxCollisionShapeBox` | `PhysxCollisionShape` | 具体 PhysX 碰撞形状。 |  |
| `sapien.physx.PhysxCollisionShapeCapsule` | `PhysxCollisionShape` | 具体 PhysX 碰撞形状。 |  |
| `sapien.physx.PhysxCollisionShapeConvexMesh` | `PhysxCollisionShape` | 具体 PhysX 碰撞形状。 |  |
| `sapien.physx.PhysxCollisionShapeCylinder` | `PhysxCollisionShape` | 具体 PhysX 碰撞形状。 |  |
| `sapien.physx.PhysxCollisionShapeHeightField` | `PhysxCollisionShape` | Low-level PhysX height-field collision shape. | Samples are int16 grid values; corner-origin local convention is `(row*row_scale, height*height_scale, column*column_scale)`. Use `Scene.add_heightfield` for z-up public coordinates plus optional render mesh. |
| `sapien.physx.PhysxCollisionShapePlane` | `PhysxCollisionShape` | 具体 PhysX 碰撞形状。 |  |
| `sapien.physx.PhysxCollisionShapeSphere` | `PhysxCollisionShape` | 具体 PhysX 碰撞形状。 |  |
| `sapien.physx.PhysxCollisionShapeTriangleMesh` | `PhysxCollisionShape` | 具体 PhysX 碰撞形状。 |  |
| `sapien.physx.PhysxContact` |  | PhysX 物理 API 对象。 |  |
| `sapien.physx.PhysxContactPoint` |  | PhysX 物理 API 对象。 |  |
| `sapien.physx.PhysxCpuSystem` | `PhysxSystem` | PhysX CPU 系统；contacts/raycast/pack/unpack。 |  |
| `sapien.physx.PhysxDistanceJointComponent` | `PhysxJointComponent` | PhysX 物理 API 对象。 |  |
| `sapien.physx.PhysxDriveComponent` | `PhysxJointComponent` | PhysX 物理 API 对象。 |  |
| `sapien.physx.PhysxEngine` |  | PhysX 物理 API 对象。 |  |
| `sapien.physx.PhysxGearComponent` | `PhysxJointComponent` | PhysX 物理 API 对象。 |  |
| `sapien.physx.PhysxJointComponent` | `PhysxBaseComponent` | PhysX 物理 API 对象。 |  |
| `sapien.physx.PhysxMaterial` |  | 物理材质；静/动摩擦和恢复系数。 |  |
| `sapien.physx.PhysxRayHit` |  | PhysX 物理 API 对象。 |  |
| `sapien.physx.PhysxRigidBaseComponent` | `PhysxBaseComponent` | 刚体碰撞形状容器基类。 |  |
| `sapien.physx.PhysxRigidBodyComponent` | `PhysxRigidBaseComponent` | 刚体通用质量、阻尼、速度、力接口。 |  |
| `sapien.physx.PhysxRigidDynamicComponent` | `PhysxRigidBodyComponent` | 动态/运动学刚体组件。 |  |
| `sapien.physx.PhysxRigidStaticComponent` | `PhysxRigidBaseComponent` | 静态刚体组件。 |  |
| `sapien.physx.PhysxSDFConfig` |  | SDF mesh collision cooking 配置。 |  |
| `sapien.physx.PhysxSceneConfig` |  | 全局 PhysX scene 配置；创建系统前设置。 |  |
| `sapien.physx.PhysxShapeConfig` |  | 默认碰撞 shape offset 配置。 |  |
| `sapien.physx.PhysxSystem` | `sapien.System` | PhysX 系统基类；管理 timestep、组件列表、scene collision id。 |  |

## `sapien.physx.PhysxArticulation`

- Use: 多关节 articulation/robot 句柄。
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `name` | `str` | 属性：name。 |  |
| `pose` | `sapien.Pose` | 属性：pose。 |  |
| `qacc` | `np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | 属性：qacc。 |  |
| `qf` | `np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | 属性：qf。 |  |
| `qpos` | `np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | 属性：qpos。 |  |
| `qvel` | `np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | 属性：qvel。 |  |
| `root_angular_velocity` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：root angular velocity。 |  |
| `root_linear_velocity` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：root linear velocity。 |  |
| `root_pose` | `sapien.Pose` | 属性：root pose。 |  |
| `sleep_threshold` | `float` | 属性：sleep threshold。 |  |
| `solver_position_iterations` | `int` | 属性：solver position iterations。 |  |
| `solver_velocity_iterations` | `int` | 属性：solver velocity iterations。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `active_joints` | property | `active_joints(self) -> list[PhysxArticulationJoint]` | 属性：active joints。 |  |
| `clone_links` | method | `clone_links(self) -> list[PhysxArticulationLinkComponent]` | 调用 clone links。 |  |
| `compute_dense_jacobian` | method | `compute_dense_jacobian(self) -> np.ndarray` | Compute dense articulation Jacobian on CPU articulations. | Not supported for Direct GPU API articulations; use `PhysxGpuSystem.gpu_compute_articulation_jacobian()` and `cuda_articulation_jacobian` instead. |
| `compute_passive_force` | method | `compute_passive_force(self, gravity: bool=True, coriolis_and_centrifugal: bool=True) -> np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | 计算 passive force。 |  |
| `create_fixed_tendon` | method | `create_fixed_tendon(self, link_chain: list[PhysxArticulationLinkComponent], coefficients: list[float], recip_coefficients: list[float], rest_length: float=0, offset: float=0, stiffness: float=0, damping: float=0, low: float=-3.4028234663852886e+38, high: float=3.4028234663852886e+38, limit_stiffness: float=0) -> None` | 创建 fixed tendon。 |  |
| `create_pinocchio_model` | staticmethod | `create_pinocchio_model(articulation: PhysxArticulation, gravity=[0, 0, -9.81]) -> sapien.pysapien_pinocchio.PinocchioModel` | 创建 pinocchio model。 |  |
| `dof` | property | `dof(self) -> int` | 属性：dof。 |  |
| `find_joint_by_name` | method | `find_joint_by_name(self, name: str) -> PhysxArticulationJoint` | 调用 find joint by name。 |  |
| `find_link_by_name` | method | `find_link_by_name(self, name: str) -> PhysxArticulationLinkComponent` | 调用 find link by name。 |  |
| `get_active_joints` | method | `get_active_joints(self) -> list[PhysxArticulationJoint]` | 读取 active joints。 |  |
| `get_dof` | method | `get_dof(self) -> int` | 读取 dof。 |  |
| `get_gpu_index` | method | `get_gpu_index(self) -> int` | 返回 PhysX GPU articulation/body 状态索引；gpu_init 后缓存。 |  |
| `get_joints` | method | `get_joints(self) -> list[PhysxArticulationJoint]` | 读取 joints。 |  |
| `get_jacobian_shape` | method | `get_jacobian_shape(self) -> tuple[int, int]` | Return valid dense Jacobian `(rows, cols)` for this articulation. | Fixed base: `((link_count - 1) * 6, dof)`; floating base: `(6 + (link_count - 1) * 6, 6 + dof)`. Use to slice `cuda_articulation_jacobian`. |
| `get_link_incoming_joint_forces` | method | `get_link_incoming_joint_forces(self) -> np.ndarray[tuple[M, Literal[6]], np.dtype[np.float32]]` | 读取 link incoming joint forces。 |  |
| `get_links` | method | `get_links(self) -> list[PhysxArticulationLinkComponent]` | 读取 links。 |  |
| `get_name` | method | `get_name(self) -> str` | 读取 name。 |  |
| `get_pose` | method | `get_pose(self) -> sapien.Pose` | 读取 pose。 |  |
| `get_qacc` | method | `get_qacc(self) -> np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | 读取 qacc。 |  |
| `get_qf` | method | `get_qf(self) -> np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | 读取 qf。 |  |
| `get_qlimit` | method | `get_qlimit(self) -> np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | same as get_qlimit |  |
| `get_qlimits` | method | `get_qlimits(self) -> np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | 读取 qlimits。 |  |
| `get_qpos` | method | `get_qpos(self) -> np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | 读取 qpos。 |  |
| `get_qvel` | method | `get_qvel(self) -> np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | 读取 qvel。 |  |
| `get_root` | method | `get_root(self) -> PhysxArticulationLinkComponent` | 读取 root。 |  |
| `get_root_angular_velocity` | method | `get_root_angular_velocity(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 读取 root angular velocity。 |  |
| `get_root_linear_velocity` | method | `get_root_linear_velocity(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 读取 root linear velocity。 |  |
| `get_root_pose` | method | `get_root_pose(self) -> sapien.Pose` | 读取 root pose。 |  |
| `get_sleep_threshold` | method | `get_sleep_threshold(self) -> float` | 读取 sleep threshold。 |  |
| `get_solver_position_iterations` | method | `get_solver_position_iterations(self) -> int` | 读取 solver position iterations。 |  |
| `get_solver_velocity_iterations` | method | `get_solver_velocity_iterations(self) -> int` | 读取 solver velocity iterations。 |  |
| `gpu_index` | property | `gpu_index(self) -> int` | 属性：gpu index。 |  |
| `joints` | property | `joints(self) -> list[PhysxArticulationJoint]` | 属性：joints。 |  |
| `jacobian_shape` | property | `jacobian_shape(self) -> tuple[int, int]` | Valid dense Jacobian shape. | Same as `get_jacobian_shape()`. |
| `link_incoming_joint_forces` | property | `link_incoming_joint_forces(self) -> np.ndarray[tuple[M, Literal[6]], np.dtype[np.float32]]` | 属性：link incoming joint forces。 |  |
| `links` | property | `links(self) -> list[PhysxArticulationLinkComponent]` | 属性：links。 |  |
| `qlimit` | property | `qlimit(self) -> np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | 属性：qlimit。 |  |
| `qlimits` | property | `qlimits(self) -> np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | 属性：qlimits。 |  |
| `root` | property | `root(self) -> PhysxArticulationLinkComponent` | 属性：root。 |  |
| `set_name` | method | `set_name(self, name: str) -> None` | 设置 name。 |  |
| `set_pose` | method | `set_pose(self, pose: sapien.Pose) -> None` | 设置 pose。 |  |
| `set_qacc` | method | `set_qacc(self, qacc: np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]] \| list \| tuple) -> None` | 设置 qacc。 |  |
| `set_qf` | method | `set_qf(self, qf: np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]] \| list \| tuple) -> None` | 设置 qf。 |  |
| `set_qpos` | method | `set_qpos(self, qpos: np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]] \| list \| tuple) -> None` | 设置 qpos。 |  |
| `set_qvel` | method | `set_qvel(self, qvel: np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]] \| list \| tuple) -> None` | 设置 qvel。 |  |
| `set_root_angular_velocity` | method | `set_root_angular_velocity(self, velocity: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | 设置 root angular velocity。 |  |
| `set_root_linear_velocity` | method | `set_root_linear_velocity(self, velocity: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | 设置 root linear velocity。 |  |
| `set_root_pose` | method | `set_root_pose(self, pose: sapien.Pose) -> None` | 设置 root pose。 |  |
| `set_sleep_threshold` | method | `set_sleep_threshold(self, threshold: float) -> None` | 设置 sleep threshold。 |  |
| `set_solver_position_iterations` | method | `set_solver_position_iterations(self, count: int) -> None` | 设置 solver position iterations。 |  |
| `set_solver_velocity_iterations` | method | `set_solver_velocity_iterations(self, count: int) -> None` | 设置 solver velocity iterations。 |  |

## `sapien.physx.PhysxArticulationJoint`

- Use: Articulation joint；drive、limit、friction、armature。
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `armature` | `np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | 属性：armature。 |  |
| `drive_target` | `np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | 属性：drive target。 |  |
| `drive_velocity_target` | `np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | 属性：drive velocity target。 |  |
| `friction` | `float` | 属性：friction。 |  |
| `limit` | `np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | 属性：limit。 |  |
| `limits` | `np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | 属性：limits。 |  |
| `name` | `str` | 属性：name。 |  |
| `pose_in_child` | `sapien.Pose` | 属性：pose in child。 |  |
| `pose_in_parent` | `sapien.Pose` | 属性：pose in parent。 |  |
| `type` | `Literal['fixed', 'revolute', 'revolute_unwrapped', 'prismatic', 'free']` | 属性：type。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `child_link` | property | `child_link(self) -> PhysxArticulationLinkComponent` | 属性：child link。 |  |
| `damping` | property | `damping(self) -> float` | 属性：damping。 |  |
| `dof` | property | `dof(self) -> int` | 属性：dof。 |  |
| `drive_mode` | property | `drive_mode(self) -> Literal['force', 'acceleration']` | 属性：drive mode。 |  |
| `force_limit` | property | `force_limit(self) -> float` | 属性：force limit。 |  |
| `get_armature` | method | `get_armature(self) -> np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | 读取 armature。 |  |
| `get_child_link` | method | `get_child_link(self) -> PhysxArticulationLinkComponent` | 读取 child link。 |  |
| `get_damping` | method | `get_damping(self) -> float` | 读取 damping。 |  |
| `get_dof` | method | `get_dof(self) -> int` | 读取 dof。 |  |
| `get_drive_mode` | method | `get_drive_mode(self) -> Literal['force', 'acceleration']` | 读取 drive mode。 |  |
| `get_drive_target` | method | `get_drive_target(self) -> np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | 读取 drive target。 |  |
| `get_drive_velocity_target` | method | `get_drive_velocity_target(self) -> np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | 读取 drive velocity target。 |  |
| `get_force_limit` | method | `get_force_limit(self) -> float` | 读取 force limit。 |  |
| `get_friction` | method | `get_friction(self) -> float` | 读取 friction。 |  |
| `get_global_pose` | method | `get_global_pose(self) -> sapien.Pose` | 读取 global pose。 |  |
| `get_limit` | method | `get_limit(self) -> np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | same as get_limits |  |
| `get_limits` | method | `get_limits(self) -> np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | 读取 limits。 |  |
| `get_name` | method | `get_name(self) -> str` | 读取 name。 |  |
| `get_parent_link` | method | `get_parent_link(self) -> PhysxArticulationLinkComponent` | 读取 parent link。 |  |
| `get_pose_in_child` | method | `get_pose_in_child(self) -> sapien.Pose` | 读取 pose in child。 |  |
| `get_pose_in_parent` | method | `get_pose_in_parent(self) -> sapien.Pose` | 读取 pose in parent。 |  |
| `get_stiffness` | method | `get_stiffness(self) -> float` | 读取 stiffness。 |  |
| `get_type` | method | `get_type(self) -> Literal['fixed', 'revolute', 'revolute_unwrapped', 'prismatic', 'free']` | 读取 type。 |  |
| `global_pose` | property | `global_pose(self) -> sapien.Pose` | 属性：global pose。 |  |
| `parent_link` | property | `parent_link(self) -> PhysxArticulationLinkComponent` | 属性：parent link。 |  |
| `set_armature` | method | `set_armature(self, armature: np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]] \| list \| tuple) -> None` | 设置 armature。 |  |
| `set_drive_properties` | method | `set_drive_properties(self, stiffness: float, damping: float, force_limit: float=3.4028234663852886e+38, mode: Literal['force', 'acceleration']='force') -> None` | 设置 drive properties。 |  |
| `set_drive_property` | method | `set_drive_property(self, stiffness: float, damping: float, force_limit: float=3.4028234663852886e+38, mode: Literal['force', 'acceleration']='force') -> None` | same as set_drive_properties |  |
| `set_drive_target` | method | `set_drive_target(self, target: float) -> None<br>set_drive_target(self, target: np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]] \| list \| tuple) -> None` | 设置 drive target。 |  |
| `set_drive_velocity_target` | method | `set_drive_velocity_target(self, velocity: float) -> None<br>set_drive_velocity_target(self, velocity: np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]] \| list \| tuple) -> None` | 设置 drive velocity target。 |  |
| `set_friction` | method | `set_friction(self, friction: float) -> None` | 设置 friction。 |  |
| `set_limit` | method | `set_limit(self, limit: np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]] \| list \| tuple) -> None` | same as set_limits |  |
| `set_limits` | method | `set_limits(self, limit: np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]] \| list \| tuple) -> None` | 设置 limits。 |  |
| `set_name` | method | `set_name(self, name: str) -> None` | 设置 name。 |  |
| `set_pose_in_child` | method | `set_pose_in_child(self, pose: sapien.Pose) -> None` | 设置 pose in child。 |  |
| `set_pose_in_parent` | method | `set_pose_in_parent(self, pose: sapien.Pose) -> None` | 设置 pose in parent。 |  |
| `set_type` | method | `set_type(self, type: Literal['fixed', 'revolute', 'revolute_unwrapped', 'prismatic', 'free']) -> None` | 设置 type。 |  |
| `stiffness` | property | `stiffness(self) -> float` | 属性：stiffness。 |  |

## `sapien.physx.PhysxArticulationLinkComponent`

- Use: Articulation link 的 PhysX 组件。
- Bases: `PhysxRigidBodyComponent`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `articulation` | property | `articulation(self) -> PhysxArticulation` | 属性：articulation。 |  |
| `children` | property | `children(self) -> list[PhysxArticulationLinkComponent]` | 属性：children。 |  |
| `get_articulation` | method | `get_articulation(self) -> PhysxArticulation` | 读取 articulation。 |  |
| `get_children` | method | `get_children(self) -> list[PhysxArticulationLinkComponent]` | 读取 children。 |  |
| `get_gpu_pose_index` | method | `get_gpu_pose_index(self) -> int` | 返回 GPU pose batch 索引；gpu_init 后缓存用于渲染。 |  |
| `get_index` | method | `get_index(self) -> int` | 读取 index。 |  |
| `get_joint` | method | `get_joint(self) -> PhysxArticulationJoint` | 读取 joint。 |  |
| `get_parent` | method | `get_parent(self) -> PhysxArticulationLinkComponent` | 读取 parent。 |  |
| `gpu_pose_index` | property | `gpu_pose_index(self) -> int` | 属性：gpu pose index。 |  |
| `index` | property | `index(self) -> int` | 属性：index。 |  |
| `is_root` | property | `is_root(self) -> bool` | 布尔状态：root。 |  |
| `joint` | property | `joint(self) -> PhysxArticulationJoint` | 属性：joint。 |  |
| `parent` | property | `parent(self) -> PhysxArticulationLinkComponent` | 属性：parent。 |  |
| `put_to_sleep` | method | `put_to_sleep(self) -> None` | 调用 put to sleep。 |  |
| `set_parent` | method | `set_parent(self, parent: PhysxArticulationLinkComponent) -> None` | 设置 parent。 |  |
| `sleeping` | property | `sleeping(self) -> bool` | 属性：sleeping。 |  |
| `wake_up` | method | `wake_up(self) -> None` | 调用 wake up。 |  |
| `__init__` | method | `__init__(self, parent: PhysxArticulationLinkComponent \| None=None) -> None` | Python special method。 |  |

## `sapien.physx.PhysxBaseComponent`

- Use: PhysX 物理 API 对象。
- Bases: `sapien.Component`

## `sapien.physx.PhysxBodyConfig`

- Use: 默认刚体 solver/sleep 配置。
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `sleep_threshold` | `float` | 属性：sleep threshold。 |  |
| `solver_position_iterations` | `int` | 属性：solver position iterations。 |  |
| `solver_velocity_iterations` | `int` | 属性：solver velocity iterations。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__getstate__` | method | `__getstate__(self) -> tuple` | Python special method。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |
| `__repr__` | method | `__repr__(self) -> str` | Python special method。 |  |
| `__setstate__` | method | `__setstate__(self, arg0: tuple) -> None` | Python special method。 |  |

## `sapien.physx.PhysxCollisionShape`

- Use: 碰撞形状基类；材质、pose、offset、collision groups。
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `contact_offset` | `float` | 属性：contact offset。 |  |
| `density` | `float` | 属性：density。 |  |
| `local_pose` | `sapien.Pose` | 属性：local pose。 |  |
| `min_patch_radius` | `float` | 属性：min patch radius。 |  |
| `patch_radius` | `float` | 属性：patch radius。 |  |
| `physical_material` | `PhysxMaterial` | 属性：physical material。 |  |
| `rest_offset` | `float` | 属性：rest offset。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `collision_groups` | property | `collision_groups(self) -> Annotated[list[int], FixedSize(4)]` | 属性：collision groups。 |  |
| `get_collision_groups` | method | `get_collision_groups(self) -> Annotated[list[int], FixedSize(4)]` | 读取 collision groups。 |  |
| `get_contact_offset` | method | `get_contact_offset(self) -> float` | 读取 contact offset。 |  |
| `get_density` | method | `get_density(self) -> float` | 读取 density。 |  |
| `get_local_pose` | method | `get_local_pose(self) -> sapien.Pose` | 读取 local pose。 |  |
| `get_min_patch_radius` | method | `get_min_patch_radius(self) -> float` | 读取 min patch radius。 |  |
| `get_patch_radius` | method | `get_patch_radius(self) -> float` | 读取 patch radius。 |  |
| `get_physical_material` | method | `get_physical_material(self) -> PhysxMaterial` | 读取 physical material。 |  |
| `get_rest_offset` | method | `get_rest_offset(self) -> float` | 读取 rest offset。 |  |
| `set_collision_groups` | method | `set_collision_groups(self, groups: Annotated[list[int], FixedSize(4)]) -> None` | collision groups determine the collision behavior of objects. Let A.gx denote the collision group x of collision shape A. Collision shape A and B will collide iff the... |  |
| `set_contact_offset` | method | `set_contact_offset(self, offset: float) -> None` | 设置 contact offset。 |  |
| `set_density` | method | `set_density(self, density: float) -> None` | 设置 density。 |  |
| `set_local_pose` | method | `set_local_pose(self, pose: sapien.Pose) -> None` | 设置 local pose。 |  |
| `set_min_patch_radius` | method | `set_min_patch_radius(self, radius: float) -> None` | 设置 min patch radius。 |  |
| `set_patch_radius` | method | `set_patch_radius(self, radius: float) -> None` | 设置 patch radius。 |  |
| `set_physical_material` | method | `set_physical_material(self, material: PhysxMaterial) -> None` | 设置 physical material。 |  |
| `set_rest_offset` | method | `set_rest_offset(self, offset: float) -> None` | 设置 rest offset。 |  |

## `sapien.physx.PhysxCollisionShapeBox`

- Use: 具体 PhysX 碰撞形状。
- Bases: `PhysxCollisionShape`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_half_size` | method | `get_half_size(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 读取 half size。 |  |
| `half_size` | property | `half_size(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：half size。 |  |
| `__init__` | method | `__init__(self, half_size: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, material: PhysxMaterial) -> None` | Python special method。 |  |

## `sapien.physx.PhysxCollisionShapeCapsule`

- Use: 具体 PhysX 碰撞形状。
- Bases: `PhysxCollisionShape`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_half_length` | method | `get_half_length(self) -> float` | 读取 half length。 |  |
| `get_radius` | method | `get_radius(self) -> float` | 读取 radius。 |  |
| `half_length` | property | `half_length(self) -> float` | 属性：half length。 |  |
| `radius` | property | `radius(self) -> float` | 属性：radius。 |  |
| `__init__` | method | `__init__(self, radius: float, half_length: float, material: PhysxMaterial) -> None` | Python special method。 |  |

## `sapien.physx.PhysxCollisionShapeConvexMesh`

- Use: 具体 PhysX 碰撞形状。
- Bases: `PhysxCollisionShape`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_scale` | method | `get_scale(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 读取 scale。 |  |
| `get_triangles` | method | `get_triangles(self) -> np.ndarray[np.uint32[M, 3]]` | 读取 triangles。 |  |
| `get_vertices` | method | `get_vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | 读取 vertices。 |  |
| `load_multiple` | staticmethod | `load_multiple(filename: str, scale: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, material: PhysxMaterial) -> list[PhysxCollisionShapeConvexMesh]` | 加载 multiple。 |  |
| `scale` | property | `scale(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：scale。 |  |
| `triangles` | property | `triangles(self) -> np.ndarray[np.uint32[M, 3]]` | 属性：triangles。 |  |
| `vertices` | property | `vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | 属性：vertices。 |  |
| `__init__` | method | `__init__(self, filename: str, scale: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, material: PhysxMaterial) -> None` | Python special method。 |  |

## `sapien.physx.PhysxCollisionShapeCylinder`

- Use: 具体 PhysX 碰撞形状。
- Bases: `PhysxCollisionShape`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_half_length` | method | `get_half_length(self) -> float` | 读取 half length。 |  |
| `get_radius` | method | `get_radius(self) -> float` | 读取 radius。 |  |
| `half_length` | property | `half_length(self) -> float` | 属性：half length。 |  |
| `radius` | property | `radius(self) -> float` | 属性：radius。 |  |
| `__init__` | method | `__init__(self, radius: float, half_length: float, material: PhysxMaterial) -> None` | Python special method。 |  |

## `sapien.physx.PhysxCollisionShapeHeightField`

- Use: Low-level PhysX height-field collision shape.
- Bases: `PhysxCollisionShape`
- Notes: samples must be 2D int16-compatible values. The low-level local convention has a corner origin: sample `(row, column)` is at local `(row * row_scale, sample * height_scale, column * column_scale)`. For ordinary z-up terrain with optional rendering, prefer `sapien.Scene.add_heightfield(...)`.

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `column_scale` | property | `column_scale(self) -> float` | Column spacing scale. | Read-only. |
| `get_column_scale` | method | `get_column_scale(self) -> float` | Read column spacing scale. |  |
| `get_height_field` | method | `get_height_field(self) -> np.ndarray` | Read int16 height samples. | Returned array uses the low-level PhysX row/column orientation. |
| `get_height_scale` | method | `get_height_scale(self) -> float` | Read vertical sample scale. |  |
| `get_row_scale` | method | `get_row_scale(self) -> float` | Read row spacing scale. |  |
| `height_field` | property | `height_field(self) -> np.ndarray` | Int16 height samples. | Read-only. |
| `height_scale` | property | `height_scale(self) -> float` | Vertical sample scale. | Read-only. |
| `row_scale` | property | `row_scale(self) -> float` | Row spacing scale. | Read-only. |
| `__init__` | method | `__init__(self, height_field: np.ndarray \| list \| tuple, row_scale: float, column_scale: float, height_scale: float, material: PhysxMaterial \| None=None) -> None` | Construct low-level height-field collision shape. | Scales must be positive; material defaults when omitted. |

## `sapien.physx.PhysxCollisionShapePlane`

- Use: 具体 PhysX 碰撞形状。
- Bases: `PhysxCollisionShape`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__init__` | method | `__init__(self, material: PhysxMaterial) -> None` | Python special method。 |  |

## `sapien.physx.PhysxCollisionShapeSphere`

- Use: 具体 PhysX 碰撞形状。
- Bases: `PhysxCollisionShape`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_radius` | method | `get_radius(self) -> float` | 读取 radius。 |  |
| `radius` | property | `radius(self) -> float` | 属性：radius。 |  |
| `__init__` | method | `__init__(self, radius: float, material: PhysxMaterial) -> None` | Python special method。 |  |

## `sapien.physx.PhysxCollisionShapeTriangleMesh`

- Use: 具体 PhysX 碰撞形状。
- Bases: `PhysxCollisionShape`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_scale` | method | `get_scale(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 读取 scale。 |  |
| `get_triangles` | method | `get_triangles(self) -> np.ndarray[np.uint32[M, 3]]` | 读取 triangles。 |  |
| `get_vertices` | method | `get_vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | 读取 vertices。 |  |
| `scale` | property | `scale(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：scale。 |  |
| `triangles` | property | `triangles(self) -> np.ndarray[np.uint32[M, 3]]` | 属性：triangles。 |  |
| `vertices` | property | `vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | 属性：vertices。 |  |
| `__init__` | method | `__init__(self, filename: str, scale: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, material: PhysxMaterial, sdf: bool=False, sdf_config: PhysxSDFConfig \| None=None) -> None<br>__init__(self, vertices: np.ndarray[Literal[3], np.dtype[np.float32]], triangles: np.ndarray[Literal[3], np.dtype[np.float32]], scale: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple=..., material: PhysxMaterial \| None=..., sdf: bool=False, sdf_config: PhysxSDFConfig \| None=None) -> None` | Python special method。 |  |

## `sapien.physx.PhysxContact`

- Use: PhysX 物理 API 对象。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `bodies` | property | `bodies(self) -> Annotated[list[PhysxRigidBaseComponent], FixedSize(2)]` | 属性：bodies。 |  |
| `points` | property | `points(self) -> list[PhysxContactPoint]` | 属性：points。 |  |
| `shapes` | property | `shapes(self) -> Annotated[list[PhysxCollisionShape], FixedSize(2)]` | 属性：shapes。 |  |
| `__repr__` | method | `__repr__(self) -> str` | Python special method。 |  |

## `sapien.physx.PhysxContactPoint`

- Use: PhysX 物理 API 对象。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `impulse` | property | `impulse(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：impulse。 |  |
| `normal` | property | `normal(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：normal。 |  |
| `position` | property | `position(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：position。 |  |
| `separation` | property | `separation(self) -> float` | 属性：separation。 |  |

## `sapien.physx.PhysxCpuSystem`

- Use: PhysX CPU 系统；contacts/raycast/pack/unpack。
- Bases: `PhysxSystem`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_contacts` | method | `get_contacts(self) -> list[PhysxContact]` | 读取 contacts。 |  |
| `pack` | method | `pack(self) -> bytes` | 调用 pack。 |  |
| `raycast` | method | `raycast(self, position: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, direction: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, distance: float) -> PhysxRayHit` | Casts a ray and returns the closest hit. Returns None if no hit |  |
| `unpack` | method | `unpack(self, data: bytes) -> None` | 调用 unpack。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.physx.PhysxDistanceJointComponent`

- Use: PhysX 物理 API 对象。
- Bases: `PhysxJointComponent`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `distance` | property | `distance(self) -> float` | 属性：distance。 |  |
| `get_distance` | method | `get_distance(self) -> float` | 读取 distance。 |  |
| `set_limit` | method | `set_limit(self, low: float, high: float, stiffness: float=0.0, damping: float=0.0) -> None` | 设置 limit。 |  |
| `__init__` | method | `__init__(self, body: PhysxRigidBodyComponent) -> None` | Python special method。 |  |

## `sapien.physx.PhysxDriveComponent`

- Use: PhysX 物理 API 对象。
- Bases: `PhysxJointComponent`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `drive_target` | `sapien.Pose` | 属性：drive target。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_drive_property_slerp` | method | `get_drive_property_slerp(self) -> tuple[float, float, float, Literal['force', 'acceleration']]` | 读取 drive property slerp。 |  |
| `get_drive_property_swing` | method | `get_drive_property_swing(self) -> tuple[float, float, float, Literal['force', 'acceleration']]` | 读取 drive property swing。 |  |
| `get_drive_property_twist` | method | `get_drive_property_twist(self) -> tuple[float, float, float, Literal['force', 'acceleration']]` | 读取 drive property twist。 |  |
| `get_drive_property_x` | method | `get_drive_property_x(self) -> tuple[float, float, float, Literal['force', 'acceleration']]` | 读取 drive property x。 |  |
| `get_drive_property_y` | method | `get_drive_property_y(self) -> tuple[float, float, float, Literal['force', 'acceleration']]` | 读取 drive property y。 |  |
| `get_drive_property_z` | method | `get_drive_property_z(self) -> tuple[float, float, float, Literal['force', 'acceleration']]` | 读取 drive property z。 |  |
| `get_drive_target` | method | `get_drive_target(self) -> sapien.Pose` | 读取 drive target。 |  |
| `get_drive_velocity_target` | method | `get_drive_velocity_target(self) -> tuple[np.ndarray[Literal[3], np.dtype[np.float32]], np.ndarray[Literal[3], np.dtype[np.float32]]]` | 读取 drive velocity target。 |  |
| `get_limit_cone` | method | `get_limit_cone(self) -> tuple[float, float, float, float]` | 读取 limit cone。 |  |
| `get_limit_pyramid` | method | `get_limit_pyramid(self) -> tuple[float, float, float, float, float, float]` | 读取 limit pyramid。 |  |
| `get_limit_twist` | method | `get_limit_twist(self) -> tuple[float, float, float, float]` | 读取 limit twist。 |  |
| `get_limit_x` | method | `get_limit_x(self) -> tuple[float, float, float, float]` | 读取 limit x。 |  |
| `get_limit_y` | method | `get_limit_y(self) -> tuple[float, float, float, float]` | 读取 limit y。 |  |
| `get_limit_z` | method | `get_limit_z(self) -> tuple[float, float, float, float]` | 读取 limit z。 |  |
| `set_drive_property_slerp` | method | `set_drive_property_slerp(self, stiffness: float, damping: float, force_limit: float=3.4028234663852886e+38, mode: Literal['force', 'acceleration']='force') -> None` | 设置 drive property slerp。 |  |
| `set_drive_property_swing` | method | `set_drive_property_swing(self, stiffness: float, damping: float, force_limit: float=3.4028234663852886e+38, mode: Literal['force', 'acceleration']='force') -> None` | 设置 drive property swing。 |  |
| `set_drive_property_twist` | method | `set_drive_property_twist(self, stiffness: float, damping: float, force_limit: float=3.4028234663852886e+38, mode: Literal['force', 'acceleration']='force') -> None` | 设置 drive property twist。 |  |
| `set_drive_property_x` | method | `set_drive_property_x(self, stiffness: float, damping: float, force_limit: float=3.4028234663852886e+38, mode: Literal['force', 'acceleration']='force') -> None` | 设置 drive property x。 |  |
| `set_drive_property_y` | method | `set_drive_property_y(self, stiffness: float, damping: float, force_limit: float=3.4028234663852886e+38, mode: Literal['force', 'acceleration']='force') -> None` | 设置 drive property y。 |  |
| `set_drive_property_z` | method | `set_drive_property_z(self, stiffness: float, damping: float, force_limit: float=3.4028234663852886e+38, mode: Literal['force', 'acceleration']='force') -> None` | 设置 drive property z。 |  |
| `set_drive_target` | method | `set_drive_target(self, target: sapien.Pose) -> None` | 设置 drive target。 |  |
| `set_drive_velocity_target` | method | `set_drive_velocity_target(self, linear: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, angular: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | 设置 drive velocity target。 |  |
| `set_limit_cone` | method | `set_limit_cone(self, angle_y: float, angle_z: float, stiffness: float=0.0, damping: float=0.0) -> None` | 设置 limit cone。 |  |
| `set_limit_pyramid` | method | `set_limit_pyramid(self, low_y: float, high_y: float, low_z: float, high_z: float, stiffness: float=0.0, damping: float=0.0) -> None` | 设置 limit pyramid。 |  |
| `set_limit_twist` | method | `set_limit_twist(self, low: float, high: float, stiffness: float=0.0, damping: float=0.0) -> None` | 设置 limit twist。 |  |
| `set_limit_x` | method | `set_limit_x(self, low: float, high: float, stiffness: float=0.0, damping: float=0.0) -> None` | 设置 limit x。 |  |
| `set_limit_y` | method | `set_limit_y(self, low: float, high: float, stiffness: float=0.0, damping: float=0.0) -> None` | 设置 limit y。 |  |
| `set_limit_z` | method | `set_limit_z(self, low: float, high: float, stiffness: float=0.0, damping: float=0.0) -> None` | 设置 limit z。 |  |
| `__init__` | method | `__init__(self, body: PhysxRigidBodyComponent) -> None` | Python special method。 |  |

## `sapien.physx.PhysxEngine`

- Use: PhysX 物理 API 对象。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__init__` | method | `__init__(self, tolerance_length: float, tolerance_speed: float) -> None` | Python special method。 |  |

## `sapien.physx.PhysxGearComponent`

- Use: PhysX 物理 API 对象。
- Bases: `PhysxJointComponent`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `gear_ratio` | `float` | 属性：gear ratio。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `enable_hinges` | method | `enable_hinges(self) -> None` | 启用 hinges。 |  |
| `get_gear_ratio` | method | `get_gear_ratio(self) -> float` | 读取 gear ratio。 |  |
| `is_hinges_enabled` | property | `is_hinges_enabled(self) -> bool` | 布尔状态：hinges enabled。 |  |
| `set_gear_ratio` | method | `set_gear_ratio(self, ratio: float) -> None` | 设置 gear ratio。 |  |
| `__init__` | method | `__init__(self, body: PhysxRigidBodyComponent) -> None` | Python special method。 |  |

## `sapien.physx.PhysxJointComponent`

- Use: PhysX 物理 API 对象。
- Bases: `PhysxBaseComponent`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `parent` | `PhysxRigidBaseComponent` | 属性：parent。 |  |
| `pose_in_child` | `sapien.Pose` | 属性：pose in child。 |  |
| `pose_in_parent` | `sapien.Pose` | 属性：pose in parent。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_parent` | method | `get_parent(self) -> PhysxRigidBaseComponent` | 读取 parent。 |  |
| `get_pose_in_child` | method | `get_pose_in_child(self) -> sapien.Pose` | 读取 pose in child。 |  |
| `get_pose_in_parent` | method | `get_pose_in_parent(self) -> sapien.Pose` | 读取 pose in parent。 |  |
| `get_relative_pose` | method | `get_relative_pose(self) -> sapien.Pose` | 读取 relative pose。 |  |
| `relative_pose` | property | `relative_pose(self) -> sapien.Pose` | 属性：relative pose。 |  |
| `set_inv_inertia_scales` | method | `set_inv_inertia_scales(self, scale0: float, scale1: float) -> None` | 设置 inv inertia scales。 |  |
| `set_inv_mass_scales` | method | `set_inv_mass_scales(self, scale0: float, scale1: float) -> None` | 设置 inv mass scales。 |  |
| `set_parent` | method | `set_parent(self, parent: PhysxRigidBaseComponent) -> None` | 设置 parent。 |  |
| `set_pose_in_child` | method | `set_pose_in_child(self, pose: sapien.Pose) -> None` | 设置 pose in child。 |  |
| `set_pose_in_parent` | method | `set_pose_in_parent(self, pose: sapien.Pose) -> None` | 设置 pose in parent。 |  |

## `sapien.physx.PhysxMaterial`

- Use: 物理材质；静/动摩擦和恢复系数。
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `dynamic_friction` | `float` | 属性：dynamic friction。 |  |
| `restitution` | `float` | 属性：restitution。 |  |
| `static_friction` | `float` | 属性：static friction。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_dynamic_friction` | method | `get_dynamic_friction(self) -> float` | 读取 dynamic friction。 |  |
| `get_restitution` | method | `get_restitution(self) -> float` | 读取 restitution。 |  |
| `get_static_friction` | method | `get_static_friction(self) -> float` | 读取 static friction。 |  |
| `set_dynamic_friction` | method | `set_dynamic_friction(self, friction: float) -> None` | 设置 dynamic friction。 |  |
| `set_restitution` | method | `set_restitution(self, restitution: float) -> None` | 设置 restitution。 |  |
| `set_static_friction` | method | `set_static_friction(self, friction: float) -> None` | 设置 static friction。 |  |
| `__init__` | method | `__init__(self, static_friction: float, dynamic_friction: float, restitution: float) -> None` | Python special method。 |  |

## `sapien.physx.PhysxRayHit`

- Use: PhysX 物理 API 对象。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `component` | property | `component(self) -> PhysxRigidBaseComponent` | 属性：component。 |  |
| `distance` | property | `distance(self) -> float` | 属性：distance。 |  |
| `normal` | property | `normal(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：normal。 |  |
| `position` | property | `position(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：position。 |  |
| `shape` | property | `shape(self) -> PhysxCollisionShape` | 属性：shape。 |  |
| `__repr__` | method | `__repr__(self) -> str` | Python special method。 |  |

## `sapien.physx.PhysxRigidBaseComponent`

- Use: 刚体碰撞形状容器基类。
- Bases: `PhysxBaseComponent`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `attach` | method | `attach(self, collision_shape: PhysxCollisionShape) -> PhysxRigidBaseComponent` | 调用 attach。 |  |
| `collision_shapes` | property | `collision_shapes(self) -> list[PhysxCollisionShape]` | 属性：collision shapes。 |  |
| `compute_global_aabb_tight` | method | `compute_global_aabb_tight(self) -> np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.float32]]` | 计算 global aabb tight。 |  |
| `get_collision_shapes` | method | `get_collision_shapes(self) -> list[PhysxCollisionShape]` | 读取 collision shapes。 |  |
| `get_global_aabb_fast` | method | `get_global_aabb_fast(self) -> np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.float32]]` | 读取 global aabb fast。 |  |
| `_physx_pointer` | property | `_physx_pointer(self) -> int` | 属性： physx pointer。 |  |

## `sapien.physx.PhysxRigidBodyComponent`

- Use: 刚体通用质量、阻尼、速度、力接口。
- Bases: `PhysxRigidBaseComponent`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `angular_damping` | `float` | 属性：angular damping。 |  |
| `cmass_local_pose` | `sapien.Pose` | 属性：cmass local pose。 |  |
| `disable_gravity` | `bool` | 禁用 gravity。 |  |
| `inertia` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：inertia。 |  |
| `linear_damping` | `float` | 属性：linear damping。 |  |
| `mass` | `float` | 属性：mass。 |  |
| `max_contact_impulse` | `float` | 属性：max contact impulse。 |  |
| `max_depenetration_velocity` | `float` | 属性：max depenetration velocity。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `add_force_at_point` | method | `add_force_at_point(self, force: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, point: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, mode: Literal['force', 'acceleration', 'velocity_change', 'impulse']='force') -> None` | 添加/创建 force at point。 |  |
| `add_force_torque` | method | `add_force_torque(self, force: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, torque: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, mode: Literal['force', 'acceleration', 'velocity_change', 'impulse']='force') -> None` | 添加/创建 force torque。 |  |
| `angular_velocity` | property | `angular_velocity(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：angular velocity。 |  |
| `auto_compute_mass` | property | `auto_compute_mass(self) -> bool` | 属性：auto compute mass。 |  |
| `get_angular_damping` | method | `get_angular_damping(self) -> float` | 读取 angular damping。 |  |
| `get_angular_velocity` | method | `get_angular_velocity(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 读取 angular velocity。 |  |
| `get_auto_compute_mass` | method | `get_auto_compute_mass(self) -> bool` | 读取 auto compute mass。 |  |
| `get_cmass_local_pose` | method | `get_cmass_local_pose(self) -> sapien.Pose` | 读取 cmass local pose。 |  |
| `get_disable_gravity` | method | `get_disable_gravity(self) -> bool` | 读取 disable gravity。 |  |
| `get_inertia` | method | `get_inertia(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 读取 inertia。 |  |
| `get_linear_damping` | method | `get_linear_damping(self) -> float` | 读取 linear damping。 |  |
| `get_linear_velocity` | method | `get_linear_velocity(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 读取 linear velocity。 |  |
| `get_mass` | method | `get_mass(self) -> float` | 读取 mass。 |  |
| `get_max_contact_impulse` | method | `get_max_contact_impulse(self) -> float` | 读取 max contact impulse。 |  |
| `get_max_depenetration_velocity` | method | `get_max_depenetration_velocity(self) -> float` | 读取 max depenetration velocity。 |  |
| `linear_velocity` | property | `linear_velocity(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：linear velocity。 |  |
| `set_angular_damping` | method | `set_angular_damping(self, damping: float) -> None` | 设置 angular damping。 |  |
| `set_cmass_local_pose` | method | `set_cmass_local_pose(self, pose: sapien.Pose) -> None` | 设置 cmass local pose。 |  |
| `set_disable_gravity` | method | `set_disable_gravity(self, disable: bool) -> None` | 设置 disable gravity。 |  |
| `set_inertia` | method | `set_inertia(self, inertia: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | 设置 inertia。 |  |
| `set_linear_damping` | method | `set_linear_damping(self, damping: float) -> None` | 设置 linear damping。 |  |
| `set_mass` | method | `set_mass(self, mass: float) -> None` | 设置 mass。 |  |
| `set_max_contact_impulse` | method | `set_max_contact_impulse(self, impulse: float) -> None` | 设置 max contact impulse。 |  |
| `set_max_depenetration_velocity` | method | `set_max_depenetration_velocity(self, velocity: float) -> None` | 设置 max depenetration velocity。 |  |

## `sapien.physx.PhysxRigidDynamicComponent`

- Use: 动态/运动学刚体组件。
- Bases: `PhysxRigidBodyComponent`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `angular_velocity` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：angular velocity。 |  |
| `gyroscopic_forces` | `bool` | 属性：gyroscopic forces。 |  |
| `kinematic` | `bool` | 属性：kinematic。 |  |
| `kinematic_target` | `sapien.Pose` | 属性：kinematic target。 |  |
| `linear_velocity` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：linear velocity。 |  |
| `sleep_threshold` | `float` | 属性：sleep threshold。 |  |
| `solver_position_iterations` | `int` | 属性：solver position iterations。 |  |
| `solver_velocity_iterations` | `int` | 属性：solver velocity iterations。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_angular_velocity` | method | `get_angular_velocity(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 读取 angular velocity。 |  |
| `get_gpu_index` | method | `get_gpu_index(self) -> int` | 返回 PhysX GPU articulation/body 状态索引；gpu_init 后缓存。 |  |
| `get_gpu_pose_index` | method | `get_gpu_pose_index(self) -> int` | 返回 GPU pose batch 索引；gpu_init 后缓存用于渲染。 |  |
| `get_kinematic` | method | `get_kinematic(self) -> bool` | 读取 kinematic。 |  |
| `get_kinematic_target` | method | `get_kinematic_target(self) -> sapien.Pose` | 读取 kinematic target。 |  |
| `get_linear_velocity` | method | `get_linear_velocity(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 读取 linear velocity。 |  |
| `get_locked_motion_axes` | method | `get_locked_motion_axes(self) -> Annotated[list[bool], FixedSize(6)]` | 读取 locked motion axes。 |  |
| `get_sleep_threshold` | method | `get_sleep_threshold(self) -> float` | 读取 sleep threshold。 |  |
| `get_solver_position_iterations` | method | `get_solver_position_iterations(self) -> int` | 读取 solver position iterations。 |  |
| `get_solver_velocity_iterations` | method | `get_solver_velocity_iterations(self) -> int` | 读取 solver velocity iterations。 |  |
| `gpu_index` | property | `gpu_index(self) -> int` | 属性：gpu index。 |  |
| `gpu_pose_index` | property | `gpu_pose_index(self) -> int` | 属性：gpu pose index。 |  |
| `is_sleeping` | property | `is_sleeping(self) -> bool` | 布尔状态：sleeping。 |  |
| `locked_motion_axes` | property | `locked_motion_axes(self) -> Annotated[list[bool], FixedSize(6)]` | 属性：locked motion axes。 |  |
| `put_to_sleep` | method | `put_to_sleep(self) -> None` | 调用 put to sleep。 |  |
| `set_angular_velocity` | method | `set_angular_velocity(self, velocity: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | 设置 angular velocity。 |  |
| `set_gyroscopic_forces` | method | `set_gyroscopic_forces(self, kinematic: bool) -> None` | 设置 gyroscopic forces。 |  |
| `set_kinematic` | method | `set_kinematic(self, kinematic: bool) -> None` | 设置 kinematic。 |  |
| `set_kinematic_target` | method | `set_kinematic_target(self, target: sapien.Pose) -> None` | 设置 kinematic target。 |  |
| `set_linear_velocity` | method | `set_linear_velocity(self, velocity: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | 设置 linear velocity。 |  |
| `set_locked_motion_axes` | method | `set_locked_motion_axes(self, axes: Annotated[list[bool], FixedSize(6)]) -> None` | set some motion axes of the dynamic rigid body to be locked Args: axes: list of 6 true/false values indicating whether which of the 6 DOFs of the body is locked. The o... |  |
| `set_sleep_threshold` | method | `set_sleep_threshold(self, threshold: float) -> None` | 设置 sleep threshold。 |  |
| `set_solver_position_iterations` | method | `set_solver_position_iterations(self, count: int) -> None` | 设置 solver position iterations。 |  |
| `set_solver_velocity_iterations` | method | `set_solver_velocity_iterations(self, count: int) -> None` | 设置 solver velocity iterations。 |  |
| `wake_up` | method | `wake_up(self) -> None` | 调用 wake up。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.physx.PhysxRigidStaticComponent`

- Use: 静态刚体组件。
- Bases: `PhysxRigidBaseComponent`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.physx.PhysxSDFConfig`

- Use: SDF mesh collision cooking 配置。
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `bitsPerSubgridPixel` | `int` | 属性：bitsPerSubgridPixel。 |  |
| `bits_per_subgrid_pixel` | `int` | 属性：bits per subgrid pixel。 |  |
| `enableRemeshing` | `bool` | 属性：enableRemeshing。 |  |
| `enable_remeshing` | `bool` | 启用 remeshing。 |  |
| `margin` | `float` | 属性：margin。 |  |
| `narrowBandThickness` | `float` | 属性：narrowBandThickness。 |  |
| `narrow_band_thickness` | `float` | 属性：narrow band thickness。 |  |
| `num_threads_for_construction` | `int` | 属性：num threads for construction。 |  |
| `resolution` | `int` | 属性：resolution。 |  |
| `spacing` | `float` | 属性：spacing。 |  |
| `subgridSize` | `int` | 属性：subgridSize。 |  |
| `subgrid_size` | `int` | 属性：subgrid size。 |  |
| `triangleCountReductionFactor` | `float` | 属性：triangleCountReductionFactor。 |  |
| `triangle_count_reduction_factor` | `float` | 属性：triangle count reduction factor。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__getstate__` | method | `__getstate__(self) -> tuple` | Python special method。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |
| `__repr__` | method | `__repr__(self) -> str` | Python special method。 |  |
| `__setstate__` | method | `__setstate__(self, arg0: tuple) -> None` | Python special method。 |  |

## `sapien.physx.PhysxSceneConfig`

- Use: 全局 PhysX scene 配置；创建系统前设置。
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `bounce_threshold` | `float` | 属性：bounce threshold。 |  |
| `cpu_workers` | `int` | 属性：cpu workers。 |  |
| `enable_ccd` | `bool` | 启用 ccd。 |  |
| `enable_enhanced_determinism` | `bool` | 启用 enhanced determinism。 |  |
| `enable_friction_every_iteration` | `bool` | 启用 friction every iteration。 |  |
| `enable_pcm` | `bool` | 启用 pcm。 |  |
| `enable_tgs` | `bool` | 启用 tgs。 |  |
| `friction_correlation_distance` | `float` | 属性：friction correlation distance。 |  |
| `friction_offset_threshold` | `float` | 属性：friction offset threshold。 |  |
| `gpu_broadphase_env_id_bits` | `int` | 属性：gpu broadphase env id bits。 |  |
| `gpu_broadphase_nb_bits_env_id_x` | `int` | 属性：gpu broadphase nb bits env id x。 |  |
| `gpu_broadphase_nb_bits_env_id_y` | `int` | 属性：gpu broadphase nb bits env id y。 |  |
| `gpu_broadphase_nb_bits_env_id_z` | `int` | 属性：gpu broadphase nb bits env id z。 |  |
| `gravity` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：gravity。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__getstate__` | method | `__getstate__(self) -> tuple` | Python special method。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |
| `__repr__` | method | `__repr__(self) -> str` | Python special method。 |  |
| `__setstate__` | method | `__setstate__(self, arg0: tuple) -> None` | Python special method。 |  |

## `sapien.physx.PhysxShapeConfig`

- Use: 默认碰撞 shape offset 配置。
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `contact_offset` | `float` | 属性：contact offset。 |  |
| `rest_offset` | `float` | 属性：rest offset。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__getstate__` | method | `__getstate__(self) -> tuple` | Python special method。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |
| `__repr__` | method | `__repr__(self) -> str` | Python special method。 |  |
| `__setstate__` | method | `__setstate__(self, arg0: tuple) -> None` | Python special method。 |  |

## `sapien.physx.PhysxSystem`

- Use: PhysX 系统基类；管理 timestep、组件列表、scene collision id。
- Bases: `sapien.System`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `scene_collision_id` | `int` | 属性：scene collision id。 |  |
| `timestep` | `float` | 属性：timestep。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `articulation_link_components` | property | `articulation_link_components(self) -> list[PhysxArticulationLinkComponent]` | 属性：articulation link components。 |  |
| `config` | property | `config(self) -> PhysxSceneConfig` | 属性：config。 |  |
| `get_articulation_link_components` | method | `get_articulation_link_components(self) -> list[PhysxArticulationLinkComponent]` | 读取 articulation link components。 |  |
| `get_config` | method | `get_config(self) -> PhysxSceneConfig` | 读取 config。 |  |
| `get_rigid_dynamic_components` | method | `get_rigid_dynamic_components(self) -> list[PhysxRigidDynamicComponent]` | 读取 rigid dynamic components。 |  |
| `get_rigid_static_components` | method | `get_rigid_static_components(self) -> list[PhysxRigidStaticComponent]` | 读取 rigid static components。 |  |
| `get_scene_collision_id` | method | `get_scene_collision_id(self) -> int` | 读取 scene collision id。 |  |
| `get_timestep` | method | `get_timestep(self) -> float` | 读取 timestep。 |  |
| `rigid_dynamic_components` | property | `rigid_dynamic_components(self) -> list[PhysxRigidDynamicComponent]` | 属性：rigid dynamic components。 |  |
| `rigid_static_components` | property | `rigid_static_components(self) -> list[PhysxRigidStaticComponent]` | 属性：rigid static components。 |  |
| `set_scene_collision_id` | method | `set_scene_collision_id(self, id: int) -> None` | 设置 scene collision id。 |  |
| `set_timestep` | method | `set_timestep(self, timestep: float) -> None` | 设置仿真步长。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |
