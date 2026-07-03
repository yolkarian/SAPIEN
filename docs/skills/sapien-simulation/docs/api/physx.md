# PhysX API (`sapien.physx`)

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.physx`
- Source files: `python/py_package/pysapien/physx.pyi`, `python/pybind/physx.cpp`
- Notes: Common/CPU PhysX APIs. GPU-specific class rows are split into physx-gpu.md too.

## Functions

| API | Signature | Use | Notes |
|---|---|---|---|
| `sapien.physx.get_body_config` | `get_body_config() -> PhysxBodyConfig` | Get body config. |  |
| `sapien.physx.get_default_material` | `get_default_material() -> PhysxMaterial` | Get default material. |  |
| `sapien.physx.get_scene_config` | `get_scene_config() -> PhysxSceneConfig` | Get scene config. |  |
| `sapien.physx.get_sdf_config` | `get_sdf_config() -> PhysxSDFConfig` | Get sdf config. |  |
| `sapien.physx.get_shape_config` | `get_shape_config() -> PhysxShapeConfig` | Get shape config. |  |
| `sapien.physx.is_gpu_enabled` | `is_gpu_enabled() -> bool` | Bool: gpu enabled. |  |
| `sapien.physx.set_body_config` | `set_body_config(solver_position_iterations: int=10, solver_velocity_iterations: int=1, sleep_threshold: float=0.004999999888241291) -> None<br>set_body_config(config: PhysxBodyConfig) -> None` | Set body config. |  |
| `sapien.physx.set_default_material` | `set_default_material(static_friction: float, dynamic_friction: float, restitution: float) -> None` | Set default material. |  |
| `sapien.physx.set_gpu_memory_config` | `set_gpu_memory_config(temp_buffer_capacity: int=16777216, max_rigid_contact_count: int=524288, max_rigid_patch_count: int=81920, heap_capacity: int=67108864, found_lost_pairs_capacity: int=262144, found_lost_aggregate_pairs_capacity: int=1024, total_aggregate_pairs_capacity: int=1024) -> None` | Set gpu memory config. |  |
| `sapien.physx.set_scene_config` | `set_scene_config(gravity: np.ndarray[Literal[3], np.dtype[np.float32]]=..., bounce_threshold: float=2.0, enable_pcm: bool=True, enable_tgs: bool=True, enable_ccd: bool=False, enable_enhanced_determinism: bool=False, enable_friction_every_iteration: bool=True, friction_offset_threshold: float=0.04, friction_correlation_distance: float=0.025, cpu_workers: int=0) -> None<br>set_scene_config(config: PhysxSceneConfig) -> None` | Set scene config. |  |
| `sapien.physx.set_sdf_config` | `set_sdf_config(spacing: float=0.009999999776482582, subgrid_size: int=6, num_threads_for_construction: int=4, resolution: int=0, bits_per_subgrid_pixel: int=16, narrow_band_thickness: float=0.009999999776482582, margin: float=0.0, enable_remeshing: bool=False, triangle_count_reduction_factor: float=1.0) -> None<br>set_sdf_config(config: PhysxSDFConfig) -> None` | Set sdf config. |  |
| `sapien.physx.set_shape_config` | `set_shape_config(contact_offset: float=0.009999999776482582, rest_offset: float=0.0) -> None<br>set_shape_config(config: PhysxShapeConfig) -> None` | Set shape config. |  |
| `sapien.physx.version` | `version() -> str` | Call version. |  |

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.physx.PhysxArticulation` |  | Multi-joint articulation/robot handle. |  |
| `sapien.physx.PhysxArticulationJoint` |  | Articulation joint; drive, limit, friction, armature. |  |
| `sapien.physx.PhysxArticulationLinkComponent` | `PhysxRigidBodyComponent` | PhysX component of an articulation link. |  |
| `sapien.physx.PhysxBaseComponent` | `sapien.Component` | PhysX physics API object. |  |
| `sapien.physx.PhysxBodyConfig` |  | Default rigid-body solver/sleep configuration. |  |
| `sapien.physx.PhysxCollisionShape` |  | Collision shape base class; material, pose, offset, collision groups. |  |
| `sapien.physx.PhysxCollisionShapeBox` | `PhysxCollisionShape` | Concrete PhysX collision shape. |  |
| `sapien.physx.PhysxCollisionShapeCapsule` | `PhysxCollisionShape` | Concrete PhysX collision shape. |  |
| `sapien.physx.PhysxCollisionShapeConvexMesh` | `PhysxCollisionShape` | Concrete PhysX collision shape. |  |
| `sapien.physx.PhysxCollisionShapeCylinder` | `PhysxCollisionShape` | Concrete PhysX collision shape. |  |
| `sapien.physx.PhysxCollisionShapeHeightField` | `PhysxCollisionShape` | Low-level PhysX height-field collision shape. | Samples are int16 grid values; corner-origin local convention is `(row*row_scale, height*height_scale, column*column_scale)`. Use `Scene.add_heightfield` for z-up public coordinates plus optional render mesh. |
| `sapien.physx.PhysxCollisionShapePlane` | `PhysxCollisionShape` | Concrete PhysX collision shape. |  |
| `sapien.physx.PhysxCollisionShapeSphere` | `PhysxCollisionShape` | Concrete PhysX collision shape. |  |
| `sapien.physx.PhysxCollisionShapeTriangleMesh` | `PhysxCollisionShape` | Concrete PhysX collision shape. |  |
| `sapien.physx.PhysxContact` |  | PhysX physics API object. |  |
| `sapien.physx.PhysxContactPoint` |  | PhysX physics API object. |  |
| `sapien.physx.PhysxCpuSystem` | `PhysxSystem` | PhysX CPU system; contacts/raycast/pack/unpack. |  |
| `sapien.physx.PhysxDistanceJointComponent` | `PhysxJointComponent` | PhysX physics API object. |  |
| `sapien.physx.PhysxDriveComponent` | `PhysxJointComponent` | PhysX physics API object. |  |
| `sapien.physx.PhysxEngine` |  | PhysX physics API object. |  |
| `sapien.physx.PhysxGearComponent` | `PhysxJointComponent` | PhysX physics API object. |  |
| `sapien.physx.PhysxJointComponent` | `PhysxBaseComponent` | PhysX physics API object. |  |
| `sapien.physx.PhysxMaterial` |  | Physical material; static/dynamic friction and restitution. |  |
| `sapien.physx.PhysxRayHit` |  | PhysX physics API object. |  |
| `sapien.physx.PhysxRigidBaseComponent` | `PhysxBaseComponent` | Rigid-body collision shape container base class. |  |
| `sapien.physx.PhysxRigidBodyComponent` | `PhysxRigidBaseComponent` | Rigid-body general mass, damping, velocity, force interface. |  |
| `sapien.physx.PhysxRigidDynamicComponent` | `PhysxRigidBodyComponent` | Dynamic/kinematic rigid-body component. |  |
| `sapien.physx.PhysxRigidStaticComponent` | `PhysxRigidBaseComponent` | Static rigid-body component. |  |
| `sapien.physx.PhysxSDFConfig` |  | SDF mesh collision cooking configuration. |  |
| `sapien.physx.PhysxSceneConfig` |  | Global PhysX scene configuration; set before creating the system. |  |
| `sapien.physx.PhysxShapeConfig` |  | Default collision shape offset configuration. |  |
| `sapien.physx.PhysxSystem` | `sapien.System` | PhysX system base class; manages timestep, component list, scene collision id. |  |

## `sapien.physx.PhysxArticulation`

- Use: Multi-joint articulation/robot handle.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `name` | `str` | Property: name. |  |
| `pose` | `sapien.Pose` | Property: pose. |  |
| `qacc` | `np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | Property: qacc. |  |
| `qf` | `np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | Property: qf. |  |
| `qpos` | `np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | Property: qpos. |  |
| `qvel` | `np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | Property: qvel. |  |
| `root_angular_velocity` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: root angular velocity. |  |
| `root_linear_velocity` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: root linear velocity. |  |
| `root_pose` | `sapien.Pose` | Property: root pose. |  |
| `sleep_threshold` | `float` | Property: sleep threshold. |  |
| `solver_position_iterations` | `int` | Property: solver position iterations. |  |
| `solver_velocity_iterations` | `int` | Property: solver velocity iterations. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `active_joints` | property | `active_joints(self) -> list[PhysxArticulationJoint]` | Property: active joints. |  |
| `clone_links` | method | `clone_links(self) -> list[PhysxArticulationLinkComponent]` | Call clone links. |  |
| `compute_dense_jacobian` | method | `compute_dense_jacobian(self) -> np.ndarray` | Compute dense articulation Jacobian on CPU articulations. | Not supported for Direct GPU API articulations; use `PhysxGpuSystem.gpu_compute_articulation_jacobian()` and `cuda_articulation_jacobian` instead. |
| `compute_passive_force` | method | `compute_passive_force(self, gravity: bool=True, coriolis_and_centrifugal: bool=True) -> np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | Compute passive forces. |  |
| `create_fixed_tendon` | method | `create_fixed_tendon(self, link_chain: list[PhysxArticulationLinkComponent], coefficients: list[float], recip_coefficients: list[float], rest_length: float=0, offset: float=0, stiffness: float=0, damping: float=0, low: float=-3.4028234663852886e+38, high: float=3.4028234663852886e+38, limit_stiffness: float=0) -> None` | Create a fixed tendon. |  |
| `create_pinocchio_model` | staticmethod | `create_pinocchio_model(articulation: PhysxArticulation, gravity=[0, 0, -9.81]) -> sapien.pysapien_pinocchio.PinocchioModel` | Create a Pinocchio model. |  |
| `dof` | property | `dof(self) -> int` | Property: dof. |  |
| `find_joint_by_name` | method | `find_joint_by_name(self, name: str) -> PhysxArticulationJoint` | Call find joint by name. |  |
| `find_link_by_name` | method | `find_link_by_name(self, name: str) -> PhysxArticulationLinkComponent` | Call find link by name. |  |
| `get_active_joints` | method | `get_active_joints(self) -> list[PhysxArticulationJoint]` | Get active joints. |  |
| `get_dof` | method | `get_dof(self) -> int` | Get dof. |  |
| `get_gpu_index` | method | `get_gpu_index(self) -> int` | Return the PhysX GPU articulation/body state index; cache after gpu_init. |  |
| `get_joints` | method | `get_joints(self) -> list[PhysxArticulationJoint]` | Get joints. |  |
| `get_jacobian_shape` | method | `get_jacobian_shape(self) -> tuple[int, int]` | Return valid dense Jacobian `(rows, cols)` for this articulation. | Fixed base: `((link_count - 1) * 6, dof)`; floating base: `(6 + (link_count - 1) * 6, 6 + dof)`. Use to slice `cuda_articulation_jacobian`. |
| `get_link_incoming_joint_forces` | method | `get_link_incoming_joint_forces(self) -> np.ndarray[tuple[M, Literal[6]], np.dtype[np.float32]]` | Get link incoming joint forces. |  |
| `get_links` | method | `get_links(self) -> list[PhysxArticulationLinkComponent]` | Get links. |  |
| `get_name` | method | `get_name(self) -> str` | Get name. |  |
| `get_pose` | method | `get_pose(self) -> sapien.Pose` | Get pose. |  |
| `get_qacc` | method | `get_qacc(self) -> np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | Get qacc. |  |
| `get_qf` | method | `get_qf(self) -> np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | Get qf. |  |
| `get_qlimit` | method | `get_qlimit(self) -> np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | same as get_qlimit |  |
| `get_qlimits` | method | `get_qlimits(self) -> np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | Get qlimits. |  |
| `get_qpos` | method | `get_qpos(self) -> np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | Get qpos. |  |
| `get_qvel` | method | `get_qvel(self) -> np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | Get qvel. |  |
| `get_root` | method | `get_root(self) -> PhysxArticulationLinkComponent` | Get root. |  |
| `get_root_angular_velocity` | method | `get_root_angular_velocity(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Get root angular velocity. |  |
| `get_root_linear_velocity` | method | `get_root_linear_velocity(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Get root linear velocity. |  |
| `get_root_pose` | method | `get_root_pose(self) -> sapien.Pose` | Get root pose. |  |
| `get_sleep_threshold` | method | `get_sleep_threshold(self) -> float` | Get sleep threshold. |  |
| `get_solver_position_iterations` | method | `get_solver_position_iterations(self) -> int` | Get solver position iterations. |  |
| `get_solver_velocity_iterations` | method | `get_solver_velocity_iterations(self) -> int` | Get solver velocity iterations. |  |
| `gpu_index` | property | `gpu_index(self) -> int` | Property: gpu index. |  |
| `joints` | property | `joints(self) -> list[PhysxArticulationJoint]` | Property: joints. |  |
| `jacobian_shape` | property | `jacobian_shape(self) -> tuple[int, int]` | Valid dense Jacobian shape. | Same as `get_jacobian_shape()`. |
| `link_incoming_joint_forces` | property | `link_incoming_joint_forces(self) -> np.ndarray[tuple[M, Literal[6]], np.dtype[np.float32]]` | Property: link incoming joint forces. |  |
| `links` | property | `links(self) -> list[PhysxArticulationLinkComponent]` | Property: links. |  |
| `qlimit` | property | `qlimit(self) -> np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | Property: qlimit. |  |
| `qlimits` | property | `qlimits(self) -> np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | Property: qlimits. |  |
| `root` | property | `root(self) -> PhysxArticulationLinkComponent` | Property: root. |  |
| `set_name` | method | `set_name(self, name: str) -> None` | Set name. |  |
| `set_pose` | method | `set_pose(self, pose: sapien.Pose) -> None` | Set pose. |  |
| `set_qacc` | method | `set_qacc(self, qacc: np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]] \| list \| tuple) -> None` | Set qacc. |  |
| `set_qf` | method | `set_qf(self, qf: np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]] \| list \| tuple) -> None` | Set qf. |  |
| `set_qpos` | method | `set_qpos(self, qpos: np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]] \| list \| tuple) -> None` | Set qpos. |  |
| `set_qvel` | method | `set_qvel(self, qvel: np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]] \| list \| tuple) -> None` | Set qvel. |  |
| `set_root_angular_velocity` | method | `set_root_angular_velocity(self, velocity: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | Set root angular velocity. |  |
| `set_root_linear_velocity` | method | `set_root_linear_velocity(self, velocity: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | Set root linear velocity. |  |
| `set_root_pose` | method | `set_root_pose(self, pose: sapien.Pose) -> None` | Set root pose. |  |
| `set_sleep_threshold` | method | `set_sleep_threshold(self, threshold: float) -> None` | Set sleep threshold. |  |
| `set_solver_position_iterations` | method | `set_solver_position_iterations(self, count: int) -> None` | Set solver position iterations. |  |
| `set_solver_velocity_iterations` | method | `set_solver_velocity_iterations(self, count: int) -> None` | Set solver velocity iterations. |  |

## `sapien.physx.PhysxArticulationJoint`

- Use: Articulation joint; drive, limit, friction, armature.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `armature` | `np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | Property: armature. |  |
| `drive_target` | `np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | Property: drive target. |  |
| `drive_velocity_target` | `np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | Property: drive velocity target. |  |
| `friction` | `float` | Property: friction. |  |
| `limit` | `np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | Property: limit. |  |
| `limits` | `np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | Property: limits. |  |
| `name` | `str` | Property: name. |  |
| `pose_in_child` | `sapien.Pose` | Property: pose in child. |  |
| `pose_in_parent` | `sapien.Pose` | Property: pose in parent. |  |
| `type` | `Literal['fixed', 'revolute', 'revolute_unwrapped', 'prismatic', 'free']` | Property: type. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `child_link` | property | `child_link(self) -> PhysxArticulationLinkComponent` | Property: child link. |  |
| `damping` | property | `damping(self) -> float` | Property: damping. |  |
| `dof` | property | `dof(self) -> int` | Property: dof. |  |
| `drive_mode` | property | `drive_mode(self) -> Literal['force', 'acceleration']` | Property: drive mode. |  |
| `force_limit` | property | `force_limit(self) -> float` | Property: force limit. |  |
| `get_armature` | method | `get_armature(self) -> np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | Get armature. |  |
| `get_child_link` | method | `get_child_link(self) -> PhysxArticulationLinkComponent` | Get child link. |  |
| `get_damping` | method | `get_damping(self) -> float` | Get damping. |  |
| `get_dof` | method | `get_dof(self) -> int` | Get dof. |  |
| `get_drive_mode` | method | `get_drive_mode(self) -> Literal['force', 'acceleration']` | Get drive mode. |  |
| `get_drive_target` | method | `get_drive_target(self) -> np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | Get drive target. |  |
| `get_drive_velocity_target` | method | `get_drive_velocity_target(self) -> np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]]` | Get drive velocity target. |  |
| `get_force_limit` | method | `get_force_limit(self) -> float` | Get force limit. |  |
| `get_friction` | method | `get_friction(self) -> float` | Get friction. |  |
| `get_global_pose` | method | `get_global_pose(self) -> sapien.Pose` | Get global pose. |  |
| `get_limit` | method | `get_limit(self) -> np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | same as get_limits |  |
| `get_limits` | method | `get_limits(self) -> np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | Get limits. |  |
| `get_name` | method | `get_name(self) -> str` | Get name. |  |
| `get_parent_link` | method | `get_parent_link(self) -> PhysxArticulationLinkComponent` | Get parent link. |  |
| `get_pose_in_child` | method | `get_pose_in_child(self) -> sapien.Pose` | Get pose in child. |  |
| `get_pose_in_parent` | method | `get_pose_in_parent(self) -> sapien.Pose` | Get pose in parent. |  |
| `get_stiffness` | method | `get_stiffness(self) -> float` | Get stiffness. |  |
| `get_type` | method | `get_type(self) -> Literal['fixed', 'revolute', 'revolute_unwrapped', 'prismatic', 'free']` | Get type. |  |
| `global_pose` | property | `global_pose(self) -> sapien.Pose` | Property: global pose. |  |
| `parent_link` | property | `parent_link(self) -> PhysxArticulationLinkComponent` | Property: parent link. |  |
| `set_armature` | method | `set_armature(self, armature: np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]] \| list \| tuple) -> None` | Set armature. |  |
| `set_drive_properties` | method | `set_drive_properties(self, stiffness: float, damping: float, force_limit: float=3.4028234663852886e+38, mode: Literal['force', 'acceleration']='force') -> None` | Set drive properties. |  |
| `set_drive_property` | method | `set_drive_property(self, stiffness: float, damping: float, force_limit: float=3.4028234663852886e+38, mode: Literal['force', 'acceleration']='force') -> None` | same as set_drive_properties |  |
| `set_drive_target` | method | `set_drive_target(self, target: float) -> None<br>set_drive_target(self, target: np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]] \| list \| tuple) -> None` | Set drive target. |  |
| `set_drive_velocity_target` | method | `set_drive_velocity_target(self, velocity: float) -> None<br>set_drive_velocity_target(self, velocity: np.ndarray[tuple[M, Literal[1]], np.dtype[np.float32]] \| list \| tuple) -> None` | Set drive velocity target. |  |
| `set_friction` | method | `set_friction(self, friction: float) -> None` | Set friction. |  |
| `set_limit` | method | `set_limit(self, limit: np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]] \| list \| tuple) -> None` | same as set_limits |  |
| `set_limits` | method | `set_limits(self, limit: np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]] \| list \| tuple) -> None` | Set limits. |  |
| `set_name` | method | `set_name(self, name: str) -> None` | Set name. |  |
| `set_pose_in_child` | method | `set_pose_in_child(self, pose: sapien.Pose) -> None` | Set pose in child. |  |
| `set_pose_in_parent` | method | `set_pose_in_parent(self, pose: sapien.Pose) -> None` | Set pose in parent. |  |
| `set_type` | method | `set_type(self, type: Literal['fixed', 'revolute', 'revolute_unwrapped', 'prismatic', 'free']) -> None` | Set type. |  |
| `stiffness` | property | `stiffness(self) -> float` | Property: stiffness. |  |

## `sapien.physx.PhysxArticulationLinkComponent`

- Use: PhysX component of an articulation link.
- Bases: `PhysxRigidBodyComponent`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `articulation` | property | `articulation(self) -> PhysxArticulation` | Property: articulation. |  |
| `children` | property | `children(self) -> list[PhysxArticulationLinkComponent]` | Property: children. |  |
| `get_articulation` | method | `get_articulation(self) -> PhysxArticulation` | Get articulation. |  |
| `get_children` | method | `get_children(self) -> list[PhysxArticulationLinkComponent]` | Get children. |  |
| `get_gpu_pose_index` | method | `get_gpu_pose_index(self) -> int` | Return the GPU pose batch index; cache after gpu_init for rendering. |  |
| `get_index` | method | `get_index(self) -> int` | Get index. |  |
| `get_joint` | method | `get_joint(self) -> PhysxArticulationJoint` | Get joint. |  |
| `get_parent` | method | `get_parent(self) -> PhysxArticulationLinkComponent` | Get parent. |  |
| `gpu_pose_index` | property | `gpu_pose_index(self) -> int` | Property: gpu pose index. |  |
| `index` | property | `index(self) -> int` | Property: index. |  |
| `is_root` | property | `is_root(self) -> bool` | Bool: root. |  |
| `joint` | property | `joint(self) -> PhysxArticulationJoint` | Property: joint. |  |
| `parent` | property | `parent(self) -> PhysxArticulationLinkComponent` | Property: parent. |  |
| `put_to_sleep` | method | `put_to_sleep(self) -> None` | Call put to sleep. |  |
| `set_parent` | method | `set_parent(self, parent: PhysxArticulationLinkComponent) -> None` | Set parent. |  |
| `sleeping` | property | `sleeping(self) -> bool` | Property: sleeping. |  |
| `wake_up` | method | `wake_up(self) -> None` | Call wake up. |  |
| `__init__` | method | `__init__(self, parent: PhysxArticulationLinkComponent \| None=None) -> None` | Python special method. |  |

## `sapien.physx.PhysxBaseComponent`

- Use: PhysX physics API object.
- Bases: `sapien.Component`

## `sapien.physx.PhysxBodyConfig`

- Use: Default rigid-body solver/sleep configuration.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `sleep_threshold` | `float` | Property: sleep threshold. |  |
| `solver_position_iterations` | `int` | Property: solver position iterations. |  |
| `solver_velocity_iterations` | `int` | Property: solver velocity iterations. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__getstate__` | method | `__getstate__(self) -> tuple` | Python special method. |  |
| `__init__` | method | `__init__(self) -> None` | Python special method. |  |
| `__repr__` | method | `__repr__(self) -> str` | Python special method. |  |
| `__setstate__` | method | `__setstate__(self, arg0: tuple) -> None` | Python special method. |  |

## `sapien.physx.PhysxCollisionShape`

- Use: Collision shape base class; material, pose, offset, collision groups.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `contact_offset` | `float` | Property: contact offset. |  |
| `density` | `float` | Property: density. |  |
| `local_pose` | `sapien.Pose` | Property: local pose. |  |
| `min_patch_radius` | `float` | Property: min patch radius. |  |
| `patch_radius` | `float` | Property: patch radius. |  |
| `physical_material` | `PhysxMaterial` | Property: physical material. |  |
| `rest_offset` | `float` | Property: rest offset. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `collision_groups` | property | `collision_groups(self) -> Annotated[list[int], FixedSize(4)]` | Property: collision groups. |  |
| `get_collision_groups` | method | `get_collision_groups(self) -> Annotated[list[int], FixedSize(4)]` | Get collision groups. |  |
| `get_contact_offset` | method | `get_contact_offset(self) -> float` | Get contact offset. |  |
| `get_density` | method | `get_density(self) -> float` | Get density. |  |
| `get_local_pose` | method | `get_local_pose(self) -> sapien.Pose` | Get local pose. |  |
| `get_min_patch_radius` | method | `get_min_patch_radius(self) -> float` | Get min patch radius. |  |
| `get_patch_radius` | method | `get_patch_radius(self) -> float` | Get patch radius. |  |
| `get_physical_material` | method | `get_physical_material(self) -> PhysxMaterial` | Get physical material. |  |
| `get_rest_offset` | method | `get_rest_offset(self) -> float` | Get rest offset. |  |
| `set_collision_groups` | method | `set_collision_groups(self, groups: Annotated[list[int], FixedSize(4)]) -> None` | collision groups determine the collision behavior of objects. Let A.gx denote the collision group x of collision shape A. Collision shape A and B will collide iff the... |  |
| `set_contact_offset` | method | `set_contact_offset(self, offset: float) -> None` | Set contact offset. |  |
| `set_density` | method | `set_density(self, density: float) -> None` | Set density. |  |
| `set_local_pose` | method | `set_local_pose(self, pose: sapien.Pose) -> None` | Set local pose. |  |
| `set_min_patch_radius` | method | `set_min_patch_radius(self, radius: float) -> None` | Set min patch radius. |  |
| `set_patch_radius` | method | `set_patch_radius(self, radius: float) -> None` | Set patch radius. |  |
| `set_physical_material` | method | `set_physical_material(self, material: PhysxMaterial) -> None` | Set physical material. |  |
| `set_rest_offset` | method | `set_rest_offset(self, offset: float) -> None` | Set rest offset. |  |

## `sapien.physx.PhysxCollisionShapeBox`

- Use: Concrete PhysX collision shape.
- Bases: `PhysxCollisionShape`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_half_size` | method | `get_half_size(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Get half size. |  |
| `half_size` | property | `half_size(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: half size. |  |
| `__init__` | method | `__init__(self, half_size: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, material: PhysxMaterial) -> None` | Python special method. |  |

## `sapien.physx.PhysxCollisionShapeCapsule`

- Use: Concrete PhysX collision shape.
- Bases: `PhysxCollisionShape`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_half_length` | method | `get_half_length(self) -> float` | Get half length. |  |
| `get_radius` | method | `get_radius(self) -> float` | Get radius. |  |
| `half_length` | property | `half_length(self) -> float` | Property: half length. |  |
| `radius` | property | `radius(self) -> float` | Property: radius. |  |
| `__init__` | method | `__init__(self, radius: float, half_length: float, material: PhysxMaterial) -> None` | Python special method. |  |

## `sapien.physx.PhysxCollisionShapeConvexMesh`

- Use: Concrete PhysX collision shape.
- Bases: `PhysxCollisionShape`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_scale` | method | `get_scale(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Get scale. |  |
| `get_triangles` | method | `get_triangles(self) -> np.ndarray[np.uint32[M, 3]]` | Get triangles. |  |
| `get_vertices` | method | `get_vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | Get vertices. |  |
| `load_multiple` | staticmethod | `load_multiple(filename: str, scale: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, material: PhysxMaterial) -> list[PhysxCollisionShapeConvexMesh]` | Load multiple. |  |
| `scale` | property | `scale(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: scale. |  |
| `triangles` | property | `triangles(self) -> np.ndarray[np.uint32[M, 3]]` | Property: triangles. |  |
| `vertices` | property | `vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | Property: vertices. |  |
| `__init__` | method | `__init__(self, filename: str, scale: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, material: PhysxMaterial) -> None` | Python special method. |  |

## `sapien.physx.PhysxCollisionShapeCylinder`

- Use: Concrete PhysX collision shape.
- Bases: `PhysxCollisionShape`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_half_length` | method | `get_half_length(self) -> float` | Get half length. |  |
| `get_radius` | method | `get_radius(self) -> float` | Get radius. |  |
| `half_length` | property | `half_length(self) -> float` | Property: half length. |  |
| `radius` | property | `radius(self) -> float` | Property: radius. |  |
| `__init__` | method | `__init__(self, radius: float, half_length: float, material: PhysxMaterial) -> None` | Python special method. |  |

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

- Use: Concrete PhysX collision shape.
- Bases: `PhysxCollisionShape`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__init__` | method | `__init__(self, material: PhysxMaterial) -> None` | Python special method. |  |

## `sapien.physx.PhysxCollisionShapeSphere`

- Use: Concrete PhysX collision shape.
- Bases: `PhysxCollisionShape`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_radius` | method | `get_radius(self) -> float` | Get radius. |  |
| `radius` | property | `radius(self) -> float` | Property: radius. |  |
| `__init__` | method | `__init__(self, radius: float, material: PhysxMaterial) -> None` | Python special method. |  |

## `sapien.physx.PhysxCollisionShapeTriangleMesh`

- Use: Concrete PhysX collision shape.
- Bases: `PhysxCollisionShape`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_scale` | method | `get_scale(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Get scale. |  |
| `get_triangles` | method | `get_triangles(self) -> np.ndarray[np.uint32[M, 3]]` | Get triangles. |  |
| `get_vertices` | method | `get_vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | Get vertices. |  |
| `scale` | property | `scale(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: scale. |  |
| `triangles` | property | `triangles(self) -> np.ndarray[np.uint32[M, 3]]` | Property: triangles. |  |
| `vertices` | property | `vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | Property: vertices. |  |
| `__init__` | method | `__init__(self, filename: str, scale: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, material: PhysxMaterial, sdf: bool=False, sdf_config: PhysxSDFConfig \| None=None) -> None<br>__init__(self, vertices: np.ndarray[Literal[3], np.dtype[np.float32]], triangles: np.ndarray[Literal[3], np.dtype[np.float32]], scale: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple=..., material: PhysxMaterial \| None=..., sdf: bool=False, sdf_config: PhysxSDFConfig \| None=None) -> None` | Python special method. |  |

## `sapien.physx.PhysxContact`

- Use: PhysX physics API object.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `bodies` | property | `bodies(self) -> Annotated[list[PhysxRigidBaseComponent], FixedSize(2)]` | Property: bodies. |  |
| `points` | property | `points(self) -> list[PhysxContactPoint]` | Property: points. |  |
| `shapes` | property | `shapes(self) -> Annotated[list[PhysxCollisionShape], FixedSize(2)]` | Property: shapes. |  |
| `__repr__` | method | `__repr__(self) -> str` | Python special method. |  |

## `sapien.physx.PhysxContactPoint`

- Use: PhysX physics API object.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `impulse` | property | `impulse(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: impulse. |  |
| `normal` | property | `normal(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: normal. |  |
| `position` | property | `position(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: position. |  |
| `separation` | property | `separation(self) -> float` | Property: separation. |  |

## `sapien.physx.PhysxCpuSystem`

- Use: PhysX CPU system; contacts/raycast/pack/unpack.
- Bases: `PhysxSystem`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_contacts` | method | `get_contacts(self) -> list[PhysxContact]` | Get contacts. |  |
| `pack` | method | `pack(self) -> bytes` | Call pack. |  |
| `raycast` | method | `raycast(self, position: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, direction: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, distance: float) -> PhysxRayHit` | Casts a ray and returns the closest hit. Returns None if no hit |  |
| `unpack` | method | `unpack(self, data: bytes) -> None` | Call unpack. |  |
| `__init__` | method | `__init__(self) -> None` | Python special method. |  |

## `sapien.physx.PhysxDistanceJointComponent`

- Use: PhysX physics API object.
- Bases: `PhysxJointComponent`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `distance` | property | `distance(self) -> float` | Property: distance. |  |
| `get_distance` | method | `get_distance(self) -> float` | Get distance. |  |
| `set_limit` | method | `set_limit(self, low: float, high: float, stiffness: float=0.0, damping: float=0.0) -> None` | Set limit. |  |
| `__init__` | method | `__init__(self, body: PhysxRigidBodyComponent) -> None` | Python special method. |  |

## `sapien.physx.PhysxDriveComponent`

- Use: PhysX physics API object.
- Bases: `PhysxJointComponent`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `drive_target` | `sapien.Pose` | Property: drive target. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_drive_property_slerp` | method | `get_drive_property_slerp(self) -> tuple[float, float, float, Literal['force', 'acceleration']]` | Get drive property slerp. |  |
| `get_drive_property_swing` | method | `get_drive_property_swing(self) -> tuple[float, float, float, Literal['force', 'acceleration']]` | Get drive property swing. |  |
| `get_drive_property_twist` | method | `get_drive_property_twist(self) -> tuple[float, float, float, Literal['force', 'acceleration']]` | Get drive property twist. |  |
| `get_drive_property_x` | method | `get_drive_property_x(self) -> tuple[float, float, float, Literal['force', 'acceleration']]` | Get drive property x. |  |
| `get_drive_property_y` | method | `get_drive_property_y(self) -> tuple[float, float, float, Literal['force', 'acceleration']]` | Get drive property y. |  |
| `get_drive_property_z` | method | `get_drive_property_z(self) -> tuple[float, float, float, Literal['force', 'acceleration']]` | Get drive property z. |  |
| `get_drive_target` | method | `get_drive_target(self) -> sapien.Pose` | Get drive target. |  |
| `get_drive_velocity_target` | method | `get_drive_velocity_target(self) -> tuple[np.ndarray[Literal[3], np.dtype[np.float32]], np.ndarray[Literal[3], np.dtype[np.float32]]]` | Get drive velocity target. |  |
| `get_limit_cone` | method | `get_limit_cone(self) -> tuple[float, float, float, float]` | Get limit cone. |  |
| `get_limit_pyramid` | method | `get_limit_pyramid(self) -> tuple[float, float, float, float, float, float]` | Get limit pyramid. |  |
| `get_limit_twist` | method | `get_limit_twist(self) -> tuple[float, float, float, float]` | Get limit twist. |  |
| `get_limit_x` | method | `get_limit_x(self) -> tuple[float, float, float, float]` | Get limit x. |  |
| `get_limit_y` | method | `get_limit_y(self) -> tuple[float, float, float, float]` | Get limit y. |  |
| `get_limit_z` | method | `get_limit_z(self) -> tuple[float, float, float, float]` | Get limit z. |  |
| `set_drive_property_slerp` | method | `set_drive_property_slerp(self, stiffness: float, damping: float, force_limit: float=3.4028234663852886e+38, mode: Literal['force', 'acceleration']='force') -> None` | Set drive property slerp. |  |
| `set_drive_property_swing` | method | `set_drive_property_swing(self, stiffness: float, damping: float, force_limit: float=3.4028234663852886e+38, mode: Literal['force', 'acceleration']='force') -> None` | Set drive property swing. |  |
| `set_drive_property_twist` | method | `set_drive_property_twist(self, stiffness: float, damping: float, force_limit: float=3.4028234663852886e+38, mode: Literal['force', 'acceleration']='force') -> None` | Set drive property twist. |  |
| `set_drive_property_x` | method | `set_drive_property_x(self, stiffness: float, damping: float, force_limit: float=3.4028234663852886e+38, mode: Literal['force', 'acceleration']='force') -> None` | Set drive property x. |  |
| `set_drive_property_y` | method | `set_drive_property_y(self, stiffness: float, damping: float, force_limit: float=3.4028234663852886e+38, mode: Literal['force', 'acceleration']='force') -> None` | Set drive property y. |  |
| `set_drive_property_z` | method | `set_drive_property_z(self, stiffness: float, damping: float, force_limit: float=3.4028234663852886e+38, mode: Literal['force', 'acceleration']='force') -> None` | Set drive property z. |  |
| `set_drive_target` | method | `set_drive_target(self, target: sapien.Pose) -> None` | Set drive target. |  |
| `set_drive_velocity_target` | method | `set_drive_velocity_target(self, linear: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, angular: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | Set drive velocity target. |  |
| `set_limit_cone` | method | `set_limit_cone(self, angle_y: float, angle_z: float, stiffness: float=0.0, damping: float=0.0) -> None` | Set limit cone. |  |
| `set_limit_pyramid` | method | `set_limit_pyramid(self, low_y: float, high_y: float, low_z: float, high_z: float, stiffness: float=0.0, damping: float=0.0) -> None` | Set limit pyramid. |  |
| `set_limit_twist` | method | `set_limit_twist(self, low: float, high: float, stiffness: float=0.0, damping: float=0.0) -> None` | Set limit twist. |  |
| `set_limit_x` | method | `set_limit_x(self, low: float, high: float, stiffness: float=0.0, damping: float=0.0) -> None` | Set limit x. |  |
| `set_limit_y` | method | `set_limit_y(self, low: float, high: float, stiffness: float=0.0, damping: float=0.0) -> None` | Set limit y. |  |
| `set_limit_z` | method | `set_limit_z(self, low: float, high: float, stiffness: float=0.0, damping: float=0.0) -> None` | Set limit z. |  |
| `__init__` | method | `__init__(self, body: PhysxRigidBodyComponent) -> None` | Python special method. |  |

## `sapien.physx.PhysxEngine`

- Use: PhysX physics API object.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__init__` | method | `__init__(self, tolerance_length: float, tolerance_speed: float) -> None` | Python special method. |  |

## `sapien.physx.PhysxGearComponent`

- Use: PhysX physics API object.
- Bases: `PhysxJointComponent`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `gear_ratio` | `float` | Property: gear ratio. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `enable_hinges` | method | `enable_hinges(self) -> None` | Enable hinges. |  |
| `get_gear_ratio` | method | `get_gear_ratio(self) -> float` | Get gear ratio. |  |
| `is_hinges_enabled` | property | `is_hinges_enabled(self) -> bool` | Bool: hinges enabled. |  |
| `set_gear_ratio` | method | `set_gear_ratio(self, ratio: float) -> None` | Set gear ratio. |  |
| `__init__` | method | `__init__(self, body: PhysxRigidBodyComponent) -> None` | Python special method. |  |

## `sapien.physx.PhysxJointComponent`

- Use: PhysX physics API object.
- Bases: `PhysxBaseComponent`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `parent` | `PhysxRigidBaseComponent` | Property: parent. |  |
| `pose_in_child` | `sapien.Pose` | Property: pose in child. |  |
| `pose_in_parent` | `sapien.Pose` | Property: pose in parent. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_parent` | method | `get_parent(self) -> PhysxRigidBaseComponent` | Get parent. |  |
| `get_pose_in_child` | method | `get_pose_in_child(self) -> sapien.Pose` | Get pose in child. |  |
| `get_pose_in_parent` | method | `get_pose_in_parent(self) -> sapien.Pose` | Get pose in parent. |  |
| `get_relative_pose` | method | `get_relative_pose(self) -> sapien.Pose` | Get relative pose. |  |
| `relative_pose` | property | `relative_pose(self) -> sapien.Pose` | Property: relative pose. |  |
| `set_inv_inertia_scales` | method | `set_inv_inertia_scales(self, scale0: float, scale1: float) -> None` | Set inv inertia scales. |  |
| `set_inv_mass_scales` | method | `set_inv_mass_scales(self, scale0: float, scale1: float) -> None` | Set inv mass scales. |  |
| `set_parent` | method | `set_parent(self, parent: PhysxRigidBaseComponent) -> None` | Set parent. |  |
| `set_pose_in_child` | method | `set_pose_in_child(self, pose: sapien.Pose) -> None` | Set pose in child. |  |
| `set_pose_in_parent` | method | `set_pose_in_parent(self, pose: sapien.Pose) -> None` | Set pose in parent. |  |

## `sapien.physx.PhysxMaterial`

- Use: Physical material; static/dynamic friction and restitution.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `dynamic_friction` | `float` | Property: dynamic friction. |  |
| `restitution` | `float` | Property: restitution. |  |
| `static_friction` | `float` | Property: static friction. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_dynamic_friction` | method | `get_dynamic_friction(self) -> float` | Get dynamic friction. |  |
| `get_restitution` | method | `get_restitution(self) -> float` | Get restitution. |  |
| `get_static_friction` | method | `get_static_friction(self) -> float` | Get static friction. |  |
| `set_dynamic_friction` | method | `set_dynamic_friction(self, friction: float) -> None` | Set dynamic friction. |  |
| `set_restitution` | method | `set_restitution(self, restitution: float) -> None` | Set restitution. |  |
| `set_static_friction` | method | `set_static_friction(self, friction: float) -> None` | Set static friction. |  |
| `__init__` | method | `__init__(self, static_friction: float, dynamic_friction: float, restitution: float) -> None` | Python special method. |  |

## `sapien.physx.PhysxRayHit`

- Use: PhysX physics API object.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `component` | property | `component(self) -> PhysxRigidBaseComponent` | Property: component. |  |
| `distance` | property | `distance(self) -> float` | Property: distance. |  |
| `normal` | property | `normal(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: normal. |  |
| `position` | property | `position(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: position. |  |
| `shape` | property | `shape(self) -> PhysxCollisionShape` | Property: shape. |  |
| `__repr__` | method | `__repr__(self) -> str` | Python special method. |  |

## `sapien.physx.PhysxRigidBaseComponent`

- Use: Rigid-body collision shape container base class.
- Bases: `PhysxBaseComponent`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `attach` | method | `attach(self, collision_shape: PhysxCollisionShape) -> PhysxRigidBaseComponent` | Call attach. |  |
| `collision_shapes` | property | `collision_shapes(self) -> list[PhysxCollisionShape]` | Property: collision shapes. |  |
| `compute_global_aabb_tight` | method | `compute_global_aabb_tight(self) -> np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.float32]]` | Compute a tight global AABB. |  |
| `get_collision_shapes` | method | `get_collision_shapes(self) -> list[PhysxCollisionShape]` | Get collision shapes. |  |
| `get_global_aabb_fast` | method | `get_global_aabb_fast(self) -> np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.float32]]` | Get global AABB fast. |  |
| `_physx_pointer` | property | `_physx_pointer(self) -> int` | Property: physx pointer. |  |

## `sapien.physx.PhysxRigidBodyComponent`

- Use: Rigid-body general mass, damping, velocity, force interface.
- Bases: `PhysxRigidBaseComponent`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `angular_damping` | `float` | Property: angular damping. |  |
| `cmass_local_pose` | `sapien.Pose` | Property: cmass local pose. |  |
| `disable_gravity` | `bool` | Disable gravity. |  |
| `inertia` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: inertia. |  |
| `linear_damping` | `float` | Property: linear damping. |  |
| `mass` | `float` | Property: mass. |  |
| `max_contact_impulse` | `float` | Property: max contact impulse. |  |
| `max_depenetration_velocity` | `float` | Property: max depenetration velocity. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `add_force_at_point` | method | `add_force_at_point(self, force: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, point: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, mode: Literal['force', 'acceleration', 'velocity_change', 'impulse']='force') -> None` | Add/create force at point. |  |
| `add_force_torque` | method | `add_force_torque(self, force: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, torque: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, mode: Literal['force', 'acceleration', 'velocity_change', 'impulse']='force') -> None` | Add/create force torque. |  |
| `angular_velocity` | property | `angular_velocity(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: angular velocity. |  |
| `auto_compute_mass` | property | `auto_compute_mass(self) -> bool` | Property: auto compute mass. |  |
| `get_angular_damping` | method | `get_angular_damping(self) -> float` | Get angular damping. |  |
| `get_angular_velocity` | method | `get_angular_velocity(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Get angular velocity. |  |
| `get_auto_compute_mass` | method | `get_auto_compute_mass(self) -> bool` | Get auto compute mass. |  |
| `get_cmass_local_pose` | method | `get_cmass_local_pose(self) -> sapien.Pose` | Get cmass local pose. |  |
| `get_disable_gravity` | method | `get_disable_gravity(self) -> bool` | Get disable gravity. |  |
| `get_inertia` | method | `get_inertia(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Get inertia. |  |
| `get_linear_damping` | method | `get_linear_damping(self) -> float` | Get linear damping. |  |
| `get_linear_velocity` | method | `get_linear_velocity(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Get linear velocity. |  |
| `get_mass` | method | `get_mass(self) -> float` | Get mass. |  |
| `get_max_contact_impulse` | method | `get_max_contact_impulse(self) -> float` | Get max contact impulse. |  |
| `get_max_depenetration_velocity` | method | `get_max_depenetration_velocity(self) -> float` | Get max depenetration velocity. |  |
| `linear_velocity` | property | `linear_velocity(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: linear velocity. |  |
| `set_angular_damping` | method | `set_angular_damping(self, damping: float) -> None` | Set angular damping. |  |
| `set_cmass_local_pose` | method | `set_cmass_local_pose(self, pose: sapien.Pose) -> None` | Set cmass local pose. |  |
| `set_disable_gravity` | method | `set_disable_gravity(self, disable: bool) -> None` | Set disable gravity. |  |
| `set_inertia` | method | `set_inertia(self, inertia: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | Set inertia. |  |
| `set_linear_damping` | method | `set_linear_damping(self, damping: float) -> None` | Set linear damping. |  |
| `set_mass` | method | `set_mass(self, mass: float) -> None` | Set mass. |  |
| `set_max_contact_impulse` | method | `set_max_contact_impulse(self, impulse: float) -> None` | Set max contact impulse. |  |
| `set_max_depenetration_velocity` | method | `set_max_depenetration_velocity(self, velocity: float) -> None` | Set max depenetration velocity. |  |

## `sapien.physx.PhysxRigidDynamicComponent`

- Use: Dynamic/kinematic rigid-body component.
- Bases: `PhysxRigidBodyComponent`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `angular_velocity` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: angular velocity. |  |
| `gyroscopic_forces` | `bool` | Property: gyroscopic forces. |  |
| `kinematic` | `bool` | Property: kinematic. |  |
| `kinematic_target` | `sapien.Pose` | Property: kinematic target. |  |
| `linear_velocity` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: linear velocity. |  |
| `sleep_threshold` | `float` | Property: sleep threshold. |  |
| `solver_position_iterations` | `int` | Property: solver position iterations. |  |
| `solver_velocity_iterations` | `int` | Property: solver velocity iterations. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_angular_velocity` | method | `get_angular_velocity(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Get angular velocity. |  |
| `get_gpu_index` | method | `get_gpu_index(self) -> int` | Return the PhysX GPU articulation/body state index; cache after gpu_init. |  |
| `get_gpu_pose_index` | method | `get_gpu_pose_index(self) -> int` | Return the GPU pose batch index; cache after gpu_init for rendering. |  |
| `get_kinematic` | method | `get_kinematic(self) -> bool` | Get kinematic. |  |
| `get_kinematic_target` | method | `get_kinematic_target(self) -> sapien.Pose` | Get kinematic target. |  |
| `get_linear_velocity` | method | `get_linear_velocity(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Get linear velocity. |  |
| `get_locked_motion_axes` | method | `get_locked_motion_axes(self) -> Annotated[list[bool], FixedSize(6)]` | Get locked motion axes. |  |
| `get_sleep_threshold` | method | `get_sleep_threshold(self) -> float` | Get sleep threshold. |  |
| `get_solver_position_iterations` | method | `get_solver_position_iterations(self) -> int` | Get solver position iterations. |  |
| `get_solver_velocity_iterations` | method | `get_solver_velocity_iterations(self) -> int` | Get solver velocity iterations. |  |
| `gpu_index` | property | `gpu_index(self) -> int` | Property: gpu index. |  |
| `gpu_pose_index` | property | `gpu_pose_index(self) -> int` | Property: gpu pose index. |  |
| `is_sleeping` | property | `is_sleeping(self) -> bool` | Bool: sleeping. |  |
| `locked_motion_axes` | property | `locked_motion_axes(self) -> Annotated[list[bool], FixedSize(6)]` | Property: locked motion axes. |  |
| `put_to_sleep` | method | `put_to_sleep(self) -> None` | Call put to sleep. |  |
| `set_angular_velocity` | method | `set_angular_velocity(self, velocity: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | Set angular velocity. |  |
| `set_gyroscopic_forces` | method | `set_gyroscopic_forces(self, kinematic: bool) -> None` | Set gyroscopic forces. |  |
| `set_kinematic` | method | `set_kinematic(self, kinematic: bool) -> None` | Set kinematic. |  |
| `set_kinematic_target` | method | `set_kinematic_target(self, target: sapien.Pose) -> None` | Set kinematic target. |  |
| `set_linear_velocity` | method | `set_linear_velocity(self, velocity: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | Set linear velocity. |  |
| `set_locked_motion_axes` | method | `set_locked_motion_axes(self, axes: Annotated[list[bool], FixedSize(6)]) -> None` | set some motion axes of the dynamic rigid body to be locked Args: axes: list of 6 true/false values indicating whether which of the 6 DOFs of the body is locked. The o... |  |
| `set_sleep_threshold` | method | `set_sleep_threshold(self, threshold: float) -> None` | Set sleep threshold. |  |
| `set_solver_position_iterations` | method | `set_solver_position_iterations(self, count: int) -> None` | Set solver position iterations. |  |
| `set_solver_velocity_iterations` | method | `set_solver_velocity_iterations(self, count: int) -> None` | Set solver velocity iterations. |  |
| `wake_up` | method | `wake_up(self) -> None` | Call wake up. |  |
| `__init__` | method | `__init__(self) -> None` | Python special method. |  |

## `sapien.physx.PhysxRigidStaticComponent`

- Use: Static rigid-body component.
- Bases: `PhysxRigidBaseComponent`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__init__` | method | `__init__(self) -> None` | Python special method. |  |

## `sapien.physx.PhysxSDFConfig`

- Use: SDF mesh collision cooking configuration.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `bitsPerSubgridPixel` | `int` | Property: bitsPerSubgridPixel. |  |
| `bits_per_subgrid_pixel` | `int` | Property: bits per subgrid pixel. |  |
| `enableRemeshing` | `bool` | Property: enableRemeshing. |  |
| `enable_remeshing` | `bool` | Enable remeshing. |  |
| `margin` | `float` | Property: margin. |  |
| `narrowBandThickness` | `float` | Property: narrowBandThickness. |  |
| `narrow_band_thickness` | `float` | Property: narrow band thickness. |  |
| `num_threads_for_construction` | `int` | Property: num threads for construction. |  |
| `resolution` | `int` | Property: resolution. |  |
| `spacing` | `float` | Property: spacing. |  |
| `subgridSize` | `int` | Property: subgridSize. |  |
| `subgrid_size` | `int` | Property: subgrid size. |  |
| `triangleCountReductionFactor` | `float` | Property: triangleCountReductionFactor. |  |
| `triangle_count_reduction_factor` | `float` | Property: triangle count reduction factor. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__getstate__` | method | `__getstate__(self) -> tuple` | Python special method. |  |
| `__init__` | method | `__init__(self) -> None` | Python special method. |  |
| `__repr__` | method | `__repr__(self) -> str` | Python special method. |  |
| `__setstate__` | method | `__setstate__(self, arg0: tuple) -> None` | Python special method. |  |

## `sapien.physx.PhysxSceneConfig`

- Use: Global PhysX scene configuration; set before creating the system.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `bounce_threshold` | `float` | Property: bounce threshold. |  |
| `cpu_workers` | `int` | Property: cpu workers. |  |
| `enable_ccd` | `bool` | Enable ccd. |  |
| `enable_enhanced_determinism` | `bool` | Enable enhanced determinism. |  |
| `enable_friction_every_iteration` | `bool` | Enable friction every iteration. |  |
| `enable_pcm` | `bool` | Enable pcm. |  |
| `enable_tgs` | `bool` | Enable tgs. |  |
| `friction_correlation_distance` | `float` | Property: friction correlation distance. |  |
| `friction_offset_threshold` | `float` | Property: friction offset threshold. |  |
| `gpu_broadphase_env_id_bits` | `int` | Property: gpu broadphase env id bits. |  |
| `gpu_broadphase_nb_bits_env_id_x` | `int` | Property: gpu broadphase nb bits env id x. |  |
| `gpu_broadphase_nb_bits_env_id_y` | `int` | Property: gpu broadphase nb bits env id y. |  |
| `gpu_broadphase_nb_bits_env_id_z` | `int` | Property: gpu broadphase nb bits env id z. |  |
| `gravity` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: gravity. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__getstate__` | method | `__getstate__(self) -> tuple` | Python special method. |  |
| `__init__` | method | `__init__(self) -> None` | Python special method. |  |
| `__repr__` | method | `__repr__(self) -> str` | Python special method. |  |
| `__setstate__` | method | `__setstate__(self, arg0: tuple) -> None` | Python special method. |  |

## `sapien.physx.PhysxShapeConfig`

- Use: Default collision shape offset configuration.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `contact_offset` | `float` | Property: contact offset. |  |
| `rest_offset` | `float` | Property: rest offset. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__getstate__` | method | `__getstate__(self) -> tuple` | Python special method. |  |
| `__init__` | method | `__init__(self) -> None` | Python special method. |  |
| `__repr__` | method | `__repr__(self) -> str` | Python special method. |  |
| `__setstate__` | method | `__setstate__(self, arg0: tuple) -> None` | Python special method. |  |

## `sapien.physx.PhysxSystem`

- Use: PhysX system base class; manages timestep, component list, scene collision id.
- Bases: `sapien.System`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `scene_collision_id` | `int` | Property: scene collision id. |  |
| `timestep` | `float` | Property: timestep. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `articulation_link_components` | property | `articulation_link_components(self) -> list[PhysxArticulationLinkComponent]` | Property: articulation link components. |  |
| `config` | property | `config(self) -> PhysxSceneConfig` | Property: config. |  |
| `get_articulation_link_components` | method | `get_articulation_link_components(self) -> list[PhysxArticulationLinkComponent]` | Get articulation link components. |  |
| `get_config` | method | `get_config(self) -> PhysxSceneConfig` | Get config. |  |
| `get_rigid_dynamic_components` | method | `get_rigid_dynamic_components(self) -> list[PhysxRigidDynamicComponent]` | Get rigid dynamic components. |  |
| `get_rigid_static_components` | method | `get_rigid_static_components(self) -> list[PhysxRigidStaticComponent]` | Get rigid static components. |  |
| `get_scene_collision_id` | method | `get_scene_collision_id(self) -> int` | Get scene collision id. |  |
| `get_timestep` | method | `get_timestep(self) -> float` | Get timestep. |  |
| `rigid_dynamic_components` | property | `rigid_dynamic_components(self) -> list[PhysxRigidDynamicComponent]` | Property: rigid dynamic components. |  |
| `rigid_static_components` | property | `rigid_static_components(self) -> list[PhysxRigidStaticComponent]` | Property: rigid static components. |  |
| `set_scene_collision_id` | method | `set_scene_collision_id(self, id: int) -> None` | Set scene collision id. |  |
| `set_timestep` | method | `set_timestep(self, timestep: float) -> None` | Set the simulation timestep. |  |
| `__init__` | method | `__init__(self) -> None` | Python special method. |  |