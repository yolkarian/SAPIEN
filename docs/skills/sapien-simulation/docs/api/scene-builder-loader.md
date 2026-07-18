# Scene / Builder / Loader API

Agent-facing high-level construction wrappers. Prefer these for ordinary Python code; direct component APIs are in `physx.md` and `render.md`.

Critical GPU-agent rule: assign Scene environment id (`scene.set_environment_id(...)` or `get_or_assign_environment_id()`) before adding bodies. Parse identical URDF once and reuse cached builder via `builder.set_scene(scene)` before each build. Add static terrain/height fields before `PhysxGpuSystem.gpu_init()`.

## Scene wrapper API

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien`
- Source files: `python/py_package/wrapper/scene.py`

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.Scene` | `_Scene` | Entity/system container; high-level wrapper also provides builder, lights, camera, ground, and other convenience functions. |  |
| `sapien.Widget` |  | Compatibility placeholder; subclassing is no longer needed. |  |

## `sapien.Scene`

- Use: Entity/system container; high-level wrapper also provides builder, lights, camera, ground, and other convenience functions.
- Bases: `_Scene`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `environment_id` | `` |  |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `add_area_light_for_ray_tracing` | method | `add_area_light_for_ray_tracing(self, pose: sapien.Pose, color, half_width: float, half_height: float)` |  |  |
| `add_camera` | method | `add_camera(self, name, width: int, height: int, fovy: float, near: float, far: float) -> RenderCameraComponent` | Create a free camera entity/component. |  |
| `add_directional_light` | method | `add_directional_light(self, direction, color, shadow=False, position=[0, 0, 0], shadow_scale=10.0, shadow_near=-10.0, shadow_far=10.0, shadow_map_size=2048)` |  |  |
| `add_ground` | method | `add_ground(self, altitude, render=True, material=None, render_material=None, render_half_size=[10, 10])` | Build a static ground actor. |  |
| `add_heightfield` | method | `add_heightfield(self, height_field, row_scale: float, column_scale: float \| None=None, height_scale: float=1.0, pose=None, render=True, material=None, render_material=None, name='heightfield')` | Add static z-up height-field terrain with optional render mesh. | `height_field` is 2D int16-compatible; public coordinates map rows to +x, columns to +y, samples to +z; add before `gpu_init()` in GPU workflows. |
| `add_mounted_camera` | method | `add_mounted_camera(self, name, mount, pose, width, height, fovy, near, far) -> RenderCameraComponent` | Attach a camera component to an existing entity. |  |
| `add_point_light` | method | `add_point_light(self, position, color, shadow=False, shadow_near=0.1, shadow_far=10.0, shadow_map_size=2048)` |  |  |
| `add_spot_light` | method | `add_spot_light(self, position, direction, inner_fov: float, outer_fov: float, color, shadow=False, shadow_near=0.1, shadow_far=10.0, shadow_map_size=2048)` |  |  |
| `ambient_light` | property | `ambient_light(self)<br>ambient_light(self, color)` |  |  |
| `create_actor_builder` | method | `create_actor_builder(self)` | Create and bind an ActorBuilder to the current Scene. |  |
| `create_articulation_builder` | method | `create_articulation_builder(self)` | Create and bind an ArticulationBuilder to the current Scene. |  |
| `create_connection` | method | `create_connection(self, body0: Optional[Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent]], pose0: sapien.Pose, body1: Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent], pose1: sapien.Pose)` | Create a connection. |  |
| `create_drive` | method | `create_drive(self, body0: Optional[Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent]], pose0: sapien.Pose, body1: Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent], pose1: sapien.Pose)` | Create a drive. |  |
| `create_gear` | method | `create_gear(self, body0: Optional[Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent]], pose0: sapien.Pose, body1: Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent], pose1: sapien.Pose)` | Create a gear. |  |
| `create_physical_material` | method | `create_physical_material(self, static_friction: float, dynamic_friction: float, restitution: float)` | Create a physical material. |  |
| `create_urdf_loader` | method | `create_urdf_loader(self)` | Create and bind a URDFLoader to the current Scene. |  |
| `create_viewer` | method | `create_viewer(self)` | Create a viewer. |  |
| `get_all_actors` | method | `get_all_actors(self)` |  |  |
| `get_all_articulations` | method | `get_all_articulations(self)` |  |  |
| `get_cameras` | method | `get_cameras(self)` |  |  |
| `get_contacts` | method | `get_contacts(self)` |  |  |
| `get_environment_id` | method | `get_environment_id(self) -> int \| None` | Return the already assigned PhysX GPU broadphase environment ID. Returns ``None`` if no environment ID has been assigned yet. This method has no side effects; use :met... |  |
| `get_mounted_cameras` | method | `get_mounted_cameras(self)` |  |  |
| `get_or_assign_environment_id` | method | `get_or_assign_environment_id(self) -> int` | Return this scene's environment ID, assigning a unique one if needed. |  |
| `get_timestep` | method | `get_timestep(self)` |  |  |
| `remove_actor` | method | `remove_actor(self, actor)` |  |  |
| `remove_articulation` | method | `remove_articulation(self, articulation)` |  |  |
| `remove_camera` | method | `remove_camera(self, camera)` |  |  |
| `remove_light` | method | `remove_light(self, light)` |  |  |
| `render_id_to_visual_name` | property | `render_id_to_visual_name(self)` |  |  |
| `set_ambient_light` | method | `set_ambient_light(self, color)` |  |  |
| `set_environment_id` | method | `set_environment_id(self, env_id: int, allow_duplicate: bool=False)` | Set the PhysX GPU broadphase environment ID for this scene. In GPU mode, all SAPIEN scenes share one PhysX scene. The environment ID is used by the GPU broadphase to a... | Must be set before adding PhysX bodies; -1/0xffffffff means shared. |
| `set_environment_map` | method | `set_environment_map(self, cubemap: str \| RenderCubemap)` |  |  |
| `set_environment_map_from_files` | method | `set_environment_map_from_files(self, px: str, nx: str, py: str, ny: str, pz: str, nz: str)` |  |  |
| `set_timestep` | method | `set_timestep(self, timestep)` | Set the simulation timestep. |  |
| `step` | method | `step(self)` | Advance the system/scene one step. |  |
| `timestep` | property | `timestep(self)<br>timestep(self, timestep)` |  |  |
| `update_render` | method | `update_render(self)` | Update CPU-pose render path; use with caution under GPU PhysX. | GPU PhysX dynamic bodies read CPU poses; prefer the CUDA pose path for offscreen. |
| `__init__` | method | `__init__(self, systems=None)` |  |  |

## `sapien.Widget`

- Use: Compatibility placeholder; subclassing is no longer needed.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__init__` | method | `__init__(self)` |  |  |

## ActorBuilder API

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien`
- Source files: `python/py_package/wrapper/actor_builder.py`

## Functions

| API | Signature | Use | Notes |
|---|---|---|---|
| `sapien.wrapper.actor_builder.preprocess_mesh_file` | `preprocess_mesh_file(filename: str)` | Process input mesh file to a SAPIEN supported format Args: filename: input mesh file Returns: filename for the generated file or original filename |  |

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.ActorBuilder` |  | High-level rigid-body/Actor builder; accumulates collision/visual records then builds. |  |
| `sapien.wrapper.actor_builder.CollisionShapeRecord` |  | SAPIEN API object. |  |
| `sapien.wrapper.actor_builder.VisualShapeRecord` |  | SAPIEN API object. |  |

## `sapien.ActorBuilder`

- Use: High-level rigid-body/Actor builder; accumulates collision/visual records then builds.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `add_box_collision` | method | `add_box_collision(self, pose: sapien.Pose=sapien.Pose(), half_size: Vec3=(1, 1, 1), material: Union[sapien.physx.PhysxMaterial, None]=None, density: float=1000, patch_radius: float=0, min_patch_radius: float=0, is_trigger: bool=False)` |  |  |
| `add_box_visual` | method | `add_box_visual(self, pose: sapien.Pose=sapien.Pose(), half_size: Vec3=(1, 1, 1), material: Union[sapien.render.RenderMaterial, None, Vec3]=None, name: str='')` |  |  |
| `add_capsule_collision` | method | `add_capsule_collision(self, pose: sapien.Pose=sapien.Pose(), radius: float=1, half_length: float=1, material: Union[sapien.physx.PhysxMaterial, None]=None, density: float=1000, patch_radius: float=0, min_patch_radius: float=0, is_trigger: bool=False)` |  |  |
| `add_capsule_visual` | method | `add_capsule_visual(self, pose: sapien.Pose=sapien.Pose(), radius: float=1, half_length: float=1, material: Union[sapien.render.RenderMaterial, None, Vec3]=None, name: str='')` |  |  |
| `add_convex_collision_from_file` | method | `add_convex_collision_from_file(self, filename, pose: sapien.Pose=sapien.Pose(), scale: Vec3=(1, 1, 1), material: Union[sapien.physx.PhysxMaterial, None]=None, density: float=1000, patch_radius: float=0, min_patch_radius: float=0, is_trigger: bool=False)` |  |  |
| `add_cylinder_collision` | method | `add_cylinder_collision(self, pose: sapien.Pose=sapien.Pose(), radius: float=1, half_length: float=1, material: Union[sapien.physx.PhysxMaterial, None]=None, density: float=1000, patch_radius: float=0, min_patch_radius: float=0, is_trigger: bool=False)` |  |  |
| `add_cylinder_visual` | method | `add_cylinder_visual(self, pose: sapien.Pose=sapien.Pose(), radius: float=1, half_length: float=1, material: Union[sapien.render.RenderMaterial, None, Vec3]=None, name: str='')` |  |  |
| `add_multiple_convex_collisions_from_file` | method | `add_multiple_convex_collisions_from_file(self, filename, pose: sapien.Pose=sapien.Pose(), scale: Vec3=(1, 1, 1), material: Union[sapien.physx.PhysxMaterial, None]=None, density: float=1000, patch_radius: float=0, min_patch_radius: float=0, is_trigger: bool=False, decomposition: Literal['none', 'coacd']='none', decomposition_params=dict())` |  |  |
| `add_nonconvex_collision_from_file` | method | `add_nonconvex_collision_from_file(self, filename: str, pose: sapien.Pose=sapien.Pose(), scale: Vec3=(1, 1, 1), material: Union[sapien.physx.PhysxMaterial, None]=None, density: float=1000, patch_radius: float=0, min_patch_radius: float=0, is_trigger: bool=False, sdf_config: Optional[sapien.physx.PhysxSDFConfig]=None)` |  |  |
| `add_plane_collision` | method | `add_plane_collision(self, pose: sapien.Pose=sapien.Pose(), material: Union[sapien.physx.PhysxMaterial, None]=None, patch_radius: float=0, min_patch_radius: float=0, is_trigger: bool=False)` |  |  |
| `add_plane_visual` | method | `add_plane_visual(self, pose: sapien.Pose=sapien.Pose(), scale: Vec3=(1, 1, 1), material: Union[sapien.render.RenderMaterial, None, Vec3]=None, name: str='')` |  |  |
| `add_sphere_collision` | method | `add_sphere_collision(self, pose: sapien.Pose=sapien.Pose(), radius: float=1, material: Union[sapien.physx.PhysxMaterial, None]=None, density: float=1000, patch_radius: float=0, min_patch_radius: float=0, is_trigger: bool=False)` |  |  |
| `add_sphere_visual` | method | `add_sphere_visual(self, pose: sapien.Pose=sapien.Pose(), radius: float=1, material: Union[sapien.render.RenderMaterial, None, Vec3]=None, name: str='')` |  |  |
| `add_visual_from_file` | method | `add_visual_from_file(self, filename: str, pose: sapien.Pose=sapien.Pose(), scale: Vec3=(1, 1, 1), material: Union[sapien.render.RenderMaterial, None, Vec3]=None, name: str='')` |  |  |
| `build` | method | `build(self, name=None)` | Build the object and add it to scene. |  |
| `build_entity` | method | `build_entity(self)` | Build an entity. |  |
| `build_kinematic` | method | `build_kinematic(self, name='')` | Build a kinematic body. |  |
| `build_physx_component` | method | `build_physx_component(self, link_parent=None)` | Build a PhysX component. |  |
| `build_render_component` | method | `build_render_component(self)` | Build a render component. |  |
| `build_static` | method | `build_static(self, name='')` | Build a static body. |  |
| `set_initial_pose` | method | `set_initial_pose(self, pose)` |  |  |
| `set_mass_and_inertia` | method | `set_mass_and_inertia(self, mass, cmass_local_pose, inertia)` |  |  |
| `set_name` | method | `set_name(self, name)` |  |  |
| `set_physx_body_type` | method | `set_physx_body_type(self, type)` |  |  |
| `set_scene` | method | `set_scene(self, scene: sapien.Scene)` |  |  |
| `__init__` | method | `__init__(self)` |  |  |

## `sapien.wrapper.actor_builder.CollisionShapeRecord`

- Use: SAPIEN API object.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `decomposition` | `str` |  |  |
| `decomposition_params` | `Union[Dict[str, Any], None]` |  |  |
| `density` | `float` |  |  |
| `filename` | `str` |  |  |
| `is_trigger` | `bool` |  |  |
| `length` | `float` |  |  |
| `material` | `Union[sapien.physx.PhysxMaterial, None]` |  |  |
| `min_patch_radius` | `float` |  |  |
| `patch_radius` | `float` |  |  |
| `pose` | `sapien.Pose` |  |  |
| `radius` | `float` |  |  |
| `scale` | `Tuple` |  |  |
| `sdf_config` | `Optional[sapien.physx.PhysxSDFConfig]` |  |  |
| `type` | `Literal['convex_mesh', 'multiple_convex_meshes', 'nonconvex_mesh', 'plane', 'box', 'capsule', 'sphere', 'cylinder']` |  |  |

## `sapien.wrapper.actor_builder.VisualShapeRecord`

- Use: SAPIEN API object.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `filename` | `str` |  |  |
| `length` | `float` |  |  |
| `material` | `Union[sapien.render.RenderMaterial, None]` |  |  |
| `name` | `str` |  |  |
| `pose` | `sapien.Pose` |  |  |
| `radius` | `float` |  |  |
| `scale` | `tuple` |  |  |
| `type` | `Literal['file', 'plane', 'box', 'capsule', 'sphere', 'cylinder']` |  |  |

## ArticulationBuilder API

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.wrapper.articulation_builder`
- Source files: `python/py_package/wrapper/articulation_builder.py`

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.ArticulationBuilder` |  | High-level articulation builder; creates a LinkBuilder tree then builds. |  |
| `sapien.wrapper.articulation_builder.JointRecord` |  | SAPIEN API object. |  |
| `sapien.wrapper.articulation_builder.LinkBuilder` | `ActorBuilder` | ArticulationBuilder link builder; subclasses ActorBuilder. |  |
| `sapien.wrapper.articulation_builder.MimicJointRecord` |  | SAPIEN API object. |  |

## `sapien.ArticulationBuilder`

- Use: High-level articulation builder; creates a LinkBuilder tree then builds.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `build` | method | `build(self, fix_root_link=None, build_mimic_joints=True) -> PhysxArticulation` | Build the object and add it to scene. |  |
| `build_entities` | method | `build_entities(self, fix_root_link=None) -> List[Entity]` |  |  |
| `create_link_builder` | method | `create_link_builder(self, parent: LinkBuilder=None)` | Create a link builder. |  |
| `set_initial_pose` | method | `set_initial_pose(self, pose)` |  |  |
| `set_scene` | method | `set_scene(self, scene: Scene)` |  |  |
| `__init__` | method | `__init__(self)` |  |  |

## `sapien.wrapper.articulation_builder.JointRecord`

- Use: SAPIEN API object.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `armature` | `Union[NDArray[np.float32], float]` |  |  |
| `damping` | `float` |  |  |
| `effort_limit` | `Optional[float]` |  |  |
| `friction` | `float` |  |  |
| `velocity_limit` | `Optional[float]` | PhysX per-axis maximum joint velocity. | Positive URDF `<limit velocity>` values populate this field; non-positive placeholders leave it unset. |
| `joint_type` | `str` |  |  |
| `limits` | `Sequence[float] \| Sequence[Sequence[float]]` | One flat limit pair or per-DOF lower/upper position limit pairs. |  |
| `name` | `str` |  |  |
| `pose_in_child` | `Pose` |  |  |
| `pose_in_parent` | `Pose` |  |  |

## `sapien.wrapper.articulation_builder.LinkBuilder`

- Use: ArticulationBuilder link builder; subclasses ActorBuilder.
- Bases: `ActorBuilder`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `set_joint_name` | method | `set_joint_name(self, name: str)` |  |  |
| `set_joint_properties` | method | `set_joint_properties(self, type: str, limits: Sequence[float] \| Sequence[Sequence[float]], pose_in_parent: Pose, pose_in_child: Pose, friction: float=0, damping: float=0, effort_limit: Optional[float]=None, velocity_limit: Optional[float]=None) -> None` | Configure joint position, effort, and velocity limits. | Set before articulation build. |
| `__init__` | method | `__init__(self, index: int, parent)` |  |  |
| `_check` | method | `_check(self)` |  |  |

## `sapien.wrapper.articulation_builder.MimicJointRecord`

- Use: SAPIEN API object.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `joint` | `str` |  |  |
| `mimic` | `str` |  |  |
| `multiplier` | `float` |  |  |
| `offset` | `float` |  |  |

## URDFLoader API

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.wrapper.urdf_loader`
- Source files: `python/py_package/wrapper/urdf_loader.py`
- Geometry toggles: set `loader.load_visuals = False` to skip render shapes and render-system requirements; set `loader.load_collisions = False` to skip collision shapes while preserving topology, joints, and inertial data for kinematics-only or IK-only workflows.

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.wrapper.urdf_loader.URDFLoader` |  | URDF/SRDF parser; produces ArticulationBuilder or loads a robot directly. |  |

## `sapien.wrapper.urdf_loader.URDFLoader`

- Use: URDF/SRDF parser; produces ArticulationBuilder or loads a robot directly.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `load_visuals` | `bool` | Whether URDF visual records create render shapes. | Default `True`; set `False` for physics/kinematics-only workflows or when no render system exists. |
| `load_collisions` | `bool` | Whether URDF collision records create collision shapes. | Default `True`; set `False` for kinematics-only or IK-only workflows that only need topology, joints, and inertial data. |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `load` | method | `load(self, urdf_file: str, srdf_file=None, package_dir=None) -> PhysxArticulation` | Load the urdf_file to a scene. Scene variable must be set before calling this method. Args: urdf_file: filename for URDL file srdf_file: SRDF for urdf_file. If srdf_fi... |  |
| `load_file_as_articulation_builder` | method | `load_file_as_articulation_builder(self, urdf_file, srdf_file=None, package_dir=None) -> ArticulationBuilder` | Load a file as an articulation builder. |  |
| `load_multiple` | method | `load_multiple(self, urdf_file: str, srdf_file=None, package_dir=None)` | Multiple Loading means it can load either Articulation or Actor(Entity) here. When the loaded asset is an Actor, load() just retures empty list. Through this API, you... |  |
| `parse` | method | `parse(self, urdf_file, srdf_file=None, package_dir=None) -> Tuple[List[ArticulationBuilder], List[ActorBuilder], List[dict]]` | Parse input and return builders/structures; can be cached for reuse. |  |
| `parse_srdf` | method | `parse_srdf(self, srdf_string: str)` | Call parse srdf. |  |
| `set_density` | method | `set_density(self, density)` |  |  |
| `set_link_density` | method | `set_link_density(self, link_name, density)` |  |  |
| `set_link_material` | method | `set_link_material(self, link_name, static_friction, dynamic_friction, restitution)` |  |  |
| `set_link_min_patch_radius` | method | `set_link_min_patch_radius(self, link_name, min_patch_radius)` |  |  |
| `set_link_patch_radius` | method | `set_link_patch_radius(self, link_name, patch_radius)` |  |  |
| `set_material` | method | `set_material(self, static_friction, dynamic_friction, restitution)` |  |  |
| `set_min_patch_radius` | method | `set_min_patch_radius(self, min_patch_radius)` |  |  |
| `set_patch_radius` | method | `set_patch_radius(self, patch_radius)` |  |  |
| `set_scene` | method | `set_scene(self, scene: Scene)` |  |  |
| `__init__` | method | `__init__(self)` |  |  |
| `_config_link_builder` | method | `_config_link_builder(self, link: Link, link_builder: ActorBuilder)` |  |  |
| `_get_density` | method | `_get_density(self, link_name, index)` | Call get density. |  |
| `_get_material` | method | `_get_material(self, link_name, index)` | Call get material. |  |
| `_get_min_patch_radius` | method | `_get_min_patch_radius(self, link_name, index)` | Call get min patch radius. |  |
| `_get_patch_radius` | method | `_get_patch_radius(self, link_name, index)` | Call get patch radius. |  |
| `_get_sdf_config` | staticmethod | `_get_sdf_config(collision: Collision)` | Call get sdf config. |  |
| `_parse_actor` | method | `_parse_actor(self, link_name: str)` | Call parse actor. |  |
| `_parse_articulation` | method | `_parse_articulation(self, root, fix_base: bool)` | Call parse articulation. |  |
| `_parse_cameras` | method | `_parse_cameras(self, extra)` | Call parse cameras. |  |
| `_parse_urdf` | method | `_parse_urdf(self, urdf_string: str) -> Tuple[List[ArticulationBuilder], List[ActorBuilder], List[dict]]` | Call parse urdf. |  |
| `_pose_from_origin` | staticmethod | `_pose_from_origin(origin, scale)` |  |  |

## PinocchioModel API

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.wrapper.pinocchio_model`
- Source files: `python/py_package/wrapper/pinocchio_model.py`, `python/py_package/wrapper/pinocchio_model.pyi`

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.PinocchioModel` |  [try] | Robot kinematics/dynamics/IK wrapper. | try |

## `sapien.PinocchioModel`

- Use: Robot kinematics/dynamics/IK wrapper.
- Bases: `-`
- Context: `try`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `compute_coriolis_matrix` | method | `compute_coriolis_matrix(self, qpos, qvel)` | Compute the Coriolis matrix. |  |
| `compute_forward_dynamics` | method | `compute_forward_dynamics(self, qpos, qvel, qf)` |  |  |
| `compute_forward_kinematics` | method | `compute_forward_kinematics(self, qpos)` | Compute and cache forward kinematics. After computation, use get_link_pose to retrieve the computed pose for a specific link. |  |
| `compute_full_jacobian` | method | `compute_full_jacobian(self, qpos)` | Compute and cache Jacobian for all links |  |
| `compute_generalized_mass_matrix` | method | `compute_generalized_mass_matrix(self, qpos)` | Compute the generalized mass matrix. |  |
| `compute_inverse_dynamics` | method | `compute_inverse_dynamics(self, qpos, qvel, qacc)` |  |  |
| `compute_inverse_kinematics` | method | `compute_inverse_kinematics(self, link_index, pose, initial_qpos=None, active_qmask=None, eps=0.0001, max_iterations=1000, dt=0.1, damp=1e-06)` | Compute inverse kinematics with CLIK algorithm. Details see https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b-examples_i-inverse-kin... |  |
| `compute_single_link_local_jacobian` | method | `compute_single_link_local_jacobian(self, qpos, link_index)` | Compute the link(body) Jacobian for a single link. It is faster than compute_full_jacobian followed by get_link_jacobian |  |
| `get_link_jacobian` | method | `get_link_jacobian(self, link_index, local=False)` | Given link index, get the Jacobian. Must be called after compute_full_jacobian. `local=True` selects the link (body) frame; otherwise the world (spatial) frame is used. |  |
| `get_link_pose` | method | `get_link_pose(self, link_index)` | Given link index, get link pose (in articulation base frame) from forward kinematics. Must be called after compute_forward_kinematics. |  |
| `get_random_qpos` | method | `get_random_qpos(self)` |  |  |
| `q_p2s` | method | `q_p2s(self, qint)` |  |  |
| `q_s2p` | method | `q_s2p(self, qext)` |  |  |
| `set_joint_order` | method | `set_joint_order(self, names)` |  |  |
| `set_link_order` | method | `set_link_order(self, names)` |  |  |
| `__init__` | method | `__init__(self, urdf, gravity)` |  |  |

## Utils re-exports

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.utils`
- Source files: `python/py_package/utils/__init__.py`

## Viewer API

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.utils.viewer.viewer`
- Source files: `python/py_package/utils/viewer/viewer.py`

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.utils.Viewer` |  | Interactive viewer. |  |

## `sapien.utils.Viewer`

- Use: Interactive viewer.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `add_bounding_box` | method | `add_bounding_box(self, pose, half_size, color)` |  |  |
| `cameras` | property | `cameras(self)` |  |  |
| `clear_scene` | method | `clear_scene(self)` | Call clear scene. |  |
| `close` | method | `close(self)` |  |  |
| `closed` | property | `closed(self)` |  |  |
| `control_window` | property | `control_window(self) -> ControlWindow` |  |  |
| `draw_aabb` | method | `draw_aabb(self, lower, upper, color)` |  |  |
| `drop` | method | `drop(self, files)` |  |  |
| `focus_camera` | method | `focus_camera(self, camera)` |  |  |
| `focus_change` | method | `focus_change(self, focused)` |  |  |
| `focus_entity` | method | `focus_entity(self, entity)` |  |  |
| `get_entity_viewer_pose` | method | `get_entity_viewer_pose(self, entity)` |  |  |
| `init_plugins` | method | `init_plugins(self, plugins)` |  |  |
| `loop` | method | `loop(self, physx_steps=0)` | A convenience method for opening a temporary viewer for a scene. Simply call scene.create_viewer().loop() |  |
| `notify_render_update` | method | `notify_render_update(self)` | notify the viewer that the camera is moved |  |
| `register_click_handler` | method | `register_click_handler(self, handler)` |  |  |
| `remove_bounding_box` | method | `remove_bounding_box(self, box)` |  |  |
| `render` | method | `render(self)` | Draw the most recently submitted Vulkan state and Viewer UI. | Does not update transforms or fetch poses. |
| `render_scene` | property | `render_scene(self)` |  |  |
| `reset_notifications` | method | `reset_notifications(self)` |  |  |
| `resolution` | property | `resolution(self)<br>resolution(self, res)` |  |  |
| `scene` | property | `scene(self) -> sapien.Scene` |  |  |
| `select_entity` | method | `select_entity(self, entity: Entity)` |  |  |
| `selected_entity_visibility` | property | `selected_entity_visibility(self)<br>selected_entity_visibility(self, v)` |  |  |
| `set_camera_pose` | method | `set_camera_pose(self, pose)` |  |  |
| `set_camera_rpy` | method | `set_camera_rpy(self, r, p, y)` |  |  |
| `set_camera_xyz` | method | `set_camera_xyz(self, x, y, z)` |  |  |
| `set_scene` | method | `set_scene(self, scene: Scene)` |  |  |
| `set_scenes` | method | `set_scenes(self, scenes: list[Scene])` | Select base scenes plus associated shared scenes once. | No render offsets are applied. |
| `configure_physx_gpu_rendering` | method | `configure_physx_gpu_rendering(self, physx_system, transport="auto")` | Configure Viewer pose transport. |  |
| `pose_transport` | property | `pose_transport(self) -> str` | Active pose transport. |  |
| `update_render` | method | `update_render(self) -> None` | Submit current simulation/render state. | Explicit update boundary before `render()`. |
| `update_aabb` | method | `update_aabb(self, aabb, lower, upper)` |  |  |
| `update_bounding_box` | method | `update_bounding_box(self, box, pose, half_size)` |  |  |
| `__init__` | method | `__init__(self, renderer: SapienRenderer=None, shader_dir='', resolutions=(1920, 1080), plugins=[PathWindow(), ContactWindow(), SettingWindow(), TransformWindow(), RenderOptionsWindow(), ControlWindow(), SceneWindow(), EntityWindow(), ArticulationWindow()])` |  |  |


## Compatibility wrappers (`Engine`, `SapienRenderer`)

| API | Signature/member | Use | Notes |
|---|---|---|---|
| `sapien.Engine` | `Engine(**args)` | deprecated compatibility engine wrapper | Prefer `sapien.Scene()` directly. |
| `sapien.Engine.set_renderer` | `set_renderer(renderer)` | stores renderer on wrapper | Deprecated path. |
| `sapien.Engine.create_physical_material` | `create_physical_material(static_friction, dynamic_friction, restitution)` | create `sapien.physx.PhysxMaterial` | Use `sapien.physx.PhysxMaterial` or `scene.create_physical_material`. |
| `sapien.Engine.create_scene` | `create_scene(config=SceneConfig())` | set global scene config then return `sapien.Scene()` | Deprecated path; config must be set before system creation. |
| `sapien.SapienRenderer` | `SapienRenderer(**args)` | deprecated renderer wrapper | Rendering now uses `sapien.render.RenderSystem`/`sapien.render` config. |
| `sapien.SapienRenderer.create_material` | `create_material()` | create `sapien.render.RenderMaterial` | Prefer `sapien.render.RenderMaterial()`. |

## Viewer plugin API

| API | Signature/member | Use | Notes |
|---|---|---|---|
| `sapien.utils.viewer.Plugin` | `init(v)`, `before_render()`, `after_render()`, `close()` | base class for Viewer plugins | Implement hooks when extending viewer UI/debug tools. |
| `sapien.utils.viewer.Plugin.get_ui_windows` | `get_ui_windows()` | return internal_renderer UI windows | Viewer consumes returned widgets. |
| `sapien.utils.viewer.Plugin.notify_scene_change` | `notify_scene_change()` | scene-change hook | no-op default. |
| `sapien.utils.viewer.Plugin.notify_selected_entity_change` | `notify_selected_entity_change()` | selected-entity hook | no-op default. |
| `sapien.utils.viewer.Plugin.notify_window_focus_change` | `notify_window_focus_change(focused)` | window focus hook | no-op default. |
| `sapien.utils.viewer.Plugin.clear_scene` | `clear_scene()` | cleanup hook | no-op default. |