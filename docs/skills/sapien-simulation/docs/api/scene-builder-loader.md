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
| `sapien.Scene` | `_Scene` | 实体/系统容器；高层 wrapper 还提供 builder、灯光、相机、地面等便捷函数。 |  |
| `sapien.Widget` |  | 兼容占位；继承已不需要。 |  |

## `sapien.Scene`

- Use: 实体/系统容器；高层 wrapper 还提供 builder、灯光、相机、地面等便捷函数。
- Bases: `_Scene`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `environment_id` | `` | 属性：environment id。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `add_area_light_for_ray_tracing` | method | `add_area_light_for_ray_tracing(self, pose: sapien.Pose, color, half_width: float, half_height: float)` | 添加/创建 area light for ray tracing。 |  |
| `add_camera` | method | `add_camera(self, name, width: int, height: int, fovy: float, near: float, far: float) -> RenderCameraComponent` | 创建 free camera entity/component。 |  |
| `add_directional_light` | method | `add_directional_light(self, direction, color, shadow=False, position=[0, 0, 0], shadow_scale=10.0, shadow_near=-10.0, shadow_far=10.0, shadow_map_size=2048)` | 添加/创建 directional light。 |  |
| `add_ground` | method | `add_ground(self, altitude, render=True, material=None, render_material=None, render_half_size=[10, 10])` | 构建静态地面 actor。 |  |
| `add_heightfield` | method | `add_heightfield(self, height_field, row_scale: float, column_scale: float \| None=None, height_scale: float=1.0, pose=None, render=True, material=None, render_material=None, name='heightfield')` | Add static z-up height-field terrain with optional render mesh. | `height_field` is 2D int16-compatible; public coordinates map rows to +x, columns to +y, samples to +z; add before `gpu_init()` in GPU workflows. |
| `add_mounted_camera` | method | `add_mounted_camera(self, name, mount, pose, width, height, fovy, near, far) -> RenderCameraComponent` | 把 camera component 挂到已有 entity。 |  |
| `add_point_light` | method | `add_point_light(self, position, color, shadow=False, shadow_near=0.1, shadow_far=10.0, shadow_map_size=2048)` | 添加/创建 point light。 |  |
| `add_spot_light` | method | `add_spot_light(self, position, direction, inner_fov: float, outer_fov: float, color, shadow=False, shadow_near=0.1, shadow_far=10.0, shadow_map_size=2048)` | 添加/创建 spot light。 |  |
| `ambient_light` | property | `ambient_light(self)<br>ambient_light(self, color)` | 属性：ambient light。 |  |
| `create_actor_builder` | method | `create_actor_builder(self)` | 创建并绑定当前 Scene 的 ActorBuilder。 |  |
| `create_articulation_builder` | method | `create_articulation_builder(self)` | 创建并绑定当前 Scene 的 ArticulationBuilder。 |  |
| `create_connection` | method | `create_connection(self, body0: Optional[Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent]], pose0: sapien.Pose, body1: Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent], pose1: sapien.Pose)` | 创建 connection。 |  |
| `create_drive` | method | `create_drive(self, body0: Optional[Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent]], pose0: sapien.Pose, body1: Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent], pose1: sapien.Pose)` | 创建 drive。 |  |
| `create_gear` | method | `create_gear(self, body0: Optional[Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent]], pose0: sapien.Pose, body1: Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent], pose1: sapien.Pose)` | 创建 gear。 |  |
| `create_physical_material` | method | `create_physical_material(self, static_friction: float, dynamic_friction: float, restitution: float)` | 创建 physical material。 |  |
| `create_urdf_loader` | method | `create_urdf_loader(self)` | 创建并绑定当前 Scene 的 URDFLoader。 |  |
| `create_viewer` | method | `create_viewer(self)` | 创建 viewer。 |  |
| `get_all_actors` | method | `get_all_actors(self)` | 读取 all actors。 |  |
| `get_all_articulations` | method | `get_all_articulations(self)` | 读取 all articulations。 |  |
| `get_cameras` | method | `get_cameras(self)` | 读取 cameras。 |  |
| `get_contacts` | method | `get_contacts(self)` | 读取 contacts。 |  |
| `get_environment_id` | method | `get_environment_id(self) -> int \| None` | Return the already assigned PhysX GPU broadphase environment ID. Returns ``None`` if no environment ID has been assigned yet. This method has no side effects; use :met... |  |
| `get_mounted_cameras` | method | `get_mounted_cameras(self)` | 读取 mounted cameras。 |  |
| `get_or_assign_environment_id` | method | `get_or_assign_environment_id(self) -> int` | Return this scene's environment ID, assigning a unique one if needed. |  |
| `get_timestep` | method | `get_timestep(self)` | 读取 timestep。 |  |
| `remove_actor` | method | `remove_actor(self, actor)` | 移除 actor。 |  |
| `remove_articulation` | method | `remove_articulation(self, articulation)` | 移除 articulation。 |  |
| `remove_camera` | method | `remove_camera(self, camera)` | 移除 camera。 |  |
| `remove_light` | method | `remove_light(self, light)` | 移除 light。 |  |
| `render_id_to_visual_name` | property | `render_id_to_visual_name(self)` | 属性：render id to visual name。 |  |
| `set_ambient_light` | method | `set_ambient_light(self, color)` | 设置 ambient light。 |  |
| `set_environment_id` | method | `set_environment_id(self, env_id: int, allow_duplicate: bool=False)` | Set the PhysX GPU broadphase environment ID for this scene. In GPU mode, all SAPIEN scenes share one PhysX scene. The environment ID is used by the GPU broadphase to a... | 必须在加 PhysX body 前；-1/0xffffffff 表示 shared。 |
| `set_environment_map` | method | `set_environment_map(self, cubemap: str \| RenderCubemap)` | 设置 environment map。 |  |
| `set_environment_map_from_files` | method | `set_environment_map_from_files(self, px: str, nx: str, py: str, ny: str, pz: str, nz: str)` | 设置 environment map from files。 |  |
| `set_timestep` | method | `set_timestep(self, timestep)` | 设置仿真步长。 |  |
| `step` | method | `step(self)` | 推进系统/场景一步。 |  |
| `timestep` | property | `timestep(self)<br>timestep(self, timestep)` | 属性：timestep。 |  |
| `update_render` | method | `update_render(self)` | 更新 CPU-pose 渲染路径；GPU PhysX 下慎用。 | GPU PhysX 动态体会读 CPU pose；offscreen 优先 CUDA pose path。 |
| `__init__` | method | `__init__(self, systems=None)` | Python special method。 |  |

## `sapien.Widget`

- Use: 兼容占位；继承已不需要。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__init__` | method | `__init__(self)` | Python special method。 |  |

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
| `sapien.ActorBuilder` |  | 高层刚体/Actor 构建器；累积 collision/visual 记录后 build。 |  |
| `sapien.wrapper.actor_builder.CollisionShapeRecord` |  | SAPIEN API 对象。 |  |
| `sapien.wrapper.actor_builder.VisualShapeRecord` |  | SAPIEN API 对象。 |  |

## `sapien.ActorBuilder`

- Use: 高层刚体/Actor 构建器；累积 collision/visual 记录后 build。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `add_box_collision` | method | `add_box_collision(self, pose: sapien.Pose=sapien.Pose(), half_size: Vec3=(1, 1, 1), material: Union[sapien.physx.PhysxMaterial, None]=None, density: float=1000, patch_radius: float=0, min_patch_radius: float=0, is_trigger: bool=False)` | 添加/创建 box collision。 |  |
| `add_box_visual` | method | `add_box_visual(self, pose: sapien.Pose=sapien.Pose(), half_size: Vec3=(1, 1, 1), material: Union[sapien.render.RenderMaterial, None, Vec3]=None, name: str='')` | 添加/创建 box visual。 |  |
| `add_capsule_collision` | method | `add_capsule_collision(self, pose: sapien.Pose=sapien.Pose(), radius: float=1, half_length: float=1, material: Union[sapien.physx.PhysxMaterial, None]=None, density: float=1000, patch_radius: float=0, min_patch_radius: float=0, is_trigger: bool=False)` | 添加/创建 capsule collision。 |  |
| `add_capsule_visual` | method | `add_capsule_visual(self, pose: sapien.Pose=sapien.Pose(), radius: float=1, half_length: float=1, material: Union[sapien.render.RenderMaterial, None, Vec3]=None, name: str='')` | 添加/创建 capsule visual。 |  |
| `add_convex_collision_from_file` | method | `add_convex_collision_from_file(self, filename, pose: sapien.Pose=sapien.Pose(), scale: Vec3=(1, 1, 1), material: Union[sapien.physx.PhysxMaterial, None]=None, density: float=1000, patch_radius: float=0, min_patch_radius: float=0, is_trigger: bool=False)` | 添加/创建 convex collision from file。 |  |
| `add_cylinder_collision` | method | `add_cylinder_collision(self, pose: sapien.Pose=sapien.Pose(), radius: float=1, half_length: float=1, material: Union[sapien.physx.PhysxMaterial, None]=None, density: float=1000, patch_radius: float=0, min_patch_radius: float=0, is_trigger: bool=False)` | 添加/创建 cylinder collision。 |  |
| `add_cylinder_visual` | method | `add_cylinder_visual(self, pose: sapien.Pose=sapien.Pose(), radius: float=1, half_length: float=1, material: Union[sapien.render.RenderMaterial, None, Vec3]=None, name: str='')` | 添加/创建 cylinder visual。 |  |
| `add_multiple_convex_collisions_from_file` | method | `add_multiple_convex_collisions_from_file(self, filename, pose: sapien.Pose=sapien.Pose(), scale: Vec3=(1, 1, 1), material: Union[sapien.physx.PhysxMaterial, None]=None, density: float=1000, patch_radius: float=0, min_patch_radius: float=0, is_trigger: bool=False, decomposition: Literal['none', 'coacd']='none', decomposition_params=dict())` | 添加/创建 multiple convex collisions from file。 |  |
| `add_nonconvex_collision_from_file` | method | `add_nonconvex_collision_from_file(self, filename: str, pose: sapien.Pose=sapien.Pose(), scale: Vec3=(1, 1, 1), material: Union[sapien.physx.PhysxMaterial, None]=None, density: float=1000, patch_radius: float=0, min_patch_radius: float=0, is_trigger: bool=False, sdf_config: Optional[sapien.physx.PhysxSDFConfig]=None)` | 添加/创建 nonconvex collision from file。 |  |
| `add_plane_collision` | method | `add_plane_collision(self, pose: sapien.Pose=sapien.Pose(), material: Union[sapien.physx.PhysxMaterial, None]=None, patch_radius: float=0, min_patch_radius: float=0, is_trigger: bool=False)` | 添加/创建 plane collision。 |  |
| `add_plane_visual` | method | `add_plane_visual(self, pose: sapien.Pose=sapien.Pose(), scale: Vec3=(1, 1, 1), material: Union[sapien.render.RenderMaterial, None, Vec3]=None, name: str='')` | 添加/创建 plane visual。 |  |
| `add_sphere_collision` | method | `add_sphere_collision(self, pose: sapien.Pose=sapien.Pose(), radius: float=1, material: Union[sapien.physx.PhysxMaterial, None]=None, density: float=1000, patch_radius: float=0, min_patch_radius: float=0, is_trigger: bool=False)` | 添加/创建 sphere collision。 |  |
| `add_sphere_visual` | method | `add_sphere_visual(self, pose: sapien.Pose=sapien.Pose(), radius: float=1, material: Union[sapien.render.RenderMaterial, None, Vec3]=None, name: str='')` | 添加/创建 sphere visual。 |  |
| `add_visual_from_file` | method | `add_visual_from_file(self, filename: str, pose: sapien.Pose=sapien.Pose(), scale: Vec3=(1, 1, 1), material: Union[sapien.render.RenderMaterial, None, Vec3]=None, name: str='')` | 添加/创建 visual from file。 |  |
| `build` | method | `build(self, name=None)` | 构建对象并加入 scene。 |  |
| `build_entity` | method | `build_entity(self)` | 构建 entity。 |  |
| `build_kinematic` | method | `build_kinematic(self, name='')` | 构建 kinematic。 |  |
| `build_physx_component` | method | `build_physx_component(self, link_parent=None)` | 构建 physx component。 |  |
| `build_render_component` | method | `build_render_component(self)` | 构建 render component。 |  |
| `build_static` | method | `build_static(self, name='')` | 构建 static。 |  |
| `set_initial_pose` | method | `set_initial_pose(self, pose)` | 设置 initial pose。 |  |
| `set_mass_and_inertia` | method | `set_mass_and_inertia(self, mass, cmass_local_pose, inertia)` | 设置 mass and inertia。 |  |
| `set_name` | method | `set_name(self, name)` | 设置 name。 |  |
| `set_physx_body_type` | method | `set_physx_body_type(self, type)` | 设置 physx body type。 |  |
| `set_scene` | method | `set_scene(self, scene: sapien.Scene)` | 设置 scene。 |  |
| `__init__` | method | `__init__(self)` | Python special method。 |  |

## `sapien.wrapper.actor_builder.CollisionShapeRecord`

- Use: SAPIEN API 对象。
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `decomposition` | `str` | 属性：decomposition。 |  |
| `decomposition_params` | `Union[Dict[str, Any], None]` | 属性：decomposition params。 |  |
| `density` | `float` | 属性：density。 |  |
| `filename` | `str` | 属性：filename。 |  |
| `is_trigger` | `bool` | 布尔状态：trigger。 |  |
| `length` | `float` | 属性：length。 |  |
| `material` | `Union[sapien.physx.PhysxMaterial, None]` | 属性：material。 |  |
| `min_patch_radius` | `float` | 属性：min patch radius。 |  |
| `patch_radius` | `float` | 属性：patch radius。 |  |
| `pose` | `sapien.Pose` | 属性：pose。 |  |
| `radius` | `float` | 属性：radius。 |  |
| `scale` | `Tuple` | 属性：scale。 |  |
| `sdf_config` | `Optional[sapien.physx.PhysxSDFConfig]` | 属性：sdf config。 |  |
| `type` | `Literal['convex_mesh', 'multiple_convex_meshes', 'nonconvex_mesh', 'plane', 'box', 'capsule', 'sphere', 'cylinder']` | 属性：type。 |  |

## `sapien.wrapper.actor_builder.VisualShapeRecord`

- Use: SAPIEN API 对象。
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `filename` | `str` | 属性：filename。 |  |
| `length` | `float` | 属性：length。 |  |
| `material` | `Union[sapien.render.RenderMaterial, None]` | 属性：material。 |  |
| `name` | `str` | 属性：name。 |  |
| `pose` | `sapien.Pose` | 属性：pose。 |  |
| `radius` | `float` | 属性：radius。 |  |
| `scale` | `tuple` | 属性：scale。 |  |
| `type` | `Literal['file', 'plane', 'box', 'capsule', 'sphere', 'cylinder']` | 属性：type。 |  |

## ArticulationBuilder API

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.wrapper.articulation_builder`
- Source files: `python/py_package/wrapper/articulation_builder.py`

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.ArticulationBuilder` |  | 高层 articulation 构建器；创建 LinkBuilder 树并 build。 |  |
| `sapien.wrapper.articulation_builder.JointRecord` |  | SAPIEN API 对象。 |  |
| `sapien.wrapper.articulation_builder.LinkBuilder` | `ActorBuilder` | ArticulationBuilder 的 link 构建器；继承 ActorBuilder。 |  |
| `sapien.wrapper.articulation_builder.MimicJointRecord` |  | SAPIEN API 对象。 |  |

## `sapien.ArticulationBuilder`

- Use: 高层 articulation 构建器；创建 LinkBuilder 树并 build。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `build` | method | `build(self, fix_root_link=None, build_mimic_joints=True) -> PhysxArticulation` | 构建对象并加入 scene。 |  |
| `build_entities` | method | `build_entities(self, fix_root_link=None) -> List[Entity]` | 构建 entities。 |  |
| `create_link_builder` | method | `create_link_builder(self, parent: LinkBuilder=None)` | 创建 link builder。 |  |
| `set_initial_pose` | method | `set_initial_pose(self, pose)` | 设置 initial pose。 |  |
| `set_scene` | method | `set_scene(self, scene: Scene)` | 设置 scene。 |  |
| `__init__` | method | `__init__(self)` | Python special method。 |  |

## `sapien.wrapper.articulation_builder.JointRecord`

- Use: SAPIEN API 对象。
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `armature` | `Union[NDArray[np.float32], float]` | 属性：armature。 |  |
| `damping` | `float` | 属性：damping。 |  |
| `effort_limit` | `Optional[float]` | 属性：effort limit。 |  |
| `friction` | `float` | 属性：friction。 |  |
| `joint_type` | `str` | 属性：joint type。 |  |
| `limits` | `Tuple[float, float]` | 属性：limits。 |  |
| `name` | `str` | 属性：name。 |  |
| `pose_in_child` | `Pose` | 属性：pose in child。 |  |
| `pose_in_parent` | `Pose` | 属性：pose in parent。 |  |

## `sapien.wrapper.articulation_builder.LinkBuilder`

- Use: ArticulationBuilder 的 link 构建器；继承 ActorBuilder。
- Bases: `ActorBuilder`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `set_joint_name` | method | `set_joint_name(self, name: str)` | 设置 joint name。 |  |
| `set_joint_properties` | method | `set_joint_properties(self, type: str, limits: Tuple[float, float], pose_in_parent: Pose, pose_in_child: Pose, friction: float=0, damping: float=0, effort_limit: Optional[float]=None)` | 设置 joint properties。 |  |
| `__init__` | method | `__init__(self, index: int, parent)` | Python special method。 |  |
| `_check` | method | `_check(self)` | 调用  check。 |  |

## `sapien.wrapper.articulation_builder.MimicJointRecord`

- Use: SAPIEN API 对象。
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `joint` | `str` | 属性：joint。 |  |
| `mimic` | `str` | 属性：mimic。 |  |
| `multiplier` | `float` | 属性：multiplier。 |  |
| `offset` | `float` | 属性：offset。 |  |

## URDFLoader API

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.wrapper.urdf_loader`
- Source files: `python/py_package/wrapper/urdf_loader.py`
- Geometry toggles: set `loader.load_visuals = False` to skip render shapes and render-system requirements; set `loader.load_collisions = False` to skip collision shapes while preserving topology, joints, and inertial data for kinematics-only or IK-only workflows.

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.wrapper.urdf_loader.URDFLoader` |  | URDF/SRDF 解析器；生成 ArticulationBuilder 或直接加载机器人。 |  |

## `sapien.wrapper.urdf_loader.URDFLoader`

- Use: URDF/SRDF 解析器；生成 ArticulationBuilder 或直接加载机器人。
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
| `load_file_as_articulation_builder` | method | `load_file_as_articulation_builder(self, urdf_file, srdf_file=None, package_dir=None) -> ArticulationBuilder` | 加载 file as articulation builder。 |  |
| `load_multiple` | method | `load_multiple(self, urdf_file: str, srdf_file=None, package_dir=None)` | Multiple Loading means it can load either Articulation or Actor(Entity) here. When the loaded asset is an Actor, load() just retures empty list. Through this API, you... |  |
| `parse` | method | `parse(self, urdf_file, srdf_file=None, package_dir=None) -> Tuple[List[ArticulationBuilder], List[ActorBuilder], List[dict]]` | 解析输入并返回 builder/结构；可缓存重用。 |  |
| `parse_srdf` | method | `parse_srdf(self, srdf_string: str)` | 调用 parse srdf。 |  |
| `set_density` | method | `set_density(self, density)` | 设置 density。 |  |
| `set_link_density` | method | `set_link_density(self, link_name, density)` | 设置 link density。 |  |
| `set_link_material` | method | `set_link_material(self, link_name, static_friction, dynamic_friction, restitution)` | 设置 link material。 |  |
| `set_link_min_patch_radius` | method | `set_link_min_patch_radius(self, link_name, min_patch_radius)` | 设置 link min patch radius。 |  |
| `set_link_patch_radius` | method | `set_link_patch_radius(self, link_name, patch_radius)` | 设置 link patch radius。 |  |
| `set_material` | method | `set_material(self, static_friction, dynamic_friction, restitution)` | 设置 material。 |  |
| `set_min_patch_radius` | method | `set_min_patch_radius(self, min_patch_radius)` | 设置 min patch radius。 |  |
| `set_patch_radius` | method | `set_patch_radius(self, patch_radius)` | 设置 patch radius。 |  |
| `set_scene` | method | `set_scene(self, scene: Scene)` | 设置 scene。 |  |
| `__init__` | method | `__init__(self)` | Python special method。 |  |
| `_config_link_builder` | method | `_config_link_builder(self, link: Link, link_builder: ActorBuilder)` | 调用  config link builder。 |  |
| `_get_density` | method | `_get_density(self, link_name, index)` | 调用  get density。 |  |
| `_get_material` | method | `_get_material(self, link_name, index)` | 调用  get material。 |  |
| `_get_min_patch_radius` | method | `_get_min_patch_radius(self, link_name, index)` | 调用  get min patch radius。 |  |
| `_get_patch_radius` | method | `_get_patch_radius(self, link_name, index)` | 调用  get patch radius。 |  |
| `_get_sdf_config` | staticmethod | `_get_sdf_config(collision: Collision)` | 调用  get sdf config。 |  |
| `_parse_actor` | method | `_parse_actor(self, link_name: str)` | 调用  parse actor。 |  |
| `_parse_articulation` | method | `_parse_articulation(self, root, fix_base: bool)` | 调用  parse articulation。 |  |
| `_parse_cameras` | method | `_parse_cameras(self, extra)` | 调用  parse cameras。 |  |
| `_parse_urdf` | method | `_parse_urdf(self, urdf_string: str) -> Tuple[List[ArticulationBuilder], List[ActorBuilder], List[dict]]` | 调用  parse urdf。 |  |
| `_pose_from_origin` | staticmethod | `_pose_from_origin(origin, scale)` | 调用  pose from origin。 |  |

## PinocchioModel API

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.wrapper.pinocchio_model`
- Source files: `python/py_package/wrapper/pinocchio_model.py`

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.PinocchioModel` |  [try] | 机器人运动学/动力学/IK 包装。 | try |

## `sapien.PinocchioModel`

- Use: 机器人运动学/动力学/IK 包装。
- Bases: `-`
- Context: `try`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `compute_coriolis_matrix` | method | `compute_coriolis_matrix(self, qpos, qvel)` | 计算 coriolis matrix。 |  |
| `compute_forward_dynamics` | method | `compute_forward_dynamics(self, qpos, qvel, qf)` | 计算 forward dynamics。 |  |
| `compute_forward_kinematics` | method | `compute_forward_kinematics(self, qpos)` | Compute and cache forward kinematics. After computation, use get_link_pose to retrieve the computed pose for a specific link. |  |
| `compute_full_jacobian` | method | `compute_full_jacobian(self, qpos)` | Compute and cache Jacobian for all links |  |
| `compute_generalized_mass_matrix` | method | `compute_generalized_mass_matrix(self, qpos)` | 计算 generalized mass matrix。 |  |
| `compute_inverse_dynamics` | method | `compute_inverse_dynamics(self, qpos, qvel, qacc)` | 计算 inverse dynamics。 |  |
| `compute_inverse_kinematics` | method | `compute_inverse_kinematics(self, link_index, pose, initial_qpos=None, active_qmask=None, eps=0.0001, max_iterations=1000, dt=0.1, damp=1e-06)` | Compute inverse kinematics with CLIK algorithm. Details see https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b-examples_i-inverse-kin... |  |
| `compute_single_link_local_jacobian` | method | `compute_single_link_local_jacobian(self, qpos, index)` | Compute the link(body) Jacobian for a single link. It is faster than compute_full_jacobian followed by get_link_jacobian |  |
| `get_link_jacobian` | method | `get_link_jacobian(self, index, local=False)` | Given link index, get the Jacobian. Must be called after compute_full_jacobian. Args: link_index: index of the link local: True for world(spatial) frame; False for lin... |  |
| `get_link_pose` | method | `get_link_pose(self, index)` | Given link index, get link pose (in articulation base frame) from forward kinematics. Must be called after compute_forward_kinematics. |  |
| `get_random_qpos` | method | `get_random_qpos(self)` | 读取 random qpos。 |  |
| `q_p2s` | method | `q_p2s(self, qint)` | 调用 q p2s。 |  |
| `q_s2p` | method | `q_s2p(self, qext)` | 调用 q s2p。 |  |
| `set_joint_order` | method | `set_joint_order(self, names)` | 设置 joint order。 |  |
| `set_link_order` | method | `set_link_order(self, names)` | 设置 link order。 |  |
| `__init__` | method | `__init__(self, urdf_string, gravity)` | Python special method。 |  |

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
| `sapien.utils.Viewer` |  | 交互式 viewer。 |  |

## `sapien.utils.Viewer`

- Use: 交互式 viewer。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `add_bounding_box` | method | `add_bounding_box(self, pose, half_size, color)` | 添加/创建 bounding box。 |  |
| `cameras` | property | `cameras(self)` | 属性：cameras。 |  |
| `clear_scene` | method | `clear_scene(self)` | 调用 clear scene。 |  |
| `close` | method | `close(self)` | 调用 close。 |  |
| `closed` | property | `closed(self)` | 属性：closed。 |  |
| `control_window` | property | `control_window(self) -> ControlWindow` | 属性：control window。 |  |
| `draw_aabb` | method | `draw_aabb(self, lower, upper, color)` | 调用 draw aabb。 |  |
| `drop` | method | `drop(self, files)` | 调用 drop。 |  |
| `focus_camera` | method | `focus_camera(self, camera)` | 调用 focus camera。 |  |
| `focus_change` | method | `focus_change(self, focused)` | 调用 focus change。 |  |
| `focus_entity` | method | `focus_entity(self, entity)` | 调用 focus entity。 |  |
| `get_entity_viewer_pose` | method | `get_entity_viewer_pose(self, entity)` | 读取 entity viewer pose。 |  |
| `init_plugins` | method | `init_plugins(self, plugins)` | 调用 init plugins。 |  |
| `loop` | method | `loop(self, physx_steps=0)` | A convenience method for opening a temporary viewer for a scene. Simply call scene.create_viewer().loop() |  |
| `notify_render_update` | method | `notify_render_update(self)` | notify the viewer that the camera is moved |  |
| `register_click_handler` | method | `register_click_handler(self, handler)` | 调用 register click handler。 |  |
| `remove_bounding_box` | method | `remove_bounding_box(self, box)` | 移除 bounding box。 |  |
| `render` | method | `render(self)` | 调用 render。 |  |
| `render_scene` | property | `render_scene(self)` | 属性：render scene。 |  |
| `reset_notifications` | method | `reset_notifications(self)` | 调用 reset notifications。 |  |
| `resolution` | property | `resolution(self)<br>resolution(self, res)` | 属性：resolution。 |  |
| `scene` | property | `scene(self) -> sapien.Scene` | 属性：scene。 |  |
| `select_entity` | method | `select_entity(self, entity: Entity)` | 调用 select entity。 |  |
| `selected_entity_visibility` | property | `selected_entity_visibility(self)<br>selected_entity_visibility(self, v)` | 属性：selected entity visibility。 |  |
| `set_camera_pose` | method | `set_camera_pose(self, pose)` | 设置 camera pose。 |  |
| `set_camera_rpy` | method | `set_camera_rpy(self, r, p, y)` | 设置 camera rpy。 |  |
| `set_camera_xyz` | method | `set_camera_xyz(self, x, y, z)` | 设置 camera xyz。 |  |
| `set_scene` | method | `set_scene(self, scene: Scene)` | 设置 scene。 |  |
| `set_scenes` | method | `set_scenes(self, scenes, offsets=None)` | 设置 scenes。 |  |
| `update_aabb` | method | `update_aabb(self, aabb, lower, upper)` | 调用 update aabb。 |  |
| `update_bounding_box` | method | `update_bounding_box(self, box, pose, half_size)` | 调用 update bounding box。 |  |
| `__init__` | method | `__init__(self, renderer: SapienRenderer=None, shader_dir='', resolutions=(1920, 1080), plugins=[PathWindow(), ContactWindow(), SettingWindow(), TransformWindow(), RenderOptionsWindow(), ControlWindow(), SceneWindow(), EntityWindow(), ArticulationWindow()])` | Python special method。 |  |


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
