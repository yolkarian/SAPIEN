# Core API (`sapien` / `sapien.pysapien`)

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien`
- Source files: `python/py_package/pysapien/__init__.pyi`
- Notes: Top-level `sapien` re-exports these common core classes; `sapien.core` is compatibility alias.

## Top-level re-exports

| API | Origin | Use |
|---|---|---|
| `sapien.__version__` | `sapien.version.__version__` | package version string |
| `sapien.warn` | `_warnings.warn` | re-export/import alias |
| `sapien.Path` | `pathlib.Path` | re-export/import alias |
| `sapien.Component` | `sapien.pysapien.Component` | re-export/import alias |
| `sapien.CudaArray` | `sapien.pysapien.CudaArray` | re-export/import alias |
| `sapien.Device` | `sapien.pysapien.Device` | re-export/import alias |
| `sapien.Entity` | `sapien.pysapien.Entity` | re-export/import alias |
| `sapien.Pose` | `sapien.pysapien.Pose` | re-export/import alias |
| `sapien.System` | `sapien.pysapien.System` | re-export/import alias |
| `sapien.math` | `sapien.pysapien.math` | re-export/import alias |
| `sapien.profile` | `sapien.pysapien.profile` | re-export/import alias |
| `sapien.set_log_level` | `sapien.pysapien.set_log_level` | re-export/import alias |
| `sapien.simsense` | `sapien.pysapien.simsense` | conditional non-Darwin SimSense module |
| `sapien.SceneConfig` | `sapien.pysapien.physx.PhysxSceneConfig` | re-export/import alias |
| `sapien.PinocchioModel` | `sapien.wrapper.pinocchio_model.PinocchioModel` | top-level robotics wrapper; may fallback to built-in `pysapien_pinocchio` on Linux |
| `sapien.ActorBuilder` | `sapien.wrapper.actor_builder.ActorBuilder` | re-export/import alias |
| `sapien.ArticulationBuilder` | `sapien.wrapper.articulation_builder.ArticulationBuilder` | re-export/import alias |
| `sapien.Engine` | `sapien.wrapper.engine.Engine` | re-export/import alias |
| `sapien.SapienRenderer` | `sapien.wrapper.renderer.SapienRenderer` | re-export/import alias |
| `sapien.Scene` | `sapien.wrapper.scene.Scene` | re-export/import alias |
| `sapien.Widget` | `sapien.wrapper.scene.Widget` | re-export/import alias |
| `sapien.asset` | `.asset` | re-export/import alias |
| `sapien.internal_renderer` | `.internal_renderer` | re-export/import alias |
| `sapien.physx` | `.physx` | re-export/import alias |
| `sapien.pysapien` | `.pysapien` | re-export/import alias |
| `sapien.pysapien_pinocchio` | `.pysapien_pinocchio` | re-export/import alias |
| `sapien.render` | `.render` | re-export/import alias |
| `sapien.utils` | `.utils` | re-export/import alias |
| `sapien.version` | `.version` | re-export/import alias |
| `sapien.wrapper` | `.wrapper` | re-export/import alias |

## Functions

| API | Signature | Use | Notes |
|---|---|---|---|
| `sapien.abi_version` | `abi_version() -> int` | 调用 abi version。 |  |
| `sapien.compiled_with_cxx11_abi` | `compiled_with_cxx11_abi() -> bool` | 调用 compiled with cxx11 abi。 |  |
| `sapien.profile` | `profile(name: str) -> Profiler<br>profile(func: Callable) -> Callable` | 调用 profile。 |  |
| `sapien.pybind11_internals_id` | `pybind11_internals_id() -> str` | 调用 pybind11 internals id。 |  |
| `sapien.pybind11_use_smart_holder` | `pybind11_use_smart_holder() -> bool` | 调用 pybind11 use smart holder。 |  |
| `sapien.set_log_level` | `set_log_level(level: str) -> None` | 设置 log level。 |  |

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.Component` |  | 挂到 Entity 的功能组件基类。 |  |
| `sapien.CudaArray` |  | CUDA 内存视图；可转 torch/cupy/jax/DLPack。 |  |
| `sapien.Device` |  | CPU/CUDA/渲染设备选择与能力查询。 |  |
| `sapien.Entity` |  | 场景中的实体；持有组件和世界位姿。 |  |
| `sapien.Pose` |  | SE(3) 位姿；p 为 xyz，q 为 wxyz 四元数。 |  |
| `sapien.Profiler` |  | profile 上下文管理器/装饰器。 |  |
| `sapien.Scene` |  | 实体/系统容器；高层 wrapper 还提供 builder、灯光、相机、地面等便捷函数。 |  |
| `sapien.System` |  | Scene 系统基类；step 驱动系统。 |  |

## `sapien.Component`

- Use: 挂到 Entity 的功能组件基类。
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `entity_pose` | `Pose` | 属性：entity pose。 |  |
| `name` | `str` | 属性：name。 |  |
| `pose` | `Pose` | 属性：pose。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `disable` | method | `disable(self) -> None` | disable the component |  |
| `enable` | method | `enable(self) -> None` | enable the component |  |
| `entity` | property | `entity(self) -> Entity` | 属性：entity。 |  |
| `get_entity` | method | `get_entity(self) -> Entity` | 读取 entity。 |  |
| `get_entity_pose` | method | `get_entity_pose(self) -> Pose` | 读取 entity pose。 |  |
| `get_name` | method | `get_name(self) -> str` | 读取 name。 |  |
| `get_pose` | method | `get_pose(self) -> Pose` | 读取 pose。 |  |
| `is_enabled` | property | `is_enabled(self) -> bool` | 布尔状态：enabled。 |  |
| `set_entity_pose` | method | `set_entity_pose(self, pose: Pose) -> None` | 设置 entity pose。 |  |
| `set_name` | method | `set_name(self, name: str) -> None` | 设置 name。 |  |
| `set_pose` | method | `set_pose(self, pose: Pose) -> None` | 设置 pose。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.CudaArray`

- Use: CUDA 内存视图；可转 torch/cupy/jax/DLPack。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `cuda_id` | property | `cuda_id(self) -> int` | CUDA 状态/渲染缓冲属性。 | gpu_init 后取 view；循环中复用 torch/cupy/jax view。 |
| `cupy` | method | `cupy(self) -> cupy.ndarray` | 返回 CuPy CUDA array view。 |  |
| `dlpack` | method | `dlpack(self) -> Any` | 返回 DLPack capsule。 |  |
| `jax` | method | `jax(self) -> jax.Array` | 返回 JAX CUDA array view。 |  |
| `ptr` | property | `ptr(self) -> int` | 属性：ptr。 |  |
| `shape` | property | `shape(self) -> list[int]` | 属性：shape。 |  |
| `strides` | property | `strides(self) -> list[int]` | 属性：strides。 |  |
| `torch` | method | `torch(self) -> torch.Tensor` | 返回 torch CUDA tensor view；循环外缓存。 | 零拷贝视图生命周期依赖原 CudaArray；不要循环创建。 |
| `typestr` | property | `typestr(self) -> str` | 属性：typestr。 |  |
| `__init__` | method | `__init__(self, data: Any) -> None` | Python special method。 |  |

## `sapien.Device`

- Use: CPU/CUDA/渲染设备选择与能力查询。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `can_present` | method | `can_present(self) -> bool` | 调用 can present。 |  |
| `can_render` | method | `can_render(self) -> bool` | 调用 can render。 |  |
| `cuda_id` | property | `cuda_id(self) -> int` | CUDA 状态/渲染缓冲属性。 | gpu_init 后取 view；循环中复用 torch/cupy/jax view。 |
| `is_cpu` | method | `is_cpu(self) -> bool` | 布尔状态：cpu。 |  |
| `is_cuda` | method | `is_cuda(self) -> bool` | 布尔状态：cuda。 |  |
| `name` | property | `name(self) -> str` | 属性：name。 |  |
| `pci_string` | property | `pci_string(self) -> str \| None` | 属性：pci string。 |  |
| `__init__` | method | `__init__(self, alias: str) -> None` | Python special method。 |  |
| `__repr__` | method | `__repr__(self) -> str` | Python special method。 |  |
| `__str__` | method | `__str__(self) -> str` | Python special method。 |  |

## `sapien.Entity`

- Use: 场景中的实体；持有组件和世界位姿。
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `name` | `str` | 属性：name。 |  |
| `pose` | `Pose` | 属性：pose。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `add_component` | method | `add_component(self, component: Component) -> Entity` | 添加/创建 component。 |  |
| `add_to_scene` | method | `add_to_scene(self, scene: Scene) -> Entity` | 添加/创建 to scene。 |  |
| `components` | property | `components(self) -> list[Component]` | 属性：components。 |  |
| `find_component_by_type` | method | `find_component_by_type(self, cls: Type[_T]) -> _T` | 调用 find component by type。 |  |
| `get_components` | method | `get_components(self) -> list[Component]` | 读取 components。 |  |
| `get_global_id` | method | `get_global_id(self) -> int` | 读取 global id。 |  |
| `get_name` | method | `get_name(self) -> str` | 读取 name。 |  |
| `get_per_scene_id` | method | `get_per_scene_id(self) -> int` | 读取 per scene id。 |  |
| `get_pose` | method | `get_pose(self) -> Pose` | 读取 pose。 |  |
| `get_scene` | method | `get_scene(self) -> Scene` | 读取 scene。 |  |
| `global_id` | property | `global_id(self) -> int` | 属性：global id。 |  |
| `per_scene_id` | property | `per_scene_id(self) -> int` | 属性：per scene id。 |  |
| `remove_component` | method | `remove_component(self, component: Component) -> None` | 移除 component。 |  |
| `remove_from_scene` | method | `remove_from_scene(self) -> None` | 移除 from scene。 |  |
| `scene` | property | `scene(self) -> Scene` | 属性：scene。 |  |
| `set_name` | method | `set_name(self, name: str) -> None` | 设置 name。 |  |
| `set_pose` | method | `set_pose(self, pose: Pose) -> None` | 设置 pose。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.Pose`

- Use: SE(3) 位姿；p 为 xyz，q 为 wxyz 四元数。
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `p` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：p。 |  |
| `q` | `np.ndarray[Literal[4], np.dtype[np.float32]]` | 属性：q。 | 四元数顺序 wxyz。 |
| `rpy` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：rpy。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_p` | method | `get_p(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 读取 p。 |  |
| `get_q` | method | `get_q(self) -> np.ndarray[Literal[4], np.dtype[np.float32]]` | 读取 q。 | 四元数顺序 wxyz。 |
| `get_rpy` | method | `get_rpy(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 读取 rpy。 |  |
| `inv` | method | `inv(self) -> Pose` | 调用 inv。 |  |
| `set_p` | method | `set_p(self, p: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | 设置 p。 |  |
| `set_q` | method | `set_q(self, q: np.ndarray[Literal[4], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | 设置 q。 | 四元数顺序 wxyz。 |
| `set_rpy` | method | `set_rpy(self, rpy: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | 设置 rpy。 |  |
| `to_transformation_matrix` | method | `to_transformation_matrix(self) -> np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float32]]` | 调用 to transformation matrix。 |  |
| `__getstate__` | method | `__getstate__(self) -> tuple` | Python special method。 |  |
| `__init__` | method | `__init__(self, p: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple=..., q: np.ndarray[Literal[4], np.dtype[np.float32]] \| list[float] \| tuple=...) -> None<br>__init__(self, matrix: np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float32]] \| list \| tuple) -> None` | Python special method。 |  |
| `__mul__` | method | `__mul__(self, other: Pose) -> Pose` | Python special method。 |  |
| `__repr__` | method | `__repr__(self) -> str` | Python special method。 |  |
| `__setstate__` | method | `__setstate__(self, arg0: tuple) -> None` | Python special method。 |  |

## `sapien.Profiler`

- Use: profile 上下文管理器/装饰器。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__call__` | method | `__call__(self, func: Callable) -> Callable` | Python special method。 |  |
| `__enter__` | method | `__enter__(self) -> None` | Python special method。 |  |
| `__exit__` | method | `__exit__(self, exec_type: type \| None, exec_value: Any \| None, traceback: Any \| None) -> None` | Python special method。 |  |
| `__init__` | method | `__init__(self, name: str) -> None` | Python special method。 |  |

## `sapien.Scene`

- Use: 实体/系统容器；高层 wrapper 还提供 builder、灯光、相机、地面等便捷函数。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `add_entity` | method | `add_entity(self, entity: Entity) -> None` | 添加/创建 entity。 |  |
| `add_system` | method | `add_system(self, system: System) -> None` | 添加/创建 system。 |  |
| `clear` | method | `clear(self) -> None` | 调用 clear。 |  |
| `entities` | property | `entities(self) -> list[Entity]` | 属性：entities。 |  |
| `get_entities` | method | `get_entities(self) -> list[Entity]` | 读取 entities。 |  |
| `get_id` | method | `get_id(self) -> int` | 读取 id。 |  |
| `get_physx_system` | method | `get_physx_system(self) -> physx.PhysxSystem` | 读取 physx system。 |  |
| `get_render_system` | method | `get_render_system(self) -> render.RenderSystem` | 读取 render system。 |  |
| `get_system` | method | `get_system(self, name: str) -> System` | 读取 system。 |  |
| `id` | property | `id(self) -> int` | 属性：id。 |  |
| `pack_poses` | method | `pack_poses(self) -> bytes` | 调用 pack poses。 |  |
| `physx_system` | property | `physx_system(self) -> physx.PhysxSystem` | 属性：physx system。 |  |
| `remove_entity` | method | `remove_entity(self, entity: Entity) -> None` | 移除 entity。 |  |
| `render_system` | property | `render_system(self) -> render.RenderSystem` | 属性：render system。 |  |
| `unpack_poses` | method | `unpack_poses(self, data: bytes) -> None` | 调用 unpack poses。 |  |
| `__init__` | method | `__init__(self, systems: list[System]) -> None` | Python special method。 |  |

## `sapien.System`

- Use: Scene 系统基类；step 驱动系统。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `step` | method | `step(self) -> None` | 推进系统/场景一步。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |
