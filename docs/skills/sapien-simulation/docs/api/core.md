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
| `sapien.simsense` | `sapien.pysapien.simsense` | SimSense module re-export |
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
| `sapien.abi_version` | `abi_version() -> int` | Call abi version. |  |
| `sapien.compiled_with_cxx11_abi` | `compiled_with_cxx11_abi() -> bool` | Call compiled with cxx11 abi. |  |
| `sapien.profile` | `profile(name: str) -> Profiler<br>profile(func: Callable) -> Callable` | Call profile. |  |
| `sapien.pybind11_internals_id` | `pybind11_internals_id() -> str` | Call pybind11 internals id. |  |
| `sapien.pybind11_use_smart_holder` | `pybind11_use_smart_holder() -> bool` | Call pybind11 use smart holder. |  |
| `sapien.set_log_level` | `set_log_level(level: str) -> None` | Set log level. |  |

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.Component` |  | Base class for functional components attached to an Entity. |  |
| `sapien.CudaArray` |  | CUDA memory view; convertible to torch/cupy/jax/DLPack. |  |
| `sapien.Device` |  | CPU/CUDA/render device selection and capability query. |  |
| `sapien.Entity` |  | Entity in a scene; holds components and world pose. |  |
| `sapien.Pose` |  | SE(3) pose; p is xyz, q is wxyz quaternion. |  |
| `sapien.Profiler` |  | profile context manager/decorator. |  |
| `sapien.Scene` |  | Entity/system container; high-level wrapper also provides builder, lights, camera, ground, and other convenience functions. |  |
| `sapien.System` |  | Scene system base class; step drives the system. |  |

## `sapien.Component`

- Use: Base class for functional components attached to an Entity.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `entity_pose` | `Pose` | Property: entity pose. |  |
| `name` | `str` | Property: name. |  |
| `pose` | `Pose` | Property: pose. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `disable` | method | `disable(self) -> None` | disable the component |  |
| `enable` | method | `enable(self) -> None` | enable the component |  |
| `entity` | property | `entity(self) -> Entity` | Property: entity. |  |
| `get_entity` | method | `get_entity(self) -> Entity` | Get entity. |  |
| `get_entity_pose` | method | `get_entity_pose(self) -> Pose` | Get entity pose. |  |
| `get_name` | method | `get_name(self) -> str` | Get name. |  |
| `get_pose` | method | `get_pose(self) -> Pose` | Get pose. |  |
| `is_enabled` | property | `is_enabled(self) -> bool` | Bool: enabled. |  |
| `set_entity_pose` | method | `set_entity_pose(self, pose: Pose) -> None` | Set entity pose. |  |
| `set_name` | method | `set_name(self, name: str) -> None` | Set name. |  |
| `set_pose` | method | `set_pose(self, pose: Pose) -> None` | Set pose. |  |
| `__init__` | method | `__init__(self) -> None` | Python special method. |  |

## `sapien.CudaArray`

- Use: CUDA memory view; convertible to torch/cupy/jax/DLPack.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `cuda_id` | property | `cuda_id(self) -> int` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cupy` | method | `cupy(self) -> cupy.ndarray` | Return a CuPy CUDA array view. |  |
| `dlpack` | method | `dlpack(self) -> Any` | Return a DLPack capsule. |  |
| `jax` | method | `jax(self) -> jax.Array` | Return a JAX CUDA array view. |  |
| `ptr` | property | `ptr(self) -> int` | Property: ptr. |  |
| `shape` | property | `shape(self) -> list[int]` | Property: shape. |  |
| `strides` | property | `strides(self) -> list[int]` | Property: strides. |  |
| `torch` | method | `torch(self) -> torch.Tensor` | Return a torch CUDA tensor view; cache outside the loop. | Zero-copy view lifetime depends on the original CudaArray; do not create it in a loop. |
| `typestr` | property | `typestr(self) -> str` | Property: typestr. |  |
| `__init__` | method | `__init__(self, data: Any) -> None` | Python special method. |  |

## `sapien.Device`

- Use: CPU/CUDA/render device selection and capability query.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `can_present` | method | `can_present(self) -> bool` | Call can present. |  |
| `can_render` | method | `can_render(self) -> bool` | Call can render. |  |
| `cuda_id` | property | `cuda_id(self) -> int` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `is_cpu` | method | `is_cpu(self) -> bool` | Bool: cpu. |  |
| `is_cuda` | method | `is_cuda(self) -> bool` | Bool: cuda. |  |
| `name` | property | `name(self) -> str` | Property: name. |  |
| `pci_string` | property | `pci_string(self) -> str \| None` | Property: pci string. |  |
| `__init__` | method | `__init__(self, alias: str) -> None` | Python special method. |  |
| `__repr__` | method | `__repr__(self) -> str` | Python special method. |  |
| `__str__` | method | `__str__(self) -> str` | Python special method. |  |

## `sapien.Entity`

- Use: Entity in a scene; holds components and world pose.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `name` | `str` | Property: name. |  |
| `pose` | `Pose` | Property: pose. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `add_component` | method | `add_component(self, component: Component) -> Entity` | Add/create component. |  |
| `add_to_scene` | method | `add_to_scene(self, scene: Scene) -> Entity` | Add/create to scene. |  |
| `components` | property | `components(self) -> list[Component]` | Property: components. |  |
| `find_component_by_type` | method | `find_component_by_type(self, cls: Type[_T]) -> _T` | Call find component by type. |  |
| `get_components` | method | `get_components(self) -> list[Component]` | Get components. |  |
| `get_global_id` | method | `get_global_id(self) -> int` | Get global id. |  |
| `get_name` | method | `get_name(self) -> str` | Get name. |  |
| `get_per_scene_id` | method | `get_per_scene_id(self) -> int` | Get per scene id. |  |
| `get_pose` | method | `get_pose(self) -> Pose` | Get pose. |  |
| `get_scene` | method | `get_scene(self) -> Scene` | Get scene. |  |
| `global_id` | property | `global_id(self) -> int` | Property: global id. |  |
| `per_scene_id` | property | `per_scene_id(self) -> int` | Property: per scene id. |  |
| `remove_component` | method | `remove_component(self, component: Component) -> None` | Remove component. |  |
| `remove_from_scene` | method | `remove_from_scene(self) -> None` | Remove from scene. |  |
| `scene` | property | `scene(self) -> Scene` | Property: scene. |  |
| `set_name` | method | `set_name(self, name: str) -> None` | Set name. |  |
| `set_pose` | method | `set_pose(self, pose: Pose) -> None` | Set pose. |  |
| `__init__` | method | `__init__(self) -> None` | Python special method. |  |

## `sapien.Pose`

- Use: SE(3) pose; p is xyz, q is wxyz quaternion.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `p` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: p. |  |
| `q` | `np.ndarray[Literal[4], np.dtype[np.float32]]` | Property: q. | Quaternion order is wxyz. |
| `rpy` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: rpy. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_p` | method | `get_p(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Get p. |  |
| `get_q` | method | `get_q(self) -> np.ndarray[Literal[4], np.dtype[np.float32]]` | Get q. | Quaternion order is wxyz. |
| `get_rpy` | method | `get_rpy(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Get rpy. |  |
| `inv` | method | `inv(self) -> Pose` | Call inv. |  |
| `set_p` | method | `set_p(self, p: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | Set p. |  |
| `set_q` | method | `set_q(self, q: np.ndarray[Literal[4], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | Set q. | Quaternion order is wxyz. |
| `set_rpy` | method | `set_rpy(self, rpy: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | Set rpy. |  |
| `to_transformation_matrix` | method | `to_transformation_matrix(self) -> np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float32]]` | Call to transformation matrix. |  |
| `__getstate__` | method | `__getstate__(self) -> tuple` | Python special method. |  |
| `__init__` | method | `__init__(self, p: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple=..., q: np.ndarray[Literal[4], np.dtype[np.float32]] \| list[float] \| tuple=...) -> None<br>__init__(self, matrix: np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float32]] \| list \| tuple) -> None` | Python special method. |  |
| `__mul__` | method | `__mul__(self, other: Pose) -> Pose` | Python special method. |  |
| `__repr__` | method | `__repr__(self) -> str` | Python special method. |  |
| `__setstate__` | method | `__setstate__(self, arg0: tuple) -> None` | Python special method. |  |

## `sapien.Profiler`

- Use: profile context manager/decorator.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__call__` | method | `__call__(self, func: Callable) -> Callable` | Python special method. |  |
| `__enter__` | method | `__enter__(self) -> None` | Python special method. |  |
| `__exit__` | method | `__exit__(self, exec_type: type \| None, exec_value: Any \| None, traceback: Any \| None) -> None` | Python special method. |  |
| `__init__` | method | `__init__(self, name: str) -> None` | Python special method. |  |

## `sapien.Scene`

- Use: Entity/system container; high-level wrapper also provides builder, lights, camera, ground, and other convenience functions.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `add_entity` | method | `add_entity(self, entity: Entity) -> None` | Add/create entity. |  |
| `add_system` | method | `add_system(self, system: System) -> None` | Add/create system. |  |
| `clear` | method | `clear(self) -> None` | Call clear. |  |
| `entities` | property | `entities(self) -> list[Entity]` | Property: entities. |  |
| `get_entities` | method | `get_entities(self) -> list[Entity]` | Get entities. |  |
| `get_id` | method | `get_id(self) -> int` | Get id. |  |
| `get_physx_system` | method | `get_physx_system(self) -> physx.PhysxSystem` | Get physx system. |  |
| `get_render_system` | method | `get_render_system(self) -> render.RenderSystem` | Get render system. |  |
| `get_system` | method | `get_system(self, name: str) -> System` | Get system. |  |
| `id` | property | `id(self) -> int` | Property: id. |  |
| `pack_poses` | method | `pack_poses(self) -> bytes` | Call pack poses. |  |
| `physx_system` | property | `physx_system(self) -> physx.PhysxSystem` | Property: physx system. |  |
| `remove_entity` | method | `remove_entity(self, entity: Entity) -> None` | Remove entity. |  |
| `render_system` | property | `render_system(self) -> render.RenderSystem` | Property: render system. |  |
| `unpack_poses` | method | `unpack_poses(self, data: bytes) -> None` | Call unpack poses. |  |
| `__init__` | method | `__init__(self, systems: list[System]) -> None` | Python special method. |  |

## `sapien.System`

- Use: Scene system base class; step drives the system.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `step` | method | `step(self) -> None` | Advance the system/scene one step. |  |
| `__init__` | method | `__init__(self) -> None` | Python special method. |  |