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
| `sapien.can_shutdown` | `can_shutdown() -> bool` | Side-effect-free render + PhysX shutdown preflight. | Diagnose false with `get_live_resources()`. |
| `sapien.compiled_with_cxx11_abi` | `compiled_with_cxx11_abi() -> bool` | Call compiled with cxx11 abi. |  |
| `sapien.get_live_resources` | `get_live_resources() -> dict[str, dict[str, object]]` | Snapshot caller-owned render and PhysX resources blocking shutdown. |  |
| `sapien.profile` | `profile(name: str) -> Profiler<br>profile(func: Callable) -> Callable` | Call profile. |  |
| `sapien.pybind11_internals_id` | `pybind11_internals_id() -> str` | Call pybind11 internals id. |  |
| `sapien.pybind11_use_smart_holder` | `pybind11_use_smart_holder() -> bool` | Call pybind11 use smart holder. |  |
| `sapien.set_log_level` | `set_log_level(level: str) -> None` | Set log level. |  |
| `sapien.shutdown` | `shutdown() -> None` | Preflight both subsystems, then terminally release render before PhysX. | No partial shutdown on failed preflight; preserves the shared CUDA primary context. |

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
| `entity_pose` | `Pose` |  |  |
| `name` | `str` |  |  |
| `pose` | `Pose` |  |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `disable` | method | `disable(self) -> None` | disable the component |  |
| `enable` | method | `enable(self) -> None` | enable the component |  |
| `entity` | property | `entity(self) -> Entity` |  |  |
| `get_entity` | method | `get_entity(self) -> Entity` |  |  |
| `get_entity_pose` | method | `get_entity_pose(self) -> Pose` |  |  |
| `get_name` | method | `get_name(self) -> str` |  |  |
| `get_pose` | method | `get_pose(self) -> Pose` |  |  |
| `is_enabled` | property | `is_enabled(self) -> bool` |  |  |
| `set_entity_pose` | method | `set_entity_pose(self, pose: Pose) -> None` |  |  |
| `set_name` | method | `set_name(self, name: str) -> None` |  |  |
| `set_pose` | method | `set_pose(self, pose: Pose) -> None` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.CudaArray`

- Use: CUDA memory view; convertible to torch/cupy/jax/DLPack.
- Bases: `-`
- Ownership: arrays exported from `PhysxGpuSystem`, `RenderSystem.cuda_object_transforms`, and `RenderCameraGroup` image/pose buffers are owner-tracked. Their `.torch()`, `.jax()`, `.cupy()`, and `.dlpack()` consumers keep the owner alive and block `close()` until released; raw `__cuda_array_interface__` export raises because it cannot carry that guard. Component/shape-owned CUDA buffers without an explicit close owner remain borrowed views whose original owner must stay alive.

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `cuda_id` | property | `cuda_id(self) -> int` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cupy` | method | `cupy(self) -> cupy.ndarray` | Return a CuPy CUDA array view. |  |
| `dlpack` | method | `dlpack(self) -> Any` | Return a DLPack capsule. |  |
| `jax` | method | `jax(self) -> jax.Array` | Return a JAX CUDA array view. |  |
| `ptr` | property | `ptr(self) -> int` |  |  |
| `shape` | property | `shape(self) -> list[int]` |  |  |
| `strides` | property | `strides(self) -> list[int]` |  |  |
| `torch` | method | `torch(self) -> torch.Tensor` | Return a torch CUDA tensor view; cache outside the loop. | Owner-tracked tensors carry the lifecycle guard themselves and must be released before owner teardown; borrowed buffers still require their original owner to remain alive. |
| `typestr` | property | `typestr(self) -> str` |  |  |
| `__cuda_array_interface__` | property | `__cuda_array_interface__(self) -> dict` | Export a borrowed raw CUDA view only when the buffer has no tracked close owner. | Raises for owner-tracked system/group buffers; use `.torch()`, `.jax()`, `.cupy()`, or `.dlpack()` instead. |
| `__init__` | method | `__init__(self, data: Any) -> None` |  |  |

## `sapien.Device`

- Use: CPU/CUDA/render device selection and capability query.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `can_access_peer` | method | `can_access_peer(self, peer: Device) -> bool` | Query CUDA peer access to another device. | Cross-device Viewer `auto` transport remains staged even when peer access is available. |
| `can_direct_cuda_vulkan_interop` | method | `can_direct_cuda_vulkan_interop(self) -> bool` | Check whether this device exposes the CUDA/Vulkan external-memory and semaphore capabilities required by direct Viewer transport. | Direct transport also requires PhysX and Vulkan to identify the same physical device. |
| `can_present` | method | `can_present(self) -> bool` |  |  |
| `can_render` | method | `can_render(self) -> bool` |  |  |
| `cuda_external_memory` | property | `cuda_external_memory(self) -> bool` | Report CUDA external-memory support. | Viewer transport diagnostic. |
| `cuda_external_semaphore` | property | `cuda_external_semaphore(self) -> bool` | Report CUDA external-semaphore support. | Viewer transport diagnostic. |
| `cuda_id` | property | `cuda_id(self) -> int` | CUDA device ordinal. |  |
| `is_cpu` | method | `is_cpu(self) -> bool` |  |  |
| `is_cuda` | method | `is_cuda(self) -> bool` |  |  |
| `name` | property | `name(self) -> str` |  |  |
| `pci_string` | property | `pci_string(self) -> str \| None` | PCI identity when available. | Used with `uuid` to diagnose CUDA/Vulkan physical-device matching. |
| `uuid` | property | `uuid(self) -> str \| None` | Physical-device UUID when available. | Used by automatic Viewer transport selection. |
| `vulkan_external_memory` | property | `vulkan_external_memory(self) -> bool` | Report Vulkan external-memory support. | Viewer transport diagnostic. |
| `vulkan_external_semaphore` | property | `vulkan_external_semaphore(self) -> bool` | Report Vulkan external-semaphore support. | Viewer transport diagnostic. |
| `__init__` | method | `__init__(self, alias: str) -> None` |  |  |
| `__repr__` | method | `__repr__(self) -> str` |  |  |
| `__str__` | method | `__str__(self) -> str` |  |  |

## `sapien.Entity`

- Use: Entity in a scene; holds components and world pose.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `name` | `str` |  |  |
| `pose` | `Pose` |  |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `add_component` | method | `add_component(self, component: Component) -> Entity` |  |  |
| `add_to_scene` | method | `add_to_scene(self, scene: Scene) -> Entity` | Add/create to scene. |  |
| `components` | property | `components(self) -> list[Component]` |  |  |
| `find_component_by_type` | method | `find_component_by_type(self, cls: Type[_T]) -> _T` | Call find component by type. |  |
| `get_components` | method | `get_components(self) -> list[Component]` |  |  |
| `get_global_id` | method | `get_global_id(self) -> int` |  |  |
| `get_name` | method | `get_name(self) -> str` |  |  |
| `get_per_scene_id` | method | `get_per_scene_id(self) -> int` |  |  |
| `get_pose` | method | `get_pose(self) -> Pose` |  |  |
| `get_scene` | method | `get_scene(self) -> Scene` |  |  |
| `global_id` | property | `global_id(self) -> int` |  |  |
| `per_scene_id` | property | `per_scene_id(self) -> int` |  |  |
| `remove_component` | method | `remove_component(self, component: Component) -> None` |  |  |
| `remove_from_scene` | method | `remove_from_scene(self) -> None` | Remove from scene. |  |
| `scene` | property | `scene(self) -> Scene` |  |  |
| `set_name` | method | `set_name(self, name: str) -> None` |  |  |
| `set_pose` | method | `set_pose(self, pose: Pose) -> None` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.Pose`

- Use: SE(3) pose; p is xyz, q is wxyz quaternion.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `p` | `np.ndarray[Literal[3], np.dtype[np.float32]]` |  |  |
| `q` | `np.ndarray[Literal[4], np.dtype[np.float32]]` |  | Quaternion order is wxyz. |
| `rpy` | `np.ndarray[Literal[3], np.dtype[np.float32]]` |  |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_p` | method | `get_p(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` |  |  |
| `get_q` | method | `get_q(self) -> np.ndarray[Literal[4], np.dtype[np.float32]]` |  | Quaternion order is wxyz. |
| `get_rpy` | method | `get_rpy(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` |  |  |
| `inv` | method | `inv(self) -> Pose` |  |  |
| `set_p` | method | `set_p(self, p: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` |  |  |
| `set_q` | method | `set_q(self, q: np.ndarray[Literal[4], np.dtype[np.float32]] \| list[float] \| tuple) -> None` |  | Quaternion order is wxyz. |
| `set_rpy` | method | `set_rpy(self, rpy: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` |  |  |
| `to_transformation_matrix` | method | `to_transformation_matrix(self) -> np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float32]]` | Call to transformation matrix. |  |
| `__getstate__` | method | `__getstate__(self) -> tuple` |  |  |
| `__init__` | method | `__init__(self, p: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple=..., q: np.ndarray[Literal[4], np.dtype[np.float32]] \| list[float] \| tuple=...) -> None<br>__init__(self, matrix: np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float32]] \| list \| tuple) -> None` |  |  |
| `__mul__` | method | `__mul__(self, other: Pose) -> Pose` |  |  |
| `__repr__` | method | `__repr__(self) -> str` |  |  |
| `__setstate__` | method | `__setstate__(self, arg0: tuple) -> None` |  |  |

## `sapien.Profiler`

- Use: profile context manager/decorator.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__call__` | method | `__call__(self, func: Callable) -> Callable` |  |  |
| `__enter__` | method | `__enter__(self) -> None` |  |  |
| `__exit__` | method | `__exit__(self, exec_type: type \| None, exec_value: Any \| None, traceback: Any \| None) -> None` |  |  |
| `__init__` | method | `__init__(self, name: str) -> None` |  |  |

## `sapien.Scene`

- Use: Entity/system container; high-level wrapper also provides builder, lights, camera, ground, and other convenience functions.
- Bases: `-`
- Lifecycle: supports a context manager; exit calls terminal, idempotent `close()`. Use `clear()` instead when the Scene must remain reusable.

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `add_entity` | method | `add_entity(self, entity: Entity) -> None` |  |  |
| `add_system` | method | `add_system(self, system: System) -> None` |  |  |
| `clear` | method | `clear(self) -> None` | Remove entities while keeping systems attached so the Scene can be reused. |  |
| `close` | method | `close(self) -> None` | Terminally remove entities, detach systems and close the Scene. | Idempotent; mutating operations reject reuse. |
| `is_closed` | property | `is_closed(self) -> bool` | Terminal lifecycle state. |  |
| `entities` | property | `entities(self) -> list[Entity]` |  |  |
| `get_entities` | method | `get_entities(self) -> list[Entity]` |  |  |
| `get_id` | method | `get_id(self) -> int` |  |  |
| `get_physx_system` | method | `get_physx_system(self) -> physx.PhysxSystem` |  |  |
| `get_render_system` | method | `get_render_system(self) -> render.RenderSystem` |  |  |
| `get_system` | method | `get_system(self, name: str) -> System` |  |  |
| `id` | property | `id(self) -> int` |  |  |
| `pack_poses` | method | `pack_poses(self) -> bytes` |  |  |
| `physx_system` | property | `physx_system(self) -> physx.PhysxSystem` |  |  |
| `remove_entity` | method | `remove_entity(self, entity: Entity) -> None` |  |  |
| `render_system` | property | `render_system(self) -> render.RenderSystem` |  |  |
| `unpack_poses` | method | `unpack_poses(self, data: bytes) -> None` |  |  |
| `__init__` | method | `__init__(self, systems: list[System]) -> None` |  |  |

## `sapien.System`

- Use: Scene system base class; step drives the system.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `step` | method | `step(self) -> None` | Advance the system/scene one step. |  |
| `__init__` | method | `__init__(self) -> None` |  |  |