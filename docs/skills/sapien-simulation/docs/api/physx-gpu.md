# PhysX GPU Direct API

Agent-facing GPU table. Pair with `../gpu-workflows.md`; this file is lookup, not workflow prose.

Hard rules: `enable_gpu()` before `PhysxGpuSystem`; configure memory/scene/body/shape before system creation; assign env ids before adding bodies; `gpu_init()` after building; fetch before reads; apply only modified buffers; `sync_poses_gpu_to_cpu()` only for explicit CPU debugging or Viewer `cpu-debug`. Exposed after `gpu_init()`: PhysX GPU link-data and dense-Jacobian buffers.

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.physx`
- Source files: `python/py_package/pysapien/physx.pyi`, `python/py_package/physx/__init__.pyi`, `python/pybind/physx.cpp`, `src/physx/physx_system.cpp`
- Notes: Use with `../gpu-workflows.md`.
- Indexed overloads accept only `sapien.CudaArray` or CUDA-array-interface objects, for example CUDA `torch.Tensor`, `cupy.ndarray`, or Numba CUDA device arrays. Index buffers must be 1D contiguous CUDA `int32` arrays on the same CUDA device as the PhysX system and contain SAPIEN `gpu_index` values, not PhysX-internal GPU indices. NumPy arrays, Python lists, CPU tensors, `int64` tensors, non-contiguous views, and cross-device CUDA arrays are invalid.

## Functions

| API | Signature | Use | Notes |
|---|---|---|---|
| `sapien.physx.enable_gpu` | `enable_gpu() -> None` | Enable the PhysX GPU runtime; call before creating a PhysxGpuSystem. | Pre-bake the PhysX GPU .so into the package to avoid runtime download. |

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.physx.PhysxGpuContactBodyImpulseQuery` |  | PhysX physics API object. |  |
| `sapien.physx.PhysxGpuContactPairImpulseQuery` |  | PhysX physics API object. |  |
| `sapien.physx.PhysxGpuSystem` | `PhysxSystem` | PhysX GPU Direct system; shares GPU scene, CUDA state buffers and batched APIs. | Call sapien.physx.enable_gpu() first |

## `sapien.physx.PhysxGpuContactBodyImpulseQuery`

- Use: PhysX physics API object.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `cuda_impulses` | property | `cuda_impulses(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |

## `sapien.physx.PhysxGpuContactPairImpulseQuery`

- Use: PhysX physics API object.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `cuda_impulses` | property | `cuda_impulses(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |

## `sapien.physx.PhysxGpuSystem`

- Use: PhysX GPU Direct system; shares GPU scene, CUDA state buffers and batched APIs.
- Bases: `PhysxSystem`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `cuda_articulation_coriolis_and_centrifugal_compensation` | property | `cuda_articulation_coriolis_and_centrifugal_compensation(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cuda_articulation_gravity_compensation` | property | `cuda_articulation_gravity_compensation(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cuda_articulation_link_data` | property | `cuda_articulation_link_data(self) -> sapien.CudaArray` | Padded articulation link pose/velocity buffer. | Shape `(articulation_count, max_links, 13)` indexed by `articulation.gpu_index` and low-level `link.index`; channels `0:3` position, `3:7` quaternion `wxyz`, `7:10` linear velocity, `10:13` angular velocity. |
| `cuda_articulation_link_force` | property | `cuda_articulation_link_force(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cuda_articulation_link_incoming_joint_forces` | property | `cuda_articulation_link_incoming_joint_forces(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cuda_articulation_link_torque` | property | `cuda_articulation_link_torque(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cuda_articulation_qacc` | property | `cuda_articulation_qacc(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cuda_articulation_qf` | property | `cuda_articulation_qf(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cuda_articulation_qpos` | property | `cuda_articulation_qpos(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cuda_articulation_jacobian` | property | `cuda_articulation_jacobian(self) -> sapien.CudaArray` | Padded dense articulation Jacobians on GPU. | Shape `(articulation_count, max_rows, max_cols)` where `max_rows = 6 + (max_links - 1) * 6` and `max_cols = 6 + max_dofs`; slice valid submatrix with `cuda_articulation_jacobian_shape` or `articulation.get_jacobian_shape()`. |
| `cuda_articulation_jacobian_shape` | property | `cuda_articulation_jacobian_shape(self) -> sapien.CudaArray` | Valid dense Jacobian shape per articulation. | Shape `(articulation_count, 2)`, dtype `uint32`, indexed by `articulation.gpu_index`; columns are `[rows, cols]` for the valid submatrix inside `cuda_articulation_jacobian`. |
| `cuda_articulation_qvel` | property | `cuda_articulation_qvel(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cuda_articulation_target_qpos` | property | `cuda_articulation_target_qpos(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cuda_articulation_target_qvel` | property | `cuda_articulation_target_qvel(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cuda_rigid_body_data` | property | `cuda_rigid_body_data(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cuda_rigid_body_force` | property | `cuda_rigid_body_force(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cuda_rigid_body_torque` | property | `cuda_rigid_body_torque(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cuda_rigid_dynamic_data` | property | `cuda_rigid_dynamic_data(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cuda_rigid_dynamic_force` | property | `cuda_rigid_dynamic_force(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cuda_rigid_dynamic_torque` | property | `cuda_rigid_dynamic_torque(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `device` | property | `device(self) -> sapien.Device` |  |  |
| `get_assigned_scene_environment_id` | method | `get_assigned_scene_environment_id(self, scene: sapien.Scene) -> int \| None` | Only read assigned env ids; no side effects. |  |
| `get_or_assign_scene_environment_id` | method | `get_or_assign_scene_environment_id(self, scene: sapien.Scene) -> int` | Read or lazily assign a unique env id. |  |
| `get_scene_environment_id` | method | `get_scene_environment_id(self, scene: sapien.Scene) -> int` |  |  |
| `get_scene_offset` | method | `get_scene_offset(self, scene: sapien.Scene) -> np.ndarray[Literal[3], np.dtype[np.float32]]` |  |  |
| `gpu_apply_articulation_link_force` | method | `gpu_apply_articulation_link_force(self) -> None<br>gpu_apply_articulation_link_force(self, index_buffer: typing.Any) -> None` | GPU: apply written cuda_* buffers to the PhysX runtime. | Only apply buffers changed this step; avoid overwriting with stale state. |
| `gpu_apply_articulation_link_torque` | method | `gpu_apply_articulation_link_torque(self) -> None<br>gpu_apply_articulation_link_torque(self, index_buffer: typing.Any) -> None` | GPU: apply written cuda_* buffers to the PhysX runtime. | Only apply buffers changed this step; avoid overwriting with stale state. |
| `gpu_apply_articulation_qf` | method | `gpu_apply_articulation_qf(self) -> None<br>gpu_apply_articulation_qf(self, index_buffer: typing.Any) -> None` | GPU: apply written cuda_* buffers to the PhysX runtime. | Only apply buffers changed this step; avoid overwriting with stale state. |
| `gpu_apply_articulation_qpos` | method | `gpu_apply_articulation_qpos(self) -> None<br>gpu_apply_articulation_qpos(self, index_buffer: typing.Any) -> None` | GPU: apply written cuda_* buffers to the PhysX runtime. | Only apply buffers changed this step; avoid overwriting with stale state. |
| `gpu_apply_articulation_qvel` | method | `gpu_apply_articulation_qvel(self) -> None<br>gpu_apply_articulation_qvel(self, index_buffer: typing.Any) -> None` | GPU: apply written cuda_* buffers to the PhysX runtime. | Only apply buffers changed this step; avoid overwriting with stale state. |
| `gpu_apply_articulation_root_pose` | method | `gpu_apply_articulation_root_pose(self) -> None<br>gpu_apply_articulation_root_pose(self, index_buffer: typing.Any) -> None` | GPU: apply written cuda_* buffers to the PhysX runtime. | Only apply buffers changed this step; avoid overwriting with stale state. |
| `gpu_apply_articulation_root_velocity` | method | `gpu_apply_articulation_root_velocity(self) -> None<br>gpu_apply_articulation_root_velocity(self, index_buffer: typing.Any) -> None` | GPU: apply written cuda_* buffers to the PhysX runtime. | Only apply buffers changed this step; avoid overwriting with stale state. |
| `gpu_apply_articulation_target_position` | method | `gpu_apply_articulation_target_position(self) -> None<br>gpu_apply_articulation_target_position(self, index_buffer: typing.Any) -> None` | GPU: apply written cuda_* buffers to the PhysX runtime. | Only apply buffers changed this step; avoid overwriting with stale state. |
| `gpu_apply_articulation_target_velocity` | method | `gpu_apply_articulation_target_velocity(self) -> None<br>gpu_apply_articulation_target_velocity(self, index_buffer: typing.Any) -> None` | GPU: apply written cuda_* buffers to the PhysX runtime. | Only apply buffers changed this step; avoid overwriting with stale state. |
| `gpu_apply_rigid_dynamic_data` | method | `gpu_apply_rigid_dynamic_data(self) -> None<br>gpu_apply_rigid_dynamic_data(self, index_buffer: typing.Any) -> None` | GPU: apply written cuda_* buffers to the PhysX runtime. | Only apply buffers changed this step; avoid overwriting with stale state. |
| `gpu_apply_rigid_dynamic_force` | method | `gpu_apply_rigid_dynamic_force(self) -> None` | GPU: apply written cuda_* buffers to the PhysX runtime. | Only apply buffers changed this step; avoid overwriting with stale state. |
| `gpu_apply_rigid_dynamic_torque` | method | `gpu_apply_rigid_dynamic_torque(self) -> None` | GPU: apply written cuda_* buffers to the PhysX runtime. | Only apply buffers changed this step; avoid overwriting with stale state. |
| `gpu_compute_articulation_coriolis_and_centrifugal_compensation` | method | `gpu_compute_articulation_coriolis_and_centrifugal_compensation(self) -> None<br>gpu_compute_articulation_coriolis_and_centrifugal_compensation(self, gpu_indices: typing.Any) -> None` | GPU: compute articulation compensation terms into cuda_* buffers. | Selected overload accepts only `sapien.CudaArray` or CUDA-array-interface objects such as CUDA `torch.Tensor`, `cupy.ndarray`, or Numba CUDA device arrays; data must be 1D contiguous CUDA `int32` SAPIEN `gpu_index` values on the PhysX device. |
| `gpu_compute_articulation_gravity_compensation` | method | `gpu_compute_articulation_gravity_compensation(self) -> None<br>gpu_compute_articulation_gravity_compensation(self, gpu_indices: typing.Any) -> None` | GPU: compute articulation compensation terms into cuda_* buffers. | Selected overload accepts only `sapien.CudaArray` or CUDA-array-interface objects such as CUDA `torch.Tensor`, `cupy.ndarray`, or Numba CUDA device arrays; data must be 1D contiguous CUDA `int32` SAPIEN `gpu_index` values on the PhysX device. |
| `gpu_compute_articulation_jacobian` | method | `gpu_compute_articulation_jacobian(self) -> None<br>gpu_compute_articulation_jacobian(self, gpu_indices: typing.Any) -> None` | Compute dense articulation Jacobians into `cuda_articulation_jacobian`. | Selected overload accepts only `sapien.CudaArray` or CUDA-array-interface objects such as CUDA `torch.Tensor`, `cupy.ndarray`, or Numba CUDA device arrays; updates only selected padded entries. |
| `gpu_create_contact_body_impulse_query` | method | `gpu_create_contact_body_impulse_query(self, bodies: list[PhysxRigidBaseComponent]) -> PhysxGpuContactBodyImpulseQuery` |  |  |
| `gpu_create_contact_pair_impulse_query` | method | `gpu_create_contact_pair_impulse_query(self, body_pairs: list[tuple[PhysxRigidBaseComponent, PhysxRigidBaseComponent]]) -> PhysxGpuContactPairImpulseQuery` |  |  |
| `gpu_fetch_articulation_link_incoming_joint_forces` | method | `gpu_fetch_articulation_link_incoming_joint_forces(self) -> None` | GPU: fetch from the PhysX runtime into the corresponding cuda_* buffers. | Fetch the corresponding buffer before reading state. |
| `gpu_fetch_articulation_link_pose` | method | `gpu_fetch_articulation_link_pose(self) -> None` | GPU: fetch from the PhysX runtime into the corresponding cuda_* buffers. | Fetch the corresponding buffer before reading state. |
| `gpu_fetch_articulation_link_velocity` | method | `gpu_fetch_articulation_link_velocity(self) -> None` | GPU: fetch from the PhysX runtime into the corresponding cuda_* buffers. | Fetch the corresponding buffer before reading state. |
| `gpu_fetch_articulation_qacc` | method | `gpu_fetch_articulation_qacc(self) -> None` | GPU: fetch from the PhysX runtime into the corresponding cuda_* buffers. | Fetch the corresponding buffer before reading state. |
| `gpu_fetch_articulation_qpos` | method | `gpu_fetch_articulation_qpos(self) -> None` | GPU: fetch from the PhysX runtime into the corresponding cuda_* buffers. | Fetch the corresponding buffer before reading state. |
| `gpu_fetch_articulation_qvel` | method | `gpu_fetch_articulation_qvel(self) -> None` | GPU: fetch from the PhysX runtime into the corresponding cuda_* buffers. | Fetch the corresponding buffer before reading state. |
| `gpu_fetch_articulation_target_qpos` | method | `gpu_fetch_articulation_target_qpos(self) -> None` | GPU: fetch from the PhysX runtime into the corresponding cuda_* buffers. | Fetch the corresponding buffer before reading state. |
| `gpu_fetch_articulation_target_qvel` | method | `gpu_fetch_articulation_target_qvel(self) -> None` | GPU: fetch from the PhysX runtime into the corresponding cuda_* buffers. | Fetch the corresponding buffer before reading state. |
| `gpu_fetch_rigid_dynamic_data` | method | `gpu_fetch_rigid_dynamic_data(self) -> None` | GPU: fetch from the PhysX runtime into the corresponding cuda_* buffers. | Fetch the corresponding buffer before reading state. |
| `gpu_init` | method | `gpu_init(self) -> None` | "Warm start" the GPU simulation by stepping the system once. This function must be called each time when actors are added or removed from the scene. One may call `gpu_... | After all actors/articulations are built; rebuild after adding/removing bodies. |
| `gpu_query_contact_body_impulses` | method | `gpu_query_contact_body_impulses(self, query: PhysxGpuContactBodyImpulseQuery) -> None` | Query net contact forces for specific bodies of the last simulation step. Usage: query = system.gpu_create_contact_body_force_query(bodies) # create force query in adv... |  |
| `gpu_query_contact_pair_impulses` | method | `gpu_query_contact_pair_impulses(self, query: PhysxGpuContactPairImpulseQuery) -> None` |  |  |
| `gpu_set_cuda_stream` | method | `gpu_set_cuda_stream(self, stream: int) -> None` | PhysX GPU APIs will be synchronized with the provided stream and SAPIEN's CUDA kernels will be launched to the provided stream. Args: stream: integer representation of... |  |
| `gpu_update_articulation_kinematics` | method | `gpu_update_articulation_kinematics(self) -> None<br>gpu_update_articulation_kinematics(self, index_buffer: typing.Any) -> None` | Update articulation kinematics after applying qpos/root state. | `index_buffer` overload updates selected articulation `gpu_index` rows; buffer must follow the indexed overload rule; call before fetching link poses or computing Jacobians after qpos writes. |
| `set_scene_environment_id` | method | `set_scene_environment_id(self, scene: sapien.Scene, env_id: int, allow_duplicate: bool=False) -> None` | Set the PhysX GPU broadphase env id; call before adding bodies. | Must be set before adding PhysX bodies; -1/0xffffffff means shared. |
| `set_scene_environment_ids` | method | `set_scene_environment_ids(self, mapping: list[tuple[sapien.Scene, int]], allow_duplicate: bool=False) -> None` |  |  |
| `set_scene_offset` | method | `set_scene_offset(self, scene: sapien.Scene, offset: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | In GPU mode, all SAPIEN scenes share the same PhysX scene. One should call this function to apply an offset to avoid bodies in different scenes interfere with each oth... | Must be set before adding PhysX bodies; prefer env-id isolation currently. |
| `step_finish` | method | `step_finish(self) -> None` |  |  |
| `step_start` | method | `step_start(self) -> None` |  |  |
| `sync_poses_gpu_to_cpu` | method | `sync_poses_gpu_to_cpu(self) -> None` | Warning: this function is super slow and for debug only. Download all poses from the GPU and copy to SAPIEN entities. | Not for normal Viewer/training/sensor/video paths; explicit CPU debugging only. |
| `__init__` | method | `__init__(self, device: str='cuda') -> None<br>__init__(self, device: sapien.Device) -> None` |  |  |

## GPU articulation link/Jacobian buffer reference

- SAPIEN has no IK solver; `sapien.physx.GpuInverseKinematicsSolver` and `sapien.physx.gpu_inverse_kinematics` do not exist.
- Exposed PhysX GPU buffers on `PhysxGpuSystem` after `gpu_init()`:
  - `cuda_articulation_link_data` for current link poses/velocities.
  - `gpu_compute_articulation_jacobian(...)` plus `cuda_articulation_jacobian` for dense Jacobians.
- Use `cuda_articulation_jacobian_shape[articulation.gpu_index]` or `articulation.get_jacobian_shape()` to slice each valid Jacobian. Do not treat the padded buffer shape as the valid matrix shape.
- Jacobian rows are `[vx, vy, vz, wx, wy, wz]` per low-level link index. Fixed-base articulations omit root rows; floating-base articulations include six root velocity columns. PhysX linear rows are at link centers of mass; shift them in application code when solving for link-frame origins.
- Selected-articulation APIs require a CUDA `int32` SAPIEN `gpu_index` buffer as described in the indexed overload rule above.