# API Changes By Release

Migration-sensitive additions, removals, and behavior changes only, newest first. Signatures and parameter meanings are discoverable at runtime with `help(...)` or `__doc__`; this file records what you would otherwise call and find missing or use with the wrong lifecycle assumptions.

## Unreleased

Multi-GPU rendering:

- `sapien.render.RenderSystem(device)` systems on **different devices now coexist in one
  process** (previously the second device raised "current SAPIEN version only supports
  single-GPU rendering"). `RenderSystem()` without a device still uses the default render
  device, so single-GPU code is unchanged.
- Materials, textures, cubemaps, meshes and loaded models are device-independent and may be
  shared across devices; each device uploads lazily and edits reach every device.
- `RenderSystemGroup` and `RenderWindow`/VR raise `RuntimeError` for render systems of
  another device; windows and VR use the default render device. Camera-group CUDA outputs
  are on the group's device.
- New optional `device` arguments: `RenderCudaMeshComponent(max_vertex_count,
  max_triangle_count, device=None)` (raises when added to a scene of another device) and
  `RenderShapeTriangleMeshPart.get_cuda_vertices/get_cuda_triangles(device=None)` (the
  `cuda_vertices`/`cuda_triangles` properties keep returning the default device's copy).
- Renderer CUDA interop restores the caller's current CUDA context; `torch.cuda.current_device()`
  is unchanged by render calls.
- Fixed: with `SAPIEN_DISABLE_RAY_TRACING=1`, render devices had no CUDA id, so
  `RenderSystem("cuda:N")` failed to find its device.

CUDA interop and threads (JAX/Torch handoff without host synchronization):

- `sapien.CudaArray` implements the array-API DLPack protocol (`__dlpack__(stream=...,
  max_version=...)`, `__dlpack_device__`): `torch.from_dlpack(x)`, `jax.dlpack.from_dlpack(x)`
  and `cupy.from_dlpack(x)` take SAPIEN arrays directly, and the consumer's stream now waits
  for the owner's stream (the PhysX system's configured stream, a camera group's stream,
  otherwise the legacy default stream). `.torch()`, `.jax()` and `.cupy()` use the same
  handshake. `.jax()` used to pass a bare capsule, which current JAX rejects.
- New `CudaArray.copy_from(src)`: copy a same-shape, same-type CUDA array (JAX, Torch, CuPy,
  CAI object, `sapien.CudaArray`) into a SAPIEN buffer on its owner's stream after the
  producer finished it; the producer stays alive until the copy ran. This is how JAX writes
  actions into e.g. `cuda_articulation_target_qpos` before an apply.
- New `PhysxGpuSystem.gpu_get_cuda_stream()` and `PhysxGpuSystem.gpu_wait_for(array)` (make
  the system's stream wait for a JAX/Torch/CuPy array without blocking the host).
- Indexed GPU calls accept any `__dlpack__` producer, JAX arrays included, order the system's
  stream after the producer and keep it alive until the stream ran past the call. Inputs used
  to be read through `__cuda_array_interface__` with no stream ordering, so a JAX index array
  could be read before JAX had written it.
- **Behavior change:** `gpu_apply_*` and the Viewer wrench/teleport helpers no longer block the
  host until PhysX has copied the data; they return once enqueued, and the system's stream is
  ordered after PhysX's read. A buffer written from another stream after an apply must be
  ordered after the system's stream (Torch's default stream shares the legacy default stream
  with an unconfigured system and needs nothing).
- **Behavior change:** `wait_idle()` waits for the in-flight step and the system's own stream
  only, no longer for the whole device (other systems, Torch/JAX streams). `close()` still
  quiesces the device before releasing.
- `gpu_set_cuda_stream()` makes the new stream wait for work already queued on the previous
  one (on the device, not the host), so `wait_idle()`, DLPack exports and `copy_from` after a
  switch include that work; it used to be left unordered. Exports and `copy_from` of
  camera-group buffers follow `RenderSystemGroup.set_cuda_stream()` the same way.
- **Behavior change:** fetch, apply, compute, contact-query, upload/download and `gpu_init()`
  calls raise `RuntimeError` while a `step_start()` is in flight; PhysX used to drop applies
  silently and fail fetches with a generic error.
- Every steady-state `PhysxGpuSystem` call and `close()` release the GIL; calls into one system
  from several threads are serialized by an internal lock instead of racing. Build scenes and
  call `gpu_init()` from one thread; scene/entity/device/default-material bookkeeping is now
  thread-safe, PhysX object creation is not.
- Indexed calls with an empty index buffer return without touching PhysX. Articulation joint
  buffers (`cuda_articulation_qpos`, `qvel`, `qf`, `qacc`, `target_qpos`, `target_qvel`, the
  compensation buffers) now start on 256-byte boundaries, as XLA expects of imported buffers.

Multi-GPU PhysX:

- Single-GPU code needs no change: the Python API is the same, and `PhysxGpuSystem()`
  without a device still uses the default `"cuda"` device, the only one when one GPU is
  visible. Pass `"cuda:k"` only to choose among several visible GPUs.
- `PhysxGpuSystem(device)` systems on **different CUDA devices now coexist in one
  process** (previously the second device raised "PhysX only supports a single GPU").
  Each device gets its own PhysX CUDA context manager on its primary CUDA context. It now
  stays alive until `sapien.physx.shutdown()` or engine destruction instead of being
  released with the device's last system (fork.15.post1 leases): PhysX keeps per-manager
  kernel handles for the `PxPhysics` lifetime, so recycled managers ran stale kernels. With
  the leases (reproduced on fork.15.post2 and fork.16), closing every GPU system while any
  PhysX object kept the engine alive and then creating another failed with CUDA error 400
  or segfaulted, also on one GPU. Workarounds verified on fork.16: keep one GPU system
  alive for the whole process, or drop every PhysX object and call
  `sapien.physx.shutdown()` (then `enable_gpu()` again) between jobs.
- **Behavior change:** SAPIEN PhysX calls no longer leave the system's device current;
  they restore the caller's current CUDA device (`torch.cuda.current_device()` is
  unchanged) and never create a context on another device. With several visible GPUs,
  code that relied on a SAPIEN call selecting the device for later `device="cuda"` tensors
  must create them on the system's device; one visible GPU is unaffected.
- `gpu_set_cuda_stream(stream)` raises `RuntimeError` for a stream of another device.
- `PhysxGpuSystem.step()`, `step_start()`, `step_finish()` and `PhysxSystem.wait_idle()`
  release the GIL. `step_start()` is asynchronous only with `cpu_workers >= 1`.
- `enable_gpu()` downloads a missing PhysX GPU library under a file lock and renames it
  into place complete, so concurrent ranks no longer race on the first download.

Stream-order fixes for a non-blocking configured stream, e.g.
`gpu_set_cuda_stream(torch.cuda.Stream().cuda_stream)` (Torch's pool streams are non-blocking):

- Fixed: the first indexed `gpu_*` call could raise a false `index buffer entry 0 is outside
  the gpu_index range`; the first contact query could report zero contacts;
  `sync_poses_gpu_to_cpu()` could copy poses from before its own fetch; GPU `articulation.qpos`
  assignment could apply a value whose upload had not landed. `gpu_init()` now allocates and
  fully initializes the index-check and contact buffers. The default stream and blocking
  streams were unaffected, except that contact-buffer growth was unordered.
- Workarounds for released versions with such a stream (checked on fork.15.post2 and fork.16):
  make one indexed call and one contact query on the default stream before
  `gpu_set_cuda_stream()`; before `sync_poses_gpu_to_cpu()`, call
  `gpu_fetch_rigid_dynamic_data()`, `gpu_fetch_articulation_link_pose()` and
  `stream.synchronize()` yourself. Contact-buffer growth and qpos/target uploads have none.

Joint friction is now **Coulomb effort**, not the deprecated PhysX load-dependent
coefficient: `PhysxArticulationJoint.friction/get_friction/set_friction`,
`set_joint_frictions`, and builder `friction` use N (prismatic) or N*m
(rotational). Static and dynamic efforts are equal, finite, non-negative, default
0.05 (builder/URDF `friction` left unset keeps it); zero disables friction. URDF dynamics friction is applied and clone/reparent
preserves it. Existing GPU between-step property updates remain supported. Material
friction coefficients are unchanged. Old coefficient values are not equivalent efforts.

Other corrected behavior:

- PhysX `gpu_init()` waits for asynchronous contact queries and invalidates the
  contact snapshot, including reinitialization after scene topology changes.
- Orthographic camera aspect ratios use floating-point division; convex collision
  clones retain their properties; rejected attached-shape pose writes are atomic.
  Missing mesh files raise a descriptive RuntimeError rather than invoking UB.
- Attached light setup edits propagate and invalidate renderer resources before
  group sealing. They remain rejected after group `gpu_init()`. An attached textured
  light now rejects `texture=None`; detached components still accept it.
- External `CudaArray` accepts zero-element null-pointer arrays. It uses source
  `__dlpack_device__` if present, otherwise the current CUDA device at import.
  Non-empty pointer checks and borrowed-input ownership are unchanged.
- C++ triangle-mesh constructors add a `char const *` filename overload; string
  literals no longer convert to `bool generateSDF`.
- Native empty CUDA-event synchronization is a no-op. Wheels declare Pillow for
  dome environment-map generation.

New material/body parameters (properties and explicit `get_*`/`set_*` methods):

| API | Contract |
|---|---|
| `PhysxMaterial.friction_combine_mode`, `restitution_combine_mode` | `Literal['average', 'min', 'multiply', 'max']`, default `average`. Invalid strings/non-strings raise `TypeError`. A contact selects the higher-priority mode (`average < min < multiply < max`); friction applies to static/dynamic coefficients independently. Shared shapes/clones observe the same material. The weak default-material cache is unchanged: retain a strong reference; these are not persistent global defaults. |
| `PhysxRigidBodyComponent.max_linear_velocity`, `max_angular_velocity` | COM speed magnitude (length units/s) and angular speed magnitude (rad/s), for rigid dynamics and articulation links. Finite float32 `[0, 1e16f]`, inclusive; otherwise `RuntimeError` without mutation. Defaults remain dynamic `1e16/100`, link `100 * tolerance length/50` (SAPIEN's default tolerance length 0.1 gives linear 10). Articulation clone/reparent preserves limits. PhysX limits before solving, not a hard post-step cap, and link limiting can change momentum. Not joint velocity limits or kinematic target caps. |

For these four new properties, CPU writes are supported between steps (all scenes
sharing a material must be idle). GPU writes are only guaranteed before the first
`gpu_init()`; post-init propagation is not guaranteed, not runtime-frozen, and
CPU readback is not evidence of GPU behavior. Never write during an in-flight step.
Constructors, batch setters, global defaults and material ownership are unchanged.

`gpu_fetch_rigid_dynamic_data()` and `gpu_fetch_articulation_link_pose()` now order PhysX scratch writes after previous consumers on the configured SAPIEN CUDA stream. This fixes earlier snapshots being overwritten by a later fetch across simulation steps. Signatures are unchanged and calls remain asynchronous with respect to the CPU; host/cross-stream consumers still need an explicit wait. See `unittest/test_physx/test_gpu_stream_order.py` for delayed-consumer and contact-order regression coverage.

New configuration:

| API | Contract |
|---|---|
| `PhysxSceneConfig.gpu_index_validation` | `bool`, default `True`. Controls the device-side range check of caller-supplied index buffers (below). Set it to `False` before creating the `PhysxGpuSystem` to remove the per-call stream synchronization for trusted indices; it is frozen at construction, host-side checks (dtype, contiguity, device, entry count) still run, and out-of-range values become undefined behavior. Pickled with the config, so the pickle tuple grows from 15 to 16 fields. |

Behavior changes (signatures unchanged):

| API | Contract |
|---|---|
| Indexed `gpu_apply_*`, `gpu_compute_articulation_*`, `gpu_update_articulation_kinematics` overloads | Index values are range-checked against the rigid-dynamic or articulation `gpu_index` count. Out-of-range or negative entries, or more entries than that count, raise `RuntimeError` before any buffer or PhysX state changes; previously they read and wrote outside SAPIEN/PhysX buffers and could poison the CUDA context. The value check synchronizes the configured SAPIEN CUDA stream once per indexed call unless `PhysxSceneConfig.gpu_index_validation` is off; the entry-count check always runs. Non-indexed overloads are unchanged. |
| `Scene.add_entity`, `Entity.add_component`, `Component.enable()` | When a component rejects an add before modifying its own scene state, previously added components are removed and the entity or component is detached. GPU rigid bodies now check the scene environment ID before registering with PhysX. |
| `RenderTexturedLightComponent` | Adding it to a scene without a `texture` raises `RuntimeError` instead of crashing. |
| Grouped cameras after `RenderSystemGroup.gpu_init()` | Removing the camera's entity (or closing its Scene) waits for in-flight GPU work instead of hanging the group's later `close()`. The group's `update_render()` / `take_picture()` then raise, and re-adding the sealed camera to a scene raises until the group is closed. |
| `sapien.Scene` wrapper | Removed the redundant `__del__` that called `clear()`; `~Scene` already closes the scene. A failing `Scene(...)` constructor now raises its exception instead of segfaulting. |

## 3.0.0+fork.15.post2

PhysX GPU contact-query scheduling:

| API | Contract |
|---|---|
| `gpu_query_contact_pair_impulses(query, synchronize=True)` | Existing behavior remains synchronous by default. `False` enqueues the query on the stream configured by `gpu_set_cuda_stream`. |
| `gpu_query_contact_body_impulses(query, synchronize=True)` | Same optional synchronization contract for net body impulses. |
| `gpu_wait_contact_queries()` | Synchronize queued contact copies/query kernels before host or cross-stream access. `step()` / `step_start()` also wait for an outstanding asynchronous query before the next simulation invalidates contact pointers. |

Contact data storage grows automatically when the reported contact count exceeds the current capacity.

Articulation qpos/qvel/qacc/target, link-velocity, and incoming-joint-force fetches are ordered onto the stream configured by `gpu_set_cuda_stream`. Existing fetch signatures are unchanged. Host download helpers wait for that stream before copying to CPU; callers performing their own host or cross-stream reads must do the same.

## 3.0.0+fork.15.post1

New terminal lifecycle surface:

| API | Contract |
|---|---|
| `Scene.close()` / `is_closed` | Terminal and idempotent; removes entities, detaches systems, returns managed environment-ID slots, and rejects reuse. Use `Scene.clear()` when the Scene must remain reusable. |
| `PhysxSystem.close()` / `is_closed` | Close every owning Scene and release exported CUDA views first. Closed systems reject steady-state APIs. `PhysxGpuSystem.wait_idle()` drains in-flight simulation/CUDA work and is called by `close()`. |
| `RenderCameraGroup.close()`, `RenderSystemGroup.close()`, `RenderSystem.close()` | Close render groups before Scenes, then close detached render systems. `RenderSystemGroup.close()` closes its retained camera groups. |
| `sapien.physx.get_live_resources()` / `can_shutdown()` / `shutdown()` | Diagnose blockers, preflight, then release PhysX caches, per-device CUDA-context-manager leases, engine, and defaults. |
| `sapien.render.get_live_resources()` / `can_shutdown()` / `shutdown()` | Diagnose blockers, clear library-owned default-ground textures, wait for Vulkan/CUDA work, and release the render engine/context. |
| `sapien.get_live_resources()` / `can_shutdown()` / `shutdown()` | Side-effect-free joint preflight; shutdown releases render before PhysX so a later job can initialize both again in the same Python process. |

Behavior and ownership changes:

- `Scene`, `PhysxSystem`, `RenderSystem`, `RenderSystemGroup`, and `RenderCameraGroup` support context managers whose exit calls terminal `close()`.
- CUDA arrays owned by `PhysxGpuSystem`, `RenderSystem.cuda_object_transforms`, and `RenderCameraGroup` image/pose buffers reject raw `__cuda_array_interface__` export. Use `.torch()`, `.jax()`, `.cupy()`, or `.dlpack()`; the derived consumer carries a lifecycle guard and must be released before the owner can close. Component/shape-owned CUDA buffers without an explicit close owner retain borrowed-view semantics.
- `sapien.shutdown()` never calls `cudaDeviceReset()`: it preserves the CUDA primary context shared with Torch/JAX. Use a fresh process when process-exit-equivalent CUDA isolation is required.

Build-only post-release fix:

- `fork.15.post1` renames the C++ parameter identifiers of inline light shadow setters from the Windows SDK macro names `near` / `far` to `value`. Python signatures and setter behavior are unchanged.

## 3.0.0+fork.14

Removed, with replacement:

| Removed | Use instead |
|---|---|
| `PhysxGpuSystem.set_scene_environment_id(s)`, bulk setter, `allow_duplicate` | `Scene.set_environment_id()`, or declare `num_scenes` |
| `Scene.environment_id` setter | `Scene.set_environment_id()`; property is read-only |
| `PhysxSceneConfig.gpu_broadphase_env_id_bits`, writable `gpu_broadphase_nb_bits_env_id_x/y/z` | `set_gpu_broadphase_env_id_bits(x, y, z)` |
| `set_gpu_broadphase_env_count`, `gpu_broadphase_env_band_count`, `gpu_broadphase_max_env_count` | declare `num_scenes` |
| module-level `broadphase_env_id_bits_for_env_count`, `max_broadphase_env_count`, `broadphase_env_id_window` | — |
| `PhysxGpuSystem.has_shared_environment_scene`, `has_managed_broadphase_env_ids`, `manages_collision_group_scene_ids`, `broadphase_env_id_window`, `broadphase_env_band_count` | — |
| `RenderMaterial.diffuse_texture` and its getter/setter | `base_color_texture` |

Behavior that no docstring will warn you about:

- Environment IDs have two disjoint modes. `num_scenes = N` makes SAPIEN the sole assigner and `Scene.set_environment_id()` raises; leaving it unset makes `Scene.get_or_assign_environment_id()` raise instead. In manual mode duplicate IDs are legal and put those scenes in one environment.
- `num_scenes` caps scenes held *at once*, not over the run; destroyed or shared scenes return their slot.
- `Scene.environment_id` is the raw ID; `PhysxGpuSystem.get_broadphase_environment_id()` is what PhysX receives. They differ only when a shared scene exists.
- `set_light_directions` rejects spot and textured lights.
- Misconfigurations raise at `PhysxGpuSystem()` construction, not at first body.

Packaging:

- PhysX Linux binaries moved to the `<version>-Linux` release tag, matching `-windows`. Affects any Docker pre-bake step; see [`../gpu-workflows.md`](../gpu-workflows.md).
- Wheels before fork.14 claimed `manylinux_2_28` but needed glibc 2.38 and failed to import on 2.28–2.37.
