# API Changes By Release

Migration-sensitive additions, removals, and behavior changes only, newest first. Signatures and parameter meanings are discoverable at runtime with `help(...)` or `__doc__`; this file records what you would otherwise call and find missing or use with the wrong lifecycle assumptions.

## Unreleased

`gpu_fetch_rigid_dynamic_data()` and `gpu_fetch_articulation_link_pose()` now order PhysX scratch writes after previous consumers on the configured SAPIEN CUDA stream. This fixes earlier snapshots being overwritten by a later fetch across simulation steps. Signatures are unchanged and calls remain asynchronous with respect to the CPU; host/cross-stream consumers still need an explicit wait. See `unittest/test_physx/test_gpu_stream_order.py` for delayed-consumer and contact-order regression coverage.

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
