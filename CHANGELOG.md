# Changelog

Notable user-facing and developer-facing changes are documented here.
Release descriptions are written from reviewed commits and diffs, then passed to the tagged release workflow as input.

## Unreleased

## 3.0.0+fork.15.post1 - 2026-08-25

### Added

- Added terminal, idempotent lifecycle APIs for job-scoped teardown: `Scene.close()` / `is_closed`, `PhysxSystem.close()` / `is_closed`, `PhysxGpuSystem.wait_idle()` and context-manager support. A PhysX system refuses to close while scenes, registered components, or exported CUDA views remain alive; closed objects reject steady-state use.
- Added explicit render lifecycle APIs: `RenderCameraGroup.close()`, `RenderSystemGroup.close()`, `RenderSystem.close()`, `sapien.render.get_live_resources()` / `can_shutdown()` / `shutdown()`, and top-level `sapien.get_live_resources()` / `can_shutdown()` / `shutdown()`. Top-level shutdown preflights both subsystems, closes render before PhysX, waits for Vulkan/CUDA work and recreates the weak RenderEngine/Vulkan context on the next job.
- Added `sapien.physx.get_live_resources()`, `can_shutdown()` and `shutdown()`. Shutdown clears SAPIEN-owned PhysX mesh/default caches, releases per-device `PxCudaContextManager` leases, destroys `PhysxEngine` (`PxPhysics` / `PxFoundation`) and resets PhysX defaults so a new job can initialize PhysX again in the same Python process.
- Added owner-backed DLPack exports for PhysX system CUDA arrays, RenderSystem transform arrays, and RenderCameraGroup image/pose arrays. `.torch()`, `.jax()`, `.cupy()` and `.dlpack()` keep a lifecycle guard until every consumer is released, and raw `__cuda_array_interface__` export is rejected for these tracked buffers to prevent silent use-after-free. Component/shape-owned CUDA buffers that have no explicit close owner retain their existing borrowed-view semantics.

### Fixed

- Fixed the Windows release build by avoiding the legacy `near` / `far` macro names in inline light shadow setters.
- Fixed CPU-only (`SAPIEN_CUDA=OFF`) pybind builds by compiling the Torch DLPack exporter and PhysX CUDA-context acquisition only when CUDA support is enabled, and by keeping the non-CUDA GPU-system stub concrete.
- Fixed render shutdown with closed-but-still-referenced `RenderSystem` objects by unregistering them from `SapienRenderEngine`; later jobs can create cameras without waiting for Python garbage collection.
- Treat the high-level Scene wrapper's default ground texture cache as library-owned render state and clear it during `sapien.render.shutdown()`, so `add_ground()` no longer permanently blocks job-scoped shutdown.

### Changed

- `PhysxEngine` now holds weak per-device CUDA-context leases instead of owning raw `PxCudaContextManager*` values forever; the manager is released after the last GPU system using it releases its `PxScene`.
- Managed environment-ID slots and per-scene GPU offset/broadphase bookkeeping are released immediately by `Scene.close()` instead of waiting for weak-pointer expiry.

### Known limitations

- `sapien.physx.shutdown()` does not call `cudaDeviceReset()` and therefore does not recreate the CUDA primary context shared with Torch/JAX. It fully releases SAPIEN-owned PhysX resources, but applications that require process-exit-equivalent CUDA isolation between jobs must still launch each job in a fresh process; resetting the primary context invalidates a live JAX PJRT client.


## 3.0.0+fork.14 - 2026-07-27

### Fixed

- Fixed environments silently losing every contact with shared objects when GPU broadphase environment-ID bits are enabled. PhysX relocates environment `e` into the encoded broadphase band `[e << (32 - b), (e + 1) << (32 - b))`, but gives objects shared by all environments (`PX_INVALID_U32`, typically a single ground plane) the fixed encoded interval `[0x01800000, 0xfe7fffff]` that does not follow the banding. The bands at either end of the range fall outside that interval, so sweep-and-prune never reported those pairs: the affected bodies generated zero contact force from the first step and fell through the ground forever, while roughly 99% of the environments kept working and the aggregate metrics still looked healthy. Measured with `gpu_broadphase_env_id_bits = ceil(log2(num_envs))`, this silently broke 1 of 64, 3 of 256, 12 of 1024 and 48 of 4096 environments — always the lowest and highest indices. SAPIEN now shifts the environment IDs it hands to PhysX past the unusable bands whenever a shared scene exists, and stores the shifted value per scene so every body of an environment lands in the same band.

### Added

- `PhysxSceneConfig.num_scenes` and `with_shared_scene` declare the environment layout; `PhysxGpuSystem` reads them at construction and freezes them there, which is the only point where either applies: PhysX bakes the broadphase bit count into the scene at `createScene`, and refuses `PxActor::setEnvironmentID` once an actor is in a scene. `num_scenes` gives each scene a unique environment ID and derives the bit counts; `with_shared_scene` shifts every ID into the bands that still overlap objects shared by all environments. The two are independent. Leaving `num_scenes` unset selects manual mode: SAPIEN numbers nothing, a scene nobody numbered is environment 0, and `Scene.set_environment_id()` is yours to drive.
- `Scene.set_shared_environment()` marks the one scene holding objects every environment collides with. It takes no ID, requires `with_shared_scene`, and must be called before that scene gets any body, since a body freezes its environment ID at `PxScene::addActor`. A shared scene sits outside the `num_scenes` budget, so an ordinary ID already assigned to it — by `get_or_assign_environment_id()`, or just by reading `environment_id` after an assignment — is released back to the pool and reused by the next scene rather than burned. `Scene.get_or_assign_environment_id()` reports `-1` for a shared scene, matching `Scene.environment_id`.
- `PhysxSceneConfig.set_gpu_broadphase_env_id_bits(bits_x, bits_y, bits_z)` is the only way to write the per-axis bit counts; the fields themselves are read-only. It validates all three at once and is ignored when `num_scenes` derives them.
- `PhysxSystemGpu.get_broadphase_environment_id(scene)` reports the ID `PxActor::setEnvironmentID` actually receives.
- `Scene.set_environment_id(env_id)` numbers a scene by hand, for simulations that manage environments themselves. Only available when the config left `num_scenes` unset — declaring it makes SAPIEN the sole assigner and this raises — and only before the scene gets a body, since PhysX freezes an actor's environment ID at `PxScene::addActor`. The value is the *raw, unshifted* ID: it is stored and read back verbatim, and the broadphase band offset is applied only on the way to `PxActor::setEnvironmentID`. Duplicates are allowed and meaningful: giving several scenes the same ID is how they end up in one environment and collide with each other, which auto-assignment cannot express.
- With `with_shared_scene`, SAPIEN writes each scene's environment ID into the high 16 bits of every shape's fourth collision group word, where its filter shader compares it, so cross-environment pairs are rejected exactly rather than only spread apart by the broadphase. This carries the *raw* environment ID with no band offset, because the field is compared for equality rather than encoded into bounds; shared scenes get `0xffff`.

### Changed

- **Breaking:** environment IDs can no longer be chosen *and* auto-assigned in the same simulation. `PhysxSystemGpu.set_scene_environment_id(s)`, the bulk setter, the `Scene.environment_id` setter and `allow_duplicate` are gone. The two modes are now disjoint: declare `num_scenes` and SAPIEN is the sole assigner, so two scenes can never accidentally claim the same value, or leave it unset and number the scenes yourself with `Scene.set_environment_id()`, where duplicates are your explicit intent. `Scene.environment_id` is read-only, `-1` for a shared scene and `None` until one is set or assigned, and `Scene.get_or_assign_environment_id()` raises unless `num_scenes` was declared.
- **Breaking:** `set_light_directions` accepts directional lights only. 3.0.0+fork.13 documented it for directional, spot, and textured lights; spot and textured are now rejected.
- **Breaking:** `RenderMaterial.diffuse_texture`, `get_diffuse_texture()` and `set_diffuse_texture()` are removed. They were Python-only aliases kept after the C++ rename to base color, and `get_diffuse_texture()` never returned anything because the wrapper dropped the `return`. Use `base_color_texture`, which is also what the underlying renderer and glTF call it: in a metallic-roughness workflow the map feeds specular F0 at `metallic = 1`, so "diffuse" describes it only at `metallic = 0`.
- Linux wheel platform tags are verified rather than assumed. `auditwheel` derives the floor from versioned glibc symbols only, so the precompiled PhysX libraries' unversioned `__isoc23_strtoul`, `__isoc23_strtoull`, `__isoc23_fscanf` and `__isoc23_vsscanf` — introduced in glibc 2.38 — were invisible to it: the wheel claimed `manylinux_2_28` while failing to import on anything below 2.38, even though every versioned symbol in those libraries topped out at `GLIBC_2.2.5`. PhysX is now built on glibc 2.31, and `scripts/build.sh` fails the build if a repaired wheel still carries an unversioned symbol the tag does not cover.
- **Breaking:** the broadphase sizing and introspection API is withdrawn from Python, leaving the declaration above as the whole surface. Removed `gpu_broadphase_env_id_bits`, writable `gpu_broadphase_nb_bits_env_id_x/y/z`, `set_gpu_broadphase_env_count`, `gpu_broadphase_env_band_count`, `gpu_broadphase_max_env_count`, `broadphase_env_id_bits_for_env_count`, `max_broadphase_env_count`, `broadphase_env_id_window`, `broadphase_env_band_count`, `has_shared_environment_scene`, `has_managed_broadphase_env_ids`, and `manages_collision_group_scene_ids`. The C++ helpers behind them remain as implementation detail.
- The environment-ID bit counts now default to `(0, 0, 4)`. Four matches PhysX's own "snap to grid" shift, so the banding costs no coordinate precision at all, and one axis is as good as three because the distinct band count is `2 ** max(x, y, z)`, never the product. With a shared scene the counts are widened to `(4, 4, 4)`. `num_scenes` derives the same shape: `(0, 0, b)` on its own, `(b, b, b)` with a shared scene, so an ordinary batched simulation keeps full broadphase coordinate precision on X and Y.
- With a shared scene every non-zero per-axis count must be equal, and mixed layouts such as `(12, 8, 0)` are rejected at construction. A narrower axis keeps fewer of the shifted ID's low bits, so an environment inside the widest axis's usable band wraps back out of it there — with `(12, 8, 0)` environment 232 becomes 256, which lands in band 0 of the 8-bit axis, outside its usable range — and silently stops colliding with the shared object.
- `num_scenes` is a hard cap on ordinary scenes held *at once*, not a sizing hint nor a lifetime total: assigning an ID to one more raises. The bit count derived from it is baked into the PhysX scene at `createScene`, so an extra environment could not be given a band afterwards. Slots come back: a scene turned shared gives its ID up, and so does a destroyed scene, so a setup/teardown loop no longer exhausts the budget without ever holding that many scenes at once. Environment IDs are assigned lazily, so a scene that never asks for one — and never gets a body — costs nothing at all. `num_scenes` must be positive, and with a shared scene it may not exceed 65535, since SAPIEN keeps the scene ID in a 16-bit collision-group field and reserves `0xffff` for the shared scene itself. Both limits are checked when the system is constructed, and both again when IDs are handed out.
- A shared scene whose bit count leaves no band inside the fixed encoded interval PhysX gives shared objects — one bit, where the boundary between the two bands falls inside it — is now rejected when the system is constructed rather than when the first body binds. The count is frozen at `createScene`, so failing later left nothing to do about it. Declaring `num_scenes` never reaches this, since the sizing always picks a width that holds it.
- Environments past the usable band window wrap into the high bits of the ID rather than being rejected. PhysX discards those bits when it places the box but compares the full 32-bit environment ID when it filters the pair — the same `updateData_envIDs` buffer feeds `translateAABBsLaunch` and `filtering()` in `broadphase.cu` — so wrapped environments still collide correctly and share a band only in the broadphase-spreading sense.
- The broadphase band offset is applied only when `with_shared_scene` is set. Without a shared scene there is no fixed encoded interval to fall outside of, so no band is unreachable and `PxActor::setEnvironmentID` receives the scene's environment ID untouched — in both managed and manual mode. Previously the managed path still ran the banding arithmetic, which was the identity but cached a redundant per-scene value.
- The usable band window is derived from `max(x, y, z)` rather than intersecting the per-axis ranges: the widest axis alone decides how environments are banded, so it alone decides which bands reach shared objects.

## 3.0.0+fork.13 - 2026-07-25

### Added

- Added batched physical-property setters in `sapien.physx` for reset-time domain randomization: `set_body_masses` (with optional inertia scaling), `set_body_inertias`, `set_body_cmass_local_poses`, `set_joint_frictions`, `set_joint_drive_properties`, `set_joint_armatures`, and `set_material_properties`. Each applies a whole batch in one call, validates every entry before mutating anything, and works on `PhysxGpuSystem` after `gpu_init()`: PhysX uploads the new properties during the next step without disturbing GPU-side poses, velocities, or joint states.
- Added an explicit `RenderSystemGroup.gpu_init()` lifecycle for grouped GPU rendering: construct, optionally `set_cuda_poses()` when GPU objects or mounted cameras exist, `create_camera_group()`, configure pose modes, then one `gpu_init()` that validates referenced PhysX GPU systems, resolves final mounted-camera indices and output render scenes (including `batched_render_shared` scenes), prepares renderer resources without a warm-up render, takes the one-time CPU snapshots, seeds CUDA camera pose rows, and seals pose ownership.
- Added grouped camera pose modes through `RenderCameraGroup.set_pose_mode(camera, mode)`: `'static'` (default; a one-time CPU snapshot at `gpu_init()`, CPU pose setters raise afterwards), `'cpu'` (the CPU pose stays authoritative and uploads when dirty at `update_render()`, for host-driven follow/anchor cameras), and `'cuda'` (a group-owned CUDA pose row `[p, q(wxyz)]` exposed as `RenderCameraGroup.cuda_poses`, with `get_cuda_pose_index()` and `set_cuda_pose()`). Only `'cuda'` cameras occupy rows in the compact group pose buffer; cameras mounted on PhysX GPU bodies are automatically CUDA-attached to their parent pose row and cannot be configured `'cpu'` or `'static'`. `RenderCameraComponent.pose_mode` reports the configured mode.
- Added light pose modes through `RenderLightComponent.set_pose_mode(mode)`: `'static'` (default; the pose is a one-time snapshot at `gpu_init()` and moving it raises) or `'cpu'` (per-frame entity/local pose updates propagate and upload when dirty). `gpu_init()` rejects a cpu-mode light that shares its entity with a PhysX GPU body, since its CPU pose would not be authoritative; attach such a light to a separate entity and drive it from downloaded state.
- Added coarse CPU dirty versions consumed by `RenderSystemGroup.update_render()`: `RenderCameraComponent.camera_state_version` (projection/intrinsics in every mode, plus the pose in `'cpu'` mode) and `RenderSystem.scene_light_state_version` (light properties, cpu-mode light poses, ambient light). A steady state with unchanged CPU state performs zero per-frame CPU uploads.
- Added batched CPU light setters in `sapien.render` for per-environment lighting randomization: `set_light_poses(lights, poses)` (`[N, 7]` xyz + wxyz local poses), `set_light_directions(lights, directions)` (`[N, 3]`, directional/spot/textured lights only), and `set_light_colors(lights, colors)` (`[N, 3]`). They follow the `sapien.physx` batched-setter conventions: one call, a flat component list, whole-batch validate-then-apply, and zero partial application on failure.

### Fixed

- Fixed a whole-process crash (SIGSEGV) when PhysX GPU scene creation fails, typically when creating an additional `PhysxGpuSystem` on a device without enough free memory (e.g. a same-device evaluation simulation next to a running training simulation): PhysX eagerly allocates the configured GPU heaps at scene creation, and SAPIEN dereferenced the null scene on failure. Creation failures now raise `RuntimeError` for both CPU and GPU systems.
- A failed GPU scene creation no longer freezes other GPU systems on the same device: SAPIEN now clears the out-of-memory abort latch PhysX sets on the shared `PxCudaContext`, so already-running simulations keep stepping and a new system can be created after memory is freed.
- PhysX CUDA context manager creation is now validated; an invalid or failed context manager raises an error instead of crashing later.
- Fixed raster free-camera capture with `RenderSystemGroup` CUDA pose updates by giving every grouped transform one GPU owner. `update_render()` now always orders prior Vulkan work before its CUDA transform writes on the shared timeline semaphore (previously only with ray tracing enabled), raster camera-batch renderers run with external transform updates so record-time uploads no longer overwrite CUDA-written poses with stale CPU model matrices, and every CPU-owned object transform is seeded once per render scene. One group update and one capture per frame now shows every articulation link at its GPU pose; the discarded warm-up capture and second `update_render()` workaround is no longer needed.
- Fixed camera render-scene re-resolution scheduling a full renderer pipeline rebuild on every `RenderSystemGroup.update_render()`; re-assigning the same render scene is now a no-op.
- Fixed `RenderCameraGroup.take_picture()` reading stale or reused render target images after a renderer rebuild; the image copy commands are re-recorded whenever the render target images change. Raster group captures previously returned segmentation-view contents instead of the requested target.
- Fixed dynamic bodies in implicitly discovered `batched_render_shared` scenes being excluded from grouped GPU updates. Final camera output selections now participate in PhysX GPU initialization validation, automatic pose-index binding, and CUDA pose-source requirements.
- Fixed multi-scene render groups uploading zero raster lights: `svulkan2` scene/light uploads now use the aggregated light lists, so lights from `batched_render_shared` member scenes reach the raster light buffers of cameras rendering a scene group.
- The grouped CPU camera upload now waits for its copy to complete. It previously resubmitted the shared upload command buffer without waiting, so a camera whose state changed on consecutive `update_render()` calls without an intervening capture could resubmit a still-pending command buffer and rewrite the staging buffer it reads. Steady state is unaffected, because grouped uploads only run when CPU state actually changed.
- Light components now clear their `svulkan2` back-pointer when removed from a scene, so the real-time color, FOV, shape, and shadow-parameter setters cannot touch a node that has already been released.
- `RenderCameraComponent.pose_mode` now reports the effective pose source. A PhysX-mounted camera auto-attached at `gpu_init()`, or a camera given an explicit `set_gpu_pose_batch_index()`, reports `'cuda'` instead of the unconfigured default `'static'`, matching the transform that actually drives it.

### Changed

- **Breaking:** grouped GPU rendering now requires the explicit `RenderSystemGroup.gpu_init()` lifecycle described above. Steady-state calls before `gpu_init()` raise, camera groups cannot be created after it, one camera cannot belong to two groups, and free cameras default to pose mode `'static'` instead of following CPU poses. Configure `set_pose_mode(camera, 'cuda')` for cameras driven by GPU tensors or `'cpu'` for host-driven cameras. Renderers owned by a group run in a dedicated grouped-GPU execution mode and never upload CPU frame state at render time; `scene.update_render()` plays no role in grouped capture.
- Camera projection and intrinsics stay CPU real-time in every grouped pose mode: `set_perspective_parameters()`, `set_fovx`/`set_fovy`, near/far, principal point, and skew all apply after `gpu_init()`. The CUDA camera kernel writes only view and inverse-view, so a projection change never overrides a GPU pose. Setup-only camera fields (`set_gpu_pose_batch_index`, `set_scenes`) raise once sealed.
- Light properties other than pose stay CPU real-time after `gpu_init()` on both raster and ray tracing: color/intensity, spot inner/outer FOV, parallelogram shape, shadow near/far, directional shadow half-size, and ambient light, each propagating to the underlying `svulkan2` light immediately. Setup-only light fields fail fast once sealed: shadow enable/disable, shadow map size, and textured-light texture; light count/type and scene membership stay frozen by the scene-version check.
- `RenderSystemGroup.update_render()` owns the per-frame grouped schedule: CPU component refresh (raising on static-pose tampering), dirty CPU camera and scene/light uploads, the Vulkan-to-CUDA handoff, the CUDA transform patch (object transforms plus mounted and cuda camera view matrices), then the CUDA-to-Vulkan handoff. `take_picture()` only draws and copies. In `svulkan2`, CPU uploads are split into `uploadCpuCameraState()` (camera only, never touching object transforms) and `uploadCpuSceneLightState()` (scene/light only, never touching the camera), and grouped ray-tracing light uploads are decoupled from the render-version bump so TLAS and instance updates no longer re-upload light or object metadata every frame. The ordinary CPU renderer path is unchanged.
- Static render bodies without a CUDA pose source, and point clouds, are sealed one-time snapshots at `gpu_init()`; moving them afterwards raises at `update_render()` or at the next CPU scene update. CPU-owned ray-tracing instance transforms are seeded once at `gpu_init()` (the Viewer's CPU-managed group keeps per-update uploads for its movable helpers).
- A Viewer can run beside a sealed `RenderSystemGroup` on the same scene: its controller camera stays CPU-managed, joins no camera group, and allocates no CUDA pose row. Open the Viewer before `gpu_init()`, because the group freezes scene topology including the controller camera node, and keep lazily-added helper overlays such as camera linesets off on the shared render scene.

## 3.0.0+fork.12 - 2026-07-19

### Added

- Added same-device direct and cross-device staged PhysX GPU pose transport for the Vulkan Viewer, including raster and ray-tracing updates, resolved-scene automatic transport selection, compact transfer-byte diagnostics, GPU-aware selection/focus/overlays, damped point-spring dragging, queued GPU teleports, explicit `Viewer.update_render()`, camera multi-scene selection, and CUDA/Vulkan device-capability discovery.
- Made existing Viewer property windows PhysX-GPU-aware: selected Entity poses and Articulation qpos/drive targets use on-demand selected-row transfers and queued pre-step edits, collapsed windows avoid readback, GPU Contact reports and CPU IK are explicitly disabled, and transport status appears in the Control window.

### Changed

- Viewer and camera scene selection now aggregate the selected base scenes plus shared render scenes once, without grid or per-scene render offsets.
- `Viewer.render()` now draws the most recently submitted state; call `Viewer.update_render()` explicitly after simulation updates.
- Improved default raster visuals with image-based studio reflections, screen-space ambient occlusion, energy-conserving ambient light, and smoother filtered shadows.
- Updated default primitive materials, terrain checker materials, scene ambient fill, and packaged example lighting for clearer color, surface detail, contact shading, and cast shadows.

### Fixed

- Fixed selected GPU articulation qpos and drive-target host transfers racing pending work on a configured non-blocking CUDA stream.
- Corrected generated Python API stubs and aligned Pinocchio wrapper signatures and packaged typing information with runtime behavior.
- Fixed wheel version normalization when the development version contains a commit hash beginning with a digit.

## 3.0.0+fork.11 - 2026-07-15

### Added

- Added ray-tracing shader support to `RenderSystemGroup` CUDA pose updates, including mounted-camera buffers, rigid-instance transforms, per-frame TLAS updates, and accumulation resets.

### Changed

- Unified rasterization and ray-tracing color management around an ACES-fitted, sRGB-encoded default while retaining gamma and plain sRGB display modes.
- Improved raster PBR energy conservation, low-roughness stability, transformed tangent frames, transparent-material lighting/shadows, and viewer exposure/tone-mapping controls.
- Improved ray-tracing path stability, normal handling, lens sampling, transmission weighting, and non-finite radiance handling.
- Clarified that SAPIEN owns packaged shader packs while `svulkan2` owns the generic runtime and renderer-internal shaders.

### Fixed

- Fixed textured raster and ray-tracing materials ignoring the material base-color and alpha factors.
- Fixed shadowed lights being associated with the wrong light when shadowed and unshadowed lights were mixed.
- Fixed raster point and line overlays drawing through closer scene geometry.
- Fixed ray-tracing alpha accumulating once per bounce, a GGX lower-hemisphere test that could never reject invalid samples, and biased directional-light softness sampling.
- Fixed `svulkan2` crashes when clearing an environment map, Vec4 property type validation, the RT post-processing descriptor-pool type, and missing barriers between compute post-processing passes.
- Fixed batched RT cameras building resources from their original scene instead of the assigned `SceneGroup`, including missing BLAS initialization and light data from shared scenes.
- Fixed CUDA batched RT support forcing ordinary RT camera and TLAS uploads through synchronous device-local staging copies.

## 3.0.0+fork.10 - 2026-07-10

### Added

- Exposed PhysX per-axis maximum articulation-joint velocity limits through `PhysxArticulationJoint.max_joint_velocity` and `get/set_max_joint_velocity()`.
- Applied positive URDF `<limit velocity="...">` values to PhysX joints, treated non-positive placeholders as unspecified, and preserved effective limits when exporting URDFs.

## 3.0.0+fork.9 - 2026-07-05

### Changed

- Streamlined the release skill preflight and dispatch flow, including string-based `gh workflow run --json` boolean inputs and no default long-running watch.
- Updated agent instructions to use the current MyST tutorial index paths and dropped the deleted serialization note reference.
- Narrowed PhysX environment-id render synchronization error handling to only ignore scenes without render systems.
- Clarified render-shape GPU transform-index overload usage for owning scenes and scene groups.
- Moved the documentation Pages build job to Ubuntu 24.04 so the published release wheel imports with the required glibc symbols.

### Fixed

- Implemented `Viewer.set_camera_xyz()` and `Viewer.set_camera_rpy()` as real public methods so scripted viewer camera setup works without relying on plugin monkey-patching.
- Added shared `0xffff` scene/ignore ID semantics to collision-group filtering, matching PhysX GPU environment-ID shared-object behavior.

## 3.0.0+fork.8 - 2026-07-03

### Added

- Added a required Markdown release notes input to the tagged release workflow.
- Added a manual-only project-level Pi skill for reviewing commits since the previous release tag, composing summarized release notes, triggering tagged releases through `gh`, and maintaining this changelog.
- Added default release-tag selection that increments the final number from the latest numeric release tag unless a version is specified.
- Added sanitized convenience copies of agent skills under `docs/skills` for repository-local use.
- Added a GitHub Pages workflow for building and deploying the Sphinx documentation.

### Changed

- Switched the current Sphinx documentation to the Furo theme for automatic light/dark mode support.
- Updated the release skill to default unspecified release targets to the current `HEAD` commit.
- Limited push-triggered nightly builds to the `main` branch.
- Moved the historical changelog out of `readme.md` and into this file.
- Moved detailed README setup, server rendering, build, and validation notes into Markdown files under `docs/`.
- Converted handwritten current documentation under `docs/source` from reStructuredText to MyST Markdown.
- Updated installation documentation to use GitHub Releases wheels instead of PyPI.

### Removed

- Removed the outdated Bitbucket Pipelines configuration.
- Removed the obsolete serialization notes.

## 3.0

- Major API and infrastructure overhaul.

## 2.2

- Renamed `VulkanRenderer` to `SapienRenderer`; `VulkanRenderer` remains an alias.
- Added ray tracing support to `SapienRenderer`.
- Deprecated `KuafuRenderer`; use the ray-tracing shader in `SapienRenderer` instead.
- Added GPU-accelerated stereo depth sensor simulation.
- Added render server support.
- Added Python 3.11 support.
- Fixed inverse kinematics default active joint mask; it now defaults to all ones.
- Fixed incorrectly exported memory in Vulkan-CUDA interop.
- Fixed joint `get_global_pose`.

## 2.1 - Python 3.10 and fixes

- Added Python 3.10 support.
- Fixed crashes when running without a renderer.
- Fixed joint force limits, which previously behaved as impulse limits.
- Fixed inertia computation for scaled URDF assets.
- Fixed point-light shadows.
- Fixed collision behavior for meshes loaded from DAE files.
- Improved utility support for `set_material`, active lights, flat shading, dynamic point rendering, envmap generation, and multi-threaded environments.

## 2.1 - renderer and camera refactor

- Refactored the light system and removed light functions from `scene.renderer_scene`.
- Refactored the camera system:
  - Cameras no longer require mounts.
  - Cameras can change their parent and mounted pose with `camera.set_parent` and `camera.set_local_pose`.
  - When a camera is not mounted, setting its local pose sets its global pose.
  - Added `scene.add_camera` and `scene.remove_camera`.
  - `add_mounted_camera` can be replaced with `add_camera` followed by `camera.set_parent` and `camera.set_local_pose`; `add_mounted_camera` remains available, but `fovx` should no longer be provided.
  - Removed mount-related functions, including `find_camera_by_mount`.
  - Cameras now support full camera parameters through `camera.near`, `camera.far`, `camera.set_fovx`, `camera.set_fovy`, `camera.set_focal_lengths`, `camera.set_principal_point`, `camera.skew`, and `camera.set_perspective_parameters`.
- Refactored the render shape system:
  - After `actor.get_visual_bodies()` and `visual_body.get_render_shapes()`, users should check `visual_body.type`.
  - When `visual_body.type` is `mesh`, `shape.scale` is replaced by `visual_body.scale`, and `shape.pose` is replaced by `visual_body.local_pose`.
  - These changes align runtime objects with `add_visual_shape` actor-builder behavior.

## pre-2.0

- Changed the default camera shader so the fourth component now stores the normalized 0-1 depth value.
- Added `critical` and `off` log levels.
- Added point cloud and line rendering for camera and point cloud visualization.
- Improved shader performance by compiling the same shader only once per process.
- Fixed articulation `setDriveTarget` reversal for prismatic joints; joint `setDriveTarget` is not affected.
- Fixed kinematic articulation loading.

## 1 to 2 migration

- Replace `scene.renderer_scene.add_xxx_light` with `scene.add_xxx_light`.
- Replace `scene.remove_mounted_camera` with `scene.remove_camera`.
- Optionally remove `fovx` from `scene.add_mounted_camera`.

## 1.1

- Added support for non-convex static and kinematic collision shapes.
- Added warnings for small mass or inertia values.
- Introduced `Entity` as the base class of actors.
- Added light classes inherited from entities, allowing light objects to be manipulated in SAPIEN scenes.
- Updated the viewer and renamed actor to entity where appropriate.
- Added partial support for the URDF material tag for primitive shapes and single colors.
- Fixed renderer issues.
- Added support for inner and outer FOV for spotlights.

## 1.0

- Replaced the old Vulkan-based renderer completely; see `sapien.core.renderer` for details.
- Exposed GUI functionality to Python.
- Reimplemented the Vulkan viewer in Python.
- Exposed PhysX shape wrappers to Python:
  - Collision shapes can be retrieved through `actor.get_collision_shapes`.
  - Collision groups on a shape can be set by `CollisionShape.set_collision_groups`.
  - Shapes are now also available in `Contact`.
- Changed render material creation to `renderer.create_material()`.
- Replaced actor-builder `add_xxx_shape` APIs with `add_xxx_collision`.
- Moved light functions from scene to `scene.renderer_scene`.
- Added centrifugal and Coriolis force support.
- Changed default physical parameters for better stability.
