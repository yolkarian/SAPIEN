# Changelog

Notable user-facing and developer-facing changes are documented here.
Release descriptions are written from reviewed commits and diffs, then passed to the tagged release workflow as input.

## Unreleased

### Added

- Added batched physical-property setters in `sapien.physx` for reset-time domain randomization: `set_body_masses` (with optional inertia scaling), `set_body_inertias`, `set_body_cmass_local_poses`, `set_joint_frictions`, `set_joint_drive_properties`, `set_joint_armatures`, and `set_material_properties`. Each applies a whole batch in one call, validates every entry before mutating anything, and works on `PhysxGpuSystem` after `gpu_init()`: PhysX uploads the new properties during the next step without disturbing GPU-side poses, velocities, or joint states.

### Fixed

- Fixed a whole-process crash (SIGSEGV) when PhysX GPU scene creation fails, typically when creating an additional `PhysxGpuSystem` on a device without enough free memory (e.g. a same-device evaluation simulation next to a running training simulation): PhysX eagerly allocates the configured GPU heaps at scene creation, and SAPIEN dereferenced the null scene on failure. Creation failures now raise `RuntimeError` for both CPU and GPU systems.
- A failed GPU scene creation no longer freezes other GPU systems on the same device: SAPIEN now clears the out-of-memory abort latch PhysX sets on the shared `PxCudaContext`, so already-running simulations keep stepping and a new system can be created after memory is freed.
- PhysX CUDA context manager creation is now validated; an invalid or failed context manager raises an error instead of crashing later.
- Fixed raster free-camera capture with `RenderSystemGroup` CUDA pose updates by giving every grouped transform one GPU owner. `BatchedRenderSystem.update()` now always orders prior Vulkan work before its CUDA transform writes on the shared timeline semaphore (previously only with ray tracing enabled), raster camera-batch renderers run with external transform updates so record-time uploads no longer overwrite CUDA-written poses with stale CPU model matrices, and `RenderSystemGroup` creation seeds every CPU-owned object transform once per render scene. One group update and one capture per frame now shows every articulation link at its GPU pose; the discarded warm-up capture and second `update_render()` workaround is no longer needed.
- Fixed camera render-scene re-resolution scheduling a full renderer pipeline rebuild on every `RenderSystemGroup.update_render()`; re-assigning the same render scene is now a no-op.
- Fixed `RenderCameraGroup.take_picture()` reading stale or reused render target images after a renderer rebuild; the image copy commands are re-recorded whenever the render target images change. Raster group captures previously returned segmentation-view contents instead of the requested target.

### Changed

- `RenderSystemGroup` now follows an explicit sealed lifecycle: construct, optionally `set_cuda_poses()` when GPU objects or mounted cameras exist, `create_camera_group()`, then one `gpu_init()` that validates referenced PhysX GPU systems, resolves final mounted-camera indices and output render scenes (including batched-render-shared scenes), prepares renderer resources without a warm-up render, takes the one-time CPU snapshots (static objects, lights, free-camera pose rows), and seals transform ownership. Steady-state calls before `gpu_init()` raise, camera groups cannot be created after it, and one camera cannot belong to multiple groups. Renderers owned by a group enter a dedicated grouped-GPU execution mode (replacing the removed per-renderer auto-upload flag), so no render or rebuild can upload CPU frame state over GPU-owned buffers; destroying the group releases its camera/shape seals and restores CPU-managed camera rendering. Scene-level ownership remains active until the last camera group using a shared transform buffer is released, preventing an earlier released camera from uploading stale CPU transforms over a live group's CUDA data. The Viewer's fixed-scene group remains CPU-managed for its helper visuals.
- Grouped camera transforms are GPU-owned after `gpu_init()`. Mounted cameras follow their PhysX GPU parent pose row; free cameras (no GPU pose batch index) receive a row in a group-owned CUDA pose buffer, seeded once from the CPU pose, and update through `RenderCameraGroup.cuda_free_camera_poses`, `get_free_camera_cuda_pose_index()`, or `set_free_camera_pose()`. `RenderCameraComponent.set_local_pose()` raises on sealed cameras, and `scene.update_render()` plays no role in grouped capture: one CUDA pose write, one `update_render()`, and one capture produce the current frame on both raster and RT.
- Static render bodies without a CUDA pose source become sealed one-time snapshots at `gpu_init()`; moving their CPU pose afterwards raises at the next CPU scene update. CPU-owned RT instance transforms are seeded once at `gpu_init()` instead of being re-uploaded every `update_render()` (the Viewer's CPU-managed group keeps per-update uploads for its movable helpers). Pose sources are tracked through an internal registry (primary source plus one free-camera source per camera group), forward-compatible with additional CUDA pose producers.

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
