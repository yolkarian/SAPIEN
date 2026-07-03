# SAPIEN API Tables for Agents

Purpose: compact, grep-friendly API map for coding agents working on SAPIEN simulation code. This is not tutorial prose.

Source policy: generated/curated from the SAPIEN repository's [`python/py_package`](https://github.com/yolkarian/SAPIEN/tree/dev/python/py_package) source tree. If docs and source disagree, prefer source/stubs/CI.

Conventions:

- Runtime/install assumption: use SAPIEN release wheels from `https://github.com/yolkarian/SAPIEN/releases`, not the PyPI SAPIEN package.
- API rows use preferred public imports (`sapien.*`, `sapien.physx.*`, `sapien.render.*`) even when implementation lives in `sapien.pysapien.*`.
- `Pose.q` is always `wxyz`.
- GPU PhysX: call `sapien.physx.enable_gpu()` before `PhysxGpuSystem`; configure global PhysX before creating systems; call `gpu_init()` after all bodies/lights are built.
- Custom GPU IK: SAPIEN exposes PhysX GPU link-data and dense-Jacobian buffers; build articulations into the target `PhysxGpuSystem` scene, call `gpu_init()`, then use `cuda_articulation_link_data`, `cuda_articulation_jacobian`, and `cuda_articulation_jacobian_shape`. The old `GpuInverseKinematicsSolver` and `gpu_inverse_kinematics` helpers are removed.
- For training/offscreen sensors, use CUDA buffers and `RenderSystemGroup.set_cuda_poses(...)`; reserve `sync_poses_gpu_to_cpu()` for viewer/debug only.
- Cache `CudaArray.torch()/cupy()/jax()` views and GPU indices after `gpu_init()`; do not recreate in loops.

## File map

| File | Contents |
|---|---|
| [`core.md`](core.md) | sapien + core entity/component/device/pose API |
| [`scene-builder-loader.md`](scene-builder-loader.md) | High-level Scene, builders, URDF loader, Pinocchio, viewer helpers |
| [`physx.md`](physx.md) | PhysX CPU/common objects, articulations, bodies, shapes, configs |
| [`physx-gpu.md`](physx-gpu.md) | PhysX GPU Direct, CUDA buffers, env IDs, custom IK buffers |
| [`render.md`](render.md) | Rendering systems, cameras, materials, shapes, lights, textures |
| [`sensors-assets-math.md`](sensors-assets-math.md) | Math utilities, sensors, SimSense, assets |
| [`internal-renderer-ui.md`](internal-renderer-ui.md) | Low-level internal renderer + ImGui widgets |
| [`misc-python-tools.md`](misc-python-tools.md) | Asset/URDF/export/show helper modules |

## Fast import map

| Need | Preferred API |
|---|---|
| Scene with default CPU physics + render | `sapien.Scene()` |
| Physics-only GPU scene | `sapien.Scene([sapien.physx.PhysxGpuSystem(device)])` |
| Render-enabled scene | `sapien.Scene([physx_system, sapien.render.RenderSystem(device)])` |
| Actors | `scene.create_actor_builder()` / `sapien.ActorBuilder()` |
| Robots/articulations | `scene.create_articulation_builder()` / `scene.create_urdf_loader()` |
| Height-field terrain | `scene.add_heightfield(height_field, row_scale, column_scale=None, height_scale=...)` before GPU `gpu_init()` |
| Rigid body component | `entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)` |
| Camera | `scene.add_camera(...)` or `sapien.render.RenderCameraComponent` |
| Batched GPU render | `sapien.render.RenderSystemGroup([...]).set_cuda_poses(physx.cuda_rigid_body_data)` |
| Custom GPU IK buffers | `physx_system.cuda_articulation_link_data`, `physx_system.gpu_compute_articulation_jacobian(...)`, `physx_system.cuda_articulation_jacobian_shape` after `physx_system.gpu_init()` |
| GPU state tensors | `physx_system.cuda_*` then `.torch()`/`.cupy()`/`.jax()` |
| Viewer | `sapien.utils.Viewer()` / `scene.create_viewer()` |
