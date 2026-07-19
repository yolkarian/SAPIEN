# SAPIEN API Reference Tables

Purpose: compact, grep-friendly API map an agent consults when building SAPIEN simulation environments for users. This is not tutorial prose.

Source policy: generated/curated from the SAPIEN repository's [`python/py_package`](https://github.com/yolkarian/SAPIEN/tree/dev/python/py_package) source tree. If docs and source disagree, prefer source/stubs/CI.

Conventions:

- Runtime/install assumption: use SAPIEN release wheels from `https://github.com/yolkarian/SAPIEN/releases`, not the PyPI SAPIEN package.
- API rows use preferred public imports (`sapien.*`, `sapien.physx.*`, `sapien.render.*`) even when implementation lives in `sapien.pysapien.*`.
- `Pose.q` is always `wxyz`.
- GPU PhysX: call `sapien.physx.enable_gpu()` before `PhysxGpuSystem`; configure global PhysX before creating systems; call `gpu_init()` after all bodies/lights are built.
- No IK solver: SAPIEN has no IK solver (`GpuInverseKinematicsSolver` and `gpu_inverse_kinematics` do not exist). Exposed PhysX GPU link-data and dense-Jacobian buffers: `cuda_articulation_link_data`, `cuda_articulation_jacobian`, `cuda_articulation_jacobian_shape` (valid after `gpu_init()`).
- For training/offscreen sensors, use CUDA buffers and `RenderSystemGroup.set_cuda_poses(...)`; use `Viewer.configure_physx_gpu_rendering(...)` for interactive GPU visualization, and reserve `sync_poses_gpu_to_cpu()` for explicit CPU debugging.
- Cache `CudaArray.torch()/cupy()/jax()` views and GPU indices after `gpu_init()`; do not recreate in loops.

## File map

| File | Contents |
|---|---|
| [`core.md`](core.md) | sapien + core entity/component/device/pose API |
| [`scene-builder-loader.md`](scene-builder-loader.md) | High-level Scene, builders, URDF loader, Pinocchio, viewer helpers |
| [`physx.md`](physx.md) | PhysX CPU/common objects, articulations, bodies, shapes, configs |
| [`physx-gpu.md`](physx-gpu.md) | PhysX GPU Direct, CUDA buffers, env IDs, exposed link/Jacobian buffers |
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
| Interactive GPU Viewer | `viewer.configure_physx_gpu_rendering(physx_system, "auto")`, then `apply_interactions()` before physics and `update_render()` before `render()` |
| Viewer device/transport diagnostics | `sapien.Device.uuid`, `can_direct_cuda_vulkan_interop()`, `can_access_peer(...)`, `viewer.pose_transport`, `viewer.pose_transfer_bytes` |
| GPU articulation link/Jacobian buffers | `physx_system.cuda_articulation_link_data`, `physx_system.gpu_compute_articulation_jacobian(...)`, `physx_system.cuda_articulation_jacobian_shape` after `physx_system.gpu_init()` |
| GPU state tensors | `physx_system.cuda_*` then `.torch()`/`.cupy()`/`.jax()` |
| Viewer | `sapien.utils.Viewer()` / `scene.create_viewer()` |
