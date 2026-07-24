---
name: sapien-simulation
description: Build, refactor, or review SAPIEN simulation environments, especially PhysX GPU setup, scene/build/light/render/sensor initialization order, direct and staged GPU Viewer rendering, GPU buffer access, batched rendering, exposed PhysX GPU articulation link/Jacobian buffers, reset workflows, step workflows, and Docker container packaging of GPU PhysX and rendering dependencies.
---

# SAPIEN Simulation

Use this skill when creating, refactoring, or reviewing SAPIEN simulation environments.

## Required behavior

1. Read `docs/gpu-workflows.md` before editing SAPIEN simulation code.
2. For API lookup, use the compact agent tables under `docs/api/README.md` before guessing names/signatures.
3. When installing SAPIEN for examples, validation, or containers, use the fork's GitHub release wheels from `https://github.com/yolkarian/SAPIEN/releases`; do not install SAPIEN from PyPI.
4. Use SAPIEN GPU PhysX and GPU APIs for parallel simulation.
5. Preserve initialization order: configure PhysX, create systems, create builders, create scenes, assign GPU scene environment IDs or offsets, build bodies, add lights, call `gpu_init()`, initialize GPU buffers, then initialize viewer/sensors/rendering.
6. For many identical URDF robots, parse the URDF once, cache the builder, and reuse it with `builder.set_scene(scene)` plus a restored initial pose before each `builder.build()`. For kinematics-only or IK-only workflows, set `loader.load_visuals = False` and `loader.load_collisions = False` when geometry is unnecessary.
7. Prefer PhysX GPU scene environment IDs for multi-env isolation when available; assign a unique env ID before adding each env's bodies, and use env ID `-1`/`0xffffffff` for shared objects such as one global ground plane.
8. Create `sapien.render.RenderSystem` only when viewer, sensors, or offscreen rendering need it; do not create it for pure physics.
9. For an interactive PhysX GPU Viewer, initialize GPU PhysX first, use automatic direct/staged pose transport, call `viewer.apply_interactions()` immediately before every physics substep that may consume Viewer commands, call `viewer.update_render()` after each displayed state, and then call `viewer.render()`. Reserve `PhysxGpuSystem.sync_poses_gpu_to_cpu()` for explicit CPU debugging or Viewer `cpu-debug`. For offscreen capture, follow the explicit `RenderSystemGroup` lifecycle: construct, `set_cuda_poses(...)`, `create_camera_group(...)`, then one `gpu_init()` that prepares resources, takes the one-time CPU snapshots, seeds free-camera CUDA pose rows, and seals transform ownership. Per frame, move a free/follow camera by writing its group CUDA pose row (`RenderCameraGroup.set_free_camera_pose(camera, pose)` or `cuda_free_camera_poses`), call `RenderSystemGroup.update_render()`, and capture once; `scene.update_render()` is not part of grouped capture.
10. Cache SAPIEN CUDA tensor views and GPU indices once after `gpu_init()`; use cached indices for batched access.
11. For indexed GPU APIs, allowed selected-index argument types are `sapien.CudaArray` or CUDA-array-interface objects such as CUDA `torch.Tensor`, `cupy.ndarray`, or Numba CUDA device arrays. They must wrap 1D contiguous CUDA `int32` buffers on the same CUDA device as the PhysX system and contain SAPIEN `gpu_index` values; NumPy arrays, Python lists, CPU tensors, `int64` tensors, non-contiguous views, and cross-device arrays are invalid. Keep external owners alive until the SAPIEN CUDA stream finishes.
12. SAPIEN has no IK solver. Exposed PhysX GPU buffers after `gpu_init()`: `cuda_articulation_link_data`, `cuda_articulation_jacobian`, `cuda_articulation_jacobian_shape`.
13. For Docker/containerized SAPIEN GPU rendering, configure NVIDIA Vulkan/EGL ICD discovery explicitly: inject minimal NVIDIA ICD JSON files when needed, set `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`, and rely on NVIDIA Container Toolkit for the actual driver libraries instead of installing drivers in the image.
14. Pre-bake the PhysX GPU runtime library into SAPIEN Docker images so `sapien.physx.enable_gpu()` does not download `physxgpu-linux-clang.zip` on every fresh container start. After SAPIEN is installed, derive `PHYSX_VERSION` from `sapien.physx.version()`, extract the matching release zip into `$HOME/.sapien/physx/<PHYSX_VERSION>/`, and verify the nested `libPhysXGpu_64.so`. Do not call `enable_gpu()` during `docker build` because it also loads `libcuda.so` and needs the runtime GPU driver. See `docs/gpu-workflows.md` for the exact Dockerfile snippet.

## Guardrails

- Do not mix CPU state APIs with GPU runtime state updates.
- Do not call `sync_poses_gpu_to_cpu()` in normal Viewer, training, reset, step, sensor, video, or offscreen capture paths; it downloads all poses to CPU and is only appropriate for explicit CPU debugging or Viewer `cpu-debug`.
- Do not use CPU pose APIs on components sealed by `RenderSystemGroup.gpu_init()`: member-camera `set_local_pose()` raises, moving a CPU-owned static body raises at the next CPU scene update, and `scene.update_render()` plays no role in grouped capture. Move free cameras through the group's CUDA pose rows, mounted cameras through their PhysX parent rows, and bind moving objects to a CUDA pose source. Do not call `update_render()`/capture before `gpu_init()` or create camera groups after it. Do not recreate `RenderCameraGroup` per frame; `BatchedRenderSystem` retains every created camera batch and would accumulate GPU/Vulkan resources.
- Do not bind app-specific interactive controls to SAPIEN viewer navigation keys such as `W/A/S/D/Q/E`; prefer non-conflicting keys such as `I/K/J/L` for planar commands, `U/O` for yaw, `C` to clear commands, and `N` to reset.
- Do not create batched rendering before render bodies/cameras have GPU pose batch indices.
- SAPIEN has no IK solver; `GpuInverseKinematicsSolver` and `gpu_inverse_kinematics` do not exist.
- Do not pass CPU, non-contiguous, non-`int32`, cross-device, or PhysX-internal GPU index buffers to indexed GPU APIs.
- Do not apply all GPU buffers during normal step; apply only buffers modified in that step.
- Remember SAPIEN quaternion convention is `wxyz`.
- Do not ignore SAPIEN's `Failed to find Vulkan ICD file` warning in GPU containers; fix Vulkan/EGL ICD setup and verify `import sapien` is warning-free under `docker run --gpus ...`.
- Do not let containers rely on SAPIEN's runtime PhysX GPU download. `enable_gpu()` downloads `physxgpu-linux-clang.zip` into `$HOME/.sapien/physx/<version>/` when the library is missing; fresh containers do not persist that writable layer, so this repeats and can fail offline. Pre-bake the extracted library into the image, or mount a persistent volume at the runtime user's `$HOME/.sapien`.
