# SAPIEN GPU Simulation Workflows

## Core rules

- Install SAPIEN from `https://github.com/yolkarian/SAPIEN/releases` (fork release wheels), not from PyPI.
- Use `sapien.physx.enable_gpu()` before creating `sapien.physx.PhysxGpuSystem`.
- Use `sapien.Device("cuda")` for PhysX; use the same device for `sapien.render.RenderSystem` when rendering exists.
- For many identical URDF robots, call `loader.parse(...)` once, cache the resulting articulation builder, and reuse it across scenes by calling `builder.set_scene(scene)`, restoring its saved `initial_pose`, and then `builder.build()` for each env. For kinematics-only or IK-only workflows, set `loader.load_visuals = False` and `loader.load_collisions = False` before parsing/loading when geometry is unnecessary.
- Prefer PhysX GPU scene environment IDs over collision-group hacks for multi-env isolation: set a unique env ID before adding each env's bodies, and set env ID `-1`/`0xffffffff` for shared bodies such as a global ground plane.
- In GPU runtime, update state through `sapien.physx.PhysxGpuSystem.cuda_*` buffers plus `gpu_apply_*()`.
- Read state only after the needed `gpu_fetch_*()` calls.
- `PhysxGpuSystem.sync_poses_gpu_to_cpu()` downloads all GPU poses to CPU SAPIEN entities; use it only for explicit CPU-state debugging or the Viewer `cpu-debug` transport, never for normal Viewer rendering, training, reset, step, sensors, video, or offscreen capture.
- For direct GPU camera/offscreen rendering, use GPU pose batch indices with `sapien.render.RenderSystemGroup.set_cuda_poses(physx_system.cuda_rigid_body_data)` and read images through `get_picture_cuda(...)` instead of syncing poses to CPU.
- `RenderSystemGroup` follows an explicit lifecycle: construct, `set_cuda_poses(...)`, `create_camera_group(...)`, then one `gpu_init()` that resolves output render scenes (including batched-render-shared scenes such as a shared ground), prepares all resources, takes the one-time CPU snapshots, seeds free-camera pose rows, and seals transform ownership. `update_render()` then owns every grouped transform: GPU-sourced objects and mounted cameras update from the bound CUDA pose buffer, free cameras from group-owned CUDA pose rows (`RenderCameraGroup.cuda_free_camera_poses`), with prior Vulkan reads and CUDA writes ordered both ways on a timeline semaphore. After `gpu_init()`, `set_local_pose()` on member cameras raises, moving a CPU-owned static body raises at the next CPU scene update, and `scene.update_render()` plays no role in grouped capture. Never recreate a `RenderCameraGroup` per frame. One group update and one capture per frame shows the current camera and all GPU articulation-link poses on both raster and RT.
- `RenderSystemGroup` supports both raster and `"rt"` camera shader packs. RT rigid-pose updates also update the TLAS and reset accumulation. SAPIEN has no deformable-body physics, but it does expose a render-only `RenderCudaMeshComponent`; supporting that component in batched RT additionally requires synchronized BLAS updates or rebuilds, shared-`SceneGroup` aggregation, and accumulation resets after vertex changes.
- Cache `sapien.CudaArray.torch()` views once after `gpu_init()`; do not recreate them in loops.
- Cache common GPU indices once after `gpu_init()`:
  - `sapien.physx.PhysxArticulation.get_gpu_index()`
  - `sapien.physx.PhysxRigidDynamicComponent.get_gpu_index()`
  - `sapien.physx.PhysxRigidDynamicComponent.get_gpu_pose_index()`
  - `sapien.physx.PhysxArticulationLinkComponent.get_gpu_pose_index()`
- SAPIEN quaternion convention is `wxyz`.
- SAPIEN has no IK solver. It exposes PhysX GPU buffers `cuda_articulation_link_data`, `cuda_articulation_jacobian`, and `cuda_articulation_jacobian_shape`; build target articulations into a `Scene` owned by the `PhysxGpuSystem`, call `gpu_init()` so `articulation.gpu_index` and CUDA buffers are valid, then use those buffers directly.
- Indexed GPU APIs accept only these Python argument types for selected-index buffers:
  - `sapien.CudaArray`
  - any CUDA object exposing `__cuda_array_interface__`, such as a CUDA `torch.Tensor`, `cupy.ndarray`, or Numba CUDA device array
  The underlying array must be 1D, contiguous, CUDA `int32`, on the same CUDA device as the PhysX system, and contain SAPIEN `gpu_index` values, not PhysX-internal GPU indices. NumPy arrays, Python lists, CPU tensors, `int64` tensors, non-contiguous views, and cross-device CUDA arrays are not valid. Keep the owner of an external index tensor alive until SAPIEN's CUDA stream has finished using it.

## Docker Vulkan/EGL setup for SAPIEN rendering

SAPIEN imports run Vulkan/EGL discovery before rendering. In NVIDIA Docker containers, `nvidia-smi` can work while SAPIEN still warns:

```text
Failed to find Vulkan ICD file. This is probably due to an incorrect or partial installation of the NVIDIA driver.
```

This commonly happens because NVIDIA Container Toolkit mounts the driver-provided Vulkan ICD at `/etc/vulkan/icd.d/nvidia_icd.json`, while SAPIEN's fallback check only looks for `/usr/share/vulkan/icd.d/nvidia_icd.json` unless `VK_ICD_FILENAMES` is already set.

For SAPIEN GPU/rendering containers:

- Install Vulkan/EGL loader packages such as `libvulkan1`, `libegl1`, `libgl1`, and project-specific X11/GL dependencies. Do not install host NVIDIA drivers in the image.
- Ensure Docker GPU runs include `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` or a broader value such as `compute,utility,graphics,display` when viewers are needed.
- Artificially inject minimal NVIDIA ICD JSON files into the image when the base image does not already provide them. These JSON files are not drivers; the referenced `libGLX_nvidia.so.0` and `libEGL_nvidia.so.0` must come from NVIDIA Container Toolkit at runtime:

  `docker/nvidia_icd.json` -> `/etc/vulkan/icd.d/nvidia_icd.json`

  ```json
  {
      "file_format_version": "1.0.0",
      "ICD": {
          "library_path": "libGLX_nvidia.so.0",
          "api_version": "1.1.95"
      }
  }
  ```

  `docker/10_nvidia.json` -> `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`

  ```json
  {
      "file_format_version": "1.0.0",
      "ICD": {
          "library_path": "libEGL_nvidia.so.0"
      }
  }
  ```

- Set `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` in the Dockerfile and pass the same env var from run/interactive scripts so SAPIEN does not fall back to its bundled ICD path.
- Do not rely on Mesa ICDs (`lvp`, `nouveau`, `radeon`, etc.) for NVIDIA SAPIEN GPU rendering; they can hide an NVIDIA ICD issue or select a CPU/software Vulkan device.
- Verify with a GPU container smoke test and fail if the warning is present:

  ```bash
  docker run --rm --gpus 'device=0' \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    -e VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    IMAGE \
    python -c 'import os, sapien; print(os.environ.get("VK_ICD_FILENAMES"), sapien.__version__)' 2>&1
  ```

## Pre-bake the PhysX GPU library into Docker images

On first GPU use, `sapien.physx.enable_gpu()` looks for the PhysX GPU shared library under `Path.home() / ".sapien" / "physx" / sapien.physx.version()`. In containers, treat this as `$SAPIEN_HOME/.sapien/physx/<version>` or `$HOME/.sapien/physx/<version>` for the runtime user. SAPIEN accepts either:

- `<parent>/libPhysXGpu_64.so`
- `<parent>/physxgpu-linux-clang/PhysX/bin/linux.x86_64/release/libPhysXGpu_64.so`

If neither exists, SAPIEN prints:

```text
Downloading PhysX GPU library to $HOME/.sapien/physx/107.3-physx-5.6.1 from Github.
This can take several minutes. If it fails to download, please manually download
https://github.com/yolkarian/physx-release/releases/download/107.3-physx-5.6.1/physxgpu-linux-clang.zip
and unzip at $HOME/.sapien/physx/107.3-physx-5.6.1.
```

Fresh `docker run` containers do not persist the writable `$HOME/.sapien` layer, so relying on this lazy download causes repeated multi-minute startup downloads and can fail offline. Bake the extracted library into the image, or mount a persistent volume at the runtime user's `~/.sapien`.

Do NOT pre-bake by calling `sapien.physx.enable_gpu()` in the Dockerfile: after downloading, `enable_gpu()` also `dlopen`s `libcuda.so` and the PhysX GPU `.so`, then runs `_enable_gpu()`. Build stages usually do not have the NVIDIA driver/GPU that this requires. Instead, after Python dependencies and SAPIEN are installed, download/extract the matching release zip directly. Prefer deriving the version from the installed wheel so the asset tag stays in sync:

```dockerfile
# Pre-bake the PhysX GPU runtime library so sapien.physx.enable_gpu()
# skips the GitHub download at container startup. Do not call enable_gpu()
# during docker build: it loads libcuda.so and needs the runtime GPU driver.
RUN export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    && physx_version="$(python -c 'import sapien.physx as physx; print(physx.version())')" \
    && sapien_home="${SAPIEN_HOME:-${HOME}}" \
    && parent="${sapien_home}/.sapien/physx/${physx_version}" \
    && url="https://github.com/yolkarian/physx-release/releases/download/${physx_version}/physxgpu-linux-clang.zip" \
    && mkdir -p "${parent}" \
    && tmp_zip="$(mktemp)" \
    && curl -fsSL "${url}" -o "${tmp_zip}" \
    && python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "${tmp_zip}" "${parent}" \
    && rm -f "${tmp_zip}" \
    && so="${parent}/physxgpu-linux-clang/PhysX/bin/linux.x86_64/release/libPhysXGpu_64.so" \
    && test -f "${so}" \
    && echo "Pre-populated SAPIEN PhysX GPU library: ${so}"
```

Notes:

- The SAPIEN wheel installed in the image must come from `https://github.com/yolkarian/SAPIEN/releases`, not PyPI.
- Put this `RUN` after the layer that installs the SAPIEN wheel, because it imports `sapien.physx.version()`.
- `SAPIEN_HOME` must match the runtime user's home because SAPIEN uses `Path.home()` and has no path override. If running as a non-root user, set `SAPIEN_HOME`, `HOME`, and ownership consistently.
- If reproducible/pinned layers are preferred, use `ARG PHYSX_VERSION=...` matching `sapien.physx.version()` instead of deriving it dynamically, but still verify the `.so`.
- Windows containers use the same `~/.sapien/physx/<version>/` directory but the `-windows` release tag and `physxgpu-windows-vc17win64.zip`, extracting `PhysXGpu_64.dll`.

Verify with a GPU container and fail if SAPIEN still tries to download:

```bash
docker run --rm --gpus 'device=0' \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  IMAGE \
  python - <<'PY' 2>&1 | tee sapien-physx.log
import contextlib
import io
import sapien.physx as physx

buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    physx.enable_gpu()
out = buf.getvalue()
print(out, end="")
assert physx.is_gpu_enabled()
assert "Downloading PhysX GPU library" not in out
print("gpu ok", physx.version())
PY
```

## Initialization workflow

1. Enable GPU PhysX:
   - `sapien.physx.enable_gpu()`
2. Configure global PhysX before creating the GPU system:
   - `sapien.physx.set_gpu_memory_config(...)`
   - `sapien.physx.set_scene_config(...)`
   - `sapien.physx.set_shape_config(...)`
   - `sapien.physx.set_body_config(...)`
3. Create device and systems:
   - `device = sapien.Device("cuda")`
   - `physx_system = sapien.physx.PhysxGpuSystem(device)`
   - `physx_system.set_timestep(dt)`
4. Create builders:
   - `sapien.ActorBuilder()`
   - `sapien.ArticulationBuilder()`
   - configure names, poses, body type, collision groups, materials, joints.
5. Create scenes:
   - physics only: `sapien.Scene(systems=[physx_system])`
   - viewer, sensors, or offscreen rendering: `sapien.Scene(systems=[physx_system, sapien.render.RenderSystem(device)])`
6. For multiple scenes sharing one `PhysxGpuSystem`, isolate envs before adding any PhysX body:
   - Preferred on SAPIEN builds with env-ID support: call `scene.set_environment_id(env_id)` or `physx_system.set_scene_environment_id(scene, env_id)` before adding bodies. Use one unique non-negative env ID per environment.
   - For one shared static world or ground, create it in a scene whose environment ID is `-1`/`0xffffffff`; PhysX GPU broadphase treats this as shared by all envs.
   - `PhysxSceneConfig.gpu_broadphase_env_id_bits` can be set before creating `PhysxGpuSystem`; use enough bits for the env count, for example `min(16, (num_envs - 1).bit_length())`.
   - Use `physx_system.set_scene_offset(scene, offset)` only when spatial separation is needed for visualization or for older SAPIEN builds without env-ID support.
7. Build all actors, articulations, and static terrain:
   - For identical URDF robots, parse the URDF once. Before each build, call `builder.set_scene(scene)`, restore the cached initial pose, apply any per-env pose/offset, then call `builder.build()`.
   - For kinematics-only or IK-only URDF workflows, set `loader.load_visuals = False` and `loader.load_collisions = False` before parsing/loading when geometry is unnecessary.
   - Positive URDF `<limit velocity="...">` values populate `PhysxArticulationJoint.max_joint_velocity`; non-positive placeholders retain the PhysX default. Set any overrides before `gpu_init()` so PhysX GPU copies the limits into its articulation data.
   - PhysX 5.6.1 has no maximum joint-acceleration constraint API; enforce acceleration bounds in the controller.
   - Do not call `loader.parse(...)` once per env for thousands of identical robots.
   - Add `scene.add_heightfield(...)` terrain before `gpu_init()`; its public coordinates are z-up with rows to +x, columns to +y, and samples to +z.
   - If not using env IDs, set scene/env collision groups on the builder or collision shapes before each build.
8. For every scene with a render system, add lights after scene creation and body build, before GPU initialization:
   - `scene.render_system.set_ambient_light(...)`
   - `sapien.Entity()`
   - `sapien.render.RenderDirectionalLightComponent()` or another SAPIEN light component
   - `entity.add_component(light)`
   - `scene.add_entity(entity)`
9. Initialize GPU PhysX after all bodies are built and lights are added:
   - `physx_system.gpu_init()`
10. Cache CUDA views from `physx_system.cuda_*`, cache required GPU indices with `get_gpu_index()` / `get_gpu_pose_index()`, and create reusable CUDA `int32` index buffers for selected indexed GPU APIs.
11. Fetch initial state, write initial GPU buffers, then apply:
    - `gpu_fetch_rigid_dynamic_data()`
    - `gpu_fetch_articulation_link_pose()` / `gpu_fetch_articulation_link_velocity()`
    - `gpu_fetch_articulation_qpos()` / `gpu_fetch_articulation_qvel()`
    - write `cuda_rigid_body_data`, `cuda_rigid_body_force`, `cuda_rigid_body_torque`
    - write `cuda_articulation_qpos`, `cuda_articulation_qvel`, `cuda_articulation_qf`
    - write `cuda_articulation_target_qpos`, `cuda_articulation_target_qvel`
    - `gpu_apply_rigid_dynamic_data()`
    - `gpu_apply_rigid_dynamic_force()` / `gpu_apply_rigid_dynamic_torque()`
    - `gpu_apply_articulation_qpos()` / `gpu_apply_articulation_qvel()` / `gpu_apply_articulation_qf()`
    - `gpu_apply_articulation_root_pose()` / `gpu_apply_articulation_root_velocity()`
    - `gpu_apply_articulation_target_position()` / `gpu_apply_articulation_target_velocity()`
    - `gpu_update_articulation_kinematics()`
12. When the environment has a viewer, initialize it here:
    - `sapien.utils.viewer.Viewer()`
    - `viewer.set_scene(scene)`
    - `viewer.configure_physx_gpu_rendering(physx_system, transport="auto")`
    - `viewer.update_render()`
    - `viewer.render()`
13. Initialize sensors/rendering after `gpu_init()` and GPU index caching:
    - create `sapien.render.RenderCameraComponent(...)`.
    - set camera pose, near/far, fov or intrinsics.
    - sibling PhysX GPU bodies/links are bound to mounted cameras and dynamic render shapes automatically; use explicit `set_gpu_pose_batch_index(...)` only for custom pose-buffer layouts.
    - direct GPU sensors/offscreen cameras: use `sapien.render.RenderSystemGroup([...])`, `set_cuda_poses(physx_system.cuda_rigid_body_data)`, `create_camera_group(...)`, `update_render()`, `take_picture()`, and `get_picture_cuda(...)`.
    - for one rendered env, still prefer `RenderSystemGroup([scene.get_render_system()])` so dynamic body poses come directly from GPU buffers without `sync_poses_gpu_to_cpu()`.

## GPU articulation link/Jacobian buffers

SAPIEN has no IK solver. It exposes these PhysX GPU buffers and dense articulation Jacobians after `gpu_init()`.

- Required setup:
  1. `sapien.physx.enable_gpu()` before creating the system.
  2. Create `physx_system = sapien.physx.PhysxGpuSystem(...)` and a physics-only scene: `scene = sapien.Scene([physx_system])`.
  3. For URDF robots that only need topology/inertial/joints, set `loader.load_visuals = False` and `loader.load_collisions = False` before parsing/loading.
  4. Build all target articulations into the scene.
  5. Call `physx_system.gpu_init()`; only then are `articulation.gpu_index`, CUDA buffers, link data, and dense Jacobian buffers valid.
  6. Cache CUDA views and create reusable selected-articulation index buffers.
- Index buffers for selected GPU APIs accept only `sapien.CudaArray` or CUDA-array-interface objects such as CUDA `torch.Tensor`, `cupy.ndarray`, or Numba CUDA device arrays. The underlying data must be 1D contiguous CUDA `int32` SAPIEN `articulation.gpu_index` values on the same CUDA device as the PhysX system. NumPy arrays, Python lists, CPU tensors, `int64` tensors, non-contiguous views, and cross-device arrays are invalid. Keep external owners alive until SAPIEN's CUDA stream completes.
- `cuda_articulation_link_data` shape is `(articulation_count, max_links, 13)` and uses `articulation.gpu_index` plus low-level `link.index`. Channels are `0:3` world position, `3:7` quaternion `wxyz`, `7:10` linear velocity, and `10:13` angular velocity.
- `gpu_compute_articulation_jacobian(...)` writes `cuda_articulation_jacobian`, a padded tensor with shape `(articulation_count, max_rows, max_cols)`, where `max_rows = 6 + (max_links - 1) * 6` and `max_cols = 6 + max_dofs`.
- Slice each valid Jacobian with `physx_system.cuda_articulation_jacobian_shape[articulation.gpu_index]` or `articulation.get_jacobian_shape()`. For fixed-base articulations, valid shape is `((link_count - 1) * 6, dof)` and root rows are omitted. For floating-base articulations, valid shape is `(6 + (link_count - 1) * 6, 6 + dof)` and the first six columns are root linear/angular velocity.
- Jacobian row order is `[vx, vy, vz, wx, wy, wz]` per link. PhysX reports the linear component at each link center of mass; shift the linear rows in application code when the task frame is the link origin.

Minimal buffer-access setup:

```python
import sapien

sapien.physx.enable_gpu()
physx_system = sapien.physx.PhysxGpuSystem()
scene = sapien.Scene([physx_system])

# Build articulations into scene here.

physx_system.gpu_init()
link_data = physx_system.cuda_articulation_link_data
jacobian = physx_system.cuda_articulation_jacobian
jacobian_shape = physx_system.cuda_articulation_jacobian_shape

# Create a CUDA int32 index buffer via sapien.CudaArray or a
# CUDA-array-interface owner to select articulations.
physx_system.gpu_compute_articulation_jacobian(index_buffer)
```

## Rendering workflow

- Policy video monitoring during training or evaluation:
  - Do not enable viewer.
  - Pick one env, usually env 0.
  - Create `sapien.render.RenderSystem(device)` only for that env's `sapien.Scene`.
  - Create cameras only in that env.
  - Use the GPU batched-render path below, even for a single rendered scene.
  - Do not create `RenderSystem` or cameras for the other training envs.
- Direct GPU camera/offscreen rendering, including one-scene capture:
  1. After `gpu_init()`, sibling PhysX GPU bodies/links are bound automatically. Use `set_gpu_pose_batch_index(...)` only for custom pose-buffer layouts.
  2. For free cameras created with `scene.add_camera(...)`, set the initial world transform in exactly one place before `gpu_init()`, usually `camera.set_local_pose(sapien.Pose(p=..., q=...))` while the owning entity remains identity. `gpu_init()` seeds the camera's CUDA pose row from this pose and seals the CPU pose afterwards.
  3. Create one `RenderSystemGroup([scene.get_render_system(), ...])`, call `set_cuda_poses(physx_system.cuda_rigid_body_data)`, create every `RenderCameraGroup`, then call `render_system_group.gpu_init()` exactly once. Camera groups cannot be created after `gpu_init()`, and steady-state calls before it raise.
  4. For a PhysX-mounted camera or another camera with a GPU pose batch index, after the required `gpu_fetch_*()` calls use `RenderSystemGroup.update_render()`, `RenderCameraGroup.take_picture()`, then `RenderCameraGroup.get_picture_cuda(name)`.
  5. For a fixed or moving free camera (raster or RT), use this order every frame:
     1. When the camera moves, write its world pose row `[px, py, pz, qw, qx, qy, qz]` in `camera_group.cuda_free_camera_poses` on the GPU, or copy one CPU-authored pose explicitly with `camera_group.set_free_camera_pose(camera, pose)`.
     2. Call `render_system_group.update_render()`; it writes PhysX CUDA body poses and all grouped camera buffers with bidirectional Vulkan/CUDA semaphore ordering.
     3. Run `camera.take_picture()` and keep `camera.get_picture_cuda(name)`, or capture through `camera_group.take_picture()`; both read the same GPU-owned transforms.
  6. Copy only the kept CUDA image to CPU when a video encoder or logger requires it. This path does not call `sync_poses_gpu_to_cpu()`.

  ```python
  # One-time setup: resolve scenes, prepare resources, seed snapshots, seal ownership.
  render_group = sapien.render.RenderSystemGroup([scene.get_render_system()])
  render_group.set_cuda_poses(physx_system.cuda_rigid_body_data)
  camera.set_local_pose(initial_pose)  # seeds the CUDA row at gpu_init()
  camera_group = render_group.create_camera_group([camera], ["Color"])
  render_group.gpu_init()

  # Each free-camera frame: one pose-row write, one ordered update, one capture.
  camera_group.set_free_camera_pose(camera, updated_pose)  # or write cuda_free_camera_poses
  render_group.update_render()   # ordered CUDA body/link + camera update
  camera.take_picture()
  color_cuda = camera.get_picture_cuda("Color")
  ```
- CPU-pose render path:
  - `scene.update_render()` / `RenderSystem.step()` alone leaves GPU PhysX bodies at stale CPU poses unless `PhysxGpuSystem.sync_poses_gpu_to_cpu()` is called.
  - Grouped capture involves no CPU transform participation: GPU-sourced body poses and all grouped camera poses come only from `RenderSystemGroup.update_render()`; `scene.update_render()` serves CPU rendering and the Viewer, not grouped capture.
- Viewer path:
  1. Call `gpu_init()` before Viewer submission. The Viewer auto-detects one unambiguous initialized `PhysxGpuSystem` in its resolved base plus shared scenes; use `Viewer.configure_physx_gpu_rendering(physx_system, transport="auto")` to choose explicitly.
  2. `"auto"` selects direct CUDA/Vulkan interop on a compatible same physical device and compact pinned-host staging on different devices. Both raster and RT Viewer shader paths are supported.
  3. For physical dragging or queued gizmo teleports, call `Viewer.apply_interactions()` immediately before every PhysX substep. Ctrl + left drag uses a damped point spring.
  4. `Viewer.update_render()` after each displayed simulation state.
  5. `Viewer.render()` to draw without another pose fetch.
  6. Inspect `Viewer.pose_transport` for the active choice and `Viewer.pose_transfer_bytes` for cumulative pose D2H bytes. Staged submission copies 28 bytes per unique rendered body/link pose and does not update CPU entities.
  7. Viewer spring composition preserves the exposed application force/torque buffers. Do not issue a later apply for the same selected body before `step()`, because it would replace the composed spring.
  8. GPU-aware Entity and Articulation windows transfer only the selected pose or articulation row, cache it for the submitted frame, and queue supported edits for `apply_interactions()`. Collapsed windows do not read GPU state. CPU contact reports and CPU Pinocchio IK are explicitly unavailable in these windows under PhysX GPU.
  9. `Viewer.set_scenes(scenes)` and camera `set_scenes(scenes)` select base scenes plus associated shared scenes once. They do not accept render offsets; place entities explicitly when scenes should appear separated.
  10. Viewer plugins may call `begin_gpu_interaction()`, `update_gpu_interaction_target()`, and `end_gpu_interaction()`, or queue teleports with `queue_gpu_rigid_dynamic_pose()` and `queue_gpu_articulation_root_pose()`. All commands remain deferred until `apply_interactions()`.
- Use `sync_poses_gpu_to_cpu()` only for explicit CPU-state debugging or the Viewer `cpu-debug` transport. SAPIEN documents it as a super-slow helper that downloads all poses from GPU to CPU entities.
- When adding policy-eval or teleoperation keyboard controls on top of the interactive viewer, do not reuse SAPIEN's built-in camera/navigation keys such as `W/A/S/D/Q/E`. Prefer a separate key cluster, for example `I/K` for forward/backward command, `J/L` for lateral command, `U/O` for yaw, `C` to clear commands, and `N` to reset.

## Reset workflow

1. Write rigid poses and velocities to `cuda_rigid_body_data`.
2. Clear `cuda_rigid_body_force` and `cuda_rigid_body_torque`.
3. Write articulation positions and velocities to `cuda_articulation_qpos` and `cuda_articulation_qvel`.
4. Clear `cuda_articulation_qf`.
5. Write drive targets to `cuda_articulation_target_qpos` and `cuda_articulation_target_qvel`.
6. Apply all reset buffers:
   - `gpu_apply_rigid_dynamic_data()`
   - `gpu_apply_rigid_dynamic_force()` / `gpu_apply_rigid_dynamic_torque()`
   - `gpu_apply_articulation_qpos()` / `gpu_apply_articulation_qvel()` / `gpu_apply_articulation_qf()`
   - `gpu_apply_articulation_root_pose()` / `gpu_apply_articulation_root_velocity()`
   - `gpu_apply_articulation_target_position()` / `gpu_apply_articulation_target_velocity()`
7. Call `gpu_update_articulation_kinematics()`.
8. Fetch state with the needed `gpu_fetch_*()` calls.
9. For full reset, call `physx_system.step()` and fetch again; for partial reset, do not advance physics for non-reset scenes.
10. Update rendering only if viewer or sensors exist. Normal Viewer rendering should use direct/staged transport, and direct sensors/offscreen capture should use `RenderSystemGroup` with CUDA pose buffers. Use `sync_poses_gpu_to_cpu()` only for explicit CPU debugging or Viewer `cpu-debug`.

## Step workflow

1. If the environment uses non-persistent rigid-body wrenches, clear stale `cuda_rigid_body_force` / `cuda_rigid_body_torque` rows at the start of step and apply the cleared buffers.
2. Compute action targets from already fetched state.
3. Write only buffers modified by this step; do not write unrelated state buffers. Common action buffers are:
   - `cuda_articulation_target_qpos`
   - `cuda_articulation_target_qvel`
   - `cuda_articulation_qf`
   - `cuda_rigid_body_force`
   - `cuda_rigid_body_torque`
4. Apply exactly the buffers modified by this step:
   - `cuda_articulation_target_qpos` -> `gpu_apply_articulation_target_position()`
   - `cuda_articulation_target_qvel` -> `gpu_apply_articulation_target_velocity()`
   - `cuda_articulation_qf` -> `gpu_apply_articulation_qf()`
   - `cuda_rigid_body_force` -> `gpu_apply_rigid_dynamic_force()` or `gpu_apply_rigid_dynamic_force(index_buffer)`
   - `cuda_rigid_body_torque` -> `gpu_apply_rigid_dynamic_torque()` or `gpu_apply_rigid_dynamic_torque(index_buffer)`
5. Call `physx_system.step()`.
6. If one control step contains multiple physics substeps and uses `cuda_articulation_qf`, reapply `gpu_apply_articulation_qf()` between substeps.
7. Fetch state with the needed `gpu_fetch_*()` calls.
8. Update viewer/render/sensors only if they exist. Normal Viewer rendering should use direct/staged transport, and direct sensors/offscreen capture should use `RenderSystemGroup` with CUDA pose buffers. Use `sync_poses_gpu_to_cpu()` only for explicit CPU debugging or Viewer `cpu-debug`.
9. Read observations, rewards, termination, or diagnostics after fetch/render update.

## Collision and scene isolation

- Prefer PhysX GPU environment IDs when available:
  - `scene.set_environment_id(env_id)` or `physx_system.set_scene_environment_id(scene, env_id)` must run before adding bodies to that scene.
  - Bodies with the same non-negative env ID collide with each other.
  - Env ID `-1` / `0xffffffff` is the PhysX GPU broadphase wildcard for shared bodies, such as a single ground plane that should collide with all envs.
- `PhysxGpuSystem.set_scene_offset(scene, offset)` offsets bodies in the shared PhysX scene; call it before adding bodies when you need spatial separation or when env IDs are unavailable.
- Use `sapien.physx.PhysxCollisionShape.set_collision_groups([word0, word1, word2, word3])` for shape-level type/affinity or ignore filtering, not as the primary env-isolation mechanism when env IDs exist.
- In SAPIEN GPU collision-group filtering:
  - `word0` and `word1` are contact type and affinity.
  - `word2` is ignore group.
  - `word3` packs a scene ID in the upper 16 bits and an ignore ID in the lower 16 bits.
  - ID `0xffff` is shared: as a scene ID it collides with all scene IDs, and as an ignore ID it does not suppress collisions.
  - Older SAPIEN env-isolation code may use `word3`; prefer PhysX env IDs on current builds.

## Common mistakes

- Calling `gpu_init()` before all bodies, articulations, and static terrain/height fields are built.
- Creating a `RenderSystem` for pure physics environments.
- Creating sensors outside the post-`gpu_init()` sensor/render phase.
- Looking for `sapien.physx.GpuInverseKinematicsSolver` or `sapien.physx.gpu_inverse_kinematics`; SAPIEN has no IK solver and these do not exist.
- Passing CPU, non-contiguous, non-`int32`, cross-device, or PhysX-internal GPU index buffers to indexed GPU APIs.
- Treating the padded `cuda_articulation_jacobian.shape` as the valid matrix shape; use `cuda_articulation_jacobian_shape` or `articulation.get_jacobian_shape()` and account for link-COM linear rows.
- Reading `cuda_*` buffers before fetch.
- Writing `cuda_*` buffers without apply.
- Recreating `sapien.CudaArray.torch()` views repeatedly.
- Forgetting `wxyz` quaternion order.
- Calling all apply functions in the normal step path and overwriting simulated state with stale buffers.
- Calling `sync_poses_gpu_to_cpu()` for normal Viewer rendering, training, reset, step, eval video, sensors, or offscreen capture; it downloads all poses to CPU and should be reserved for explicit CPU debugging or Viewer `cpu-debug`.
- Using `scene.update_render()` / `RenderSystem.step()` alone for GPU-PhysX camera capture; it reads CPU entity poses and renders stale dynamic bodies.
- Calling `set_local_pose()` on a camera in a `RenderCameraGroup`; `gpu_init()` seals camera transforms for GPU ownership and the setter raises. Write the group CUDA pose row instead.
- Adding `scene.update_render()`, discarded warm-up captures, or a second `update_render()` to the grouped capture loop. `RenderSystemGroup.update_render()` performs the bidirectional Vulkan/CUDA semaphore handoff and owns every grouped transform, so one pose write, one group update, and one capture per frame is the contract.
- Calling `update_render()` or capturing before `RenderSystemGroup.gpu_init()`, or creating camera groups after it; the lifecycle is configure, `gpu_init()`, then steady state.
- Moving a CPU-owned static body (no CUDA pose source) after `gpu_init()`; its transform is a sealed one-time snapshot and the next `scene.update_render()` raises.
- Recreating `RenderCameraGroup` every frame to move a free camera. `BatchedRenderSystem::mCameraBatches` retains every created batch, so this accumulates GPU/Vulkan resources; create once and use direct `RenderCameraComponent` capture after the ordered camera/body updates.
- Shipping a SAPIEN GPU container without pre-baking `$HOME/.sapien/physx/<version>/`, so every fresh `docker run` re-downloads `physxgpu-linux-clang.zip`; bake the extracted library into the image (see "Pre-bake the PhysX GPU library into Docker images" above).
