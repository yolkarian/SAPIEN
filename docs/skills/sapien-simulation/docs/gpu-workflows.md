# SAPIEN GPU Simulation Workflows

## Core rules

- Install SAPIEN from `https://github.com/yolkarian/SAPIEN/releases` (fork release wheels), not from PyPI.
- Use `sapien.physx.enable_gpu()` before creating `sapien.physx.PhysxGpuSystem`.
- Use `sapien.Device("cuda")` for PhysX; use the same device for `sapien.render.RenderSystem` when rendering exists.
- For many identical URDF robots, call `loader.parse(...)` once, cache the resulting articulation builder, and reuse it across scenes by calling `builder.set_scene(scene)`, restoring its saved `initial_pose`, and then `builder.build()` for each env. For kinematics-only or IK-only workflows, set `loader.load_visuals = False` and `loader.load_collisions = False` before parsing/loading when geometry is unnecessary.
- Prefer PhysX GPU scene environment IDs over collision-group hacks for multi-env isolation: set a unique env ID before adding each env's bodies, and set env ID `-1`/`0xffffffff` for shared bodies such as a global ground plane.
- In GPU runtime, update state through `sapien.physx.PhysxGpuSystem.cuda_*` buffers plus `gpu_apply_*()`.
- Read state only after the needed `gpu_fetch_*()` calls.
- `PhysxGpuSystem.sync_poses_gpu_to_cpu()` downloads all GPU poses to CPU SAPIEN entities; use it only for viewer/debug paths that need CPU entity poses, never for training, reset, step, sensors, video, or offscreen capture.
- For direct GPU camera/offscreen rendering, use GPU pose batch indices with `sapien.render.RenderSystemGroup.set_cuda_poses(physx_system.cuda_rigid_body_data)` and read images through `get_picture_cuda(...)` instead of syncing poses to CPU.
- Cache `sapien.CudaArray.torch()` views once after `gpu_init()`; do not recreate them in loops.
- Cache common GPU indices once after `gpu_init()`:
  - `sapien.physx.PhysxArticulation.get_gpu_index()`
  - `sapien.physx.PhysxRigidDynamicComponent.get_gpu_index()`
  - `sapien.physx.PhysxRigidDynamicComponent.get_gpu_pose_index()`
  - `sapien.physx.PhysxArticulationLinkComponent.get_gpu_pose_index()`
- SAPIEN quaternion convention is `wxyz`.
- SAPIEN no longer provides `GpuInverseKinematicsSolver` or `gpu_inverse_kinematics`. For custom batched GPU IK, build target articulations into a `Scene` owned by the `PhysxGpuSystem`, call `gpu_init()` so `articulation.gpu_index` and CUDA buffers are valid, then use `cuda_articulation_link_data`, `cuda_articulation_jacobian`, and `cuda_articulation_jacobian_shape` directly.
- Indexed GPU APIs accept only these Python argument types for selected-index buffers:
  - `sapien.CudaArray`
  - any CUDA object exposing `__cuda_array_interface__`, such as a CUDA `torch.Tensor`, `cupy.ndarray`, or Numba CUDA device array
  The underlying array must be 1D, contiguous, CUDA `int32`, on the same CUDA device as the PhysX system, and contain SAPIEN `gpu_index` values, not PhysX-internal GPU indices. NumPy arrays, Python lists, CPU tensors, `int64` tensors, non-contiguous views, and cross-device CUDA arrays are not valid. Keep the owner of an external index tensor alive until SAPIEN's CUDA stream has finished using it.
- Custom IK writes shared PhysX GPU buffers when it applies qpos/root state and updates kinematics. Use a scratch physics-only scene/system when IK must not touch the main simulation buffers or concurrent GPU work.

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
    - `physx_system.sync_poses_gpu_to_cpu()`
    - `viewer.render()`
13. Initialize sensors/rendering after `gpu_init()` and GPU index caching:
    - create `sapien.render.RenderCameraComponent(...)`.
    - set camera pose, near/far, fov or intrinsics.
    - for mounted cameras, call `RenderCameraComponent.set_gpu_pose_batch_index(link_gpu_pose_index)`.
    - for dynamic render shapes, call `RenderShape.set_gpu_pose_batch_index(gpu_pose_index)`.
    - direct GPU sensors/offscreen cameras: use `sapien.render.RenderSystemGroup([...])`, `set_cuda_poses(physx_system.cuda_rigid_body_data)`, `create_camera_group(...)`, `update_render()`, `take_picture()`, and `get_picture_cuda(...)`.
    - for one rendered env, still prefer `RenderSystemGroup([scene.get_render_system()])` so dynamic body poses come directly from GPU buffers without `sync_poses_gpu_to_cpu()`.

## Custom GPU IK workflow

SAPIEN does not provide a built-in batched IK policy. Build custom batched IK on top of PhysX GPU buffers and dense articulation Jacobians.

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
- A typical custom IK loop writes `cuda_articulation_qpos`, calls `gpu_apply_articulation_qpos(index_buffer)`, calls `gpu_update_articulation_kinematics(index_buffer)`, fetches link poses, calls `gpu_compute_articulation_jacobian(index_buffer)`, solves externally, and repeats.

Minimal IK-only setup:

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
# CUDA-array-interface owner, then run custom IK math in the caller.
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
  1. Bind dynamic render shapes with `RenderShape.set_gpu_pose_batch_index(body.get_gpu_pose_index())`.
  2. For mounted cameras, bind them with `RenderCameraComponent.set_gpu_pose_batch_index(link_gpu_pose_index)`; for free/follow cameras, update their CPU pose explicitly before rendering.
  3. Create `RenderSystemGroup([scene.get_render_system(), ...])` and `create_camera_group(cameras, picture_names)`.
  4. Call `RenderSystemGroup.set_cuda_poses(physx_system.cuda_rigid_body_data)`.
  5. After the required `gpu_fetch_*()` calls, call `RenderSystemGroup.update_render()`.
  6. Call `RenderCameraGroup.take_picture()`.
  7. Read frames with `RenderCameraGroup.get_picture_cuda(name)`; copy to CPU only if a video encoder or logger needs CPU arrays.
- CPU-pose render path:
  - `scene.update_render()` / `RenderSystem.step()` and `RenderCameraComponent.get_picture(...)` read CPU SAPIEN entity poses. Under GPU PhysX these poses are stale unless `PhysxGpuSystem.sync_poses_gpu_to_cpu()` is called first.
  - This path is acceptable for viewer/debug rendering only; do not use it for normal offscreen video/camera capture.
- Viewer path:
  1. `PhysxGpuSystem.sync_poses_gpu_to_cpu()`
  2. `Viewer.render()`
- Use `sync_poses_gpu_to_cpu()` only in viewer/debug paths. SAPIEN documents it as a super-slow debug helper that downloads all poses from GPU to CPU entities.
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
10. Update rendering only if viewer or sensors exist. Use `sync_poses_gpu_to_cpu()` only for viewer/debug; direct sensors/offscreen capture should use `RenderSystemGroup` with CUDA pose buffers.

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
   - `cuda_rigid_body_force` -> `gpu_apply_rigid_dynamic_force()`
   - `cuda_rigid_body_torque` -> `gpu_apply_rigid_dynamic_torque()`
5. Call `physx_system.step()`.
6. If one control step contains multiple physics substeps and uses `cuda_articulation_qf`, reapply `gpu_apply_articulation_qf()` between substeps.
7. Fetch state with the needed `gpu_fetch_*()` calls.
8. Update viewer/render/sensors only if they exist. Use `sync_poses_gpu_to_cpu()` only for viewer/debug; direct sensors/offscreen capture should use `RenderSystemGroup` with CUDA pose buffers.
9. Read observations, rewards, termination, or diagnostics after fetch/render update.

## Collision and scene isolation

- Prefer PhysX GPU environment IDs when available:
  - `scene.set_environment_id(env_id)` or `physx_system.set_scene_environment_id(scene, env_id)` must run before adding bodies to that scene.
  - Bodies with the same non-negative env ID collide with each other.
  - Env ID `-1` / `0xffffffff` is the PhysX GPU broadphase wildcard for shared bodies, such as a single ground plane that should collide with all envs.
- `PhysxGpuSystem.set_scene_offset(scene, offset)` offsets bodies in the shared PhysX scene; call it before adding bodies when you need spatial separation or when env IDs are unavailable.
- Use `sapien.physx.PhysxCollisionShape.set_collision_groups([word0, word1, word2, word3])` for shape-level type/affinity or ignore filtering, not as the primary env-isolation mechanism when env IDs exist.
- In SAPIEN GPU collision-group filtering:
  - `word2` is ignore group.
  - `word0` and `word1` are contact type and affinity.
  - Older SAPIEN env-isolation code may also use `word3`; prefer PhysX env IDs on current builds.

## Common mistakes

- Calling `gpu_init()` before all bodies, articulations, and static terrain/height fields are built.
- Creating a `RenderSystem` for pure physics environments.
- Creating sensors outside the post-`gpu_init()` sensor/render phase.
- Looking for `sapien.physx.GpuInverseKinematicsSolver` or `sapien.physx.gpu_inverse_kinematics`; these helpers were removed. Use low-level GPU buffers for custom IK.
- Passing CPU, non-contiguous, non-`int32`, cross-device, or PhysX-internal GPU index buffers to indexed GPU APIs.
- Treating the padded `cuda_articulation_jacobian.shape` as the valid matrix shape; use `cuda_articulation_jacobian_shape` or `articulation.get_jacobian_shape()` and account for link-COM linear rows.
- Reading `cuda_*` buffers before fetch.
- Writing `cuda_*` buffers without apply.
- Recreating `sapien.CudaArray.torch()` views repeatedly.
- Forgetting `wxyz` quaternion order.
- Calling all apply functions in the normal step path and overwriting simulated state with stale buffers.
- Calling `sync_poses_gpu_to_cpu()` for training, reset, step, eval video, sensors, or offscreen capture; it downloads all poses to CPU and should be reserved for viewer/debug paths.
- Using `scene.update_render()` / `RenderSystem.step()` alone for GPU-PhysX camera capture; it reads CPU entity poses and will render stale dynamic bodies unless you first sync, so prefer `RenderSystemGroup` with CUDA pose buffers.
- Shipping a SAPIEN GPU container without pre-baking `$HOME/.sapien/physx/<version>/`, so every fresh `docker run` re-downloads `physxgpu-linux-clang.zip`; bake the extracted library into the image (see "Pre-bake the PhysX GPU library into Docker images" above).
