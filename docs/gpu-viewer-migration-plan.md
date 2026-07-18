# PhysX GPU Viewer Migration Plan

## Purpose

This plan upgrades the existing SAPIEN Viewer to visualize and interact with `PhysxGpuSystem` state without requiring a full `sync_poses_gpu_to_cpu()` on every displayed frame.

The migration is deliberately narrow in scope:

1. Keep the current Vulkan renderer.
2. Keep the current SAPIEN Viewer UI, plugin system, windows, and widgets.
3. Prioritize interactive visualization for PhysX GPU simulation.
4. Change only the pose-data transport, synchronization, and GPU-aware handling inside existing interaction and property windows.
5. Preserve the existing `Viewer.set_scene` / `Viewer.set_scenes` scene-selection model while removing all render-layer offset behavior.
6. Give scene cameras the same default-scene and explicit-multi-scene selection semantics as the Viewer.

Newton is only an architectural reference for the explicit frame contract (`begin_frame` / `log_state` / `end_frame` / `apply_forces`, caller-driven, no background thread) and for the Stage 5.1 physical picking force model. Its OpenGL renderer, UI implementation, and GL upload code will not be migrated.

Two concepts in this plan are deliberately not taken from Newton and have no reference implementation there:

- Zero-copy pose transport. Newton moves poses through a per-frame D2H copy into pinned memory followed by a GL buffer upload. The same-device zero-copy reference is SAPIEN's own `RenderSystemGroup`.
- Scratch-buffer wrench composition (Stage 5.3). Newton accumulates picking forces directly into the solver force buffer and relies on the application clearing forces every frame. The scratch-buffer design here is new, required by PhysX Direct GPU replace semantics.

## Explicit Non-Goals

This migration will not introduce:

- an OpenGL renderer or OpenGL fallback;
- a second Viewer UI framework;
- a new UI component library;
- a generic property registry or reflection system;
- new ECS components whose only purpose is Viewer state;
- a rewrite of `ControlWindow`, `EntityWindow`, `ArticulationWindow`, or `TransformWindow`;
- unrelated CPU Viewer features;
- renderer-side grid layout or per-scene display offsets;
- continuous synchronization of every CPU `Entity.pose` under PhysX GPU.

The existing UI windows will remain responsible for presentation and callbacks. Their data access and command application will become GPU-aware through internal Viewer services, not through additional scene components.

The legacy multi-scene offset mechanism is intentionally replaced rather than retained as a fallback. The Viewer retains its scene-selection API, and scene cameras gain the equivalent API. These APIs select render content only: every object is rendered at its existing pose. PhysX environment IDs control collision isolation and shared-world behavior, but never produce a render transform.

## Render-Scene Selection Contract

Viewer and camera rendering use the same resolver:

1. The base scene set is selected as follows:
   - a default Viewer renders its current scene;
   - `viewer.set_scenes(scenes)` renders exactly the specified base scenes;
   - a scene camera renders its owning scene by default;
   - `camera.set_scenes(scenes)` renders exactly the specified base scenes.
2. The effective render set is the ordered, deduplicated union of the base scene set and the shared scenes registered with the same render context. A render system with `batched_render_shared=True` is included once even if it is also present in the base scene set.
3. All objects and lights in the effective render set are aggregated into one render scene and one output coordinate system.
4. Scene selection never applies a transform. There is no implicit grid, no per-scene display offset, and no offset parameter in the new Viewer or camera `set_scenes` contract. Objects from different scenes may overlap visually.
5. Camera multi-scene rendering is opt-in. If `camera.set_scenes()` is never called, camera ownership, owning-scene rendering, shared-scene inclusion, mounted pose behavior, render properties and targets, update flow, and `take_picture()` behavior remain unchanged. Calling `set_scenes()` changes only the base render-scene selection.

Implement this resolution once as an internal service shared by `SapienRendererWindow`, scene cameras, and `RenderSystemGroup`. Shared-scene discovery must use explicit render-context or system membership rather than a process-wide scene scan.

## Current State

SAPIEN currently has two relevant GPU rendering paths:

1. The interactive Viewer reads CPU `Entity.pose` values. With `PhysxGpuSystem`, applications normally call `sync_poses_gpu_to_cpu()` first. This fetches all GPU poses, copies the full rigid-body buffer to host memory, and updates every CPU Entity.
2. `RenderSystemGroup` can use CUDA to update Vulkan object-transform buffers directly. This is already used for same-device batched cameras and offscreen rendering, but it is not integrated into the complete interactive Viewer lifecycle.
3. `Viewer.set_scenes()` currently creates a CPU-managed `SceneGroup` and applies explicit or implicit grid offsets. That layout behavior will be removed while retaining `set_scenes` as explicit scene selection. Scene cameras currently lack the equivalent selection API and will gain it. Parallel scenes are isolated by PhysX environment IDs, shared scenes are represented by `batched_render_shared`, and each Viewer/camera output aggregates its resolved base-plus-shared scene set without changing object poses.

The direct CUDA–Vulkan path is the preferred fast path, but the Viewer must also support configurations such as:

```text
Discrete NVIDIA GPU: PhysX and CUDA computation
Integrated GPU:      Vulkan window rendering
```

In this case, the compute GPU generally cannot write Vulkan memory owned by the integrated GPU. The required path is:

```text
Compute GPU → pinned host memory → Vulkan rendering GPU
```

The final Viewer therefore needs both same-device direct transport and cross-device staged transport.

## Target Architecture

```text
PhysX GPU pose buffers
        │
        ├─ same device ── CUDA writes Vulkan transform buffers directly
        │
        └─ cross device ─ CUDA packs poses
                          → pinned host staging
                          → Vulkan upload/compute
        │
        ▼
Resolved Vulkan render scene per output
(current or explicitly selected base scenes + shared scenes once)
        │
        ├─ Viewer viewport + current UI/plugins
        │      ├─ ControlWindow selection and focus
        │      ├─ EntityWindow values
        │      ├─ ArticulationWindow values and controls
        │      └─ TransformWindow gizmo and interaction
        │
        └─ scene-camera image outputs
               │
               ▼
        GPU-aware reads or queued PhysX GPU commands
```

No new Viewer-facing component model is required. The transport implementation may use internal C++ classes or private Python helper objects, but they must not be added to scene Entities or exposed as another UI framework.

Each Viewer or camera output has one coordinate system and one destination transform buffer. There is no scene transform in the pose path: the direct and staged kernels compute only `body pose × shape local pose`. If applications want environments to appear spatially separated, they must place the entities at separated poses; otherwise objects from different scenes may intentionally overlap.

## Transport Selection

The Viewer selects a transport in `auto` mode:

| Simulation and rendering configuration | Transport |
| --- | --- |
| PhysX CUDA and Vulkan use the same compatible physical device | Same-device CUDA–Vulkan direct |
| PhysX CUDA and Vulkan use different physical devices | Pinned-host staged transfer |
| Vulkan device has no matching CUDA device | Pinned-host staged transfer |
| CPU PhysX | Existing CPU Entity path |
| Explicit GPU debugging that requires all CPU poses | Explicit `sync_poses_gpu_to_cpu()` |
| Two peer-capable NVIDIA GPUs | Staged initially; optional P2P optimization later |

Zero-copy rendering is a fast path, not a Viewer semantic requirement. The staged path is required hardware support rather than a legacy compatibility path.

## Explicit Render-State Update Boundary

The Viewer should expose one explicit semantic boundary for submitting simulation state while keeping transport-specific synchronization internal:

```python
while not viewer.closed:
    viewer.apply_interactions()
    physx_system.step()
    viewer.update_render()
    viewer.render()
```

The methods have distinct responsibilities:

- `viewer.update_render()` submits the latest simulation state to the selected pose transport for the Viewer's resolved scene set. It performs or reuses the required PhysX GPU pose fetches, updates existing render objects that are not bound to PhysX GPU poses, and finally updates GPU-bound Vulkan transforms.
- `viewer.render()` draws the most recently submitted Vulkan state and the existing Viewer UI. It does not implicitly fetch poses or submit another transform update.
- Scene cameras keep their existing render/update lifecycle; `camera.set_scenes()` only changes the camera's resolved render content and never applies offsets. The camera renderer and `RenderSystemGroup` consume the same internal scene resolver as the Viewer.
- `viewer.apply_interactions()` applies commands collected by the existing Viewer controls before the next physics step.

There should be no autonomous background thread that watches simulation state. The explicit call defines which physics step is submitted, prevents unwanted updates when simulation and display rates differ, and makes pause, reset, multiple substeps, custom CUDA streams, profiling, and debugging deterministic.

The implementation inside `viewer.update_render()` should remain asynchronous where possible:

```text
1. Update existing render objects not bound to PhysX GPU poses
2. Fetch or reuse the required PhysX GPU pose buffers
3. Enqueue the selected direct or staged transport
4. Signal CUDA/Vulkan synchronization primitives
5. Return without a device-wide synchronization
```

`viewer.render()` then consumes that submission through the existing Vulkan synchronization. Calling `viewer.render()` repeatedly while paused must not repeat pose fetches or transform updates.

The implementation must track whether the required pose buffers were already fetched for the submitted simulation frame, so application-side observation fetches are not duplicated.

## Zero-Copy Performance Expectations

Same-device zero-copy removes the most expensive parts of the current Viewer path: full pose D2H, host-side Entity updates, and host-to-render-device upload. It should therefore make `viewer.update_render()` relatively inexpensive for raster rendering, but it does not make the call free.

A same-device direct update still performs some or all of the following work:

- PhysX Direct GPU pose fetches when the application has not already fetched them;
- PhysX-to-SAPIEN pose-layout conversion kernels (already implemented; they run inside the `gpu_fetch_*` calls);
- one transform kernel over the rendered shapes;
- mounted-camera transform updates;
- writes to Vulkan object-transform buffers;
- CUDA/Vulkan semaphore signal and wait operations;
- scene-version and topology checks.

For raster rendering with moderate shape counts, this work is expected to be small relative to full rendering and should impose little CPU overhead because it requires no pose D2H or device-wide synchronization. The cost remains proportional to the number of fetched bodies and rendered shapes, so large batched scenes must still be benchmarked.

RT updates can be materially more expensive because `viewer.update_render()` may also update or refit the TLAS and reset accumulation. These costs are not eliminated by zero-copy.

Performance acceptance must therefore be based on measured GPU and CPU timings rather than the assumption that zero-copy has no cost. Benchmarks should separately report:

1. PhysX pose fetch time;
2. pose conversion time;
3. object and camera transform-update time;
4. synchronization overhead;
5. RT acceleration-structure update time;
6. total Viewer update time excluding draw;
7. total Vulkan draw time.

---

## Stage 0: Lock the Scope and Establish GPU Baselines

**Priority: Immediate** — **Complexity: Small**

### Tasks

1. Record the following decisions:
   - the renderer remains Vulkan;
   - the existing Viewer UI remains unchanged at the framework level;
   - PhysX GPU visualization is the primary migration target;
   - same-device direct and cross-device staged transports are production paths;
   - CPU `Entity.pose` may remain stale during normal GPU Viewer operation;
   - Viewer commands are applied before a later physics step;
   - a default Viewer/camera renders its current or owning scene plus the associated shared scenes;
   - `viewer.set_scenes(scenes)` and `camera.set_scenes(scenes)` replace the base scene set with the specified scenes, then include associated shared scenes once;
   - every resolved scene set is aggregated into one output render scene without per-scene transforms;
   - objects render at their existing poses, and shared render systems contribute their complete content once;
   - PhysX environment IDs affect physics membership only, not render placement;
   - topology changes require transport rebinding.
2. Add device-capability discovery for:
   - PhysX CUDA device;
   - Vulkan physical device;
   - Vulkan device CUDA ID when one exists (already exposed: `src/device.cpp` records a `cudaId` per Vulkan device);
   - physical-device UUID or PCI bus identity (PCI parsing already exists in `src/device.cpp`);
   - CUDA external-memory support;
   - CUDA external-semaphore support;
   - optional CUDA peer-access capability.

   Reuse and extend the existing discovery in `src/device.cpp` rather than adding a parallel mechanism. `findBestRenderDevice` already ranks CUDA-capable Vulkan devices highest, so the Milestone A same-device default matches the existing selection bias. Fix the CUDA/Vulkan device merge loop in `src/device.cpp`, which currently pushes a duplicate entry for a GPU that is both Vulkan- and CUDA-capable.
3. Establish GPU-focused benchmarks for:
   - current `sync_poses_gpu_to_cpu() + Viewer.render()`;
   - current `RenderSystemGroup` direct updates;
   - default current/owning scene plus shared scenes;
   - explicitly selected multiple scenes plus shared scenes;
   - rigid dynamics and articulations;
   - raster and RT shader packs;
   - separate timings for PhysX pose fetch, pose conversion, object/camera transform updates, synchronization, RT acceleration-structure updates, Viewer update, and Vulkan draw;
   - D2H bytes and device-wide synchronization count.
4. Record the verified semantics of Direct GPU force and torque writes (established from `PxDirectGPUAPI`, `PxRigidBody.h`, and GPU testing):
   - writes are replace, not accumulate: `setRigidDynamicData` / `setArticulationData` have SET semantics;
   - PhysX clears applied forces and torques after they act on the next simulation step unless `PxRigidBodyFlag::eRETAIN_ACCELERATIONS` is raised; SAPIEN never sets that flag, so a single `gpu_apply_*` force acts on exactly one step (verified empirically for rigid dynamics and articulation links);
   - sustained interaction therefore requires `viewer.apply_interactions()` to run before every physics step; releasing a pick clears the Viewer scratch so the next composition applies a zero Viewer contribution;
   - `gpu_apply_rigid_dynamic_force` / `gpu_apply_rigid_dynamic_torque` have no `index_buffer` overload, so composition with application-provided wrench buffers must write the full rigid-body force/torque buffers;
   - if `eRETAIN_ACCELERATIONS` support is added later, that mode requires an explicit zero-write on release.
5. Add test instrumentation that fails if the normal GPU Viewer path calls `sync_poses_gpu_to_cpu()`.

### Exit Criteria

- Same-device and cross-device baselines are reproducible.
- Transport selection has all required device information.
- Wrench ownership and clearing rules are documented and tested.
- The scope explicitly excludes UI-framework and OpenGL work.

### Baseline Harness

`manualtest/gpu_viewer.py` reproduces the Stage 0 paths. Run at least the following matrix
on the target hardware:

```bash
# Existing full pose D2H path and existing direct camera path.
python manualtest/gpu_viewer.py --transport cpu-debug --selection default
python manualtest/gpu_viewer.py --transport render-system-group --selection default
python manualtest/gpu_viewer.py --transport render-system-group --selection explicit

# Same-device Viewer paths, scene aggregation, articulation, and RT coverage.
python manualtest/gpu_viewer.py --transport direct --selection explicit
python manualtest/gpu_viewer.py --transport direct --articulations
python manualtest/gpu_viewer.py --transport direct --shader rt

# Existing cross-device CPU-debug baseline before the Stage 3 staged transport.
python manualtest/gpu_viewer.py --transport cpu-debug \
  --render-device pci:0000:00:02.0
```

Replace the PCI alias with the Vulkan rendering device under test. The script reports host-side
fetch, render-update, and draw/capture timings, pose D2H bytes, fetch counts, and full-sync
counts. It also asserts that direct paths leave CPU entity poses stale and do not perform a full
pose sync. Capture the finer GPU stages with Nsight Systems; the implementation emits NVTX
ranges for PhysX copy and conversion, object/camera transforms, CUDA–Vulkan synchronization,
RT render-version updates, Viewer update, and draw.

---

## Stage 1: Separate Vulkan Scene Update From Viewer Drawing

**Priority: Immediate prerequisite** — **Complexity: Medium**

This stage creates a minimal insertion point for GPU pose transport. It must not redesign the existing Viewer windows.

### Tasks

1. Split `RenderWindow` operations into:
   - resolving the base scene selection plus associated shared scenes;
   - updating render-system state;
   - assembling and updating one output render scene from the resolved set, with identity composition and shared content included once;
   - drawing the Vulkan viewport and existing UI.
2. Expose `viewer.update_render()` as the explicit state-submission phase and make `viewer.render()` draw only the most recently submitted state. A partial split already exists: `Viewer.render()` gates `window.update_render()` behind the single-shot `render_updated` flag, so the paused loop already avoids repeated state pushes; build on that gate.
3. Ensure each `RenderSystem` is stepped at most once per submitted display frame.
4. Introduce a small internal pose-transport interface used only by `viewer.update_render()`.
5. Add one internal render-scene resolver shared by the Viewer, scene cameras, and `RenderSystemGroup`. It owns base-scene selection, shared-scene inclusion, stable ordering, and deduplication.
6. Retain `viewer.set_scene(scene)` and `viewer.set_scenes(scenes)` as scene-selection APIs, remove only their offset behavior, and add `camera.set_scenes(scenes)` with the same semantics. A camera with no override uses its owning scene as the base set; scene selection does not otherwise change the existing camera update and `take_picture()` lifecycle.
7. Preserve the current single-scene CPU path. CPU multi-scene rendering adopts the same no-offset base-plus-shared aggregation as GPU multi-scene rendering; do not preserve the old grid-offset path through a compatibility layer.
8. Define transform ownership:
   - every scene in the resolved base-plus-shared set contributes objects to the same output render scene, with no scene-level transform;
   - a render system marked `batched_render_shared` contributes its objects and lights once rather than once per base scene;
   - objects, lights, and cameras not bound to a PhysX GPU pose continue to use the existing RenderSystem update logic;
   - PhysX-GPU-bound dynamic shapes are updated afterward by the selected transport;
   - do not rely on write ordering alone: suppress the per-frame CPU upload of stale `Entity.pose` for GPU-bound shapes (restore a mechanism like the commented-out `disable_auto_upload` binding) instead of letting `RenderSystem::step()` write transforms the transport immediately overwrites;
   - the svulkan2 renderer re-uploads object transforms during draw when auto-upload is enabled (`Renderer::render()` → `uploadGpuResources()` → `uploadObjectTransforms()`), which would overwrite the direct GPU write with CPU matrices; move that upload into `viewer.update_render()` — CPU-owned transforms first, GPU final write after — and disable it during `viewer.render()` (svulkan2 already exposes `setAutoUploadEnabled`), while `viewer.render()` still handles camera, light, and object metadata updates;
   - `viewer.render()` must not perform another transform update.
9. Do not create an autonomous background update thread.
10. Handle topology or scene-selection changes with the semantics the code has today: `RenderSystemGroup.update_render()` throws after any scene add/remove, and any PhysX component add/remove clears the GPU-initialized flag and requires a full `gpu_init()` re-run (including its warmup step). Changing the Viewer/camera base scenes or shared-scene membership invalidates the aggregate render scene and transport indices; rebuild those render mappings explicitly. PhysX GPU state only needs reinitialization when PhysX topology changed.
11. Keep the current Viewer plugin list and UI windows unchanged.
12. Validate transport inputs: `set_cuda_poses` (`setPoseSource`) already checks contiguity and a 4-byte last-dimension stride, but does not check the `float32` dtype, a minimum of 7 channels per row, or `poseIndex` range against the pose buffer length. Add those validations so a wrong buffer fails loudly instead of silently reading garbage.

### Exit Criteria

- Single-scene CPU Viewer behavior has no regression, and CPU/GPU Viewer and camera outputs use the documented base-plus-shared, no-offset semantics.
- Default and explicit multi-scene selection resolve to stable, deduplicated render sets.
- Scene update and Vulkan draw are independently callable.
- A GPU pose transport can become the final writer of dynamic transforms.
- Topology changes cannot silently reuse stale GPU indices.
- No new UI framework, ECS component, or OpenGL dependency is introduced.

---

## Stage 2: Same-Device PhysX GPU Visualization in the Current Vulkan Viewer

**Priority: Highest user-visible milestone** — **Complexity: Large**

The first production result is raster visualization of `PhysxGpuSystem` in the current Viewer on one compatible NVIDIA GPU.

### Configuration

The exact API name can be finalized during implementation, but the behavior should be equivalent to:

```python
viewer.configure_physx_gpu_rendering(
    physx_system,
    transport="auto",
)
```

The Viewer should also auto-detect a shared `PhysxGpuSystem` from its resolved scenes when unambiguous.

Scene selection remains independent of pose transport:

```python
viewer.set_scene(scene)                  # scene + associated shared scenes
viewer.set_scenes([scene0, scene1])      # specified scenes + shared scenes
camera.set_scenes([scene0, scene1])      # same contract for this camera
```

Without a camera override, a camera renders its owning scene plus associated shared scenes.

Supported overrides:

```text
auto
direct
staged
cpu-debug
```

`cpu-debug` means an explicit full CPU pose synchronization. It is not the normal GPU Viewer fallback.

### Stage 2A: Raster

#### Tasks

1. After `gpu_init()`, build internal mappings for:
   - rigid dynamic `gpu_pose_index`;
   - articulation link `gpu_pose_index`;
   - existing sibling `RenderBodyComponent` shapes;
   - shape local pose and scale;
   - originating SAPIEN scene and environment ID for selection and interaction bookkeeping only;
   - destination transform-buffer index in the resolved Viewer or camera output scene;
   - existing mounted cameras that depend on a GPU body pose.
2. Reuse the direct transform-update machinery from `RenderSystemGroup` and bind it to `cuda_rigid_body_data` by calling `set_cuda_poses` explicitly; nothing binds automatically. `cuda_rigid_body_data` is already SAPIEN-layout `[count, 13]` with the pose at row offsets 0-6 (p.xyz, q.wxyz), the PhysX-to-SAPIEN conversion runs inside `gpu_fetch_*`, and the existing kernel already computes the required `body pose × shape local pose`. Do not reuse the current per-system output-scene layout unchanged. For every Viewer or camera output:
   - resolve either its default base scene or its explicit `set_scenes` base set, then append associated `batched_render_shared` systems exactly once;
   - build one aggregate render scene for that resolved set;
   - bind the transport to that exact output render scene and its single transform buffer;
   - compute all destination transform and RT instance indices against that aggregate scene;
   - use identity composition for every source scene and add no scene-offset field or kernel term;
   - retain Viewer `set_scenes` as selection, add camera `set_scenes`, and remove only the old grid/offset composition;
   - cover both raster object transforms and RT instance transforms.
3. Centralize the pose fetches needed for displayed frames:
   - `gpu_fetch_rigid_dynamic_data()`;
   - `gpu_fetch_articulation_link_pose()`.
4. Avoid duplicate fetches when application logic has already fetched the same data in the current frame: track `last_fetched_step = mTotalSteps` per buffer, treat the buffer as fresh only when it equals the current step count, and invalidate it on `gpu_init()`, topology rebuild, and any apply or kinematic operation that changes poses, so a teleport or reset within the same step does not reuse a stale fetch.
5. Implement the explicit phases as:

   ```text
   viewer.update_render():
       existing RenderSystem update for objects not sourced from PhysX GPU poses
       PhysX GPU pose fetches, unless already completed for this simulation frame
       RenderSystemGroup.update_render() as the final update for GPU-bound shapes

   viewer.render():
       current Vulkan Viewer and UI draw only
   ```

6. Make `viewer.update_render()` enqueue work asynchronously where possible and return without a device-wide synchronization. This requires fixing the fetch path: `gpuFetchRigidDynamicData()` / `gpuFetchArticulationLinkPose()` currently call the PhysX Direct GPU API without a `finishEvent`, and PhysX documents a NULL `finishEvent` as "wait for the copy to finish before returning" — today's fetches are host-blocking. Pass a CUDA `finishEvent`, make the SAPIEN stream wait on that event, then run the conversion and transform kernels and signal the CUDA/Vulkan timeline semaphore.
7. Connect to the application CUDA stream and preserve CUDA–Vulkan timeline-semaphore ordering.
8. Do not call `cudaDeviceSynchronize()` or perform pose D2H.
9. Preserve the current Vulkan outputs used by the Viewer, including Segmentation, Position/Depth, color, and shadows.

#### Exit Criteria

- A moving PhysX GPU body is rendered correctly while its CPU Entity pose remains stale.
- The normal Viewer frame does not call `sync_poses_gpu_to_cpu()`.
- `viewer.render()` does not trigger another pose fetch or transform update.
- Current selection IDs, shadows, and mounted cameras continue to work.
- A default Viewer/camera renders its current or owning scene plus shared scenes; an explicit `set_scenes` override renders the specified base scenes plus shared scenes.
- Objects retain their poses, overlapping objects from different scenes remain overlapping, and shared content appears once.
- The current Viewer UI is rendered without changes to its framework or widgets.
- Direct-update CPU and GPU timings are reported separately and remain within the performance gates established from Stage 0 rather than being assumed to be zero.

### Stage 2B: RT

Most of this stage already exists in `RenderSystemGroup`: with an RT shader pack, `update_render()` already writes RT instance transforms (`VkTransformMatrixKHR`) from the same CUDA kernel, bumps the scene render version so the RT renderer refits the TLAS from external transforms, and resets accumulation. Stage 2B is therefore Viewer wiring and verification, not new transport implementation.

#### Tasks

1. Wire the Viewer RT path to the same `RenderSystemGroup` submission used by Stage 2A, so RT instance transforms come from the same CUDA pose source.
2. Verify TLAS update/refit and accumulation reset trigger through the existing render-version path when transforms change.
3. Validate Viewer and camera default-scene selection, explicit multi-scene selection, shared-scene inclusion, mounted cameras, and selection changes at runtime.

#### Exit Criteria

- Raster and RT produce matching object transforms.
- RT updates require no pose D2H.
- No OpenGL interop is introduced.

---

## Stage 3: Cross-Device PhysX GPU Visualization Through Vulkan

**Priority: Second visualization milestone** — **Complexity: Very large**

This stage supports discrete-GPU PhysX computation with rendering on an integrated GPU or another discrete GPU.

### Preconditions for Cross-Device Operation

The engine singletons do not block this configuration: `PhysxSystemGpu` takes an explicit CUDA device, `PhysxEngine` only requires all PhysX scenes to share one CUDA device, and `SapienRenderEngine` only requires one Vulkan render device per process — one PhysX GPU plus one different render GPU is constructible today. What actually blocks the direct path is `RenderSystemGroup.update_render()`, which throws when the pose buffer's CUDA device differs from the renderer's device.

The staged transport therefore does not need a generalized multi-device engine rework. It needs:

- independent, explicit selection of the PhysX CUDA device and the Vulkan render device;
- the staged upload/compute transport of this stage;
- bypassing the `RenderSystemGroup` CUDA direct path entirely on this route.

### Data Path

Pack only the pose channels required for visualization:

```text
[position xyz, quaternion wxyz]
```

Use this pipeline:

```text
PhysX cuda_rigid_body_data
    ↓ CUDA pack/gather kernel on the compute GPU
compact pose buffer [rendered_body_count, 7]
    ↓ asynchronous D2H
reusable double-buffered pinned host memory
    ↓ Vulkan staging-buffer upload
pose buffer on the Vulkan rendering device
    ↓ Vulkan compute shader
Vulkan object-transform buffers
    ↓ current Vulkan Viewer or camera draw
```

The Vulkan compute shader uses the same metadata as the aggregate same-device transform path:

- pose index;
- local pose;
- scale;
- destination transform index in the resolved Viewer or camera output scene.

Source scene and environment IDs remain CPU-side bookkeeping for selection and interaction; they do not select another output buffer or modify the render transform. This transfers one compact pose per rendered body rather than one matrix per shape.

### Stage 3A: Raster

#### Tasks

1. Build a compact list of bodies used by each Viewer's or camera's resolved base-plus-shared scene set.
2. Pack only those poses into a contiguous CUDA buffer.
3. Allocate reusable pinned host slots rather than allocating per frame.
4. Upload completed host slots through Vulkan staging buffers.
5. Add a Vulkan compute pass that writes object-transform buffers. The current buffers cannot be written by compute directly: object-transform buffers are created with only `eUniformBuffer | eTransferDst` usage, and RT instance buffers likewise lack storage usage. Choose explicitly: add `eStorageBuffer` usage to the affected buffers, or have compute write a temporary storage buffer that is copied into the existing buffers with `vkCmdCopyBuffer` (the `eTransferDst` usage already permits this). Either way, add the required compute/transfer → vertex/fragment and acceleration-structure-build barriers.
6. Preserve:
   - default current/owning scene plus shared scenes;
   - explicit Viewer/camera multi-scene selection plus shared scenes;
   - each object's existing pose without renderer-generated offsets;
   - shared-scene geometry and lights included once;
   - bound and unbound objects;
   - shadows and Segmentation output.
7. Use explicit synchronization with a CPU bridge; CUDA external semaphores generally cannot be shared with Intel/AMD Vulkan drivers, so the boundary is:

   ```text
   CUDA event marks a pinned-host slot ready
       → CPU polls/waits on the event
       → Vulkan staging copy is submitted
       → Vulkan timeline/fence gates the draw and the slot reuse
   ```

8. Initially display the latest complete state. If the same-frame host wait on the CUDA event is unacceptable, one-frame-latency pipelining is the required asynchronous design, not an optional later optimization.

#### Exit Criteria

- PhysX can run on a discrete NVIDIA GPU while the current Vulkan Viewer renders on an integrated GPU.
- The Viewer does not call `sync_poses_gpu_to_cpu()` or update all CPU Entities.
- Per-frame pose D2H is approximately `28 × rendered_body_count` bytes, excluding small fixed metadata.
- Direct and staged transports produce matching transforms.
- No OpenGL upload path exists.

### Stage 3B: RT

#### Tasks

1. Feed staged poses into Vulkan RT instance transforms.
2. Update/refit the TLAS on the rendering device.
3. Reset RT accumulation after staged transform changes.
4. Report unsupported renderer/driver combinations explicitly.

Until Stage 3B is complete, cross-device RT must select `cpu-debug` explicitly or report that staged RT is unavailable. It must not silently display stale transforms.

### Exit Criteria

- Cross-device raster and RT agree on object transforms.
- Unsupported combinations fail with actionable diagnostics.
- The implementation remains entirely Vulkan-based.

---

## Stage 4: Adapt Existing Selection, Focus, and Overlay Handling

**Priority: After core GPU visualization** — **Complexity: Medium**

This stage updates the existing `ControlWindow` and related Viewer helpers. It does not add a new selection component or replace the current UI.

### Tasks

1. Keep the current Vulkan Segmentation-based selection path.
2. Continue exposing `viewer.selected_entity` so existing plugins and user code remain unchanged.
3. Build private mappings from existing Entities/components to:
   - `gpu_pose_index`;
   - rigid dynamic `gpu_index`;
   - articulation `gpu_index`;
   - articulation link index.
4. Add an internal Viewer method for retrieving the current selected pose:
   - direct mode: download only the selected 7-float pose when CPU code needs it;
   - staged mode: read it from the current host staging slot;
   - CPU mode: use the existing Entity pose.
5. Update existing focus, selected-frame, joint-axis, coordinate-axis, and bounding-overlay logic to use the current transport pose rather than stale CPU Entity pose.
6. Refresh CPU-visible selected values only when the corresponding existing window is open or a low-frequency UI update is due.
7. Account for the paused loop: plugin `before_render`/`after_render` hooks (focus re-centering, joint-axis, coordinate-axis, and camera-lineset overlays) run on every paused frame, so overlay pose reads must respect the same open-window and low-frequency gating while paused.

### Exit Criteria

- Existing left-click selection works for moving GPU bodies without full CPU pose sync.
- `viewer.selected_entity` and existing plugin notifications still work.
- Focus and overlays follow current GPU state.
- Closing the relevant UI windows eliminates selected-pose D2H.
- No new scene component or UI abstraction is introduced.

---

## Stage 5: Adapt Existing Interaction Handling for PhysX GPU

**Priority: After GPU visualization and selection** — **Complexity: Large**

This stage modifies the behavior behind the current Viewer controls and `TransformWindow`. It does not replace their UI.

### 5.1 Physical Picking and Dragging

Use the existing Vulkan render outputs:

```text
viewport input
  → Segmentation identifies the existing Entity/body/link
  → Position/Depth identifies the world-space hit point
  → current GPU pose defines the local anchor
  → private GPU picking state
```

Note: the current Viewer picking reads only the Segmentation target; reading Position/Depth for the hit point is new work, including verifying the target exists in the active shader pack. Pixel readback can also block on the render, so hit-point reads stay on the click path only and must be benchmarked rather than assumed cheap.

During dragging, transfer only small control data:

- ray origin and direction;
- target position;
- active or released state;
- stiffness and damping settings already owned by the Viewer.

This small command path may pass through CPU when rendering and simulation use different GPUs.

The GPU force model should use a damped point spring:

```text
F = mass_gain × [stiffness × (target - point) - damping × point_velocity]
torque = (point - center_of_mass) × F
```

Include:

- maximum-acceleration clamping;
- effective articulation mass for lightweight links;
- static and kinematic filtering;
- release-time clearing of the Viewer scratch, so the next composition applies a zero Viewer contribution;
- correct originating-scene and environment-ID mapping for physics commands, without any renderer-offset conversion;
- velocity availability: `gpu_fetch_articulation_link_pose()` fetches poses only, so dragging articulation links additionally requires `gpu_fetch_articulation_link_velocity()` for the damping term (rigid dynamics already carry velocity in `cuda_rigid_body_data`).

Input bindings must respect the current Viewer camera controls. Physical drag may use an existing transform interaction, a modifier, or a configurable binding rather than taking over a current camera binding unconditionally.

### 5.2 Existing TransformWindow Gizmo

Keep the current gizmo and widgets. Change only what happens when they edit a transform:

| Existing selected object | GPU-mode handling |
| --- | --- |
| Dynamic rigid body | Physical target by default; explicit teleport when requested |
| Kinematic rigid body | Teleport via a GPU pose write, or disable the edit; no GPU kinematic-target API exists (`PxDirectGPUAPI` has no kinematic write type; `setKinematicTarget` is CPU-only), and a GPU pose write loses kinematic sweep semantics |
| Static body | Keep current valid CPU behavior or report rebuild required |
| Articulation link | Physical target |
| Articulation root | Write the selected GPU root-pose row and apply it |
| Existing IK mode | Keep the current supported path; do not add GPU IK in this migration |

The gizmo callback should queue a private Viewer command. It must not call ineffective dynamic `Entity.set_pose()` in Direct GPU mode.

Teleporting a dynamic body must account for `gpu_apply_rigid_dynamic_data` writing pose and linear/angular velocity together: fetch current data first to preserve velocities, or write explicit zero velocities.

### 5.3 Wrench Composition

Viewer interaction must not overwrite application data in:

- `cuda_rigid_body_force`;
- `cuda_rigid_body_torque`;
- `cuda_articulation_link_force`;
- `cuda_articulation_link_torque`.

Use private interaction scratch buffers and a defined GPU composition step:

```text
application wrench (cuda_*_force/torque, written by the application)
+ Viewer interaction wrench (private scratch)
→ private final buffer
→ internal apply-from-buffer call
```

The composition must not write in place into the exposed `cuda_*_force` / `cuda_*_torque` buffers: they are simultaneously the application's input and the source `gpu_apply_*()` reads, so an in-place `application + viewer` write would re-add the Viewer wrench on every subsequent composition. Preferred design: a private final buffer plus an internal apply-from-buffer API (the current `gpu_apply_*` entry points read only the fixed SAPIEN buffers). A snapshot/compose/apply/restore sequence over the shared buffers is possible but has more complex ordering and is not recommended.

Because `gpu_apply_rigid_dynamic_force` / `gpu_apply_rigid_dynamic_torque` have no `index_buffer` overload, the composition step writes the full rigid-body force/torque buffers, not only the selected body.

Expose one application phase:

```python
viewer.apply_interactions()
```

Apply ownership must be explicit. Recommended contract: while Viewer interaction is active, the application writes the exposed buffers, and `viewer.apply_interactions()` is the sole caller of the force/torque apply operations, immediately before `physx.step()` — the application does not call the corresponding `gpu_apply_*()` itself. A later application apply would overwrite the Viewer wrench, and a Viewer full-buffer apply can re-submit application data not intended for this step, especially for indexed articulation applies. The alternative is application-applies-first with the Viewer overwriting only the picked body/link through the internal indexed apply-from-buffer path; since an internal apply-from-buffer API is being added anyway, an indexed rigid-dynamic variant also avoids full-buffer composition every step.

PhysX clears applied forces after each simulation step (SAPIEN does not set `eRETAIN_ACCELERATIONS`), so the interaction wrench acts on exactly one step: `viewer.apply_interactions()` must run before every physics step, and before each step of a multi-step control frame. On release or selection change, clear the Viewer scratch so the next composition applies a zero Viewer contribution; no interaction force outlives the step it was composed for.

### Exit Criteria

- The existing Viewer UI can physically drag rigid dynamics and articulation links.
- Render transforms are not used as a substitute for physics state.
- Releasing a pick leaves no residual wrench.
- Application wrench values are preserved.
- The current Transform window remains the UI surface.
- No Viewer-only ECS component or replacement UI framework is added.

---

## Stage 6: Make Existing Entity and Articulation Windows GPU-Aware

**Priority: After visualization and interaction are usable** — **Complexity: Medium to large**

This stage adapts only fields already exposed by the current windows. It does not introduce a property registry or generic Inspector system.

### EntityWindow

1. Keep the current Entity and component sections.
2. Read dynamic pose from the Viewer transport when PhysX GPU is active.
3. Keep static render and collision information on the current CPU access path.
4. Route supported GPU pose edits through queued GPU commands rather than `Entity.set_pose()`.
5. Mark fields as read-only when the current PhysX GPU API cannot safely modify them at runtime.
6. Remove the stale `self.viewer.system._internal_scene` reference in `entity_window.py` (dead code; the attribute no longer exists).

### ArticulationWindow

1. Keep the current joint list and controls.
2. Read the selected articulation row from cached CUDA views or a selected-row transfer.
3. Do not poll slow whole-articulation CPU helpers every rendered frame. The current window also rebuilds its entire joint UI, including per-joint callback closures, on every frame; the rework must address the O(dof) rebuild, not only the data reads.
4. Route currently supported controls to the corresponding buffers and apply calls:
   - qpos;
   - drive position target;
   - drive velocity target;
5. Keep drive stiffness, damping, force limit, friction, and drive mode on their existing runtime API where valid; clearly disable or defer fields that require GPU reinitialization.
6. Preserve the current CPU PhysX callbacks.

### TransformWindow

1. Keep the current window, gizmo, ghost preview, and options.
2. Use transport pose for the selected link/body, including the follow path, which currently overwrites the gizmo pose from `selected_entity.pose` on every build.
3. Route final actions through the interaction command queue introduced in Stage 5.
4. Keep the current IK implementation and limitations; GPU IK is out of scope.

### ContactWindow

ContactWindow reads CPU PhysX contact reports, which do not exist in the same form under `PhysxGpuSystem`. In GPU mode, disable it (or the affected sections) with an explicit notice; porting it to GPU contact queries is out of scope for this migration. SettingWindow needs no special handling: it only reads `timestep` and `config` values that remain valid on `PhysxGpuSystem`.

### Exit Criteria

- Existing windows remain visually and structurally familiar.
- Closed windows cause no GPU state readback.
- Open windows read only the selected body or articulation data.
- Supported edits use GPU buffers and safe apply points.
- Unsupported runtime edits are clearly disabled instead of silently modifying only CPU state.
- The CPU-PhysX-only ContactWindow is explicitly disabled or degraded under PhysX GPU instead of showing wrong or empty data.
- No generic property registry is added.

---

## Stage 7: Default Selection, Cleanup, Documentation, and Release

**Priority: Final integration** — **Complexity: Medium**

### Default Behavior

```text
PhysX GPU + same compatible Vulkan device → direct
PhysX GPU + different Vulkan device       → staged
CPU PhysX                                 → existing CPU path

Viewer without set_scenes override        → current scene + shared scenes
Viewer with set_scenes override           → specified scenes + shared scenes
Camera without set_scenes override        → owning scene + shared scenes
Camera with set_scenes override           → specified scenes + shared scenes
```

Every row above uses identity scene composition. Different scenes may render overlapping objects at the same coordinates.

Expose the active transport through existing Viewer diagnostics or an existing settings/status area, for example:

```text
Pose transport: CUDA 0 → Vulkan direct
Pose transport: CUDA 0 → pinned host → Vulkan Intel
Pose transport: CPU Entity → Vulkan
```

Do not add a new window solely for this status.

### Cleanup

1. Remove implicit Viewer calls to `sync_poses_gpu_to_cpu()`.
2. Keep `sync_poses_gpu_to_cpu()` only as an explicit debug/API operation.
3. Retain `Viewer.set_scene` and `Viewer.set_scenes` as scene-selection APIs, but remove the `offsets` argument, implicit grid layout, `scene_offset` bookkeeping, and offset multiplication in helpers such as `get_entity_viewer_pose`. Add and document `camera.set_scenes(scenes)` with the same selection-only semantics. Keep no offset fallback; applications that want visual separation must place their entities explicitly.
4. Remove manual GPU Viewer scripts and code fragments replaced by the production transports.
5. Remove tests that exist only to preserve the superseded full-scene Viewer sync or offset-layout paths.
6. Keep no OpenGL experiment, dormant GL transport, replacement UI framework, property registry, or Viewer-only ECS component.
7. Update:
   - Viewer tutorial;
   - rendering tutorial;
   - PhysX GPU workflow documentation;
   - RL documentation;
   - Python stubs;
   - CHANGELOG.

### Validation Matrix

| Simulation | Vulkan rendering device | Scene configuration | Shader path |
| --- | --- | --- | --- |
| CPU PhysX | Integrated or discrete GPU | Viewer/camera default scene + shared; explicit multi-scene + shared | Raster |
| PhysX GPU | Same NVIDIA GPU | Viewer/camera default scene + shared; explicit multi-scene + shared | Raster/RT |
| PhysX GPU on NVIDIA | Intel/AMD integrated GPU | Default and explicit multi-scene selection | Staged raster |
| PhysX GPU on NVIDIA A | NVIDIA B | Default and explicit multi-scene selection | Staged raster/RT |
| GPU articulations | Same or different rendering GPU | Overlapping environments plus shared scene | Raster |

Also validate:

- pause and single-step;
- current selection and focus behavior;
- physical interaction;
- current Transform window behavior;
- wrench composition;
- topology rebuild;
- environment IDs without render offsets;
- default Viewer scene plus shared scenes;
- explicit Viewer multi-scene selection plus shared scenes;
- default camera owning scene plus shared scenes;
- explicit camera multi-scene selection plus shared scenes;
- multiple environments at identical poses rendering in place;
- shared-scene geometry and lights appearing exactly once;
- mounted cameras;
- Viewer close and reopen;
- automatic staged fallback when CUDA–Vulkan interop is unavailable.

### Exit Criteria

- `auto` selects the correct transport on supported device combinations.
- Same-device GPU Viewer performs no pose D2H.
- Cross-device GPU Viewer transfers compact poses instead of synchronizing all Entities.
- Current single-scene CPU Viewer behavior remains supported, and CPU/GPU Viewer and camera outputs follow the same default-or-explicit base scenes plus shared scenes, no-offset contract.
- Current Viewer UI remains the only UI scheme.
- Vulkan remains the only Viewer renderer.

---

## Milestones

### Milestone A: Same-GPU PhysX GPU Raster Viewer

Complete Stages 0, 1, and 2A.

Result:

- the current Vulkan Viewer displays PhysX GPU state directly on the same GPU;
- current raster outputs, shadows, selection IDs, cameras, and UI continue to work;
- default Viewer/camera rendering uses the current or owning scene plus shared scenes;
- explicit Viewer/camera `set_scenes` rendering uses the specified scenes plus shared scenes;
- every output aggregates its resolved scenes without synthetic offsets, and shared content appears once;
- no full CPU pose synchronization is required.

This is the first release target and has priority over interaction changes, property-window changes, and CPU Viewer enhancements.

### Milestone B: Complete Same-GPU Vulkan Viewer

Complete Stage 2B.

Result:

- raster and RT consume the same PhysX GPU pose source;
- RT transform and accumulation behavior are correct.

### Milestone C: Cross-Device PhysX GPU Viewer

Complete Stage 3A, followed by Stage 3B.

Result:

- PhysX can run on a discrete GPU while the current Vulkan Viewer renders on an integrated or different GPU;
- transport uses pinned-host staging and Vulkan upload/compute;
- no OpenGL bridge is introduced.

Risk: the staged transport must bypass the `RenderSystemGroup` CUDA direct path and bridge CUDA-to-Vulkan synchronization through the CPU (see the Stage 3 preconditions); device selection itself is already independent.

### Milestone D: Existing Viewer Interaction on GPU State

Complete Stages 4 and 5.

Result:

- the current selection, focus, and Transform-window interactions operate on PhysX GPU state without full CPU synchronization.

### Milestone E: Existing Property Windows on GPU State

Complete Stages 6 and 7.

Result:

- current Entity, Articulation, and Transform windows use GPU-aware reads and safe commands where supported;
- no new UI framework or component architecture is required;
- the new transport becomes the default GPU Viewer path.

## Initial Implementation Order

The implementation order must remain GPU-visualization-first:

1. Same-device Vulkan raster visualization.
2. Same-device Vulkan RT visualization.
3. Cross-device Vulkan raster visualization.
4. Cross-device Vulkan RT visualization.
5. Selection, focus, and overlays on GPU state.
6. Physical interaction and current gizmo handling.
7. GPU-aware data access in current property windows.
8. CPU regression validation and final cleanup.

This ordering keeps the migration focused on data transport, synchronization, and interaction handling while preserving the existing Vulkan renderer and Viewer UI.
