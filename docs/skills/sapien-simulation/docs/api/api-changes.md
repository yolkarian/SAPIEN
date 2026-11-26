# API Changes By Release

Migration-sensitive additions, removals, and behavior changes only, newest first. Signatures and parameter meanings are discoverable at runtime with `help(...)` or `__doc__`; this file records what you would otherwise call and find missing or use with the wrong lifecycle assumptions.

## Unreleased

Half-resolution GTAO in `default` and `vr_default` rasterization packs:

- **Behavior change:** `default` and `vr_default` replace inline SSAO with a half-resolution `ao` pass (`vulkan_shader/common/gtao.glsl`) plus a 3×3 depth-aware bilateral filter/upsample in `deferred.frag`. Controlled by the existing `ambientOcclusionStrength` (default `0.65`, `0` disables) and `ambientOcclusionRadius` (default `0.3` m) parameters.
- `camera.get_picture_names()` on `default` / `vr_default` now includes `AmbientOcclusion` (`float32`, half-resolution shape `(ceil(H/2), ceil(W/2))`, raw GTAO visibility in `[0, 1]`).

Rasterization shadow upgrades and scene-owned shared shadow maps:

- **New light softness / radius properties (default `0.0`, real-time):**
  - `RenderDirectionalLightComponent.softness` (`get_softness`/`set_softness`, optional `softness=` on `Scene.add_directional_light`): angular softness for raster PCSS soft shadows and RT directional shadow jitter.
  - `RenderPointLightComponent.radius` and `RenderSpotLightComponent.radius` (`get_radius`/`set_radius`, optional `radius=` on `Scene.add_point_light`/`Scene.add_spot_light`): emitter radius in metres for raster PCSS soft shadows and RT shadow jitter.
- **Behavior change:** directional raster shadows use 3 concentric cascades (`0.15x`, `0.40x`, `1.0x` of `shadow_half_size`), point shadows use 16-tap cube PCF with slope bias instead of 1-tap un-biased depth comparison, and all raster shadow maps are owned by `Scene` and shared across cameras viewing the same scene (rendering shadows once per scene update instead of once per camera).
- **Breaking (custom raster packs with shadows):** `samplerDirectionalLightDepths` is now `sampler2DArray` (3 cascade layers per directional light) and `ShadowBuffer.directionalLightBuffers` holds `3 * N` `LightBuffer` entries (matching `pointLightBuffers` `6 * N`). `SpotLight` in `common/light_data.glsl` uses its last field as `float radius`.

`RenderBodyComponent.cast_shadow` and full post-processing passes:

- `RenderBodyComponent.cast_shadow` (default `True`): controls shadow casting in rasterization, `rt`, `shadow_catcher` and `rt_batch` (via TLAS instance mask bit `0x02`). Setting it to `False` keeps the body visible to camera rays while skipping shadow rays and shadow maps.
- `toneMapper` now accepts `0..5` across built-in packs: `0` gamma 2.2, `1` sRGB, `2` ACES (default), `3` Khronos PBR Neutral, `4` AgX, `5` Filmic (Hable).
- Built-in `vulkan_shader/post/` adds `bloom`, `color_grade`, `lens` and `fxaa`. The `rt` pack runs `["bloom", "color_grade", "tonemap", "lens"]` (identity defaults; all parameters settable via `RenderConfig` and `camera.set_property` without resetting accumulation) and declares presets `"preview"` (`spp=4, maxDepth=4`) and `"production"` (`spp=64, maxDepth=12, clampIndirect=10.0`).

The `rt` path tracer outputs first-batch AOVs:

- New `rt` pictures from `camera.get_picture(name)`:
  - `Depth` (`float32`, shape `(H, W)`): linear depth in metres along the camera view axis, `0` on sky (matches `rt_batch`).
  - `MotionVector` (`float32`, shape `(H, W, 2)`): screen-space optical flow `(dx, dy)` in pixels (`curr - prev`, `+x` right, `+y` down), covering moving objects, moving cameras and rotating sky; `0` on the initial frame and after an idle `update_render()`.
  - `Emission`, `DiffuseDirect`, `SpecularDirect`, `DiffuseIndirect`, `SpecularIndirect` (`float32`, shape `(H, W, 4)`): accumulated over frames; their RGB sum equals `HdrColor` (before denoising) everywhere in the image.
- Custom RT packs may declare `rg32f`/`rg16f`/`rgba16f` `out<Name>` storage images and bind `PrevTlasInstances` in the scene set (`rt` uses binding 20).

Materials follow the first batch of OpenPBR Surface 1.1:

- **Breaking:** `RenderMaterial(specular=...)` and `.specular` are gone. Use `specular_weight`,
  which scales the normal-incidence reflectance of `ior`; `specular=s` is `specular_weight=2*s`
  at the new default `ior=1.5` (was 1.45). `transmission_roughness` is removed.
- New `RenderMaterial` members: `base_weight`, `specular_weight`, `specular_color`,
  `specular_roughness_anisotropy`, `transmission_color`, `transmission_depth` (scene units),
  `coat_weight`, `coat_color`, `coat_roughness`, `coat_ior`, `fuzz_weight`, `fuzz_color`,
  `fuzz_roughness`, `thin_walled`, `occlusion_texture`. `metallic`, `roughness`, `ior`,
  `transmission`, `base_color` and `emission` keep their meaning.
- **Behavior change (`rt`, `shadow_catcher`):** base-colour alpha is coverage, so a glass made
  with `base_color` alpha now looks cut-out; use `transmission=1.0` (and
  `transmission_color`). Transmission is tinted by `transmission_color`, not the base colour.
  Coat, fuzz, anisotropy, specular colour, thin walls and `transmission_depth` absorption are
  shaded by the path tracers and by `rt_batch`'s `pbr` tier (direct light). The raster packs
  shade coat, fuzz and specular colour over their base BRDF (anisotropy averaged into the
  roughness, no transmission) and the occlusion texture; materials without coat or fuzz render
  as before. Deferred raster packs have a new `uvec4` picture `Material` (packed layers).
- Custom packs: `Material` (`common/material.glsl`) is 512 bytes; `fresnel`, `textureMask`
  and `emission` became `specularWeight` (with `ior`), slot tests (`material.xTexture >= 0`)
  and `emissionColor`/`emissionStrength`. `common/bsdf.glsl` is replaced by
  `common/openpbr.glsl`.

Rasterization lights have no fixed limit:

- **Behavior change:** the built-in raster packs draw every light of the scene instead of at
  most 3 directional, 10 point and 10 spot lights (`minimal`: 10 point). The bound is the
  device's uniform-buffer range (on NVIDIA 2047 point or directional, 371 spot and textured
  lights); beyond it the camera raises `RuntimeError`. Shadow casters and textured lights
  remain bounded by the pack's shadow maps. In `minimal`, textured lights now sample their
  shadow map.
- **Breaking (custom raster packs):** lights are the uniform buffers `PointLights`,
  `DirectionalLights` and `SpotLights` of the scene set (records of
  `common/light_data.glsl`, formerly `common/rt_light.glsl`), sized by the specialization
  constants `NUM_POINT_LIGHTS`, `NUM_DIRECTIONAL_LIGHTS`, `NUM_SPOT_LIGHTS`; `SceneBuffer` keeps
  only `ambientLight`; textured lights follow the spot lights. Copy `default/scene_set.glsl`
  and use `common/raster_lighting.glsl` (`sapienAddDirectLighting`).

Ray tracing reads the rasterizers' material records:

- **Breaking (custom ray-tracing packs):** `rt`, `shadow_catcher` and `rt_batch` read one
  storage buffer `Materials { Material m[]; } materials` (records of
  `common/material.glsl`; `rt` binding 2, `rt_batch` binding 5). `TextureIndices` is gone:
  read `materials.m[i].baseColorTexture` (and `roughnessTexture`, `normalTexture`,
  `metallicTexture`, `emissionTexture`, `transmissionTexture`, -1 for none).
- **Behavior change:** editing a material after the first `rt` frame restarts accumulation
  (old samples were kept); an emission edit relights the scene through light sampling; a
  texture new to the scene's materials rebuilds its ray-tracing tables (it was ignored).
- `rt_batch` groups apply material edits before each frame. A texture the group did not
  have at `gpu_init()` raises `RuntimeError` from `take_picture()`; assign textures before
  `gpu_init()`.

Shader packs post-process through `pack.json`:

- **Breaking:** `"post"` in `pack.json` lists compute passes (`<pack>/post/<name>.comp`, else
  the built-in `vulkan_shader/post/<name>.comp`) run in order. `rt` and `shadow_catcher`
  declare `"post": ["tonemap"]`; `exposure` and `toneMapper` are parameters of that pass and
  are set as before (`cfg["exposure"]`, `camera.set_property`).
- Ray tracing: the first pass reads `HdrColor`, the last writes `Color`; `postprocessing.comp`
  is no longer read and a pack without `"post"` has no `Color` picture. Custom packs: add
  `"post": ["tonemap"]` and remove `exposure`/`toneMapper` from the push constants.
- Rasterization packs may opt in: the chain reads the `Lighting` target and writes
  `PostColor`, sampled by a later pass as `samplerPostColor`. The built-in rasterization
  packs do not use it (it costs throughput with many small cameras) and tone-map in
  `composite0.frag` as before.
- `"post"` raises `RuntimeError` in an `rt_batch` pack and with MSAA. Pass contract and an
  example: `docs/source/tutorial/rendering/customize_shaders.md` ("Post-processing passes").

The `rt` path tracer samples its lights:

- **Behavior change:** the environment map and emissive meshes are light-sampled and combined
  with BSDF sampling by multiple importance sampling; environment radiance is no longer
  clamped to 100. HDR maps and emissive meshes light the scene at full strength, and images
  converge faster and look different. Emissive meshes are sampled at their current pose;
  meshes with an emission texture are only found by BSDF rays.
- **Breaking (default):** `russianRoulette` defaults to `true` in `rt` and `shadow_catcher`.
  Set `config["russianRoulette"] = False` for the previous termination.
- New `rt` uniform `clampIndirect` (float, default `0` = off): clamps the brightest channel of
  every contribution after the first bounce (firefly control).
- New `rt` variant `lightSampling` (`"mis"` default, `"nee"`, `"bsdf"`) for debugging; set it
  on the `RenderConfig` before creating the camera (`camera.set_property` raises for it). The
  three converge to the same image.
- Shadow rays pass transparent and transmissive occluders with the light they let through
  (base-colour alpha, object `transparency`, `transmission` tinted by the base colour)
  instead of being blocked by them.
- Building needs the Zig module `src/sapien_renderer/alias_table.zig` (linked into the
  renderer like `pack_manifest.zig`; wheels and the SDK need no Zig at run time).

Shader packs declare their parameters in `pack.json`:

- **Breaking:** every shader pack directory, built-in or custom, must contain a
  `pack.json` next to `gbuffer.vert` / `camera.rgen` / `batch.comp`. It declares the
  parameters a `RenderConfig` may assign (`kind` `"uniform"` or `"variant"`, `type`,
  `default`, optional `minimum`/`maximum`, `values` for enums, `define` for variants,
  `description`) and optional `presets`. A missing or malformed `pack.json` raises
  `RuntimeError` naming the directory.
- **Breaking:** `RenderConfig.defines` is gone. Preprocessor definitions now come
  from variant parameters of `pack.json`, so `cfg.defines = {"SHADOWS": ""}` becomes
  `cfg["shadows"] = True` and `{"SHADING_PBR": ""}` becomes `cfg["shading"] = "pbr"`
  (or `RenderConfig("rt_batch", preset="pbr")`). `cfg` values are now
  `bool | int | float | str`.
- `cfg[name] = value` is validated immediately against the manifest; `cfg[name]`
  returns the effective value (assignment, else the default); `name in cfg` is true
  only for explicit assignments; `del cfg[name]` reverts to the default. New
  read-only `cfg.parameters` (file order) and `cfg.presets`.
- Defaults moved into `pack.json`: the hard-coded `rt` defaults (spp 32, maxDepth 8,
  aperture 0, focusPlane 1, russianRoulette off, exposure 1, toneMapper 2) and the
  batched renderer's `exposure` are now manifest defaults. A pack that declares
  different defaults renders them.
- `camera.set_property` accepts uniform parameters only and also accepts `bool`; a
  variant parameter raises (set it on the `RenderConfig` before creating the camera).
  `camera.get_property` returns the runtime override or the config's effective value.
- `sapien.render.clear_cache(shaders=True)` also drops the cached parsed manifests.
- `RenderWindow.set_render_config` carries over only explicit assignments that the new
  pack declares with a compatible type; the Viewer render panel is built from
  `cfg.parameters`, so its widgets follow the pack (`spp`, `maxDepth`, `aperture`,
  `focusPlane`, `exposure`, `toneMapper`, `ambientOcclusion*`).

Rasterization shader packs read one scene-wide descriptor set:

- **Breaking:** a custom rasterization pack includes `common/scene.glsl` once per pass
  (`#define SET_NUM n` first). Each draw binds it at its own instance and material: shaders
  read the drawn object's `instance` (segmentation, `transformIndex`, transparency, ...) and
  `material` (uniform buffers), and `transforms[]`, `materials[]`, the bindless `textures[]`
  for any entry; shadow passes read `transforms[instance.transformIndex]` too. The
  `object_set.glsl` / `material_set.glsl` sets are removed; packs written against them fail to
  load until ported. `Material` is 256 B; a material texture index of `-1` means no texture.
- **Breaking:** `max_num_textures` (`set_global_config`, default 512) is the number of
  `textures[]` slots; slot 0 is the white default texture and a scene needing more distinct
  textures raises `RuntimeError`. `max_num_materials` is only the initial size of
  `materials[]`, which grows.
- `render_body.set_property` writes pack-declared `CustomObjectData` fields by name;
  `set_texture` stores a `textures[]` slot in an `int` field and `set_texture_array` in an
  `int` array field. These textures must be 2D (1D/3D raise); an unset texture field is 0,
  the default texture; a texture assigned after the first frame is drawn.
- Rasterization requires `runtimeDescriptorArray` and non-uniform descriptor indexing;
  unsupported devices raise a clear error.
- Material edits reach `RenderCameraGroup` cameras after `gpu_init()`, as before.

Batched ray tracing:

- `RenderConfig("rt_batch")` (or any directory holding `batch.comp`) is a **batched**
  ray-tracing pack. `RenderConfig.kind` reports `"batch_ray_tracing"`; `is_ray_tracing` stays
  true and `is_batch_ray_tracing` is new. Such a pack has no per-camera renderer, so
  `camera.take_picture()`, `camera.get_picture()` / `get_picture_cuda()`, `RenderWindow`,
  `Viewer` and VR raise `RuntimeError` telling you to use a `RenderCameraGroup`.
- One `RenderCameraGroup` builds one group renderer for its batch cameras: per frame one
  command buffer refits every environment's acceleration structure and dispatches over all
  cameras, submitted once, and the pictures are written straight into the CUDA-exported
  `[camera][height][width][channel]` buffers. `take_picture()` therefore copies nothing.
- Every camera of one camera group must copy an **equal** `RenderConfig` and have equal width
  and height (one group per resolution); mixing batch and non-batch cameras raises.
- Pictures of the built-in pack: `Color` (8-bit `uint8` by default, `float32` with
  `picture_formats={"Color": "r32g32b32a32Sfloat"}`), `Albedo`, `Depth` (linear depth in metres,
  positive, `0` on a miss), `Normal` (camera space), `Position` (camera-space xyz + normalized depth),
  `Segmentation` (`uvec4`, same semantics as rasterization).
- Shading tier is the `shading` variant parameter: `"visibility"`, `"lambert"` (default) or
  `"pbr"`; the `shadows` bool variant adds hard shadow rays and the `spp` uniform adds
  deterministic supersampling. There is no `SPP` preprocessor definition.
- Uniform parameters of a batch pack come only from the `RenderConfig` and apply to the whole
  group: `exposure` (default 1.0), `toneMapper` (default 0) and `spp` (default 1, 1..32). Variants
  (`shading`, `shadows`, `maxShadowLights`) select shader source and are fixed when the group
  renderer is built. A uniform the shader does not declare raises naming the parameter, its
  declared type and the pack (for a batch pack at `gpu_init()`); `camera.set_property` raises
  for a batch camera.

Render configuration (breaking):

- **Breaking:** the process-global render setters are gone. Replace them with
  `sapien.render.RenderConfig` and the per-target defaults:

  | Removed | Use |
  |---|---|
  | `set_camera_shader_dir` / `get_camera_shader_dir` | `RenderConfig(pack)`; `set_default_render_config("camera", cfg)` / `get_default_render_config("camera")` |
  | `set_viewer_shader_dir` / `get_viewer_shader_dir` | same with `"viewer"` |
  | `RenderCameraComponent(w, h, shader_dir)`, `RenderWindow(w, h, shader_dir)`, `Viewer(shader_dir=...)` | same constructors with `render_config=cfg` |
  | `RenderWindow.set_shader_dir` | `RenderWindow.set_render_config(cfg)` |
  | `set_ray_tracing_samples_per_pixel` / `..._path_depth` / `..._denoiser` / `..._dof_aperture` / `..._dof_plane` and getters | `cfg["spp"]`, `cfg["maxDepth"]`, `cfg.denoiser`, `cfg["aperture"]`, `cfg["focusPlane"]` |
  | `set_picture_format` | `cfg.picture_formats = {name: format}` |
  | `set_msaa` / `get_msaa` | `cfg.msaa` |

  Consumers copy the config they are given, so one process can render rasterization and ray
  tracing side by side. `camera.set_property` stays as the runtime per-camera setter.
- **Behavior change:** an unknown property name or a wrong value type raises `RuntimeError`
  naming the key and the shader pack, both from a config when its renderer is created and from
  `camera.set_property` once the camera has a renderer. Unknown names used to be silently
  dropped.
- New C++ `sapien::sapien_renderer::RenderConfig` with
  `createRenderer(context, config)` / `buildRendererConfig(config)`;
  `SapienRendererDefault::setDefaultRenderConfig(target, config)` /
  `getDefaultRenderConfig(target)` replace `setViewerShaderDirectory`,
  `setCameraShaderDirectory` and the ray-tracing/MSAA setters.
- New `svulkan2::renderer::RendererBase::getCustomPropertyType(name)` reports the declared type
  of a shader-pack property (`CustomPropertyType`), which is how SAPIEN validates configs.

Rendering:

- **Behavior change:** ray-traced cameras (`"rt"` pack) use a new random sequence on every render.
  Before, each accumulation restart replayed one sequence, so a `RenderSystemGroup` (which restarts
  accumulation on every `update_render()`) showed the same noise pattern in every frame. Do not
  expect two captures of the same state to be bit-identical in `"rt"`; the n-th render of a camera
  is still reproducible across runs. Custom RT packs can read the counter as push constant
  `int frameSeed`.
- **Behavior change:** `camera.set_property(name, value)` on a ray-traced camera restarts
  accumulation when the value changes (except `exposure` and `toneMapper`).
- **Behavior change:** `sapien.render.clear_cache(shaders=True)` now really reloads edited shader
  files and includes, for raster and ray-tracing packs; shader compile errors raise `RuntimeError`
  with `file:line` of each error.
- **Behavior change:** more lights than the raster pack stored no longer add garbage lighting;
  see "Rasterization lights have no fixed limit" above for the current behavior.
- **Behavior change:** HDR environment maps keep radiance above 1 in rough reflections (raster
  image-based lighting); scenes lit by bright HDR maps render brighter than before.
- **Behavior change:** glTF `KHR_materials_specular` now sets `RenderMaterial.specular` to
  `0.5 * specularFactor * luminance(specularColor)`.


C++ programs (native trainers such as PufferLib 5.0, no Python):

- **New C++ SDK:** use `sapien-sdk-<version>-linux_x86_64.tar.gz` from the release, not the
  wheel. Unpacked anywhere it holds `include/` (SAPIEN, PhysX, Eigen, glm, Vulkan headers),
  `lib/` (`libsapien.so`, the libraries it loads, `libPhysXGpu_64.so`, the CMake package in
  `lib/cmake/sapien`) and `share/sapien/vulkan_shader`. Compile with
  `-std=c++20 -I<sdk>/include -I<sdk>/include/physx/include` and no `-D` flags, link
  `-L<sdk>/lib -lsapien -Wl,-rpath,<sdk>/lib`; or `find_package(sapien)` with
  `CMAKE_PREFIX_PATH=<sdk>` and link `sapien::sapien`. The libraries load each other through
  `$ORIGIN` RUNPATHs, so no `LD_LIBRARY_PATH` is needed. Rendering programs first make the calls
  the Python package makes on import: `SapienRendererDefault::internalSetShaderSearchPath(
  "<sdk>/share/sapien/vulkan_shader")`, then
  `SapienRendererDefault::setDefaultRenderConfig("camera", RenderConfig("default"))` and the
  same for `"viewer"`.
- New C++ `PhysxDefault::EnableGPU(physxGpuLibrary = "")` selects the PhysX GPU runtime through
  PhysX's own load hook; empty searches `PhysxDefault::GetPhysxGpuLibraryCandidates()`: on Linux
  `libPhysXGpu_64.so` next to `libsapien.so` (the SDK ships it), then the cache
  `sapien.physx.enable_gpu()` downloads into (`GetPhysxGpuCacheDirectory()`). C++ programs no longer
  have to `dlopen` libcuda and the runtime themselves. A process uses one runtime file; concurrent
  calls are serialized, but enabling must still precede every other use of PhysX.
- **Behavior change:** `sapien.physx.enable_gpu()` no longer preloads `libcuda.so` / `cuda.dll` and
  the runtime with `ctypes`; it passes the runtime's path to C++, and PhysX loads the CUDA driver
  (`libcuda.so.1` on Linux) itself when the first GPU system is created. Machines that have only
  the versioned `libcuda.so.1` (no `libcuda.so` development link) now work.
- The installed headers (the SDK's `include/`, also `sapien/include` in the wheel) need no `-D`
  flags: generated `sapien/config.h` and `svulkan2/build_options.h` carry `SAPIEN_CUDA`,
  `SVULKAN2_CUDA_INTEROP`, `VK_NO_PROTOTYPES` and `VULKAN_HPP_STORAGE_SHARED`, and the Eigen 3.4
  headers ship alongside. Before, a program built without `-DSAPIEN_CUDA` silently saw other class
  layouts than `libsapien.so`.

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
- **Breaking:** `sapien.render.SapienRenderer()` takes no `device` argument. The render engine
  is process-wide and never selected a device for later systems in multi-GPU builds, so a
  passed device was silently ignored by `RenderSystem()`; old calls now raise `TypeError`. Pass
  the device to `RenderSystem(device)` instead.
- Renderer CUDA interop restores the caller's current CUDA context; `torch.cuda.current_device()`
  is unchanged by render calls.
- Fixed: with `SAPIEN_DISABLE_RAY_TRACING=1`, render devices had no CUDA id, so
  `RenderSystem("cuda:N")` failed to find its device.
- `RenderSystemGroup.update_render()`, `gpu_init()`, `close()` and `RenderCameraGroup.take_picture()`,
  `set_cuda_pose()`, `close()` release the GIL, so groups render concurrently from one thread
  each: groups on different devices, and groups on one device that share no render system.
  Calls into one group and its camera groups from several threads are serialized by a per-group
  lock. Build and `gpu_init()` everything on one thread.
- **Behavior change:** `RenderCameraGroup.take_picture()`'s copy into the picture buffers waits
  for all work already submitted to the group's CUDA stream; it used to overwrite pictures that
  a consumer on that stream had not read yet. New `RenderCameraGroup.gpu_wait_for(array)` orders
  that stream after a consumer on another stream (JAX).
- `RenderSystemGroup.set_cuda_stream()` raises `RuntimeError` for a stream of another device.

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
- **Behavior change:** non-indexed `gpu_apply_*` and the Viewer wrench/teleport helpers no
  longer block the host until PhysX has copied the data; they return once enqueued, and the
  system's stream is ordered after PhysX's read. Indexed overloads still synchronize the
  system's stream once for their index check while `gpu_index_validation` is on. A buffer
  written from another stream after an apply must be ordered after the system's stream
  (Torch's default stream shares the legacy default stream with an unconfigured system and
  needs nothing).
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

## 3.0.0+fork.16

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
