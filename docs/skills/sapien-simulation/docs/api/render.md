# Render API (`sapien.render`)

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.render`
- Source files: `python/py_package/pysapien/render.pyi`
- Notes: Only instantiate RenderSystem for viewer/sensors/offscreen rendering. For GPU PhysX dynamic bodies prefer RenderSystemGroup + CUDA poses.

## Functions

| API | Signature | Use | Notes |
|---|---|---|---|
| `sapien.render.clear_cache` | `clear_cache(models: bool=True, images: bool=True, shaders: bool=False) -> None` | Call clear cache. |  |
| `sapien.render.enable_vr` | `enable_vr() -> None` | Enable VR via Steam. Must be called before creating RenderSystem or sapien Scene. |  |
| `sapien.render.get_camera_shader_dir` | `get_camera_shader_dir() -> str` | Get camera shader dir. |  |
| `sapien.render.get_device_summary` | `get_device_summary() -> str` | Get device summary. |  |
| `sapien.render.get_imgui_ini_filename` | `get_imgui_ini_filename() -> str` | Get imgui ini filename. |  |
| `sapien.render.get_msaa` | `get_msaa() -> int` | Get msaa. |  |
| `sapien.render.get_ray_tracing_denoiser` | `get_ray_tracing_denoiser() -> Literal['none', 'oidn', 'optix']` | Get ray tracing denoiser. |  |
| `sapien.render.get_ray_tracing_dof_aperture` | `get_ray_tracing_dof_aperture() -> float` | Get ray tracing dof aperture. |  |
| `sapien.render.get_ray_tracing_dof_plane` | `get_ray_tracing_dof_plane() -> float` | Get ray tracing dof plane. |  |
| `sapien.render.get_ray_tracing_path_depth` | `get_ray_tracing_path_depth() -> int` | Get ray tracing path depth. |  |
| `sapien.render.get_ray_tracing_samples_per_pixel` | `get_ray_tracing_samples_per_pixel() -> int` | Get ray tracing samples per pixel. |  |
| `sapien.render.get_viewer_shader_dir` | `get_viewer_shader_dir() -> str` | Get viewer shader dir. |  |
| `sapien.render.get_vr_action_manifest_filename` | `get_vr_action_manifest_filename() -> str` | Get vr action manifest filename. |  |
| `sapien.render.get_vr_enabled` | `get_vr_enabled() -> bool` | Get vr enabled. |  |
| `sapien.render.load_scene` | `load_scene(filename: str, apply_scale: bool=True) -> RenderSceneLoaderNode` | Load a scene. |  |
| `sapien.render.set_camera_shader_dir` | `set_camera_shader_dir(dir: str) -> None` | Set camera shader dir. |  |
| `sapien.render.set_global_config` | `set_global_config(max_num_materials: int=128, max_num_textures: int=512, default_mipmap_levels: int=1, do_not_load_texture: bool=False) -> None` | Sets global properties for SAPIEN renderers. This function should only be called before creating any renderer-related objects. |  |
| `sapien.render.set_imgui_ini_filename` | `set_imgui_ini_filename(filename: str) -> None` | Set imgui ini filename. |  |
| `sapien.render.set_log_level` | `set_log_level(level: str) -> None` | Set log level. |  |
| `sapien.render.set_msaa` | `set_msaa(msaa: int) -> None` | Set msaa. |  |
| `sapien.render.set_picture_format` | `set_picture_format(name: str, format: str) -> None` | Set picture format. |  |
| `sapien.render.set_ray_tracing_denoiser` | `set_ray_tracing_denoiser(name: Literal['none', 'oidn', 'optix']) -> None` | Set ray tracing denoiser. |  |
| `sapien.render.set_ray_tracing_dof_aperture` | `set_ray_tracing_dof_aperture(radius: float) -> None` | Set ray tracing dof aperture. |  |
| `sapien.render.set_ray_tracing_dof_plane` | `set_ray_tracing_dof_plane(depth: float) -> None` | Set ray tracing dof plane. |  |
| `sapien.render.set_ray_tracing_path_depth` | `set_ray_tracing_path_depth(depth: int) -> None` | Set ray tracing path depth. |  |
| `sapien.render.set_ray_tracing_samples_per_pixel` | `set_ray_tracing_samples_per_pixel(spp: int) -> None` | Set ray tracing samples per pixel. |  |
| `sapien.render.set_viewer_shader_dir` | `set_viewer_shader_dir(dir: str) -> None` | Set viewer shader dir. |  |
| `sapien.render.set_vr_action_manifest_filename` | `set_vr_action_manifest_filename(filename: str) -> None` | Set vr action manifest filename. |  |

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.render.RenderBodyComponent` | `sapien.Component` | Render body; holds RenderShape. |  |
| `sapien.render.RenderCameraComponent` | `sapien.Component` | Camera component; read image/CUDA image after take_picture. |  |
| `sapien.render.RenderCameraGroup` |  | Batched camera group; take/read CUDA for multiple cameras at once. |  |
| `sapien.render.RenderCubemap` |  | Environment cubemap. |  |
| `sapien.render.RenderCudaMeshComponent` | `sapien.Component` | SAPIEN render API object. |  |
| `sapien.render.RenderDirectionalLightComponent` | `RenderLightComponent` | Concrete render light component. |  |
| `sapien.render.RenderLightComponent` | `sapien.Component` | Light component base class. |  |
| `sapien.render.RenderMaterial` |  | PBR render material. |  |
| `sapien.render.RenderParallelogramLightComponent` | `RenderLightComponent` | Concrete render light component. |  |
| `sapien.render.RenderPointCloudComponent` | `sapien.Component` | SAPIEN render API object. |  |
| `sapien.render.RenderPointLightComponent` | `RenderLightComponent` | Concrete render light component. |  |
| `sapien.render.RenderSceneLoaderNode` |  | SAPIEN render API object. |  |
| `sapien.render.RenderShape` |  | Render shape base class; supports GPU pose batch index. |  |
| `sapien.render.RenderShapeBox` | `RenderShapePrimitive` | Concrete render shape. |  |
| `sapien.render.RenderShapeCapsule` | `RenderShapePrimitive` | Concrete render shape. |  |
| `sapien.render.RenderShapeCylinder` | `RenderShapePrimitive` | Concrete render shape. |  |
| `sapien.render.RenderShapePlane` | `RenderShapePrimitive` | Concrete render shape. |  |
| `sapien.render.RenderShapePrimitive` | `RenderShape` | Concrete render shape. |  |
| `sapien.render.RenderShapeSphere` | `RenderShapePrimitive` | Concrete render shape. |  |
| `sapien.render.RenderShapeTriangleMesh` | `RenderShape` | Concrete render shape. |  |
| `sapien.render.RenderShapeTriangleMeshPart` |  | Concrete render shape. |  |
| `sapien.render.RenderSpotLightComponent` | `RenderLightComponent` | Concrete render light component. |  |
| `sapien.render.RenderSystem` | `sapien.System` | Render system; only needed for viewer/sensor/offscreen. | create only when rendering/viewer/sensors needed |
| `sapien.render.RenderSystemGroup` |  | Batched render group over multiple RenderSystems; can bind CUDA poses. |  |
| `sapien.render.RenderTexture` |  | SAPIEN render API object. |  |
| `sapien.render.RenderTexture2D` |  | 2D render texture. |  |
| `sapien.render.RenderTexturedLightComponent` | `RenderSpotLightComponent` | Concrete render light component. |  |
| `sapien.render.RenderVRDisplay` |  | SAPIEN render API object. |  |
| `sapien.render.RenderWindow` |  | SAPIEN render API object. |  |
| `sapien.render.SapienRenderer` |  | SAPIEN API object. |  |

## `sapien.render.RenderBodyComponent`

- Use: Render body; holds RenderShape.
- Bases: `sapien.Component`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `shading_mode` | `int` |  |  |
| `visibility` | `float` |  |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `attach` | method | `attach(self, shape: RenderShape) -> RenderBodyComponent` |  |  |
| `clone` | method | `clone(self) -> RenderBodyComponent` |  |  |
| `compute_global_aabb_tight` | method | `compute_global_aabb_tight(self) -> np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.float32]]` | Compute a tight global AABB. |  |
| `disable_render_id` | method | `disable_render_id(self) -> None` |  |  |
| `enable_render_id` | method | `enable_render_id(self) -> None` |  |  |
| `get_global_aabb_fast` | method | `get_global_aabb_fast(self) -> np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.float32]]` |  |  |
| `is_render_id_disabled` | property | `is_render_id_disabled(self) -> bool` |  |  |
| `render_shapes` | property | `render_shapes(self) -> list[RenderShape]` |  |  |
| `set_property` | method | `set_property(self, name: str, value: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None<br>set_property(self, name: str, value: float) -> None<br>set_property(self, name: str, value: int) -> None` |  |  |
| `set_texture` | method | `set_texture(self, name: str, texture: RenderTexture) -> None` |  |  |
| `set_texture_array` | method | `set_texture_array(self, name: str, textures: list[RenderTexture]) -> None` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |
| `_internal_node` | property | `_internal_node(self) -> sapien.internal_renderer.Node` |  |  |

## `sapien.render.RenderCameraComponent`

- Use: Camera component; read image/CUDA image after take_picture.
- Bases: `sapien.Component`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `far` | `float` |  |  |
| `local_pose` | `sapien.Pose` |  |  |
| `near` | `float` |  |  |
| `skew` | `float` |  |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `cx` | property | `cx(self) -> float` |  |  |
| `cy` | property | `cy(self) -> float` |  |  |
| `fovx` | property | `fovx(self) -> float` |  |  |
| `fovy` | property | `fovy(self) -> float` |  |  |
| `fx` | property | `fx(self) -> float` |  |  |
| `fy` | property | `fy(self) -> float` |  |  |
| `get_extrinsic_matrix` | method | `get_extrinsic_matrix(self) -> np.ndarray[tuple[Literal[3], Literal[4]], np.dtype[np.float32]]` | Get 3x4 extrinsic camera matrix in OpenCV format. |  |
| `get_far` | method | `get_far(self) -> float` |  |  |
| `get_global_pose` | method | `get_global_pose(self) -> sapien.Pose` |  |  |
| `get_height` | method | `get_height(self) -> int` |  |  |
| `get_intrinsic_matrix` | method | `get_intrinsic_matrix(self) -> np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float32]]` | Get 3x3 intrinsic camera matrix in OpenCV format. |  |
| `get_local_pose` | method | `get_local_pose(self) -> sapien.Pose` |  |  |
| `get_mode` | method | `get_mode(self) -> Literal['perspective', 'orthographic']` |  |  |
| `get_model_matrix` | method | `get_model_matrix(self) -> np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float32]]` | Get model matrix (inverse of extrinsic matrix) used in rendering (Y up, Z back) |  |
| `get_near` | method | `get_near(self) -> float` |  |  |
| `get_picture` | method | `get_picture(self, name: str) -> np.ndarray[Any, np.dtype[Any]]` |  | GPU PhysX dynamic bodies read CPU poses; prefer the CUDA pose path for offscreen. |
| `get_picture_cuda` | method | `get_picture_cuda(self, name: str) -> sapien.CudaArray` | This function transfers the rendered image into a CUDA buffer. Usage: # use torch backend sapien.set_cuda_tensor_backend("torch") # called once per process image: torc... | Direct GPU render path; no sync_poses_gpu_to_cpu needed. |
| `get_picture_names` | method | `get_picture_names(self) -> list[str]` |  |  |
| `get_projection_matrix` | method | `get_projection_matrix(self) -> np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float32]]` | Get projection matrix in used in rendering (right-handed NDC with [-1,1] XY and [0,1] Z) |  |
| `get_skew` | method | `get_skew(self) -> float` |  |  |
| `get_width` | method | `get_width(self) -> int` |  |  |
| `global_pose` | property | `global_pose(self) -> sapien.Pose` |  |  |
| `height` | property | `height(self) -> int` |  |  |
| `mode` | property | `mode(self) -> Literal['perspective', 'orthographic']` |  |  |
| `ortho_bottom` | property | `ortho_bottom(self) -> float` |  |  |
| `ortho_left` | property | `ortho_left(self) -> float` |  |  |
| `ortho_right` | property | `ortho_right(self) -> float` |  |  |
| `ortho_top` | property | `ortho_top(self) -> float` |  |  |
| `set_far` | method | `set_far(self, far: float) -> None` |  |  |
| `set_focal_lengths` | method | `set_focal_lengths(self, fx: float, fy: float) -> None` |  |  |
| `set_fovx` | method | `set_fovx(self, fov: float, compute_y: bool=True) -> None` |  |  |
| `set_fovy` | method | `set_fovy(self, fov: float, compute_x: bool=True) -> None` |  |  |
| `set_gpu_pose_batch_index` | method | `set_gpu_pose_batch_index(self, index: int) -> None` | Bind GPU pose index for batched/direct GPU rendering. | Direct GPU render path; no sync_poses_gpu_to_cpu needed. |
| `set_local_pose` | method | `set_local_pose(self, pose: sapien.Pose) -> None` |  |  |
| `set_scenes` | method | `set_scenes(self, scenes: list[sapien.Scene]) -> None` | Select base scenes; associated shared scenes are included once. | No render offsets are applied. |
| `set_near` | method | `set_near(self, near: float) -> None` |  |  |
| `set_orthographic_parameters` | method | `set_orthographic_parameters(self, near: float, far: float, top: float) -> None<br>set_orthographic_parameters(self, near: float, far: float, left: float, right: float, bottom: float, top: float) -> None` |  |  |
| `set_perspective_parameters` | method | `set_perspective_parameters(self, near: float, far: float, fx: float, fy: float, cx: float, cy: float, skew: float) -> None` |  |  |
| `set_principal_point` | method | `set_principal_point(self, cx: float, cy: float) -> None` |  |  |
| `set_property` | method | `set_property(self, name: str, value: float) -> None<br>set_property(self, name: str, value: int) -> None` |  |  |
| `set_skew` | method | `set_skew(self, skew: float) -> None` |  |  |
| `set_texture` | method | `set_texture(self, name: str, texture: RenderTexture) -> None` |  |  |
| `set_texture_array` | method | `set_texture_array(self, name: str, textures: list[RenderTexture]) -> None` |  |  |
| `take_picture` | method | `take_picture(self) -> None` | Trigger camera/camera-group rendering. |  |
| `width` | property | `width(self) -> int` |  |  |
| `__init__` | method | `__init__(self, width: int, height: int, shader_dir: str='') -> None` |  |  |
| `_cuda_buffer` | property | `_cuda_buffer(self) -> sapien.CudaArray` | Debug only. Get the CUDA buffer containing GPU data for this camera, including transformaion matrices, sizes, and user-defined shader fields. |  |
| `_internal_renderer` | property | `_internal_renderer(self) -> sapien.internal_renderer.Renderer` |  |  |

## `sapien.render.RenderCameraGroup`

- Use: Batched camera group; take/read CUDA for multiple cameras at once.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_picture_cuda` | method | `get_picture_cuda(self, name: str) -> sapien.CudaArray` | Read CUDA image buffer; avoids CPU copy. | Direct GPU render path; no sync_poses_gpu_to_cpu needed. |
| `take_picture` | method | `take_picture(self) -> None` | Trigger camera/camera-group rendering. |  |

## `sapien.render.RenderCubemap`

- Use: Environment cubemap.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `export` | method | `export(self, filename: str) -> None` |  |  |
| `__init__` | method | `__init__(self, filename: str) -> None<br>__init__(self, px: str, nx: str, py: str, ny: str, pz: str, nz: str) -> None` |  |  |
| `_internal_cubemap` | property | `_internal_cubemap(self) -> sapien.internal_renderer.Cubemap` |  |  |

## `sapien.render.RenderCudaMeshComponent`

- Use: SAPIEN render API object.
- Bases: `sapien.Component`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `material` | `RenderMaterial` |  |  |
| `triangle_count` | `int` |  |  |
| `vertex_count` | `int` |  |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `cuda_triangles` | property | `cuda_triangles(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cuda_vertices` | property | `cuda_vertices(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `get_cuda_triangles` | method | `get_cuda_triangles(self) -> sapien.CudaArray` |  |  |
| `get_cuda_vertices` | method | `get_cuda_vertices(self) -> sapien.CudaArray` |  |  |
| `get_material` | method | `get_material(self) -> RenderMaterial` |  |  |
| `get_triangle_count` | method | `get_triangle_count(self) -> int` |  |  |
| `get_vertex_count` | method | `get_vertex_count(self) -> int` |  |  |
| `notify_vertex_updated` | method | `notify_vertex_updated(self, cuda_stream: int=0) -> None` |  |  |
| `set_material` | method | `set_material(self, material: RenderMaterial) -> None` |  |  |
| `set_triangle_count` | method | `set_triangle_count(self, count: int) -> None` |  |  |
| `set_triangles` | method | `set_triangles(self, triangles: np.ndarray[np.uint32[M, 3]]) -> None` |  |  |
| `set_vertex_count` | method | `set_vertex_count(self, count: int) -> None` |  |  |
| `__init__` | method | `__init__(self, max_vertex_count: int, max_triangle_count: int) -> None` |  |  |

## `sapien.render.RenderDirectionalLightComponent`

- Use: Concrete render light component.
- Bases: `RenderLightComponent`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `shadow_half_size` | `float` |  |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_shadow_half_size` | method | `get_shadow_half_size(self) -> float` |  |  |
| `set_shadow_half_size` | method | `set_shadow_half_size(self, size: float) -> None` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.render.RenderLightComponent`

- Use: Light component base class.
- Bases: `sapien.Component`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `color` | `np.ndarray[Literal[3], np.dtype[np.float32]]` |  |  |
| `local_pose` | `sapien.Pose` |  |  |
| `shadow` | `bool` |  |  |
| `shadow_far` | `float` |  |  |
| `shadow_map_size` | `int` |  |  |
| `shadow_near` | `float` |  |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `disable_shadow` | method | `disable_shadow(self) -> None` |  |  |
| `enable_shadow` | method | `enable_shadow(self) -> None` |  |  |
| `get_color` | method | `get_color(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` |  |  |
| `get_global_pose` | method | `get_global_pose(self) -> sapien.Pose` |  |  |
| `get_local_pose` | method | `get_local_pose(self) -> sapien.Pose` |  |  |
| `get_shadow_far` | method | `get_shadow_far(self) -> float` |  |  |
| `get_shadow_map_size` | method | `get_shadow_map_size(self) -> int` |  |  |
| `get_shadow_near` | method | `get_shadow_near(self) -> float` |  |  |
| `global_pose` | property | `global_pose(self) -> sapien.Pose` |  |  |
| `set_color` | method | `set_color(self, color: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` |  |  |
| `set_local_pose` | method | `set_local_pose(self, pose: sapien.Pose) -> None` |  |  |
| `set_shadow_far` | method | `set_shadow_far(self, far: float) -> None` |  |  |
| `set_shadow_map_size` | method | `set_shadow_map_size(self, size: int) -> None` |  |  |
| `set_shadow_near` | method | `set_shadow_near(self, near: float) -> None` |  |  |

## `sapien.render.RenderMaterial`

- Use: PBR render material.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `base_color` | `Annotated[list[float], FixedSize(4)]` |  |  |
| `base_color_texture` | `RenderTexture2D` |  |  |
| `diffuse_texture` | `` |  |  |
| `emission` | `Annotated[list[float], FixedSize(4)]` |  |  |
| `emission_texture` | `RenderTexture2D` |  |  |
| `ior` | `float` |  |  |
| `metallic` | `float` |  |  |
| `metallic_texture` | `RenderTexture2D` |  |  |
| `normal_texture` | `RenderTexture2D` |  |  |
| `roughness` | `float` |  |  |
| `roughness_texture` | `RenderTexture2D` |  |  |
| `specular` | `float` |  |  |
| `transmission` | `float` |  |  |
| `transmission_roughness` | `float` |  |  |
| `transmission_texture` | `RenderTexture2D` |  |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_base_color` | method | `get_base_color(self) -> Annotated[list[float], FixedSize(4)]` |  |  |
| `get_base_color_texture` | method | `get_base_color_texture(self) -> RenderTexture2D` |  |  |
| `get_diffuse_texture` | method | `get_diffuse_texture(self)` |  |  |
| `get_emission` | method | `get_emission(self) -> Annotated[list[float], FixedSize(4)]` |  |  |
| `get_emission_texture` | method | `get_emission_texture(self) -> RenderTexture2D` |  |  |
| `get_ior` | method | `get_ior(self) -> float` |  |  |
| `get_metallic` | method | `get_metallic(self) -> float` |  |  |
| `get_metallic_texture` | method | `get_metallic_texture(self) -> RenderTexture2D` |  |  |
| `get_normal_texture` | method | `get_normal_texture(self) -> RenderTexture2D` |  |  |
| `get_roughness` | method | `get_roughness(self) -> float` |  |  |
| `get_roughness_texture` | method | `get_roughness_texture(self) -> RenderTexture2D` |  |  |
| `get_specular` | method | `get_specular(self) -> float` |  |  |
| `get_transmission` | method | `get_transmission(self) -> float` |  |  |
| `get_transmission_roughness` | method | `get_transmission_roughness(self) -> float` |  |  |
| `get_transmission_texture` | method | `get_transmission_texture(self) -> RenderTexture2D` |  |  |
| `set_base_color` | method | `set_base_color(self, color: Annotated[list[float], FixedSize(4)]) -> None` |  |  |
| `set_base_color_texture` | method | `set_base_color_texture(self, texture: RenderTexture2D) -> None` |  |  |
| `set_diffuse_texture` | method | `set_diffuse_texture(self, texture)` |  |  |
| `set_emission` | method | `set_emission(self, emission: Annotated[list[float], FixedSize(4)]) -> None` |  |  |
| `set_emission_texture` | method | `set_emission_texture(self, texture: RenderTexture2D) -> None` |  |  |
| `set_ior` | method | `set_ior(self, ior: float) -> None` |  |  |
| `set_metallic` | method | `set_metallic(self, metallic: float) -> None` |  |  |
| `set_metallic_texture` | method | `set_metallic_texture(self, texture: RenderTexture2D) -> None` |  |  |
| `set_normal_texture` | method | `set_normal_texture(self, texture: RenderTexture2D) -> None` |  |  |
| `set_roughness` | method | `set_roughness(self, roughness: float) -> None` |  |  |
| `set_roughness_texture` | method | `set_roughness_texture(self, texture: RenderTexture2D) -> None` |  |  |
| `set_specular` | method | `set_specular(self, specular: float) -> None` |  |  |
| `set_transmission` | method | `set_transmission(self, transmission: float) -> None` |  |  |
| `set_transmission_roughness` | method | `set_transmission_roughness(self, roughness: float) -> None` |  |  |
| `set_transmission_texture` | method | `set_transmission_texture(self, texture: RenderTexture2D) -> None` |  |  |
| `__eq__` | method | `__eq__(self, arg0: RenderMaterial) -> bool` |  |  |
| `__init__` | method | `__init__(self, emission: Annotated[list[float], FixedSize(4)]=[0.0, 0.0, 0.0, 0.0], base_color: Annotated[list[float], FixedSize(4)]=[1.0, 1.0, 1.0, 1.0], specular: float=0.0, roughness: float=1.0, metallic: float=0.0, transmission: float=0.0, ior: float=1.4500000476837158, transmission_roughness: float=0.0) -> None` |  |  |

## `sapien.render.RenderParallelogramLightComponent`

- Use: Concrete render light component.
- Bases: `RenderLightComponent`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `angle` | property | `angle(self) -> float` |  |  |
| `get_angle` | method | `get_angle(self) -> float` |  |  |
| `get_half_height` | method | `get_half_height(self) -> float` |  |  |
| `get_half_width` | method | `get_half_width(self) -> float` |  |  |
| `half_height` | property | `half_height(self) -> float` |  |  |
| `half_width` | property | `half_width(self) -> float` |  |  |
| `set_shape` | method | `set_shape(self, half_width: float, half_height: float, angle: float=1.5707963705062866) -> None` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.render.RenderPointCloudComponent`

- Use: SAPIEN render API object.
- Bases: `sapien.Component`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_cuda_aabb` | method | `get_cuda_aabb(self) -> sapien.CudaArray \| None` | this function is a temporary hack to help update the AABBs used for ray tracing BLAS. returns None if ray tracing has not been initialized |  |
| `get_cuda_vertices` | method | `get_cuda_vertices(self) -> sapien.CudaArray` |  |  |
| `get_vertices` | method | `get_vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | Get previously set vertices. This function does not reflect any changes directly made to the GPU. |  |
| `set_attribute` | method | `set_attribute(self, name: str, attribute: np.ndarray[tuple[M, N], np.dtype[np.float32]] \| list \| tuple) -> RenderPointCloudComponent` |  |  |
| `set_vertices` | method | `set_vertices(self, vertices: np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]] \| list \| tuple) -> RenderPointCloudComponent` |  |  |
| `__init__` | method | `__init__(self, capacity: int=0) -> None` |  |  |

## `sapien.render.RenderPointLightComponent`

- Use: Concrete render light component.
- Bases: `RenderLightComponent`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.render.RenderSceneLoaderNode`

- Use: SAPIEN render API object.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `children` | property | `children(self) -> list[RenderSceneLoaderNode]` |  |  |
| `flatten` | method | `flatten(self) -> tuple[list[RenderShapeTriangleMesh], list[RenderLightComponent]]` |  |  |
| `light` | property | `light(self) -> RenderLightComponent` |  |  |
| `mesh` | property | `mesh(self) -> RenderShapeTriangleMesh` |  |  |
| `name` | property | `name(self) -> str` |  |  |
| `pose` | property | `pose(self) -> sapien.Pose` |  |  |
| `scale` | property | `scale(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` |  |  |

## `sapien.render.RenderShape`

- Use: Render shape base class; supports GPU pose batch index.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `front_face` | `Literal['counterclockwise', 'clockwise']` |  |  |
| `local_pose` | `sapien.Pose` |  |  |
| `name` | `str` |  |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `clone` | method | `clone(self) -> RenderShape` |  |  |
| `get_front_face` | method | `get_front_face(self) -> Literal['counterclockwise', 'clockwise']` |  |  |
| `get_local_pose` | method | `get_local_pose(self) -> sapien.Pose` |  |  |
| `get_material` | method | `get_material(self) -> RenderMaterial` |  |  |
| `get_name` | method | `get_name(self) -> str` |  |  |
| `get_parts` | method | `get_parts(self) -> list[RenderShapeTriangleMeshPart]` |  |  |
| `get_per_scene_id` | method | `get_per_scene_id(self) -> int` |  |  |
| `material` | property | `material(self) -> RenderMaterial` |  |  |
| `parts` | property | `parts(self) -> list[RenderShapeTriangleMeshPart]` |  |  |
| `per_scene_id` | property | `per_scene_id(self) -> int` |  |  |
| `set_front_face` | method | `set_front_face(self, front_face: Literal['counterclockwise', 'clockwise']) -> None` |  |  |
| `set_gpu_pose_batch_index` | method | `set_gpu_pose_batch_index(self, index: int) -> None` | Bind GPU pose index for batched/direct GPU rendering. | Direct GPU render path; no sync_poses_gpu_to_cpu needed. |
| `set_local_pose` | method | `set_local_pose(self, pose: sapien.Pose) -> None` |  |  |
| `set_name` | method | `set_name(self, name: str) -> None` |  |  |

## `sapien.render.RenderShapeBox`

- Use: Concrete render shape.
- Bases: `RenderShapePrimitive`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_half_size` | method | `get_half_size(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` |  |  |
| `half_size` | property | `half_size(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` |  |  |
| `__init__` | method | `__init__(self, half_size: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, material: RenderMaterial) -> None` |  |  |

## `sapien.render.RenderShapeCapsule`

- Use: Concrete render shape.
- Bases: `RenderShapePrimitive`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_half_length` | method | `get_half_length(self) -> float` |  |  |
| `get_radius` | method | `get_radius(self) -> float` |  |  |
| `half_length` | property | `half_length(self) -> float` |  |  |
| `radius` | property | `radius(self) -> float` |  |  |
| `__init__` | method | `__init__(self, radius: float, half_length: float, material: RenderMaterial) -> None` |  |  |

## `sapien.render.RenderShapeCylinder`

- Use: Concrete render shape.
- Bases: `RenderShapePrimitive`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_half_length` | method | `get_half_length(self) -> float` |  |  |
| `get_radius` | method | `get_radius(self) -> float` |  |  |
| `half_length` | property | `half_length(self) -> float` |  |  |
| `radius` | property | `radius(self) -> float` |  |  |
| `__init__` | method | `__init__(self, radius: float, half_length: float, material: RenderMaterial) -> None` |  |  |

## `sapien.render.RenderShapePlane`

- Use: Concrete render shape.
- Bases: `RenderShapePrimitive`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_scale` | method | `get_scale(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` |  |  |
| `scale` | property | `scale(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` |  |  |
| `__init__` | method | `__init__(self, scale: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, material: RenderMaterial) -> None` |  |  |

## `sapien.render.RenderShapePrimitive`

- Use: Concrete render shape.
- Bases: `RenderShape`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_triangles` | method | `get_triangles(self) -> np.ndarray[np.uint32[M, 3]]` |  |  |
| `get_vertex_normal` | method | `get_vertex_normal(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` |  |  |
| `get_vertex_uv` | method | `get_vertex_uv(self) -> np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` |  |  |
| `get_vertices` | method | `get_vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` |  |  |
| `triangles` | property | `triangles(self) -> np.ndarray[np.uint32[M, 3]]` |  |  |
| `vertex_normal` | property | `vertex_normal(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` |  |  |
| `vertex_uv` | property | `vertex_uv(self) -> np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` |  |  |
| `vertices` | property | `vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` |  |  |

## `sapien.render.RenderShapeSphere`

- Use: Concrete render shape.
- Bases: `RenderShapePrimitive`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_radius` | method | `get_radius(self) -> float` |  |  |
| `radius` | property | `radius(self) -> float` |  |  |
| `__init__` | method | `__init__(self, radius: float, material: RenderMaterial) -> None` |  |  |

## `sapien.render.RenderShapeTriangleMesh`

- Use: Concrete render shape.
- Bases: `RenderShape`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `scale` | `np.ndarray[Literal[3], np.dtype[np.float32]]` |  |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `filename` | property | `filename(self) -> str` |  |  |
| `get_filename` | method | `get_filename(self) -> str` |  |  |
| `get_scale` | method | `get_scale(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` |  |  |
| `set_scale` | method | `set_scale(self, scale: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | Note: this function only works when the shape is not added to scene |  |
| `__init__` | method | `__init__(self, vertices: np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]] \| list \| tuple, triangles: np.ndarray[np.uint32[M, 3]], normals: np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]] \| list \| tuple, uvs: np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]] \| list \| tuple, material: RenderMaterial) -> None<br>__init__(self, filename: str, scale: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple=..., material: RenderMaterial \| None=None) -> None` |  |  |

## `sapien.render.RenderShapeTriangleMeshPart`

- Use: Concrete render shape.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `cuda_triangles` | property | `cuda_triangles(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cuda_vertices` | property | `cuda_vertices(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `get_cuda_triangles` | method | `get_cuda_triangles(self) -> sapien.CudaArray` |  |  |
| `get_cuda_vertices` | method | `get_cuda_vertices(self) -> sapien.CudaArray` |  |  |
| `get_material` | method | `get_material(self) -> RenderMaterial` |  |  |
| `get_triangles` | method | `get_triangles(self) -> np.ndarray[np.uint32[M, 3]]` |  |  |
| `get_vertex_normal` | method | `get_vertex_normal(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` |  |  |
| `get_vertex_uv` | method | `get_vertex_uv(self) -> np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` |  |  |
| `get_vertices` | method | `get_vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` |  |  |
| `material` | property | `material(self) -> RenderMaterial` |  |  |
| `set_vertex_normal` | method | `set_vertex_normal(self, normal: np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]] \| list \| tuple) -> None` |  |  |
| `set_vertex_uv` | method | `set_vertex_uv(self, uv: np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]] \| list \| tuple) -> None` |  |  |
| `triangles` | property | `triangles(self) -> np.ndarray[np.uint32[M, 3]]` |  |  |
| `vertices` | property | `vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` |  |  |
| `__eq__` | method | `__eq__(self, arg0: RenderShapeTriangleMeshPart) -> bool` |  |  |

## `sapien.render.RenderSpotLightComponent`

- Use: Concrete render light component.
- Bases: `RenderLightComponent`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `inner_fov` | `float` |  |  |
| `outer_fov` | `float` |  |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_inner_fov` | method | `get_inner_fov(self) -> float` |  |  |
| `get_outer_fov` | method | `get_outer_fov(self) -> float` |  |  |
| `set_inner_fov` | method | `set_inner_fov(self, fov: float) -> None` |  |  |
| `set_outer_fov` | method | `set_outer_fov(self, fov: float) -> None` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.render.RenderSystem`

- Use: Render system; only needed for viewer/sensor/offscreen.
- Bases: `sapien.System`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `ambient_light` | `np.ndarray[Literal[3], np.dtype[np.float32]]` |  |  |
| `cubemap` | `RenderCubemap` |  |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `cameras` | property | `cameras(self) -> list[RenderCameraComponent]` |  |  |
| `cuda_object_transforms` | property | `cuda_object_transforms(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `device` | property | `device(self) -> sapien.Device` |  |  |
| `get_ambient_light` | method | `get_ambient_light(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` |  |  |
| `get_cameras` | method | `get_cameras(self) -> list[RenderCameraComponent]` |  |  |
| `get_cubemap` | method | `get_cubemap(self) -> RenderCubemap` |  |  |
| `get_lights` | method | `get_lights(self) -> list[RenderLightComponent]` |  |  |
| `get_point_clouds` | method | `get_point_clouds(self) -> list[RenderPointCloudComponent]` |  |  |
| `get_render_bodies` | method | `get_render_bodies(self) -> list[RenderBodyComponent]` |  |  |
| `lights` | property | `lights(self) -> list[RenderLightComponent]` |  |  |
| `point_clouds` | property | `point_clouds(self) -> list[RenderPointCloudComponent]` |  |  |
| `render_bodies` | property | `render_bodies(self) -> list[RenderBodyComponent]` |  |  |
| `set_ambient_light` | method | `set_ambient_light(self, color: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` |  |  |
| `set_cubemap` | method | `set_cubemap(self, cubemap: RenderCubemap) -> None` |  |  |
| `__init__` | method | `__init__(self, device: sapien.Device \| None=None) -> None<br>__init__(self, device: str) -> None` |  |  |
| `_internal_scene` | property | `_internal_scene(self) -> sapien.internal_renderer.Scene` |  |  |

## `sapien.render.RenderSystemGroup`

- Use: Batched render group over multiple RenderSystems; can bind CUDA poses.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `create_camera_group` | method | `create_camera_group(self, cameras: list[RenderCameraComponent], picture_names: list[str]) -> RenderCameraGroup` | Create a batched camera group. |  |
| `set_cuda_poses` | method | `set_cuda_poses(self, pose_buffer: sapien.CudaArray) -> None` | Bind a RenderSystemGroup to a PhysX CUDA pose buffer. | Direct GPU render path; no sync_poses_gpu_to_cpu needed. |
| `set_cuda_stream` | method | `set_cuda_stream(self, stream: int) -> None` |  |  |
| `update_render` | method | `update_render(self) -> None` | This function performs CUDA operations to transfer poses from the CUDA buffer provided by :func:`set_cuda_poses` into render systems. It updates the transformation mat... | GPU PhysX dynamic bodies read CPU poses; prefer the CUDA pose path for offscreen. |
| `__init__` | method | `__init__(self, systems: list[RenderSystem]) -> None` |  |  |

## `sapien.render.RenderTexture`

- Use: SAPIEN render API object.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `address_mode` | property | `address_mode(self) -> Literal['repeat', 'border', 'edge', 'mirror']` |  |  |
| `channels` | property | `channels(self) -> int` |  |  |
| `depth` | property | `depth(self) -> int` |  |  |
| `download` | method | `download(self) -> np.ndarray[Any, np.dtype[Any]]` |  |  |
| `filter_mode` | property | `filter_mode(self) -> Literal['nearest', 'linear']` |  |  |
| `format` | property | `format(self) -> str` |  |  |
| `get_address_mode` | method | `get_address_mode(self) -> Literal['repeat', 'border', 'edge', 'mirror']` |  |  |
| `get_channels` | method | `get_channels(self) -> int` |  |  |
| `get_depth` | method | `get_depth(self) -> int` |  |  |
| `get_filter_mode` | method | `get_filter_mode(self) -> Literal['nearest', 'linear']` |  |  |
| `get_format` | method | `get_format(self) -> str` |  |  |
| `get_height` | method | `get_height(self) -> int` |  |  |
| `get_mipmap_levels` | method | `get_mipmap_levels(self) -> int` |  |  |
| `get_width` | method | `get_width(self) -> int` |  |  |
| `height` | property | `height(self) -> int` |  |  |
| `is_srgb` | property | `is_srgb(self) -> bool` |  |  |
| `mipmap_levels` | property | `mipmap_levels(self) -> int` |  |  |
| `upload` | method | `upload(self, data: np.ndarray[Any, np.dtype[Any]] \| list \| tuple) -> None` |  |  |
| `width` | property | `width(self) -> int` |  |  |
| `__eq__` | method | `__eq__(self, arg0: RenderTexture) -> bool` |  |  |
| `__init__` | method | `__init__(self, array: np.ndarray[Any, np.dtype[Any]] \| list \| tuple, dim: int, format: str, mipmap_levels: int=1, filter_mode: Literal['nearest', 'linear']='linear', address_mode: Literal['repeat', 'border', 'edge', 'mirror']='repeat', srgb: bool=False) -> None` |  |  |

## `sapien.render.RenderTexture2D`

- Use: 2D render texture.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `address_mode` | property | `address_mode(self) -> Literal['repeat', 'border', 'edge', 'mirror']` |  |  |
| `channels` | property | `channels(self) -> int` |  |  |
| `download` | method | `download(self) -> np.ndarray[Any, np.dtype[Any]]` |  |  |
| `filename` | property | `filename(self) -> str` |  |  |
| `filter_mode` | property | `filter_mode(self) -> Literal['nearest', 'linear']` |  |  |
| `format` | property | `format(self) -> str` |  |  |
| `get_address_mode` | method | `get_address_mode(self) -> Literal['repeat', 'border', 'edge', 'mirror']` |  |  |
| `get_channels` | method | `get_channels(self) -> int` |  |  |
| `get_filename` | method | `get_filename(self) -> str` |  |  |
| `get_filter_mode` | method | `get_filter_mode(self) -> Literal['nearest', 'linear']` |  |  |
| `get_format` | method | `get_format(self) -> str` |  |  |
| `get_height` | method | `get_height(self) -> int` |  |  |
| `get_mipmap_levels` | method | `get_mipmap_levels(self) -> int` |  |  |
| `get_width` | method | `get_width(self) -> int` |  |  |
| `height` | property | `height(self) -> int` |  |  |
| `is_srgb` | property | `is_srgb(self) -> bool` |  |  |
| `mipmap_levels` | property | `mipmap_levels(self) -> int` |  |  |
| `upload` | method | `upload(self, data: np.ndarray[Any, np.dtype[Any]] \| list \| tuple) -> None` |  |  |
| `width` | property | `width(self) -> int` |  |  |
| `__eq__` | method | `__eq__(self, arg0: RenderTexture2D) -> bool` |  |  |
| `__init__` | method | `__init__(self, array: np.ndarray[Any, np.dtype[Any]] \| list \| tuple, format: str, mipmap_levels: int=1, filter_mode: Literal['nearest', 'linear']='linear', address_mode: Literal['repeat', 'border', 'edge', 'mirror']='repeat', srgb: bool=False) -> None<br>__init__(self, filename: str, mipmap_levels: int=1, filter_mode: Literal['nearest', 'linear']='linear', address_mode: Literal['repeat', 'border', 'edge', 'mirror']='repeat', srgb: bool=True) -> None` | Create texture from file. The srgb parameter only affects files in uint8 format; it should be true for color textures (diffuse, emission) and false for others (normal,... |  |

## `sapien.render.RenderTexturedLightComponent`

- Use: Concrete render light component.
- Bases: `RenderSpotLightComponent`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `texture` | `RenderTexture2D` |  |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_texture` | method | `get_texture(self) -> RenderTexture2D` |  |  |
| `set_texture` | method | `set_texture(self, texture: RenderTexture2D) -> None` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.render.RenderVRDisplay`

- Use: SAPIEN render API object.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `root_pose` | `sapien.Pose` |  |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `fetch_poses` | method | `fetch_poses(self) -> None` | fetches poses of HMD and controllers |  |
| `get_controller_axis_state` | method | `get_controller_axis_state(self, id: int, axis: int) -> Annotated[list[float], FixedSize(2)]` |  |  |
| `get_controller_button_pressed` | method | `get_controller_button_pressed(self, id: int) -> int` |  |  |
| `get_controller_button_touched` | method | `get_controller_button_touched(self, id: int) -> int` |  |  |
| `get_controller_ids` | method | `get_controller_ids(self) -> list[int]` |  |  |
| `get_controller_pose` | method | `get_controller_pose(self, id: int) -> sapien.Pose` | Gets the local pose of a controller. It should be called immediately after fetch_poses |  |
| `get_hmd_pose` | method | `get_hmd_pose(self) -> sapien.Pose` | Gets the local pose of the head set. It should be called immediately after fetch_poses |  |
| `get_left_hand_root_pose` | method | `get_left_hand_root_pose(self) -> sapien.Pose` |  |  |
| `get_left_hand_skeletal_poses` | method | `get_left_hand_skeletal_poses(self) -> list[sapien.Pose]` |  |  |
| `get_right_hand_root_pose` | method | `get_right_hand_root_pose(self) -> sapien.Pose` |  |  |
| `get_right_hand_skeletal_poses` | method | `get_right_hand_skeletal_poses(self) -> list[sapien.Pose]` |  |  |
| `get_root_pose` | method | `get_root_pose(self) -> sapien.Pose` |  |  |
| `render` | method | `render(self) -> None` |  |  |
| `set_camera_parameters` | method | `set_camera_parameters(self, near: float, far: float) -> None` |  |  |
| `set_root_pose` | method | `set_root_pose(self, pose: sapien.Pose) -> None` |  |  |
| `set_scene` | method | `set_scene(self, scene: sapien.Scene) -> None` |  |  |
| `update_render` | method | `update_render(self) -> None` | update_render implicitly calls fetch_poses to make sure the HMD pose is up-to-date | GPU PhysX dynamic bodies read CPU poses; prefer the CUDA pose path for offscreen. |
| `__init__` | method | `__init__(self) -> None` |  |  |
| `_internal_scene` | property | `_internal_scene(self) -> sapien.internal_renderer.Scene` |  |  |

## `sapien.render.RenderWindow`

- Use: SAPIEN render API object.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `cursor` | `bool` |  |  |
| `denoiser` | `Literal['none', 'oidn', 'optix']` |  |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `alt` | property | `alt(self) -> bool` |  |  |
| `camera_mode` | property | `camera_mode(self) -> Literal['perspective', 'orthographic']` |  |  |
| `ctrl` | property | `ctrl(self) -> bool` |  |  |
| `display_picture_names` | property | `display_picture_names(self) -> list[str]` | Names for available display targets that can be displayed in the render function |  |
| `far` | property | `far(self) -> float` |  |  |
| `fovy` | property | `fovy(self) -> float` |  |  |
| `fps` | property | `fps(self) -> float` |  |  |
| `get_camera_model_matrix` | method | `get_camera_model_matrix(self) -> np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float32]]` |  |  |
| `get_camera_pose` | method | `get_camera_pose(self) -> sapien.Pose` |  |  |
| `get_camera_position` | method | `get_camera_position(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` |  |  |
| `get_camera_projection_matrix` | method | `get_camera_projection_matrix(self) -> np.ndarray[Any, np.dtype[np.float32]]` |  |  |
| `get_camera_property_float` | method | `get_camera_property_float(self, key: str) -> float` |  |  |
| `get_camera_property_int` | method | `get_camera_property_int(self, key: str) -> int` |  |  |
| `get_camera_rotation` | method | `get_camera_rotation(self) -> np.ndarray[Literal[4], np.dtype[np.float32]]` |  |  |
| `get_content_scale` | method | `get_content_scale(self) -> float` |  |  |
| `get_picture` | method | `get_picture(self, name: str) -> np.ndarray[Any, np.dtype[Any]]` |  | GPU PhysX dynamic bodies read CPU poses; prefer the CUDA pose path for offscreen. |
| `get_picture_pixel` | method | `get_picture_pixel(self, name: str, x: int, y: int) -> np.ndarray[Any, np.dtype[Any]]` |  |  |
| `get_picture_size` | method | `get_picture_size(self, name: str) -> Annotated[list[int], FixedSize(2)]` |  |  |
| `hide` | method | `hide(self) -> None` |  |  |
| `key_down` | method | `key_down(self, key: str) -> bool` |  |  |
| `key_press` | method | `key_press(self, key: str) -> bool` |  |  |
| `mouse_click` | method | `mouse_click(self, key: int) -> bool` |  |  |
| `mouse_delta` | property | `mouse_delta(self) -> Annotated[list[float], FixedSize(2)]` |  |  |
| `mouse_down` | method | `mouse_down(self, key: int) -> bool` |  |  |
| `mouse_position` | property | `mouse_position(self) -> Annotated[list[float], FixedSize(2)]` |  |  |
| `mouse_wheel_delta` | property | `mouse_wheel_delta(self) -> Annotated[list[float], FixedSize(2)]` |  |  |
| `near` | property | `near(self) -> float` |  |  |
| `ortho_top` | property | `ortho_top(self) -> float` |  |  |
| `pose_transport` | property | `pose_transport(self) -> str` | Active Viewer pose transport. |  |
| `pose_transfer_bytes` | property | `pose_transfer_bytes(self) -> int` | Cumulative pose D2H bytes for the active transport. | Staged transfers are 28 bytes per unique rendered GPU pose per submission. |
| `render` | method | `render(self, target_name: str, ui_windows: list[sapien.internal_renderer.UIWidget]=[]) -> None` |  |  |
| `resize` | method | `resize(self, width: int, height: int) -> None` |  |  |
| `set_camera_orthographic_parameters` | method | `set_camera_orthographic_parameters(self, near: float, far: float, top: float) -> None` |  |  |
| `set_camera_parameters` | method | `set_camera_parameters(self, near: float, far: float, fovy: float) -> None` |  |  |
| `set_camera_pose` | method | `set_camera_pose(self, pose: sapien.Pose) -> None` |  |  |
| `set_camera_position` | method | `set_camera_position(self, position: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` |  |  |
| `set_camera_property` | method | `set_camera_property(self, key: str, value: float) -> None<br>set_camera_property(self, key: str, value: int) -> None` |  |  |
| `set_camera_rotation` | method | `set_camera_rotation(self, quat: np.ndarray[Literal[4], np.dtype[np.float32]] \| list[float] \| tuple) -> None` |  |  |
| `set_camera_texture` | method | `set_camera_texture(self, name: str, texture: RenderTexture2D) -> None` |  |  |
| `set_camera_texture_array` | method | `set_camera_texture_array(self, name: str, textures: list[RenderTexture2D]) -> None` |  |  |
| `set_content_scale` | method | `set_content_scale(self, scale: float) -> None` |  |  |
| `set_drop_callback` | method | `set_drop_callback(self, callback: Callable[[list[str]], None]) -> None` |  |  |
| `set_focus_callback` | method | `set_focus_callback(self, callback: Callable[[int], None]) -> None` |  |  |
| `set_intrinsic_parameters` | method | `set_intrinsic_parameters(self, near: float, far: float, fx: float, fy: float, cx: float, cy: float, skew: float) -> None` |  |  |
| `set_scene` | method | `set_scene(self, scene: sapien.Scene) -> None` |  |  |
| `set_scenes` | method | `set_scenes(self, scenes: list[sapien.Scene]) -> None` | Select base scenes plus associated shared scenes once. | No render offsets are applied. |
| `configure_physx_gpu_rendering` | method | `configure_physx_gpu_rendering(self, physx_system: sapien.physx.PhysxGpuSystem, transport: Literal['auto', 'direct', 'staged', 'cpu-debug']='auto') -> None` | Configure Viewer pose transport. | Auto selects same-device direct or cross-device staged raster/RT transport. |
| `set_shader_dir` | method | `set_shader_dir(self, shader_dir: str) -> None` |  |  |
| `shift` | property | `shift(self) -> bool` |  |  |
| `should_close` | property | `should_close(self) -> bool` |  |  |
| `show` | method | `show(self) -> None` |  |  |
| `size` | property | `size(self) -> Annotated[list[int], FixedSize(2)]` |  |  |
| `super` | property | `super(self) -> bool` |  |  |
| `unset_drop_callback` | method | `unset_drop_callback(self) -> None` |  |  |
| `unset_focus_callback` | method | `unset_focus_callback(self) -> None` |  |  |
| `update_render` | method | `update_render(self) -> None` | Submit render-system state and the configured Viewer pose transport. | Call before `render()` after each displayed simulation state. |
| `__init__` | method | `__init__(self, width: int, height: int, shader_dir: str) -> None` |  |  |
| `_internal_renderer` | property | `_internal_renderer(self) -> sapien.internal_renderer.Renderer` |  |  |
| `_internal_scene` | property | `_internal_scene(self) -> sapien.internal_renderer.Scene` |  |  |

## `sapien.render.SapienRenderer`

- Use: SAPIEN API object.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__init__` | method | `__init__(self, device: sapien.Device \| None=None) -> None` |  |  |
| `_internal_context` | property | `_internal_context(self) -> sapien.internal_renderer.Context` |  |  |