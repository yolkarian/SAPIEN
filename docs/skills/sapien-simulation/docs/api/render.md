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
| `shading_mode` | `int` | Property: shading mode. |  |
| `visibility` | `float` | Property: visibility. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `attach` | method | `attach(self, shape: RenderShape) -> RenderBodyComponent` | Call attach. |  |
| `clone` | method | `clone(self) -> RenderBodyComponent` | Call clone. |  |
| `compute_global_aabb_tight` | method | `compute_global_aabb_tight(self) -> np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.float32]]` | Compute a tight global AABB. |  |
| `disable_render_id` | method | `disable_render_id(self) -> None` | Disable render id. |  |
| `enable_render_id` | method | `enable_render_id(self) -> None` | Enable render id. |  |
| `get_global_aabb_fast` | method | `get_global_aabb_fast(self) -> np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.float32]]` | Get global AABB fast. |  |
| `is_render_id_disabled` | property | `is_render_id_disabled(self) -> bool` | Bool: render id disabled. |  |
| `render_shapes` | property | `render_shapes(self) -> list[RenderShape]` | Property: render shapes. |  |
| `set_property` | method | `set_property(self, name: str, value: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None<br>set_property(self, name: str, value: float) -> None<br>set_property(self, name: str, value: int) -> None` | Set property. |  |
| `set_texture` | method | `set_texture(self, name: str, texture: RenderTexture) -> None` | Set texture. |  |
| `set_texture_array` | method | `set_texture_array(self, name: str, textures: list[RenderTexture]) -> None` | Set texture array. |  |
| `__init__` | method | `__init__(self) -> None` | Python special method. |  |
| `_internal_node` | property | `_internal_node(self) -> sapien.internal_renderer.Node` | Property: internal node. |  |

## `sapien.render.RenderCameraComponent`

- Use: Camera component; read image/CUDA image after take_picture.
- Bases: `sapien.Component`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `far` | `float` | Property: far. |  |
| `local_pose` | `sapien.Pose` | Property: local pose. |  |
| `near` | `float` | Property: near. |  |
| `skew` | `float` | Property: skew. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `cx` | property | `cx(self) -> float` | Property: cx. |  |
| `cy` | property | `cy(self) -> float` | Property: cy. |  |
| `fovx` | property | `fovx(self) -> float` | Property: fovx. |  |
| `fovy` | property | `fovy(self) -> float` | Property: fovy. |  |
| `fx` | property | `fx(self) -> float` | Property: fx. |  |
| `fy` | property | `fy(self) -> float` | Property: fy. |  |
| `get_extrinsic_matrix` | method | `get_extrinsic_matrix(self) -> np.ndarray[tuple[Literal[3], Literal[4]], np.dtype[np.float32]]` | Get 3x4 extrinsic camera matrix in OpenCV format. |  |
| `get_far` | method | `get_far(self) -> float` | Get far. |  |
| `get_global_pose` | method | `get_global_pose(self) -> sapien.Pose` | Get global pose. |  |
| `get_height` | method | `get_height(self) -> int` | Get height. |  |
| `get_intrinsic_matrix` | method | `get_intrinsic_matrix(self) -> np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float32]]` | Get 3x3 intrinsic camera matrix in OpenCV format. |  |
| `get_local_pose` | method | `get_local_pose(self) -> sapien.Pose` | Get local pose. |  |
| `get_mode` | method | `get_mode(self) -> Literal['perspective', 'orthographic']` | Get mode. |  |
| `get_model_matrix` | method | `get_model_matrix(self) -> np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float32]]` | Get model matrix (inverse of extrinsic matrix) used in rendering (Y up, Z back) |  |
| `get_near` | method | `get_near(self) -> float` | Get near. |  |
| `get_picture` | method | `get_picture(self, name: str) -> np.ndarray[Any, np.dtype[Any]]` | Get picture. | GPU PhysX dynamic bodies read CPU poses; prefer the CUDA pose path for offscreen. |
| `get_picture_cuda` | method | `get_picture_cuda(self, name: str) -> sapien.CudaArray` | This function transfers the rendered image into a CUDA buffer. Usage: # use torch backend sapien.set_cuda_tensor_backend("torch") # called once per process image: torc... | Direct GPU render path; no sync_poses_gpu_to_cpu needed. |
| `get_picture_names` | method | `get_picture_names(self) -> list[str]` | Get picture names. |  |
| `get_projection_matrix` | method | `get_projection_matrix(self) -> np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float32]]` | Get projection matrix in used in rendering (right-handed NDC with [-1,1] XY and [0,1] Z) |  |
| `get_skew` | method | `get_skew(self) -> float` | Get skew. |  |
| `get_width` | method | `get_width(self) -> int` | Get width. |  |
| `global_pose` | property | `global_pose(self) -> sapien.Pose` | Property: global pose. |  |
| `height` | property | `height(self) -> int` | Property: height. |  |
| `mode` | property | `mode(self) -> Literal['perspective', 'orthographic']` | Property: mode. |  |
| `ortho_bottom` | property | `ortho_bottom(self) -> float` | Property: ortho bottom. |  |
| `ortho_left` | property | `ortho_left(self) -> float` | Property: ortho left. |  |
| `ortho_right` | property | `ortho_right(self) -> float` | Property: ortho right. |  |
| `ortho_top` | property | `ortho_top(self) -> float` | Property: ortho top. |  |
| `set_far` | method | `set_far(self, far: float) -> None` | Set far. |  |
| `set_focal_lengths` | method | `set_focal_lengths(self, fx: float, fy: float) -> None` | Set focal lengths. |  |
| `set_fovx` | method | `set_fovx(self, fov: float, compute_y: bool=True) -> None` | Set fovx. |  |
| `set_fovy` | method | `set_fovy(self, fov: float, compute_x: bool=True) -> None` | Set fovy. |  |
| `set_gpu_pose_batch_index` | method | `set_gpu_pose_batch_index(self, index: int) -> None` | Bind GPU pose index for batched/direct GPU rendering. | Direct GPU render path; no sync_poses_gpu_to_cpu needed. |
| `set_local_pose` | method | `set_local_pose(self, pose: sapien.Pose) -> None` | Set local pose. |  |
| `set_near` | method | `set_near(self, near: float) -> None` | Set near. |  |
| `set_orthographic_parameters` | method | `set_orthographic_parameters(self, near: float, far: float, top: float) -> None<br>set_orthographic_parameters(self, near: float, far: float, left: float, right: float, bottom: float, top: float) -> None` | Set orthographic parameters. |  |
| `set_perspective_parameters` | method | `set_perspective_parameters(self, near: float, far: float, fx: float, fy: float, cx: float, cy: float, skew: float) -> None` | Set perspective parameters. |  |
| `set_principal_point` | method | `set_principal_point(self, cx: float, cy: float) -> None` | Set principal point. |  |
| `set_property` | method | `set_property(self, name: str, value: float) -> None<br>set_property(self, name: str, value: int) -> None` | Set property. |  |
| `set_skew` | method | `set_skew(self, skew: float) -> None` | Set skew. |  |
| `set_texture` | method | `set_texture(self, name: str, texture: RenderTexture) -> None` | Set texture. |  |
| `set_texture_array` | method | `set_texture_array(self, name: str, textures: list[RenderTexture]) -> None` | Set texture array. |  |
| `take_picture` | method | `take_picture(self) -> None` | Trigger camera/camera-group rendering. |  |
| `width` | property | `width(self) -> int` | Property: width. |  |
| `__init__` | method | `__init__(self, width: int, height: int, shader_dir: str='') -> None` | Python special method. |  |
| `_cuda_buffer` | property | `_cuda_buffer(self) -> sapien.CudaArray` | Debug only. Get the CUDA buffer containing GPU data for this camera, including transformaion matrices, sizes, and user-defined shader fields. |  |
| `_internal_renderer` | property | `_internal_renderer(self) -> sapien.internal_renderer.Renderer` | Property: internal renderer. |  |

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
| `export` | method | `export(self, filename: str) -> None` | Call export. |  |
| `__init__` | method | `__init__(self, filename: str) -> None<br>__init__(self, px: str, nx: str, py: str, ny: str, pz: str, nz: str) -> None` | Python special method. |  |
| `_internal_cubemap` | property | `_internal_cubemap(self) -> sapien.internal_renderer.Cubemap` | Property: internal cubemap. |  |

## `sapien.render.RenderCudaMeshComponent`

- Use: SAPIEN render API object.
- Bases: `sapien.Component`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `material` | `RenderMaterial` | Property: material. |  |
| `triangle_count` | `int` | Property: triangle count. |  |
| `vertex_count` | `int` | Property: vertex count. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `cuda_triangles` | property | `cuda_triangles(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cuda_vertices` | property | `cuda_vertices(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `get_cuda_triangles` | method | `get_cuda_triangles(self) -> sapien.CudaArray` | Get cuda triangles. |  |
| `get_cuda_vertices` | method | `get_cuda_vertices(self) -> sapien.CudaArray` | Get cuda vertices. |  |
| `get_material` | method | `get_material(self) -> RenderMaterial` | Get material. |  |
| `get_triangle_count` | method | `get_triangle_count(self) -> int` | Get triangle count. |  |
| `get_vertex_count` | method | `get_vertex_count(self) -> int` | Get vertex count. |  |
| `notify_vertex_updated` | method | `notify_vertex_updated(self, cuda_stream: int=0) -> None` | Call notify vertex updated. |  |
| `set_material` | method | `set_material(self, material: RenderMaterial) -> None` | Set material. |  |
| `set_triangle_count` | method | `set_triangle_count(self, count: int) -> None` | Set triangle count. |  |
| `set_triangles` | method | `set_triangles(self, triangles: np.ndarray[np.uint32[M, 3]]) -> None` | Set triangles. |  |
| `set_vertex_count` | method | `set_vertex_count(self, count: int) -> None` | Set vertex count. |  |
| `__init__` | method | `__init__(self, max_vertex_count: int, max_triangle_count: int) -> None` | Python special method. |  |

## `sapien.render.RenderDirectionalLightComponent`

- Use: Concrete render light component.
- Bases: `RenderLightComponent`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `shadow_half_size` | `float` | Property: shadow half size. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_shadow_half_size` | method | `get_shadow_half_size(self) -> float` | Get shadow half size. |  |
| `set_shadow_half_size` | method | `set_shadow_half_size(self, size: float) -> None` | Set shadow half size. |  |
| `__init__` | method | `__init__(self) -> None` | Python special method. |  |

## `sapien.render.RenderLightComponent`

- Use: Light component base class.
- Bases: `sapien.Component`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `color` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: color. |  |
| `local_pose` | `sapien.Pose` | Property: local pose. |  |
| `shadow` | `bool` | Property: shadow. |  |
| `shadow_far` | `float` | Property: shadow far. |  |
| `shadow_map_size` | `int` | Property: shadow map size. |  |
| `shadow_near` | `float` | Property: shadow near. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `disable_shadow` | method | `disable_shadow(self) -> None` | Disable shadow. |  |
| `enable_shadow` | method | `enable_shadow(self) -> None` | Enable shadow. |  |
| `get_color` | method | `get_color(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Get color. |  |
| `get_global_pose` | method | `get_global_pose(self) -> sapien.Pose` | Get global pose. |  |
| `get_local_pose` | method | `get_local_pose(self) -> sapien.Pose` | Get local pose. |  |
| `get_shadow_far` | method | `get_shadow_far(self) -> float` | Get shadow far. |  |
| `get_shadow_map_size` | method | `get_shadow_map_size(self) -> int` | Get shadow map size. |  |
| `get_shadow_near` | method | `get_shadow_near(self) -> float` | Get shadow near. |  |
| `global_pose` | property | `global_pose(self) -> sapien.Pose` | Property: global pose. |  |
| `set_color` | method | `set_color(self, color: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | Set color. |  |
| `set_local_pose` | method | `set_local_pose(self, pose: sapien.Pose) -> None` | Set local pose. |  |
| `set_shadow_far` | method | `set_shadow_far(self, far: float) -> None` | Set shadow far. |  |
| `set_shadow_map_size` | method | `set_shadow_map_size(self, size: int) -> None` | Set shadow map size. |  |
| `set_shadow_near` | method | `set_shadow_near(self, near: float) -> None` | Set shadow near. |  |

## `sapien.render.RenderMaterial`

- Use: PBR render material.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `base_color` | `Annotated[list[float], FixedSize(4)]` | Property: base color. |  |
| `base_color_texture` | `RenderTexture2D` | Property: base color texture. |  |
| `diffuse_texture` | `` | Property: diffuse texture. |  |
| `emission` | `Annotated[list[float], FixedSize(4)]` | Property: emission. |  |
| `emission_texture` | `RenderTexture2D` | Property: emission texture. |  |
| `ior` | `float` | Property: ior. |  |
| `metallic` | `float` | Property: metallic. |  |
| `metallic_texture` | `RenderTexture2D` | Property: metallic texture. |  |
| `normal_texture` | `RenderTexture2D` | Property: normal texture. |  |
| `roughness` | `float` | Property: roughness. |  |
| `roughness_texture` | `RenderTexture2D` | Property: roughness texture. |  |
| `specular` | `float` | Property: specular. |  |
| `transmission` | `float` | Property: transmission. |  |
| `transmission_roughness` | `float` | Property: transmission roughness. |  |
| `transmission_texture` | `RenderTexture2D` | Property: transmission texture. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_base_color` | method | `get_base_color(self) -> Annotated[list[float], FixedSize(4)]` | Get base color. |  |
| `get_base_color_texture` | method | `get_base_color_texture(self) -> RenderTexture2D` | Get base color texture. |  |
| `get_diffuse_texture` | method | `get_diffuse_texture(self)` | Get diffuse texture. |  |
| `get_emission` | method | `get_emission(self) -> Annotated[list[float], FixedSize(4)]` | Get emission. |  |
| `get_emission_texture` | method | `get_emission_texture(self) -> RenderTexture2D` | Get emission texture. |  |
| `get_ior` | method | `get_ior(self) -> float` | Get ior. |  |
| `get_metallic` | method | `get_metallic(self) -> float` | Get metallic. |  |
| `get_metallic_texture` | method | `get_metallic_texture(self) -> RenderTexture2D` | Get metallic texture. |  |
| `get_normal_texture` | method | `get_normal_texture(self) -> RenderTexture2D` | Get normal texture. |  |
| `get_roughness` | method | `get_roughness(self) -> float` | Get roughness. |  |
| `get_roughness_texture` | method | `get_roughness_texture(self) -> RenderTexture2D` | Get roughness texture. |  |
| `get_specular` | method | `get_specular(self) -> float` | Get specular. |  |
| `get_transmission` | method | `get_transmission(self) -> float` | Get transmission. |  |
| `get_transmission_roughness` | method | `get_transmission_roughness(self) -> float` | Get transmission roughness. |  |
| `get_transmission_texture` | method | `get_transmission_texture(self) -> RenderTexture2D` | Get transmission texture. |  |
| `set_base_color` | method | `set_base_color(self, color: Annotated[list[float], FixedSize(4)]) -> None` | Set base color. |  |
| `set_base_color_texture` | method | `set_base_color_texture(self, texture: RenderTexture2D) -> None` | Set base color texture. |  |
| `set_diffuse_texture` | method | `set_diffuse_texture(self, texture)` | Set diffuse texture. |  |
| `set_emission` | method | `set_emission(self, emission: Annotated[list[float], FixedSize(4)]) -> None` | Set emission. |  |
| `set_emission_texture` | method | `set_emission_texture(self, texture: RenderTexture2D) -> None` | Set emission texture. |  |
| `set_ior` | method | `set_ior(self, ior: float) -> None` | Set ior. |  |
| `set_metallic` | method | `set_metallic(self, metallic: float) -> None` | Set metallic. |  |
| `set_metallic_texture` | method | `set_metallic_texture(self, texture: RenderTexture2D) -> None` | Set metallic texture. |  |
| `set_normal_texture` | method | `set_normal_texture(self, texture: RenderTexture2D) -> None` | Set normal texture. |  |
| `set_roughness` | method | `set_roughness(self, roughness: float) -> None` | Set roughness. |  |
| `set_roughness_texture` | method | `set_roughness_texture(self, texture: RenderTexture2D) -> None` | Set roughness texture. |  |
| `set_specular` | method | `set_specular(self, specular: float) -> None` | Set specular. |  |
| `set_transmission` | method | `set_transmission(self, transmission: float) -> None` | Set transmission. |  |
| `set_transmission_roughness` | method | `set_transmission_roughness(self, roughness: float) -> None` | Set transmission roughness. |  |
| `set_transmission_texture` | method | `set_transmission_texture(self, texture: RenderTexture2D) -> None` | Set transmission texture. |  |
| `__eq__` | method | `__eq__(self, arg0: RenderMaterial) -> bool` | Python special method. |  |
| `__init__` | method | `__init__(self, emission: Annotated[list[float], FixedSize(4)]=[0.0, 0.0, 0.0, 0.0], base_color: Annotated[list[float], FixedSize(4)]=[1.0, 1.0, 1.0, 1.0], specular: float=0.0, roughness: float=1.0, metallic: float=0.0, transmission: float=0.0, ior: float=1.4500000476837158, transmission_roughness: float=0.0) -> None` | Python special method. |  |

## `sapien.render.RenderParallelogramLightComponent`

- Use: Concrete render light component.
- Bases: `RenderLightComponent`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `angle` | property | `angle(self) -> float` | Property: angle. |  |
| `get_angle` | method | `get_angle(self) -> float` | Get angle. |  |
| `get_half_height` | method | `get_half_height(self) -> float` | Get half height. |  |
| `get_half_width` | method | `get_half_width(self) -> float` | Get half width. |  |
| `half_height` | property | `half_height(self) -> float` | Property: half height. |  |
| `half_width` | property | `half_width(self) -> float` | Property: half width. |  |
| `set_shape` | method | `set_shape(self, half_width: float, half_height: float, angle: float=1.5707963705062866) -> None` | Set shape. |  |
| `__init__` | method | `__init__(self) -> None` | Python special method. |  |

## `sapien.render.RenderPointCloudComponent`

- Use: SAPIEN render API object.
- Bases: `sapien.Component`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_cuda_aabb` | method | `get_cuda_aabb(self) -> sapien.CudaArray \| None` | this function is a temporary hack to help update the AABBs used for ray tracing BLAS. returns None if ray tracing has not been initialized |  |
| `get_cuda_vertices` | method | `get_cuda_vertices(self) -> sapien.CudaArray` | Get cuda vertices. |  |
| `get_vertices` | method | `get_vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | Get previously set vertices. This function does not reflect any changes directly made to the GPU. |  |
| `set_attribute` | method | `set_attribute(self, name: str, attribute: np.ndarray[tuple[M, N], np.dtype[np.float32]] \| list \| tuple) -> RenderPointCloudComponent` | Set attribute. |  |
| `set_vertices` | method | `set_vertices(self, vertices: np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]] \| list \| tuple) -> RenderPointCloudComponent` | Set vertices. |  |
| `__init__` | method | `__init__(self, capacity: int=0) -> None` | Python special method. |  |

## `sapien.render.RenderPointLightComponent`

- Use: Concrete render light component.
- Bases: `RenderLightComponent`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__init__` | method | `__init__(self) -> None` | Python special method. |  |

## `sapien.render.RenderSceneLoaderNode`

- Use: SAPIEN render API object.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `children` | property | `children(self) -> list[RenderSceneLoaderNode]` | Property: children. |  |
| `flatten` | method | `flatten(self) -> tuple[list[RenderShapeTriangleMesh], list[RenderLightComponent]]` | Call flatten. |  |
| `light` | property | `light(self) -> RenderLightComponent` | Property: light. |  |
| `mesh` | property | `mesh(self) -> RenderShapeTriangleMesh` | Property: mesh. |  |
| `name` | property | `name(self) -> str` | Property: name. |  |
| `pose` | property | `pose(self) -> sapien.Pose` | Property: pose. |  |
| `scale` | property | `scale(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: scale. |  |

## `sapien.render.RenderShape`

- Use: Render shape base class; supports GPU pose batch index.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `front_face` | `Literal['counterclockwise', 'clockwise']` | Property: front face. |  |
| `local_pose` | `sapien.Pose` | Property: local pose. |  |
| `name` | `str` | Property: name. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `clone` | method | `clone(self) -> RenderShape` | Call clone. |  |
| `get_front_face` | method | `get_front_face(self) -> Literal['counterclockwise', 'clockwise']` | Get front face. |  |
| `get_local_pose` | method | `get_local_pose(self) -> sapien.Pose` | Get local pose. |  |
| `get_material` | method | `get_material(self) -> RenderMaterial` | Get material. |  |
| `get_name` | method | `get_name(self) -> str` | Get name. |  |
| `get_parts` | method | `get_parts(self) -> list[RenderShapeTriangleMeshPart]` | Get parts. |  |
| `get_per_scene_id` | method | `get_per_scene_id(self) -> int` | Get per scene id. |  |
| `material` | property | `material(self) -> RenderMaterial` | Property: material. |  |
| `parts` | property | `parts(self) -> list[RenderShapeTriangleMeshPart]` | Property: parts. |  |
| `per_scene_id` | property | `per_scene_id(self) -> int` | Property: per scene id. |  |
| `set_front_face` | method | `set_front_face(self, front_face: Literal['counterclockwise', 'clockwise']) -> None` | Set front face. |  |
| `set_gpu_pose_batch_index` | method | `set_gpu_pose_batch_index(self, index: int) -> None` | Bind GPU pose index for batched/direct GPU rendering. | Direct GPU render path; no sync_poses_gpu_to_cpu needed. |
| `set_local_pose` | method | `set_local_pose(self, pose: sapien.Pose) -> None` | Set local pose. |  |
| `set_name` | method | `set_name(self, name: str) -> None` | Set name. |  |

## `sapien.render.RenderShapeBox`

- Use: Concrete render shape.
- Bases: `RenderShapePrimitive`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_half_size` | method | `get_half_size(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Get half size. |  |
| `half_size` | property | `half_size(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: half size. |  |
| `__init__` | method | `__init__(self, half_size: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, material: RenderMaterial) -> None` | Python special method. |  |

## `sapien.render.RenderShapeCapsule`

- Use: Concrete render shape.
- Bases: `RenderShapePrimitive`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_half_length` | method | `get_half_length(self) -> float` | Get half length. |  |
| `get_radius` | method | `get_radius(self) -> float` | Get radius. |  |
| `half_length` | property | `half_length(self) -> float` | Property: half length. |  |
| `radius` | property | `radius(self) -> float` | Property: radius. |  |
| `__init__` | method | `__init__(self, radius: float, half_length: float, material: RenderMaterial) -> None` | Python special method. |  |

## `sapien.render.RenderShapeCylinder`

- Use: Concrete render shape.
- Bases: `RenderShapePrimitive`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_half_length` | method | `get_half_length(self) -> float` | Get half length. |  |
| `get_radius` | method | `get_radius(self) -> float` | Get radius. |  |
| `half_length` | property | `half_length(self) -> float` | Property: half length. |  |
| `radius` | property | `radius(self) -> float` | Property: radius. |  |
| `__init__` | method | `__init__(self, radius: float, half_length: float, material: RenderMaterial) -> None` | Python special method. |  |

## `sapien.render.RenderShapePlane`

- Use: Concrete render shape.
- Bases: `RenderShapePrimitive`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_scale` | method | `get_scale(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Get scale. |  |
| `scale` | property | `scale(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: scale. |  |
| `__init__` | method | `__init__(self, scale: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, material: RenderMaterial) -> None` | Python special method. |  |

## `sapien.render.RenderShapePrimitive`

- Use: Concrete render shape.
- Bases: `RenderShape`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_triangles` | method | `get_triangles(self) -> np.ndarray[np.uint32[M, 3]]` | Get triangles. |  |
| `get_vertex_normal` | method | `get_vertex_normal(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | Get vertex normal. |  |
| `get_vertex_uv` | method | `get_vertex_uv(self) -> np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | Get vertex uv. |  |
| `get_vertices` | method | `get_vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | Get vertices. |  |
| `triangles` | property | `triangles(self) -> np.ndarray[np.uint32[M, 3]]` | Property: triangles. |  |
| `vertex_normal` | property | `vertex_normal(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | Property: vertex normal. |  |
| `vertex_uv` | property | `vertex_uv(self) -> np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | Property: vertex uv. |  |
| `vertices` | property | `vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | Property: vertices. |  |

## `sapien.render.RenderShapeSphere`

- Use: Concrete render shape.
- Bases: `RenderShapePrimitive`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_radius` | method | `get_radius(self) -> float` | Get radius. |  |
| `radius` | property | `radius(self) -> float` | Property: radius. |  |
| `__init__` | method | `__init__(self, radius: float, material: RenderMaterial) -> None` | Python special method. |  |

## `sapien.render.RenderShapeTriangleMesh`

- Use: Concrete render shape.
- Bases: `RenderShape`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `scale` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: scale. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `filename` | property | `filename(self) -> str` | Property: filename. |  |
| `get_filename` | method | `get_filename(self) -> str` | Get filename. |  |
| `get_scale` | method | `get_scale(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Get scale. |  |
| `set_scale` | method | `set_scale(self, scale: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | Note: this function only works when the shape is not added to scene |  |
| `__init__` | method | `__init__(self, vertices: np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]] \| list \| tuple, triangles: np.ndarray[np.uint32[M, 3]], normals: np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]] \| list \| tuple, uvs: np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]] \| list \| tuple, material: RenderMaterial) -> None<br>__init__(self, filename: str, scale: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple=..., material: RenderMaterial \| None=None) -> None` | Python special method. |  |

## `sapien.render.RenderShapeTriangleMeshPart`

- Use: Concrete render shape.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `cuda_triangles` | property | `cuda_triangles(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `cuda_vertices` | property | `cuda_vertices(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `get_cuda_triangles` | method | `get_cuda_triangles(self) -> sapien.CudaArray` | Get cuda triangles. |  |
| `get_cuda_vertices` | method | `get_cuda_vertices(self) -> sapien.CudaArray` | Get cuda vertices. |  |
| `get_material` | method | `get_material(self) -> RenderMaterial` | Get material. |  |
| `get_triangles` | method | `get_triangles(self) -> np.ndarray[np.uint32[M, 3]]` | Get triangles. |  |
| `get_vertex_normal` | method | `get_vertex_normal(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | Get vertex normal. |  |
| `get_vertex_uv` | method | `get_vertex_uv(self) -> np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | Get vertex uv. |  |
| `get_vertices` | method | `get_vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | Get vertices. |  |
| `material` | property | `material(self) -> RenderMaterial` | Property: material. |  |
| `set_vertex_normal` | method | `set_vertex_normal(self, normal: np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]] \| list \| tuple) -> None` | Set vertex normal. |  |
| `set_vertex_uv` | method | `set_vertex_uv(self, uv: np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]] \| list \| tuple) -> None` | Set vertex uv. |  |
| `triangles` | property | `triangles(self) -> np.ndarray[np.uint32[M, 3]]` | Property: triangles. |  |
| `vertices` | property | `vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | Property: vertices. |  |
| `__eq__` | method | `__eq__(self, arg0: RenderShapeTriangleMeshPart) -> bool` | Python special method. |  |

## `sapien.render.RenderSpotLightComponent`

- Use: Concrete render light component.
- Bases: `RenderLightComponent`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `inner_fov` | `float` | Property: inner fov. |  |
| `outer_fov` | `float` | Property: outer fov. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_inner_fov` | method | `get_inner_fov(self) -> float` | Get inner fov. |  |
| `get_outer_fov` | method | `get_outer_fov(self) -> float` | Get outer fov. |  |
| `set_inner_fov` | method | `set_inner_fov(self, fov: float) -> None` | Set inner fov. |  |
| `set_outer_fov` | method | `set_outer_fov(self, fov: float) -> None` | Set outer fov. |  |
| `__init__` | method | `__init__(self) -> None` | Python special method. |  |

## `sapien.render.RenderSystem`

- Use: Render system; only needed for viewer/sensor/offscreen.
- Bases: `sapien.System`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `ambient_light` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | Property: ambient light. |  |
| `cubemap` | `RenderCubemap` | Property: cubemap. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `cameras` | property | `cameras(self) -> list[RenderCameraComponent]` | Property: cameras. |  |
| `cuda_object_transforms` | property | `cuda_object_transforms(self) -> sapien.CudaArray` | CUDA state/render buffer property. | Get view after gpu_init; reuse torch/cupy/jax view in loop. |
| `device` | property | `device(self) -> sapien.Device` | Property: device. |  |
| `get_ambient_light` | method | `get_ambient_light(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Get ambient light. |  |
| `get_cameras` | method | `get_cameras(self) -> list[RenderCameraComponent]` | Get cameras. |  |
| `get_cubemap` | method | `get_cubemap(self) -> RenderCubemap` | Get cubemap. |  |
| `get_lights` | method | `get_lights(self) -> list[RenderLightComponent]` | Get lights. |  |
| `get_point_clouds` | method | `get_point_clouds(self) -> list[RenderPointCloudComponent]` | Get point clouds. |  |
| `get_render_bodies` | method | `get_render_bodies(self) -> list[RenderBodyComponent]` | Get render bodies. |  |
| `lights` | property | `lights(self) -> list[RenderLightComponent]` | Property: lights. |  |
| `point_clouds` | property | `point_clouds(self) -> list[RenderPointCloudComponent]` | Property: point clouds. |  |
| `render_bodies` | property | `render_bodies(self) -> list[RenderBodyComponent]` | Property: render bodies. |  |
| `set_ambient_light` | method | `set_ambient_light(self, color: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | Set ambient light. |  |
| `set_cubemap` | method | `set_cubemap(self, cubemap: RenderCubemap) -> None` | Set cubemap. |  |
| `__init__` | method | `__init__(self, device: sapien.Device \| None=None) -> None<br>__init__(self, device: str) -> None` | Python special method. |  |
| `_internal_scene` | property | `_internal_scene(self) -> sapien.internal_renderer.Scene` | Property: internal scene. |  |

## `sapien.render.RenderSystemGroup`

- Use: Batched render group over multiple RenderSystems; can bind CUDA poses.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `create_camera_group` | method | `create_camera_group(self, cameras: list[RenderCameraComponent], picture_names: list[str]) -> RenderCameraGroup` | Create a batched camera group. |  |
| `set_cuda_poses` | method | `set_cuda_poses(self, pose_buffer: sapien.CudaArray) -> None` | Bind a RenderSystemGroup to a PhysX CUDA pose buffer. | Direct GPU render path; no sync_poses_gpu_to_cpu needed. |
| `set_cuda_stream` | method | `set_cuda_stream(self, stream: int) -> None` | Set cuda stream. |  |
| `update_render` | method | `update_render(self) -> None` | This function performs CUDA operations to transfer poses from the CUDA buffer provided by :func:`set_cuda_poses` into render systems. It updates the transformation mat... | GPU PhysX dynamic bodies read CPU poses; prefer the CUDA pose path for offscreen. |
| `__init__` | method | `__init__(self, systems: list[RenderSystem]) -> None` | Python special method. |  |

## `sapien.render.RenderTexture`

- Use: SAPIEN render API object.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `address_mode` | property | `address_mode(self) -> Literal['repeat', 'border', 'edge', 'mirror']` | Property: address mode. |  |
| `channels` | property | `channels(self) -> int` | Property: channels. |  |
| `depth` | property | `depth(self) -> int` | Property: depth. |  |
| `download` | method | `download(self) -> np.ndarray[Any, np.dtype[Any]]` | Call download. |  |
| `filter_mode` | property | `filter_mode(self) -> Literal['nearest', 'linear']` | Property: filter mode. |  |
| `format` | property | `format(self) -> str` | Property: format. |  |
| `get_address_mode` | method | `get_address_mode(self) -> Literal['repeat', 'border', 'edge', 'mirror']` | Get address mode. |  |
| `get_channels` | method | `get_channels(self) -> int` | Get channels. |  |
| `get_depth` | method | `get_depth(self) -> int` | Get depth. |  |
| `get_filter_mode` | method | `get_filter_mode(self) -> Literal['nearest', 'linear']` | Get filter mode. |  |
| `get_format` | method | `get_format(self) -> str` | Get format. |  |
| `get_height` | method | `get_height(self) -> int` | Get height. |  |
| `get_mipmap_levels` | method | `get_mipmap_levels(self) -> int` | Get mipmap levels. |  |
| `get_width` | method | `get_width(self) -> int` | Get width. |  |
| `height` | property | `height(self) -> int` | Property: height. |  |
| `is_srgb` | property | `is_srgb(self) -> bool` | Bool: srgb. |  |
| `mipmap_levels` | property | `mipmap_levels(self) -> int` | Property: mipmap levels. |  |
| `upload` | method | `upload(self, data: np.ndarray[Any, np.dtype[Any]] \| list \| tuple) -> None` | Call upload. |  |
| `width` | property | `width(self) -> int` | Property: width. |  |
| `__eq__` | method | `__eq__(self, arg0: RenderTexture) -> bool` | Python special method. |  |
| `__init__` | method | `__init__(self, array: np.ndarray[Any, np.dtype[Any]] \| list \| tuple, dim: int, format: str, mipmap_levels: int=1, filter_mode: Literal['nearest', 'linear']='linear', address_mode: Literal['repeat', 'border', 'edge', 'mirror']='repeat', srgb: bool=False) -> None` | Python special method. |  |

## `sapien.render.RenderTexture2D`

- Use: 2D render texture.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `address_mode` | property | `address_mode(self) -> Literal['repeat', 'border', 'edge', 'mirror']` | Property: address mode. |  |
| `channels` | property | `channels(self) -> int` | Property: channels. |  |
| `download` | method | `download(self) -> np.ndarray[Any, np.dtype[Any]]` | Call download. |  |
| `filename` | property | `filename(self) -> str` | Property: filename. |  |
| `filter_mode` | property | `filter_mode(self) -> Literal['nearest', 'linear']` | Property: filter mode. |  |
| `format` | property | `format(self) -> str` | Property: format. |  |
| `get_address_mode` | method | `get_address_mode(self) -> Literal['repeat', 'border', 'edge', 'mirror']` | Get address mode. |  |
| `get_channels` | method | `get_channels(self) -> int` | Get channels. |  |
| `get_filename` | method | `get_filename(self) -> str` | Get filename. |  |
| `get_filter_mode` | method | `get_filter_mode(self) -> Literal['nearest', 'linear']` | Get filter mode. |  |
| `get_format` | method | `get_format(self) -> str` | Get format. |  |
| `get_height` | method | `get_height(self) -> int` | Get height. |  |
| `get_mipmap_levels` | method | `get_mipmap_levels(self) -> int` | Get mipmap levels. |  |
| `get_width` | method | `get_width(self) -> int` | Get width. |  |
| `height` | property | `height(self) -> int` | Property: height. |  |
| `is_srgb` | property | `is_srgb(self) -> bool` | Bool: srgb. |  |
| `mipmap_levels` | property | `mipmap_levels(self) -> int` | Property: mipmap levels. |  |
| `upload` | method | `upload(self, data: np.ndarray[Any, np.dtype[Any]] \| list \| tuple) -> None` | Call upload. |  |
| `width` | property | `width(self) -> int` | Property: width. |  |
| `__eq__` | method | `__eq__(self, arg0: RenderTexture2D) -> bool` | Python special method. |  |
| `__init__` | method | `__init__(self, array: np.ndarray[Any, np.dtype[Any]] \| list \| tuple, format: str, mipmap_levels: int=1, filter_mode: Literal['nearest', 'linear']='linear', address_mode: Literal['repeat', 'border', 'edge', 'mirror']='repeat', srgb: bool=False) -> None<br>__init__(self, filename: str, mipmap_levels: int=1, filter_mode: Literal['nearest', 'linear']='linear', address_mode: Literal['repeat', 'border', 'edge', 'mirror']='repeat', srgb: bool=True) -> None` | Create texture from file. The srgb parameter only affects files in uint8 format; it should be true for color textures (diffuse, emission) and false for others (normal,... |  |

## `sapien.render.RenderTexturedLightComponent`

- Use: Concrete render light component.
- Bases: `RenderSpotLightComponent`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `texture` | `RenderTexture2D` | Property: texture. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_texture` | method | `get_texture(self) -> RenderTexture2D` | Get texture. |  |
| `set_texture` | method | `set_texture(self, texture: RenderTexture2D) -> None` | Set texture. |  |
| `__init__` | method | `__init__(self) -> None` | Python special method. |  |

## `sapien.render.RenderVRDisplay`

- Use: SAPIEN render API object.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `root_pose` | `sapien.Pose` | Property: root pose. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `fetch_poses` | method | `fetch_poses(self) -> None` | fetches poses of HMD and controllers |  |
| `get_controller_axis_state` | method | `get_controller_axis_state(self, id: int, axis: int) -> Annotated[list[float], FixedSize(2)]` | Get controller axis state. |  |
| `get_controller_button_pressed` | method | `get_controller_button_pressed(self, id: int) -> int` | Get controller button pressed. |  |
| `get_controller_button_touched` | method | `get_controller_button_touched(self, id: int) -> int` | Get controller button touched. |  |
| `get_controller_ids` | method | `get_controller_ids(self) -> list[int]` | Get controller ids. |  |
| `get_controller_pose` | method | `get_controller_pose(self, id: int) -> sapien.Pose` | Gets the local pose of a controller. It should be called immediately after fetch_poses |  |
| `get_hmd_pose` | method | `get_hmd_pose(self) -> sapien.Pose` | Gets the local pose of the head set. It should be called immediately after fetch_poses |  |
| `get_left_hand_root_pose` | method | `get_left_hand_root_pose(self) -> sapien.Pose` | Get left hand root pose. |  |
| `get_left_hand_skeletal_poses` | method | `get_left_hand_skeletal_poses(self) -> list[sapien.Pose]` | Get left hand skeletal poses. |  |
| `get_right_hand_root_pose` | method | `get_right_hand_root_pose(self) -> sapien.Pose` | Get right hand root pose. |  |
| `get_right_hand_skeletal_poses` | method | `get_right_hand_skeletal_poses(self) -> list[sapien.Pose]` | Get right hand skeletal poses. |  |
| `get_root_pose` | method | `get_root_pose(self) -> sapien.Pose` | Get root pose. |  |
| `render` | method | `render(self) -> None` | Call render. |  |
| `set_camera_parameters` | method | `set_camera_parameters(self, near: float, far: float) -> None` | Set camera parameters. |  |
| `set_root_pose` | method | `set_root_pose(self, pose: sapien.Pose) -> None` | Set root pose. |  |
| `set_scene` | method | `set_scene(self, scene: sapien.Scene) -> None` | Set scene. |  |
| `update_render` | method | `update_render(self) -> None` | update_render implicitly calls fetch_poses to make sure the HMD pose is up-to-date | GPU PhysX dynamic bodies read CPU poses; prefer the CUDA pose path for offscreen. |
| `__init__` | method | `__init__(self) -> None` | Python special method. |  |
| `_internal_scene` | property | `_internal_scene(self) -> sapien.internal_renderer.Scene` | Property: internal scene. |  |

## `sapien.render.RenderWindow`

- Use: SAPIEN render API object.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `cursor` | `bool` | Property: cursor. |  |
| `denoiser` | `Literal['none', 'oidn', 'optix']` | Property: denoiser. |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `alt` | property | `alt(self) -> bool` | Property: alt. |  |
| `camera_mode` | property | `camera_mode(self) -> Literal['perspective', 'orthographic']` | Property: camera mode. |  |
| `ctrl` | property | `ctrl(self) -> bool` | Property: ctrl. |  |
| `display_picture_names` | property | `display_picture_names(self) -> list[str]` | Names for available display targets that can be displayed in the render function |  |
| `far` | property | `far(self) -> float` | Property: far. |  |
| `fovy` | property | `fovy(self) -> float` | Property: fovy. |  |
| `fps` | property | `fps(self) -> float` | Property: fps. |  |
| `get_camera_model_matrix` | method | `get_camera_model_matrix(self) -> np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float32]]` | Get camera model matrix. |  |
| `get_camera_pose` | method | `get_camera_pose(self) -> sapien.Pose` | Get camera pose. |  |
| `get_camera_position` | method | `get_camera_position(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | Get camera position. |  |
| `get_camera_projection_matrix` | method | `get_camera_projection_matrix(self) -> np.ndarray[Any, np.dtype[np.float32]]` | Get camera projection matrix. |  |
| `get_camera_property_float` | method | `get_camera_property_float(self, key: str) -> float` | Get camera property float. |  |
| `get_camera_property_int` | method | `get_camera_property_int(self, key: str) -> int` | Get camera property int. |  |
| `get_camera_rotation` | method | `get_camera_rotation(self) -> np.ndarray[Literal[4], np.dtype[np.float32]]` | Get camera rotation. |  |
| `get_content_scale` | method | `get_content_scale(self) -> float` | Get content scale. |  |
| `get_picture` | method | `get_picture(self, name: str) -> np.ndarray[Any, np.dtype[Any]]` | Get picture. | GPU PhysX dynamic bodies read CPU poses; prefer the CUDA pose path for offscreen. |
| `get_picture_pixel` | method | `get_picture_pixel(self, name: str, x: int, y: int) -> np.ndarray[Any, np.dtype[Any]]` | Get picture pixel. |  |
| `get_picture_size` | method | `get_picture_size(self, name: str) -> Annotated[list[int], FixedSize(2)]` | Get picture size. |  |
| `hide` | method | `hide(self) -> None` | Call hide. |  |
| `key_down` | method | `key_down(self, key: str) -> bool` | Call key down. |  |
| `key_press` | method | `key_press(self, key: str) -> bool` | Call key press. |  |
| `mouse_click` | method | `mouse_click(self, key: int) -> bool` | Call mouse click. |  |
| `mouse_delta` | property | `mouse_delta(self) -> Annotated[list[float], FixedSize(2)]` | Property: mouse delta. |  |
| `mouse_down` | method | `mouse_down(self, key: int) -> bool` | Call mouse down. |  |
| `mouse_position` | property | `mouse_position(self) -> Annotated[list[float], FixedSize(2)]` | Property: mouse position. |  |
| `mouse_wheel_delta` | property | `mouse_wheel_delta(self) -> Annotated[list[float], FixedSize(2)]` | Property: mouse wheel delta. |  |
| `near` | property | `near(self) -> float` | Property: near. |  |
| `ortho_top` | property | `ortho_top(self) -> float` | Property: ortho top. |  |
| `render` | method | `render(self, target_name: str, ui_windows: list[sapien.internal_renderer.UIWidget]=[]) -> None` | Call render. |  |
| `resize` | method | `resize(self, width: int, height: int) -> None` | Call resize. |  |
| `set_camera_orthographic_parameters` | method | `set_camera_orthographic_parameters(self, near: float, far: float, top: float) -> None` | Set camera orthographic parameters. |  |
| `set_camera_parameters` | method | `set_camera_parameters(self, near: float, far: float, fovy: float) -> None` | Set camera parameters. |  |
| `set_camera_pose` | method | `set_camera_pose(self, pose: sapien.Pose) -> None` | Set camera pose. |  |
| `set_camera_position` | method | `set_camera_position(self, position: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | Set camera position. |  |
| `set_camera_property` | method | `set_camera_property(self, key: str, value: float) -> None<br>set_camera_property(self, key: str, value: int) -> None` | Set camera property. |  |
| `set_camera_rotation` | method | `set_camera_rotation(self, quat: np.ndarray[Literal[4], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | Set camera rotation. |  |
| `set_camera_texture` | method | `set_camera_texture(self, name: str, texture: RenderTexture2D) -> None` | Set camera texture. |  |
| `set_camera_texture_array` | method | `set_camera_texture_array(self, name: str, textures: list[RenderTexture2D]) -> None` | Set camera texture array. |  |
| `set_content_scale` | method | `set_content_scale(self, scale: float) -> None` | Set content scale. |  |
| `set_drop_callback` | method | `set_drop_callback(self, callback: Callable[[list[str]], None]) -> None` | Set drop callback. |  |
| `set_focus_callback` | method | `set_focus_callback(self, callback: Callable[[int], None]) -> None` | Set focus callback. |  |
| `set_intrinsic_parameters` | method | `set_intrinsic_parameters(self, near: float, far: float, fx: float, fy: float, cx: float, cy: float, skew: float) -> None` | Set intrinsic parameters. |  |
| `set_scene` | method | `set_scene(self, scene: sapien.Scene) -> None` | Set scene. |  |
| `set_scenes` | method | `set_scenes(self, scenes: list[sapien.Scene], offsets: list[np.ndarray[Literal[3], np.dtype[np.float32]]]) -> None` | Set scenes. |  |
| `set_shader_dir` | method | `set_shader_dir(self, shader_dir: str) -> None` | Set shader dir. |  |
| `shift` | property | `shift(self) -> bool` | Property: shift. |  |
| `should_close` | property | `should_close(self) -> bool` | Property: should close. |  |
| `show` | method | `show(self) -> None` | Call show. |  |
| `size` | property | `size(self) -> Annotated[list[int], FixedSize(2)]` | Property: size. |  |
| `super` | property | `super(self) -> bool` | Property: super. |  |
| `unset_drop_callback` | method | `unset_drop_callback(self) -> None` | Call unset drop callback. |  |
| `unset_focus_callback` | method | `unset_focus_callback(self) -> None` | Call unset focus callback. |  |
| `update_render` | method | `update_render(self) -> None` | Equivalent to calling the update_render function for all added scene | GPU PhysX dynamic bodies read CPU poses; prefer the CUDA pose path for offscreen. |
| `__init__` | method | `__init__(self, width: int, height: int, shader_dir: str) -> None` | Python special method. |  |
| `_internal_renderer` | property | `_internal_renderer(self) -> sapien.internal_renderer.Renderer` | Property: internal renderer. |  |
| `_internal_scene` | property | `_internal_scene(self) -> sapien.internal_renderer.Scene` | Property: internal scene. |  |

## `sapien.render.SapienRenderer`

- Use: SAPIEN API object.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__init__` | method | `__init__(self, device: sapien.Device \| None=None) -> None` | Python special method. |  |
| `_internal_context` | property | `_internal_context(self) -> sapien.internal_renderer.Context` | Property: internal context. |  |