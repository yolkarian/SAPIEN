# Render API (`sapien.render`)

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.render`
- Source files: `python/py_package/pysapien/render.pyi`
- Notes: Only instantiate RenderSystem for viewer/sensors/offscreen rendering. For GPU PhysX dynamic bodies prefer RenderSystemGroup + CUDA poses.

## Functions

| API | Signature | Use | Notes |
|---|---|---|---|
| `sapien.render.clear_cache` | `clear_cache(models: bool=True, images: bool=True, shaders: bool=False) -> None` | 调用 clear cache。 |  |
| `sapien.render.enable_vr` | `enable_vr() -> None` | Enable VR via Steam. Must be called before creating RenderSystem or sapien Scene. |  |
| `sapien.render.get_camera_shader_dir` | `get_camera_shader_dir() -> str` | 读取 camera shader dir。 |  |
| `sapien.render.get_device_summary` | `get_device_summary() -> str` | 读取 device summary。 |  |
| `sapien.render.get_imgui_ini_filename` | `get_imgui_ini_filename() -> str` | 读取 imgui ini filename。 |  |
| `sapien.render.get_msaa` | `get_msaa() -> int` | 读取 msaa。 |  |
| `sapien.render.get_ray_tracing_denoiser` | `get_ray_tracing_denoiser() -> Literal['none', 'oidn', 'optix']` | 读取 ray tracing denoiser。 |  |
| `sapien.render.get_ray_tracing_dof_aperture` | `get_ray_tracing_dof_aperture() -> float` | 读取 ray tracing dof aperture。 |  |
| `sapien.render.get_ray_tracing_dof_plane` | `get_ray_tracing_dof_plane() -> float` | 读取 ray tracing dof plane。 |  |
| `sapien.render.get_ray_tracing_path_depth` | `get_ray_tracing_path_depth() -> int` | 读取 ray tracing path depth。 |  |
| `sapien.render.get_ray_tracing_samples_per_pixel` | `get_ray_tracing_samples_per_pixel() -> int` | 读取 ray tracing samples per pixel。 |  |
| `sapien.render.get_viewer_shader_dir` | `get_viewer_shader_dir() -> str` | 读取 viewer shader dir。 |  |
| `sapien.render.get_vr_action_manifest_filename` | `get_vr_action_manifest_filename() -> str` | 读取 vr action manifest filename。 |  |
| `sapien.render.get_vr_enabled` | `get_vr_enabled() -> bool` | 读取 vr enabled。 |  |
| `sapien.render.load_scene` | `load_scene(filename: str, apply_scale: bool=True) -> RenderSceneLoaderNode` | 加载 scene。 |  |
| `sapien.render.set_camera_shader_dir` | `set_camera_shader_dir(dir: str) -> None` | 设置 camera shader dir。 |  |
| `sapien.render.set_global_config` | `set_global_config(max_num_materials: int=128, max_num_textures: int=512, default_mipmap_levels: int=1, do_not_load_texture: bool=False) -> None` | Sets global properties for SAPIEN renderer. This function should only be called before creating any renderer-related objects. |  |
| `sapien.render.set_imgui_ini_filename` | `set_imgui_ini_filename(filename: str) -> None` | 设置 imgui ini filename。 |  |
| `sapien.render.set_log_level` | `set_log_level(level: str) -> None` | 设置 log level。 |  |
| `sapien.render.set_msaa` | `set_msaa(msaa: int) -> None` | 设置 msaa。 |  |
| `sapien.render.set_picture_format` | `set_picture_format(name: str, format: str) -> None` | 设置 picture format。 |  |
| `sapien.render.set_ray_tracing_denoiser` | `set_ray_tracing_denoiser(name: Literal['none', 'oidn', 'optix']) -> None` | 设置 ray tracing denoiser。 |  |
| `sapien.render.set_ray_tracing_dof_aperture` | `set_ray_tracing_dof_aperture(radius: float) -> None` | 设置 ray tracing dof aperture。 |  |
| `sapien.render.set_ray_tracing_dof_plane` | `set_ray_tracing_dof_plane(depth: float) -> None` | 设置 ray tracing dof plane。 |  |
| `sapien.render.set_ray_tracing_path_depth` | `set_ray_tracing_path_depth(depth: int) -> None` | 设置 ray tracing path depth。 |  |
| `sapien.render.set_ray_tracing_samples_per_pixel` | `set_ray_tracing_samples_per_pixel(spp: int) -> None` | 设置 ray tracing samples per pixel。 |  |
| `sapien.render.set_viewer_shader_dir` | `set_viewer_shader_dir(dir: str) -> None` | 设置 viewer shader dir。 |  |
| `sapien.render.set_vr_action_manifest_filename` | `set_vr_action_manifest_filename(filename: str) -> None` | 设置 vr action manifest filename。 |  |

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.render.RenderBodyComponent` | `sapien.Component` | 渲染 body；挂 RenderShape。 |  |
| `sapien.render.RenderCameraComponent` | `sapien.Component` | 相机组件；take_picture 后读取图片/ CUDA 图片。 |  |
| `sapien.render.RenderCameraGroup` |  | batched camera group；多相机一次 take/read CUDA。 |  |
| `sapien.render.RenderCubemap` |  | 环境 cubemap。 |  |
| `sapien.render.RenderCudaMeshComponent` | `sapien.Component` | SAPIEN 渲染 API 对象。 |  |
| `sapien.render.RenderDirectionalLightComponent` | `RenderLightComponent` | 具体渲染灯光组件。 |  |
| `sapien.render.RenderLightComponent` | `sapien.Component` | 灯光组件基类。 |  |
| `sapien.render.RenderMaterial` |  | PBR 渲染材质。 |  |
| `sapien.render.RenderParallelogramLightComponent` | `RenderLightComponent` | 具体渲染灯光组件。 |  |
| `sapien.render.RenderPointCloudComponent` | `sapien.Component` | SAPIEN 渲染 API 对象。 |  |
| `sapien.render.RenderPointLightComponent` | `RenderLightComponent` | 具体渲染灯光组件。 |  |
| `sapien.render.RenderSceneLoaderNode` |  | SAPIEN 渲染 API 对象。 |  |
| `sapien.render.RenderShape` |  | 渲染形状基类；支持 GPU pose batch index。 |  |
| `sapien.render.RenderShapeBox` | `RenderShapePrimitive` | 具体渲染形状。 |  |
| `sapien.render.RenderShapeCapsule` | `RenderShapePrimitive` | 具体渲染形状。 |  |
| `sapien.render.RenderShapeCylinder` | `RenderShapePrimitive` | 具体渲染形状。 |  |
| `sapien.render.RenderShapePlane` | `RenderShapePrimitive` | 具体渲染形状。 |  |
| `sapien.render.RenderShapePrimitive` | `RenderShape` | 具体渲染形状。 |  |
| `sapien.render.RenderShapeSphere` | `RenderShapePrimitive` | 具体渲染形状。 |  |
| `sapien.render.RenderShapeTriangleMesh` | `RenderShape` | 具体渲染形状。 |  |
| `sapien.render.RenderShapeTriangleMeshPart` |  | 具体渲染形状。 |  |
| `sapien.render.RenderSpotLightComponent` | `RenderLightComponent` | 具体渲染灯光组件。 |  |
| `sapien.render.RenderSystem` | `sapien.System` | 渲染系统；只有 viewer/sensor/offscreen 需要。 | create only when rendering/viewer/sensors needed |
| `sapien.render.RenderSystemGroup` |  | 多 RenderSystem batched 渲染组；可绑定 CUDA poses。 |  |
| `sapien.render.RenderTexture` |  | SAPIEN 渲染 API 对象。 |  |
| `sapien.render.RenderTexture2D` |  | 2D 渲染纹理。 |  |
| `sapien.render.RenderTexturedLightComponent` | `RenderSpotLightComponent` | 具体渲染灯光组件。 |  |
| `sapien.render.RenderVRDisplay` |  | SAPIEN 渲染 API 对象。 |  |
| `sapien.render.RenderWindow` |  | SAPIEN 渲染 API 对象。 |  |
| `sapien.render.SapienRenderer` |  | SAPIEN API 对象。 |  |

## `sapien.render.RenderBodyComponent`

- Use: 渲染 body；挂 RenderShape。
- Bases: `sapien.Component`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `shading_mode` | `int` | 属性：shading mode。 |  |
| `visibility` | `float` | 属性：visibility。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `attach` | method | `attach(self, shape: RenderShape) -> RenderBodyComponent` | 调用 attach。 |  |
| `clone` | method | `clone(self) -> RenderBodyComponent` | 调用 clone。 |  |
| `compute_global_aabb_tight` | method | `compute_global_aabb_tight(self) -> np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.float32]]` | 计算 global aabb tight。 |  |
| `disable_render_id` | method | `disable_render_id(self) -> None` | 禁用 render id。 |  |
| `enable_render_id` | method | `enable_render_id(self) -> None` | 启用 render id。 |  |
| `get_global_aabb_fast` | method | `get_global_aabb_fast(self) -> np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.float32]]` | 读取 global aabb fast。 |  |
| `is_render_id_disabled` | property | `is_render_id_disabled(self) -> bool` | 布尔状态：render id disabled。 |  |
| `render_shapes` | property | `render_shapes(self) -> list[RenderShape]` | 属性：render shapes。 |  |
| `set_property` | method | `set_property(self, name: str, value: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None<br>set_property(self, name: str, value: float) -> None<br>set_property(self, name: str, value: int) -> None` | 设置 property。 |  |
| `set_texture` | method | `set_texture(self, name: str, texture: RenderTexture) -> None` | 设置 texture。 |  |
| `set_texture_array` | method | `set_texture_array(self, name: str, textures: list[RenderTexture]) -> None` | 设置 texture array。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |
| `_internal_node` | property | `_internal_node(self) -> sapien.internal_renderer.Node` | 属性： internal node。 |  |

## `sapien.render.RenderCameraComponent`

- Use: 相机组件；take_picture 后读取图片/ CUDA 图片。
- Bases: `sapien.Component`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `far` | `float` | 属性：far。 |  |
| `local_pose` | `sapien.Pose` | 属性：local pose。 |  |
| `near` | `float` | 属性：near。 |  |
| `skew` | `float` | 属性：skew。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `cx` | property | `cx(self) -> float` | 属性：cx。 |  |
| `cy` | property | `cy(self) -> float` | 属性：cy。 |  |
| `fovx` | property | `fovx(self) -> float` | 属性：fovx。 |  |
| `fovy` | property | `fovy(self) -> float` | 属性：fovy。 |  |
| `fx` | property | `fx(self) -> float` | 属性：fx。 |  |
| `fy` | property | `fy(self) -> float` | 属性：fy。 |  |
| `get_extrinsic_matrix` | method | `get_extrinsic_matrix(self) -> np.ndarray[tuple[Literal[3], Literal[4]], np.dtype[np.float32]]` | Get 3x4 extrinsic camera matrix in OpenCV format. |  |
| `get_far` | method | `get_far(self) -> float` | 读取 far。 |  |
| `get_global_pose` | method | `get_global_pose(self) -> sapien.Pose` | 读取 global pose。 |  |
| `get_height` | method | `get_height(self) -> int` | 读取 height。 |  |
| `get_intrinsic_matrix` | method | `get_intrinsic_matrix(self) -> np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float32]]` | Get 3x3 intrinsic camera matrix in OpenCV format. |  |
| `get_local_pose` | method | `get_local_pose(self) -> sapien.Pose` | 读取 local pose。 |  |
| `get_mode` | method | `get_mode(self) -> Literal['perspective', 'orthographic']` | 读取 mode。 |  |
| `get_model_matrix` | method | `get_model_matrix(self) -> np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float32]]` | Get model matrix (inverse of extrinsic matrix) used in rendering (Y up, Z back) |  |
| `get_near` | method | `get_near(self) -> float` | 读取 near。 |  |
| `get_picture` | method | `get_picture(self, name: str) -> np.ndarray[Any, np.dtype[Any]]` | 读取 picture。 | GPU PhysX 动态体会读 CPU pose；offscreen 优先 CUDA pose path。 |
| `get_picture_cuda` | method | `get_picture_cuda(self, name: str) -> sapien.CudaArray` | This function transfers the rendered image into a CUDA buffer. Usage: # use torch backend sapien.set_cuda_tensor_backend("torch") # called once per process image: torc... | direct GPU render path；无需 sync_poses_gpu_to_cpu。 |
| `get_picture_names` | method | `get_picture_names(self) -> list[str]` | 读取 picture names。 |  |
| `get_projection_matrix` | method | `get_projection_matrix(self) -> np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float32]]` | Get projection matrix in used in rendering (right-handed NDC with [-1,1] XY and [0,1] Z) |  |
| `get_skew` | method | `get_skew(self) -> float` | 读取 skew。 |  |
| `get_width` | method | `get_width(self) -> int` | 读取 width。 |  |
| `global_pose` | property | `global_pose(self) -> sapien.Pose` | 属性：global pose。 |  |
| `height` | property | `height(self) -> int` | 属性：height。 |  |
| `mode` | property | `mode(self) -> Literal['perspective', 'orthographic']` | 属性：mode。 |  |
| `ortho_bottom` | property | `ortho_bottom(self) -> float` | 属性：ortho bottom。 |  |
| `ortho_left` | property | `ortho_left(self) -> float` | 属性：ortho left。 |  |
| `ortho_right` | property | `ortho_right(self) -> float` | 属性：ortho right。 |  |
| `ortho_top` | property | `ortho_top(self) -> float` | 属性：ortho top。 |  |
| `set_far` | method | `set_far(self, far: float) -> None` | 设置 far。 |  |
| `set_focal_lengths` | method | `set_focal_lengths(self, fx: float, fy: float) -> None` | 设置 focal lengths。 |  |
| `set_fovx` | method | `set_fovx(self, fov: float, compute_y: bool=True) -> None` | 设置 fovx。 |  |
| `set_fovy` | method | `set_fovy(self, fov: float, compute_x: bool=True) -> None` | 设置 fovy。 |  |
| `set_gpu_pose_batch_index` | method | `set_gpu_pose_batch_index(self, index: int) -> None` | 绑定 GPU pose index 供 batched/direct GPU 渲染。 | direct GPU render path；无需 sync_poses_gpu_to_cpu。 |
| `set_local_pose` | method | `set_local_pose(self, pose: sapien.Pose) -> None` | 设置 local pose。 |  |
| `set_near` | method | `set_near(self, near: float) -> None` | 设置 near。 |  |
| `set_orthographic_parameters` | method | `set_orthographic_parameters(self, near: float, far: float, top: float) -> None<br>set_orthographic_parameters(self, near: float, far: float, left: float, right: float, bottom: float, top: float) -> None` | 设置 orthographic parameters。 |  |
| `set_perspective_parameters` | method | `set_perspective_parameters(self, near: float, far: float, fx: float, fy: float, cx: float, cy: float, skew: float) -> None` | 设置 perspective parameters。 |  |
| `set_principal_point` | method | `set_principal_point(self, cx: float, cy: float) -> None` | 设置 principal point。 |  |
| `set_property` | method | `set_property(self, name: str, value: float) -> None<br>set_property(self, name: str, value: int) -> None` | 设置 property。 |  |
| `set_skew` | method | `set_skew(self, skew: float) -> None` | 设置 skew。 |  |
| `set_texture` | method | `set_texture(self, name: str, texture: RenderTexture) -> None` | 设置 texture。 |  |
| `set_texture_array` | method | `set_texture_array(self, name: str, textures: list[RenderTexture]) -> None` | 设置 texture array。 |  |
| `take_picture` | method | `take_picture(self) -> None` | 触发相机/相机组渲染。 |  |
| `width` | property | `width(self) -> int` | 属性：width。 |  |
| `__init__` | method | `__init__(self, width: int, height: int, shader_dir: str='') -> None` | Python special method。 |  |
| `_cuda_buffer` | property | `_cuda_buffer(self) -> sapien.CudaArray` | Debug only. Get the CUDA buffer containing GPU data for this camera, including transformaion matrices, sizes, and user-defined shader fields. |  |
| `_internal_renderer` | property | `_internal_renderer(self) -> sapien.internal_renderer.Renderer` | 属性： internal renderer。 |  |

## `sapien.render.RenderCameraGroup`

- Use: batched camera group；多相机一次 take/read CUDA。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_picture_cuda` | method | `get_picture_cuda(self, name: str) -> sapien.CudaArray` | 读取 CUDA 图片缓冲；避免 CPU copy。 | direct GPU render path；无需 sync_poses_gpu_to_cpu。 |
| `take_picture` | method | `take_picture(self) -> None` | 触发相机/相机组渲染。 |  |

## `sapien.render.RenderCubemap`

- Use: 环境 cubemap。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `export` | method | `export(self, filename: str) -> None` | 调用 export。 |  |
| `__init__` | method | `__init__(self, filename: str) -> None<br>__init__(self, px: str, nx: str, py: str, ny: str, pz: str, nz: str) -> None` | Python special method。 |  |
| `_internal_cubemap` | property | `_internal_cubemap(self) -> sapien.internal_renderer.Cubemap` | 属性： internal cubemap。 |  |

## `sapien.render.RenderCudaMeshComponent`

- Use: SAPIEN 渲染 API 对象。
- Bases: `sapien.Component`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `material` | `RenderMaterial` | 属性：material。 |  |
| `triangle_count` | `int` | 属性：triangle count。 |  |
| `vertex_count` | `int` | 属性：vertex count。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `cuda_triangles` | property | `cuda_triangles(self) -> sapien.CudaArray` | CUDA 状态/渲染缓冲属性。 | gpu_init 后取 view；循环中复用 torch/cupy/jax view。 |
| `cuda_vertices` | property | `cuda_vertices(self) -> sapien.CudaArray` | CUDA 状态/渲染缓冲属性。 | gpu_init 后取 view；循环中复用 torch/cupy/jax view。 |
| `get_cuda_triangles` | method | `get_cuda_triangles(self) -> sapien.CudaArray` | 读取 cuda triangles。 |  |
| `get_cuda_vertices` | method | `get_cuda_vertices(self) -> sapien.CudaArray` | 读取 cuda vertices。 |  |
| `get_material` | method | `get_material(self) -> RenderMaterial` | 读取 material。 |  |
| `get_triangle_count` | method | `get_triangle_count(self) -> int` | 读取 triangle count。 |  |
| `get_vertex_count` | method | `get_vertex_count(self) -> int` | 读取 vertex count。 |  |
| `notify_vertex_updated` | method | `notify_vertex_updated(self, cuda_stream: int=0) -> None` | 调用 notify vertex updated。 |  |
| `set_material` | method | `set_material(self, material: RenderMaterial) -> None` | 设置 material。 |  |
| `set_triangle_count` | method | `set_triangle_count(self, count: int) -> None` | 设置 triangle count。 |  |
| `set_triangles` | method | `set_triangles(self, triangles: np.ndarray[np.uint32[M, 3]]) -> None` | 设置 triangles。 |  |
| `set_vertex_count` | method | `set_vertex_count(self, count: int) -> None` | 设置 vertex count。 |  |
| `__init__` | method | `__init__(self, max_vertex_count: int, max_triangle_count: int) -> None` | Python special method。 |  |

## `sapien.render.RenderDirectionalLightComponent`

- Use: 具体渲染灯光组件。
- Bases: `RenderLightComponent`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `shadow_half_size` | `float` | 属性：shadow half size。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_shadow_half_size` | method | `get_shadow_half_size(self) -> float` | 读取 shadow half size。 |  |
| `set_shadow_half_size` | method | `set_shadow_half_size(self, size: float) -> None` | 设置 shadow half size。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.render.RenderLightComponent`

- Use: 灯光组件基类。
- Bases: `sapien.Component`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `color` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：color。 |  |
| `local_pose` | `sapien.Pose` | 属性：local pose。 |  |
| `shadow` | `bool` | 属性：shadow。 |  |
| `shadow_far` | `float` | 属性：shadow far。 |  |
| `shadow_map_size` | `int` | 属性：shadow map size。 |  |
| `shadow_near` | `float` | 属性：shadow near。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `disable_shadow` | method | `disable_shadow(self) -> None` | 禁用 shadow。 |  |
| `enable_shadow` | method | `enable_shadow(self) -> None` | 启用 shadow。 |  |
| `get_color` | method | `get_color(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 读取 color。 |  |
| `get_global_pose` | method | `get_global_pose(self) -> sapien.Pose` | 读取 global pose。 |  |
| `get_local_pose` | method | `get_local_pose(self) -> sapien.Pose` | 读取 local pose。 |  |
| `get_shadow_far` | method | `get_shadow_far(self) -> float` | 读取 shadow far。 |  |
| `get_shadow_map_size` | method | `get_shadow_map_size(self) -> int` | 读取 shadow map size。 |  |
| `get_shadow_near` | method | `get_shadow_near(self) -> float` | 读取 shadow near。 |  |
| `global_pose` | property | `global_pose(self) -> sapien.Pose` | 属性：global pose。 |  |
| `set_color` | method | `set_color(self, color: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | 设置 color。 |  |
| `set_local_pose` | method | `set_local_pose(self, pose: sapien.Pose) -> None` | 设置 local pose。 |  |
| `set_shadow_far` | method | `set_shadow_far(self, far: float) -> None` | 设置 shadow far。 |  |
| `set_shadow_map_size` | method | `set_shadow_map_size(self, size: int) -> None` | 设置 shadow map size。 |  |
| `set_shadow_near` | method | `set_shadow_near(self, near: float) -> None` | 设置 shadow near。 |  |

## `sapien.render.RenderMaterial`

- Use: PBR 渲染材质。
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `base_color` | `Annotated[list[float], FixedSize(4)]` | 属性：base color。 |  |
| `base_color_texture` | `RenderTexture2D` | 属性：base color texture。 |  |
| `diffuse_texture` | `` | 属性：diffuse texture。 |  |
| `emission` | `Annotated[list[float], FixedSize(4)]` | 属性：emission。 |  |
| `emission_texture` | `RenderTexture2D` | 属性：emission texture。 |  |
| `ior` | `float` | 属性：ior。 |  |
| `metallic` | `float` | 属性：metallic。 |  |
| `metallic_texture` | `RenderTexture2D` | 属性：metallic texture。 |  |
| `normal_texture` | `RenderTexture2D` | 属性：normal texture。 |  |
| `roughness` | `float` | 属性：roughness。 |  |
| `roughness_texture` | `RenderTexture2D` | 属性：roughness texture。 |  |
| `specular` | `float` | 属性：specular。 |  |
| `transmission` | `float` | 属性：transmission。 |  |
| `transmission_roughness` | `float` | 属性：transmission roughness。 |  |
| `transmission_texture` | `RenderTexture2D` | 属性：transmission texture。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_base_color` | method | `get_base_color(self) -> Annotated[list[float], FixedSize(4)]` | 读取 base color。 |  |
| `get_base_color_texture` | method | `get_base_color_texture(self) -> RenderTexture2D` | 读取 base color texture。 |  |
| `get_diffuse_texture` | method | `get_diffuse_texture(self)` | 读取 diffuse texture。 |  |
| `get_emission` | method | `get_emission(self) -> Annotated[list[float], FixedSize(4)]` | 读取 emission。 |  |
| `get_emission_texture` | method | `get_emission_texture(self) -> RenderTexture2D` | 读取 emission texture。 |  |
| `get_ior` | method | `get_ior(self) -> float` | 读取 ior。 |  |
| `get_metallic` | method | `get_metallic(self) -> float` | 读取 metallic。 |  |
| `get_metallic_texture` | method | `get_metallic_texture(self) -> RenderTexture2D` | 读取 metallic texture。 |  |
| `get_normal_texture` | method | `get_normal_texture(self) -> RenderTexture2D` | 读取 normal texture。 |  |
| `get_roughness` | method | `get_roughness(self) -> float` | 读取 roughness。 |  |
| `get_roughness_texture` | method | `get_roughness_texture(self) -> RenderTexture2D` | 读取 roughness texture。 |  |
| `get_specular` | method | `get_specular(self) -> float` | 读取 specular。 |  |
| `get_transmission` | method | `get_transmission(self) -> float` | 读取 transmission。 |  |
| `get_transmission_roughness` | method | `get_transmission_roughness(self) -> float` | 读取 transmission roughness。 |  |
| `get_transmission_texture` | method | `get_transmission_texture(self) -> RenderTexture2D` | 读取 transmission texture。 |  |
| `set_base_color` | method | `set_base_color(self, color: Annotated[list[float], FixedSize(4)]) -> None` | 设置 base color。 |  |
| `set_base_color_texture` | method | `set_base_color_texture(self, texture: RenderTexture2D) -> None` | 设置 base color texture。 |  |
| `set_diffuse_texture` | method | `set_diffuse_texture(self, texture)` | 设置 diffuse texture。 |  |
| `set_emission` | method | `set_emission(self, emission: Annotated[list[float], FixedSize(4)]) -> None` | 设置 emission。 |  |
| `set_emission_texture` | method | `set_emission_texture(self, texture: RenderTexture2D) -> None` | 设置 emission texture。 |  |
| `set_ior` | method | `set_ior(self, ior: float) -> None` | 设置 ior。 |  |
| `set_metallic` | method | `set_metallic(self, metallic: float) -> None` | 设置 metallic。 |  |
| `set_metallic_texture` | method | `set_metallic_texture(self, texture: RenderTexture2D) -> None` | 设置 metallic texture。 |  |
| `set_normal_texture` | method | `set_normal_texture(self, texture: RenderTexture2D) -> None` | 设置 normal texture。 |  |
| `set_roughness` | method | `set_roughness(self, roughness: float) -> None` | 设置 roughness。 |  |
| `set_roughness_texture` | method | `set_roughness_texture(self, texture: RenderTexture2D) -> None` | 设置 roughness texture。 |  |
| `set_specular` | method | `set_specular(self, specular: float) -> None` | 设置 specular。 |  |
| `set_transmission` | method | `set_transmission(self, transmission: float) -> None` | 设置 transmission。 |  |
| `set_transmission_roughness` | method | `set_transmission_roughness(self, roughness: float) -> None` | 设置 transmission roughness。 |  |
| `set_transmission_texture` | method | `set_transmission_texture(self, texture: RenderTexture2D) -> None` | 设置 transmission texture。 |  |
| `__eq__` | method | `__eq__(self, arg0: RenderMaterial) -> bool` | Python special method。 |  |
| `__init__` | method | `__init__(self, emission: Annotated[list[float], FixedSize(4)]=[0.0, 0.0, 0.0, 0.0], base_color: Annotated[list[float], FixedSize(4)]=[1.0, 1.0, 1.0, 1.0], specular: float=0.0, roughness: float=1.0, metallic: float=0.0, transmission: float=0.0, ior: float=1.4500000476837158, transmission_roughness: float=0.0) -> None` | Python special method。 |  |

## `sapien.render.RenderParallelogramLightComponent`

- Use: 具体渲染灯光组件。
- Bases: `RenderLightComponent`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `angle` | property | `angle(self) -> float` | 属性：angle。 |  |
| `get_angle` | method | `get_angle(self) -> float` | 读取 angle。 |  |
| `get_half_height` | method | `get_half_height(self) -> float` | 读取 half height。 |  |
| `get_half_width` | method | `get_half_width(self) -> float` | 读取 half width。 |  |
| `half_height` | property | `half_height(self) -> float` | 属性：half height。 |  |
| `half_width` | property | `half_width(self) -> float` | 属性：half width。 |  |
| `set_shape` | method | `set_shape(self, half_width: float, half_height: float, angle: float=1.5707963705062866) -> None` | 设置 shape。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.render.RenderPointCloudComponent`

- Use: SAPIEN 渲染 API 对象。
- Bases: `sapien.Component`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_cuda_aabb` | method | `get_cuda_aabb(self) -> sapien.CudaArray \| None` | this function is a temporary hack to help update the AABBs used for ray tracing BLAS. returns None if ray tracing has not been initialized |  |
| `get_cuda_vertices` | method | `get_cuda_vertices(self) -> sapien.CudaArray` | 读取 cuda vertices。 |  |
| `get_vertices` | method | `get_vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | Get previously set vertices. This function does not reflect any changes directly made to the GPU. |  |
| `set_attribute` | method | `set_attribute(self, name: str, attribute: np.ndarray[tuple[M, N], np.dtype[np.float32]] \| list \| tuple) -> RenderPointCloudComponent` | 设置 attribute。 |  |
| `set_vertices` | method | `set_vertices(self, vertices: np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]] \| list \| tuple) -> RenderPointCloudComponent` | 设置 vertices。 |  |
| `__init__` | method | `__init__(self, capacity: int=0) -> None` | Python special method。 |  |

## `sapien.render.RenderPointLightComponent`

- Use: 具体渲染灯光组件。
- Bases: `RenderLightComponent`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.render.RenderSceneLoaderNode`

- Use: SAPIEN 渲染 API 对象。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `children` | property | `children(self) -> list[RenderSceneLoaderNode]` | 属性：children。 |  |
| `flatten` | method | `flatten(self) -> tuple[list[RenderShapeTriangleMesh], list[RenderLightComponent]]` | 调用 flatten。 |  |
| `light` | property | `light(self) -> RenderLightComponent` | 属性：light。 |  |
| `mesh` | property | `mesh(self) -> RenderShapeTriangleMesh` | 属性：mesh。 |  |
| `name` | property | `name(self) -> str` | 属性：name。 |  |
| `pose` | property | `pose(self) -> sapien.Pose` | 属性：pose。 |  |
| `scale` | property | `scale(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：scale。 |  |

## `sapien.render.RenderShape`

- Use: 渲染形状基类；支持 GPU pose batch index。
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `front_face` | `Literal['counterclockwise', 'clockwise']` | 属性：front face。 |  |
| `local_pose` | `sapien.Pose` | 属性：local pose。 |  |
| `name` | `str` | 属性：name。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `clone` | method | `clone(self) -> RenderShape` | 调用 clone。 |  |
| `get_front_face` | method | `get_front_face(self) -> Literal['counterclockwise', 'clockwise']` | 读取 front face。 |  |
| `get_local_pose` | method | `get_local_pose(self) -> sapien.Pose` | 读取 local pose。 |  |
| `get_material` | method | `get_material(self) -> RenderMaterial` | 读取 material。 |  |
| `get_name` | method | `get_name(self) -> str` | 读取 name。 |  |
| `get_parts` | method | `get_parts(self) -> list[RenderShapeTriangleMeshPart]` | 读取 parts。 |  |
| `get_per_scene_id` | method | `get_per_scene_id(self) -> int` | 读取 per scene id。 |  |
| `material` | property | `material(self) -> RenderMaterial` | 属性：material。 |  |
| `parts` | property | `parts(self) -> list[RenderShapeTriangleMeshPart]` | 属性：parts。 |  |
| `per_scene_id` | property | `per_scene_id(self) -> int` | 属性：per scene id。 |  |
| `set_front_face` | method | `set_front_face(self, front_face: Literal['counterclockwise', 'clockwise']) -> None` | 设置 front face。 |  |
| `set_gpu_pose_batch_index` | method | `set_gpu_pose_batch_index(self, index: int) -> None` | 绑定 GPU pose index 供 batched/direct GPU 渲染。 | direct GPU render path；无需 sync_poses_gpu_to_cpu。 |
| `set_local_pose` | method | `set_local_pose(self, pose: sapien.Pose) -> None` | 设置 local pose。 |  |
| `set_name` | method | `set_name(self, name: str) -> None` | 设置 name。 |  |

## `sapien.render.RenderShapeBox`

- Use: 具体渲染形状。
- Bases: `RenderShapePrimitive`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_half_size` | method | `get_half_size(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 读取 half size。 |  |
| `half_size` | property | `half_size(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：half size。 |  |
| `__init__` | method | `__init__(self, half_size: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, material: RenderMaterial) -> None` | Python special method。 |  |

## `sapien.render.RenderShapeCapsule`

- Use: 具体渲染形状。
- Bases: `RenderShapePrimitive`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_half_length` | method | `get_half_length(self) -> float` | 读取 half length。 |  |
| `get_radius` | method | `get_radius(self) -> float` | 读取 radius。 |  |
| `half_length` | property | `half_length(self) -> float` | 属性：half length。 |  |
| `radius` | property | `radius(self) -> float` | 属性：radius。 |  |
| `__init__` | method | `__init__(self, radius: float, half_length: float, material: RenderMaterial) -> None` | Python special method。 |  |

## `sapien.render.RenderShapeCylinder`

- Use: 具体渲染形状。
- Bases: `RenderShapePrimitive`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_half_length` | method | `get_half_length(self) -> float` | 读取 half length。 |  |
| `get_radius` | method | `get_radius(self) -> float` | 读取 radius。 |  |
| `half_length` | property | `half_length(self) -> float` | 属性：half length。 |  |
| `radius` | property | `radius(self) -> float` | 属性：radius。 |  |
| `__init__` | method | `__init__(self, radius: float, half_length: float, material: RenderMaterial) -> None` | Python special method。 |  |

## `sapien.render.RenderShapePlane`

- Use: 具体渲染形状。
- Bases: `RenderShapePrimitive`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_scale` | method | `get_scale(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 读取 scale。 |  |
| `scale` | property | `scale(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：scale。 |  |
| `__init__` | method | `__init__(self, scale: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple, material: RenderMaterial) -> None` | Python special method。 |  |

## `sapien.render.RenderShapePrimitive`

- Use: 具体渲染形状。
- Bases: `RenderShape`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_triangles` | method | `get_triangles(self) -> np.ndarray[np.uint32[M, 3]]` | 读取 triangles。 |  |
| `get_vertex_normal` | method | `get_vertex_normal(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | 读取 vertex normal。 |  |
| `get_vertex_uv` | method | `get_vertex_uv(self) -> np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | 读取 vertex uv。 |  |
| `get_vertices` | method | `get_vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | 读取 vertices。 |  |
| `triangles` | property | `triangles(self) -> np.ndarray[np.uint32[M, 3]]` | 属性：triangles。 |  |
| `vertex_normal` | property | `vertex_normal(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | 属性：vertex normal。 |  |
| `vertex_uv` | property | `vertex_uv(self) -> np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | 属性：vertex uv。 |  |
| `vertices` | property | `vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | 属性：vertices。 |  |

## `sapien.render.RenderShapeSphere`

- Use: 具体渲染形状。
- Bases: `RenderShapePrimitive`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_radius` | method | `get_radius(self) -> float` | 读取 radius。 |  |
| `radius` | property | `radius(self) -> float` | 属性：radius。 |  |
| `__init__` | method | `__init__(self, radius: float, material: RenderMaterial) -> None` | Python special method。 |  |

## `sapien.render.RenderShapeTriangleMesh`

- Use: 具体渲染形状。
- Bases: `RenderShape`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `scale` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：scale。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `filename` | property | `filename(self) -> str` | 属性：filename。 |  |
| `get_filename` | method | `get_filename(self) -> str` | 读取 filename。 |  |
| `get_scale` | method | `get_scale(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 读取 scale。 |  |
| `set_scale` | method | `set_scale(self, scale: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | Note: this function only works when the shape is not added to scene |  |
| `__init__` | method | `__init__(self, vertices: np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]] \| list \| tuple, triangles: np.ndarray[np.uint32[M, 3]], normals: np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]] \| list \| tuple, uvs: np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]] \| list \| tuple, material: RenderMaterial) -> None<br>__init__(self, filename: str, scale: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple=..., material: RenderMaterial \| None=None) -> None` | Python special method。 |  |

## `sapien.render.RenderShapeTriangleMeshPart`

- Use: 具体渲染形状。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `cuda_triangles` | property | `cuda_triangles(self) -> sapien.CudaArray` | CUDA 状态/渲染缓冲属性。 | gpu_init 后取 view；循环中复用 torch/cupy/jax view。 |
| `cuda_vertices` | property | `cuda_vertices(self) -> sapien.CudaArray` | CUDA 状态/渲染缓冲属性。 | gpu_init 后取 view；循环中复用 torch/cupy/jax view。 |
| `get_cuda_triangles` | method | `get_cuda_triangles(self) -> sapien.CudaArray` | 读取 cuda triangles。 |  |
| `get_cuda_vertices` | method | `get_cuda_vertices(self) -> sapien.CudaArray` | 读取 cuda vertices。 |  |
| `get_material` | method | `get_material(self) -> RenderMaterial` | 读取 material。 |  |
| `get_triangles` | method | `get_triangles(self) -> np.ndarray[np.uint32[M, 3]]` | 读取 triangles。 |  |
| `get_vertex_normal` | method | `get_vertex_normal(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | 读取 vertex normal。 |  |
| `get_vertex_uv` | method | `get_vertex_uv(self) -> np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]]` | 读取 vertex uv。 |  |
| `get_vertices` | method | `get_vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | 读取 vertices。 |  |
| `material` | property | `material(self) -> RenderMaterial` | 属性：material。 |  |
| `set_vertex_normal` | method | `set_vertex_normal(self, normal: np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]] \| list \| tuple) -> None` | 设置 vertex normal。 |  |
| `set_vertex_uv` | method | `set_vertex_uv(self, uv: np.ndarray[tuple[M, Literal[2]], np.dtype[np.float32]] \| list \| tuple) -> None` | 设置 vertex uv。 |  |
| `triangles` | property | `triangles(self) -> np.ndarray[np.uint32[M, 3]]` | 属性：triangles。 |  |
| `vertices` | property | `vertices(self) -> np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]]` | 属性：vertices。 |  |
| `__eq__` | method | `__eq__(self, arg0: RenderShapeTriangleMeshPart) -> bool` | Python special method。 |  |

## `sapien.render.RenderSpotLightComponent`

- Use: 具体渲染灯光组件。
- Bases: `RenderLightComponent`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `inner_fov` | `float` | 属性：inner fov。 |  |
| `outer_fov` | `float` | 属性：outer fov。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_inner_fov` | method | `get_inner_fov(self) -> float` | 读取 inner fov。 |  |
| `get_outer_fov` | method | `get_outer_fov(self) -> float` | 读取 outer fov。 |  |
| `set_inner_fov` | method | `set_inner_fov(self, fov: float) -> None` | 设置 inner fov。 |  |
| `set_outer_fov` | method | `set_outer_fov(self, fov: float) -> None` | 设置 outer fov。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.render.RenderSystem`

- Use: 渲染系统；只有 viewer/sensor/offscreen 需要。
- Bases: `sapien.System`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `ambient_light` | `np.ndarray[Literal[3], np.dtype[np.float32]]` | 属性：ambient light。 |  |
| `cubemap` | `RenderCubemap` | 属性：cubemap。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `cameras` | property | `cameras(self) -> list[RenderCameraComponent]` | 属性：cameras。 |  |
| `cuda_object_transforms` | property | `cuda_object_transforms(self) -> sapien.CudaArray` | CUDA 状态/渲染缓冲属性。 | gpu_init 后取 view；循环中复用 torch/cupy/jax view。 |
| `device` | property | `device(self) -> sapien.Device` | 属性：device。 |  |
| `get_ambient_light` | method | `get_ambient_light(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 读取 ambient light。 |  |
| `get_cameras` | method | `get_cameras(self) -> list[RenderCameraComponent]` | 读取 cameras。 |  |
| `get_cubemap` | method | `get_cubemap(self) -> RenderCubemap` | 读取 cubemap。 |  |
| `get_lights` | method | `get_lights(self) -> list[RenderLightComponent]` | 读取 lights。 |  |
| `get_point_clouds` | method | `get_point_clouds(self) -> list[RenderPointCloudComponent]` | 读取 point clouds。 |  |
| `get_render_bodies` | method | `get_render_bodies(self) -> list[RenderBodyComponent]` | 读取 render bodies。 |  |
| `lights` | property | `lights(self) -> list[RenderLightComponent]` | 属性：lights。 |  |
| `point_clouds` | property | `point_clouds(self) -> list[RenderPointCloudComponent]` | 属性：point clouds。 |  |
| `render_bodies` | property | `render_bodies(self) -> list[RenderBodyComponent]` | 属性：render bodies。 |  |
| `set_ambient_light` | method | `set_ambient_light(self, color: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | 设置 ambient light。 |  |
| `set_cubemap` | method | `set_cubemap(self, cubemap: RenderCubemap) -> None` | 设置 cubemap。 |  |
| `__init__` | method | `__init__(self, device: sapien.Device \| None=None) -> None<br>__init__(self, device: str) -> None` | Python special method。 |  |
| `_internal_scene` | property | `_internal_scene(self) -> sapien.internal_renderer.Scene` | 属性： internal scene。 |  |

## `sapien.render.RenderSystemGroup`

- Use: 多 RenderSystem batched 渲染组；可绑定 CUDA poses。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `create_camera_group` | method | `create_camera_group(self, cameras: list[RenderCameraComponent], picture_names: list[str]) -> RenderCameraGroup` | 创建 batched camera group。 |  |
| `set_cuda_poses` | method | `set_cuda_poses(self, pose_buffer: sapien.CudaArray) -> None` | RenderSystemGroup 绑定 PhysX CUDA pose buffer。 | direct GPU render path；无需 sync_poses_gpu_to_cpu。 |
| `set_cuda_stream` | method | `set_cuda_stream(self, stream: int) -> None` | 设置 cuda stream。 |  |
| `update_render` | method | `update_render(self) -> None` | This function performs CUDA operations to transfer poses from the CUDA buffer provided by :func:`set_cuda_poses` into render systems. It updates the transformation mat... | GPU PhysX 动态体会读 CPU pose；offscreen 优先 CUDA pose path。 |
| `__init__` | method | `__init__(self, systems: list[RenderSystem]) -> None` | Python special method。 |  |

## `sapien.render.RenderTexture`

- Use: SAPIEN 渲染 API 对象。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `address_mode` | property | `address_mode(self) -> Literal['repeat', 'border', 'edge', 'mirror']` | 属性：address mode。 |  |
| `channels` | property | `channels(self) -> int` | 属性：channels。 |  |
| `depth` | property | `depth(self) -> int` | 属性：depth。 |  |
| `download` | method | `download(self) -> np.ndarray[Any, np.dtype[Any]]` | 调用 download。 |  |
| `filter_mode` | property | `filter_mode(self) -> Literal['nearest', 'linear']` | 属性：filter mode。 |  |
| `format` | property | `format(self) -> str` | 属性：format。 |  |
| `get_address_mode` | method | `get_address_mode(self) -> Literal['repeat', 'border', 'edge', 'mirror']` | 读取 address mode。 |  |
| `get_channels` | method | `get_channels(self) -> int` | 读取 channels。 |  |
| `get_depth` | method | `get_depth(self) -> int` | 读取 depth。 |  |
| `get_filter_mode` | method | `get_filter_mode(self) -> Literal['nearest', 'linear']` | 读取 filter mode。 |  |
| `get_format` | method | `get_format(self) -> str` | 读取 format。 |  |
| `get_height` | method | `get_height(self) -> int` | 读取 height。 |  |
| `get_mipmap_levels` | method | `get_mipmap_levels(self) -> int` | 读取 mipmap levels。 |  |
| `get_width` | method | `get_width(self) -> int` | 读取 width。 |  |
| `height` | property | `height(self) -> int` | 属性：height。 |  |
| `is_srgb` | property | `is_srgb(self) -> bool` | 布尔状态：srgb。 |  |
| `mipmap_levels` | property | `mipmap_levels(self) -> int` | 属性：mipmap levels。 |  |
| `upload` | method | `upload(self, data: np.ndarray[Any, np.dtype[Any]] \| list \| tuple) -> None` | 调用 upload。 |  |
| `width` | property | `width(self) -> int` | 属性：width。 |  |
| `__eq__` | method | `__eq__(self, arg0: RenderTexture) -> bool` | Python special method。 |  |
| `__init__` | method | `__init__(self, array: np.ndarray[Any, np.dtype[Any]] \| list \| tuple, dim: int, format: str, mipmap_levels: int=1, filter_mode: Literal['nearest', 'linear']='linear', address_mode: Literal['repeat', 'border', 'edge', 'mirror']='repeat', srgb: bool=False) -> None` | Python special method。 |  |

## `sapien.render.RenderTexture2D`

- Use: 2D 渲染纹理。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `address_mode` | property | `address_mode(self) -> Literal['repeat', 'border', 'edge', 'mirror']` | 属性：address mode。 |  |
| `channels` | property | `channels(self) -> int` | 属性：channels。 |  |
| `download` | method | `download(self) -> np.ndarray[Any, np.dtype[Any]]` | 调用 download。 |  |
| `filename` | property | `filename(self) -> str` | 属性：filename。 |  |
| `filter_mode` | property | `filter_mode(self) -> Literal['nearest', 'linear']` | 属性：filter mode。 |  |
| `format` | property | `format(self) -> str` | 属性：format。 |  |
| `get_address_mode` | method | `get_address_mode(self) -> Literal['repeat', 'border', 'edge', 'mirror']` | 读取 address mode。 |  |
| `get_channels` | method | `get_channels(self) -> int` | 读取 channels。 |  |
| `get_filename` | method | `get_filename(self) -> str` | 读取 filename。 |  |
| `get_filter_mode` | method | `get_filter_mode(self) -> Literal['nearest', 'linear']` | 读取 filter mode。 |  |
| `get_format` | method | `get_format(self) -> str` | 读取 format。 |  |
| `get_height` | method | `get_height(self) -> int` | 读取 height。 |  |
| `get_mipmap_levels` | method | `get_mipmap_levels(self) -> int` | 读取 mipmap levels。 |  |
| `get_width` | method | `get_width(self) -> int` | 读取 width。 |  |
| `height` | property | `height(self) -> int` | 属性：height。 |  |
| `is_srgb` | property | `is_srgb(self) -> bool` | 布尔状态：srgb。 |  |
| `mipmap_levels` | property | `mipmap_levels(self) -> int` | 属性：mipmap levels。 |  |
| `upload` | method | `upload(self, data: np.ndarray[Any, np.dtype[Any]] \| list \| tuple) -> None` | 调用 upload。 |  |
| `width` | property | `width(self) -> int` | 属性：width。 |  |
| `__eq__` | method | `__eq__(self, arg0: RenderTexture2D) -> bool` | Python special method。 |  |
| `__init__` | method | `__init__(self, array: np.ndarray[Any, np.dtype[Any]] \| list \| tuple, format: str, mipmap_levels: int=1, filter_mode: Literal['nearest', 'linear']='linear', address_mode: Literal['repeat', 'border', 'edge', 'mirror']='repeat', srgb: bool=False) -> None<br>__init__(self, filename: str, mipmap_levels: int=1, filter_mode: Literal['nearest', 'linear']='linear', address_mode: Literal['repeat', 'border', 'edge', 'mirror']='repeat', srgb: bool=True) -> None` | Create texture from file. The srgb parameter only affects files in uint8 format; it should be true for color textures (diffuse, emission) and false for others (normal,... |  |

## `sapien.render.RenderTexturedLightComponent`

- Use: 具体渲染灯光组件。
- Bases: `RenderSpotLightComponent`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `texture` | `RenderTexture2D` | 属性：texture。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_texture` | method | `get_texture(self) -> RenderTexture2D` | 读取 texture。 |  |
| `set_texture` | method | `set_texture(self, texture: RenderTexture2D) -> None` | 设置 texture。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.render.RenderVRDisplay`

- Use: SAPIEN 渲染 API 对象。
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `root_pose` | `sapien.Pose` | 属性：root pose。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `fetch_poses` | method | `fetch_poses(self) -> None` | fetches poses of HMD and controllers |  |
| `get_controller_axis_state` | method | `get_controller_axis_state(self, id: int, axis: int) -> Annotated[list[float], FixedSize(2)]` | 读取 controller axis state。 |  |
| `get_controller_button_pressed` | method | `get_controller_button_pressed(self, id: int) -> int` | 读取 controller button pressed。 |  |
| `get_controller_button_touched` | method | `get_controller_button_touched(self, id: int) -> int` | 读取 controller button touched。 |  |
| `get_controller_ids` | method | `get_controller_ids(self) -> list[int]` | 读取 controller ids。 |  |
| `get_controller_pose` | method | `get_controller_pose(self, id: int) -> sapien.Pose` | Gets the local pose of a controller. It should be called immediately after fetch_poses |  |
| `get_hmd_pose` | method | `get_hmd_pose(self) -> sapien.Pose` | Gets the local pose of the head set. It should be called immediately after fetch_poses |  |
| `get_left_hand_root_pose` | method | `get_left_hand_root_pose(self) -> sapien.Pose` | 读取 left hand root pose。 |  |
| `get_left_hand_skeletal_poses` | method | `get_left_hand_skeletal_poses(self) -> list[sapien.Pose]` | 读取 left hand skeletal poses。 |  |
| `get_right_hand_root_pose` | method | `get_right_hand_root_pose(self) -> sapien.Pose` | 读取 right hand root pose。 |  |
| `get_right_hand_skeletal_poses` | method | `get_right_hand_skeletal_poses(self) -> list[sapien.Pose]` | 读取 right hand skeletal poses。 |  |
| `get_root_pose` | method | `get_root_pose(self) -> sapien.Pose` | 读取 root pose。 |  |
| `render` | method | `render(self) -> None` | 调用 render。 |  |
| `set_camera_parameters` | method | `set_camera_parameters(self, near: float, far: float) -> None` | 设置 camera parameters。 |  |
| `set_root_pose` | method | `set_root_pose(self, pose: sapien.Pose) -> None` | 设置 root pose。 |  |
| `set_scene` | method | `set_scene(self, scene: sapien.Scene) -> None` | 设置 scene。 |  |
| `update_render` | method | `update_render(self) -> None` | update_render implicitly calls fetch_poses to make sure the HMD pose is up-to-date | GPU PhysX 动态体会读 CPU pose；offscreen 优先 CUDA pose path。 |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |
| `_internal_scene` | property | `_internal_scene(self) -> sapien.internal_renderer.Scene` | 属性： internal scene。 |  |

## `sapien.render.RenderWindow`

- Use: SAPIEN 渲染 API 对象。
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `cursor` | `bool` | 属性：cursor。 |  |
| `denoiser` | `Literal['none', 'oidn', 'optix']` | 属性：denoiser。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `alt` | property | `alt(self) -> bool` | 属性：alt。 |  |
| `camera_mode` | property | `camera_mode(self) -> Literal['perspective', 'orthographic']` | 属性：camera mode。 |  |
| `ctrl` | property | `ctrl(self) -> bool` | 属性：ctrl。 |  |
| `display_picture_names` | property | `display_picture_names(self) -> list[str]` | Names for available display targets that can be displayed in the render function |  |
| `far` | property | `far(self) -> float` | 属性：far。 |  |
| `fovy` | property | `fovy(self) -> float` | 属性：fovy。 |  |
| `fps` | property | `fps(self) -> float` | 属性：fps。 |  |
| `get_camera_model_matrix` | method | `get_camera_model_matrix(self) -> np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float32]]` | 读取 camera model matrix。 |  |
| `get_camera_pose` | method | `get_camera_pose(self) -> sapien.Pose` | 读取 camera pose。 |  |
| `get_camera_position` | method | `get_camera_position(self) -> np.ndarray[Literal[3], np.dtype[np.float32]]` | 读取 camera position。 |  |
| `get_camera_projection_matrix` | method | `get_camera_projection_matrix(self) -> np.ndarray[Any, np.dtype[np.float32]]` | 读取 camera projection matrix。 |  |
| `get_camera_property_float` | method | `get_camera_property_float(self, key: str) -> float` | 读取 camera property float。 |  |
| `get_camera_property_int` | method | `get_camera_property_int(self, key: str) -> int` | 读取 camera property int。 |  |
| `get_camera_rotation` | method | `get_camera_rotation(self) -> np.ndarray[Literal[4], np.dtype[np.float32]]` | 读取 camera rotation。 |  |
| `get_content_scale` | method | `get_content_scale(self) -> float` | 读取 content scale。 |  |
| `get_picture` | method | `get_picture(self, name: str) -> np.ndarray[Any, np.dtype[Any]]` | 读取 picture。 | GPU PhysX 动态体会读 CPU pose；offscreen 优先 CUDA pose path。 |
| `get_picture_pixel` | method | `get_picture_pixel(self, name: str, x: int, y: int) -> np.ndarray[Any, np.dtype[Any]]` | 读取 picture pixel。 |  |
| `get_picture_size` | method | `get_picture_size(self, name: str) -> Annotated[list[int], FixedSize(2)]` | 读取 picture size。 |  |
| `hide` | method | `hide(self) -> None` | 调用 hide。 |  |
| `key_down` | method | `key_down(self, key: str) -> bool` | 调用 key down。 |  |
| `key_press` | method | `key_press(self, key: str) -> bool` | 调用 key press。 |  |
| `mouse_click` | method | `mouse_click(self, key: int) -> bool` | 调用 mouse click。 |  |
| `mouse_delta` | property | `mouse_delta(self) -> Annotated[list[float], FixedSize(2)]` | 属性：mouse delta。 |  |
| `mouse_down` | method | `mouse_down(self, key: int) -> bool` | 调用 mouse down。 |  |
| `mouse_position` | property | `mouse_position(self) -> Annotated[list[float], FixedSize(2)]` | 属性：mouse position。 |  |
| `mouse_wheel_delta` | property | `mouse_wheel_delta(self) -> Annotated[list[float], FixedSize(2)]` | 属性：mouse wheel delta。 |  |
| `near` | property | `near(self) -> float` | 属性：near。 |  |
| `ortho_top` | property | `ortho_top(self) -> float` | 属性：ortho top。 |  |
| `render` | method | `render(self, target_name: str, ui_windows: list[sapien.internal_renderer.UIWidget]=[]) -> None` | 调用 render。 |  |
| `resize` | method | `resize(self, width: int, height: int) -> None` | 调用 resize。 |  |
| `set_camera_orthographic_parameters` | method | `set_camera_orthographic_parameters(self, near: float, far: float, top: float) -> None` | 设置 camera orthographic parameters。 |  |
| `set_camera_parameters` | method | `set_camera_parameters(self, near: float, far: float, fovy: float) -> None` | 设置 camera parameters。 |  |
| `set_camera_pose` | method | `set_camera_pose(self, pose: sapien.Pose) -> None` | 设置 camera pose。 |  |
| `set_camera_position` | method | `set_camera_position(self, position: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | 设置 camera position。 |  |
| `set_camera_property` | method | `set_camera_property(self, key: str, value: float) -> None<br>set_camera_property(self, key: str, value: int) -> None` | 设置 camera property。 |  |
| `set_camera_rotation` | method | `set_camera_rotation(self, quat: np.ndarray[Literal[4], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | 设置 camera rotation。 |  |
| `set_camera_texture` | method | `set_camera_texture(self, name: str, texture: RenderTexture2D) -> None` | 设置 camera texture。 |  |
| `set_camera_texture_array` | method | `set_camera_texture_array(self, name: str, textures: list[RenderTexture2D]) -> None` | 设置 camera texture array。 |  |
| `set_content_scale` | method | `set_content_scale(self, scale: float) -> None` | 设置 content scale。 |  |
| `set_drop_callback` | method | `set_drop_callback(self, callback: Callable[[list[str]], None]) -> None` | 设置 drop callback。 |  |
| `set_focus_callback` | method | `set_focus_callback(self, callback: Callable[[int], None]) -> None` | 设置 focus callback。 |  |
| `set_intrinsic_parameters` | method | `set_intrinsic_parameters(self, near: float, far: float, fx: float, fy: float, cx: float, cy: float, skew: float) -> None` | 设置 intrinsic parameters。 |  |
| `set_scene` | method | `set_scene(self, scene: sapien.Scene) -> None` | 设置 scene。 |  |
| `set_scenes` | method | `set_scenes(self, scenes: list[sapien.Scene], offsets: list[np.ndarray[Literal[3], np.dtype[np.float32]]]) -> None` | 设置 scenes。 |  |
| `set_shader_dir` | method | `set_shader_dir(self, shader_dir: str) -> None` | 设置 shader dir。 |  |
| `shift` | property | `shift(self) -> bool` | 属性：shift。 |  |
| `should_close` | property | `should_close(self) -> bool` | 属性：should close。 |  |
| `show` | method | `show(self) -> None` | 调用 show。 |  |
| `size` | property | `size(self) -> Annotated[list[int], FixedSize(2)]` | 属性：size。 |  |
| `super` | property | `super(self) -> bool` | 属性：super。 |  |
| `unset_drop_callback` | method | `unset_drop_callback(self) -> None` | 调用 unset drop callback。 |  |
| `unset_focus_callback` | method | `unset_focus_callback(self) -> None` | 调用 unset focus callback。 |  |
| `update_render` | method | `update_render(self) -> None` | Equivalent to calling the update_render function for all added scene | GPU PhysX 动态体会读 CPU pose；offscreen 优先 CUDA pose path。 |
| `__init__` | method | `__init__(self, width: int, height: int, shader_dir: str) -> None` | Python special method。 |  |
| `_internal_renderer` | property | `_internal_renderer(self) -> sapien.internal_renderer.Renderer` | 属性： internal renderer。 |  |
| `_internal_scene` | property | `_internal_scene(self) -> sapien.internal_renderer.Scene` | 属性： internal scene。 |  |

## `sapien.render.SapienRenderer`

- Use: SAPIEN API 对象。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__init__` | method | `__init__(self, device: sapien.Device \| None=None) -> None` | Python special method。 |  |
| `_internal_context` | property | `_internal_context(self) -> sapien.internal_renderer.Context` | 属性： internal context。 |  |
