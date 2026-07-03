# Sensors / Assets / Math API

Compact lookup for utility APIs outside core PhysX/render.

## Math API (`sapien.math`)

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.math`
- Source files: `python/py_package/pysapien/math.pyi`

## Constants / module attributes

| API | Type | Use | Notes |
|---|---|---|---|
| `sapien.math.pose_gl_to_ros` | `sapien.Pose` | Property: pose gl to ros. |  |
| `sapien.math.pose_ros_to_gl` | `sapien.Pose` | Property: pose ros to gl. |  |

## Functions

| API | Signature | Use | Notes |
|---|---|---|---|
| `sapien.math.compute_box_mass_properties` | `compute_box_mass_properties(half_size: np.ndarray[Literal[3], np.dtype[np.float32]]) -> MassProperties` | Compute box mass properties. |  |
| `sapien.math.compute_capsule_mass_properties` | `compute_capsule_mass_properties(radius: float, half_length: float) -> MassProperties` | Compute capsule mass properties. |  |
| `sapien.math.compute_cylinder_mass_properties` | `compute_cylinder_mass_properties(radius: float, half_length: float) -> MassProperties` | Compute cylinder mass properties. |  |
| `sapien.math.compute_mesh_mass_properties` | `compute_mesh_mass_properties(vertices: np.ndarray[tuple[M, Literal[3]], np.dtype[np.float32]], triangles: np.ndarray[np.uint32[M, 3]]) -> MassProperties` | Compute mesh mass properties. |  |
| `sapien.math.compute_sphere_mass_properties` | `compute_sphere_mass_properties(radius: float) -> MassProperties` | Compute sphere mass properties. |  |
| `sapien.math.shortest_rotation` | `shortest_rotation(source: np.ndarray[Literal[3], np.dtype[np.float32]], target: np.ndarray[Literal[3], np.dtype[np.float32]]) -> np.ndarray[Literal[4], np.dtype[np.float32]]` | Call shortest rotation. |  |

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.math.MassProperties` |  | Mass, center-of-mass and inertia utility object. |  |

## `sapien.math.MassProperties`

- Use: Mass, center-of-mass and inertia utility object.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `cm` | `np.ndarray[tuple[Literal[3], Literal[1]], np.dtype[np.float32]]` |  |  |
| `cm_inertia` | `np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float32]]` |  |  |
| `mass` | `float` |  |  |
| `origin_inertia` | `np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float32]]` |  |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `decompose` | method | `decompose(self) -> tuple[float, sapien.Pose, np.ndarray[Literal[3], np.dtype[np.float32]]]` | decompose mass properties into mass, cmass_local_pose, and principal inertia |  |
| `scale_mass` | method | `scale_mass(self, scale: float) -> MassProperties` | compute new mass properties as if the object density is scaled uniformly |  |
| `scale_size` | method | `scale_size(self, scale: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> MassProperties` | compute new mass properties as if the object volume is scaled around the origin while keeping density the same |  |
| `transform` | method | `transform(self, pose: sapien.Pose) -> MassProperties` | compute new mass properties as if the origin of the current object is moved to given pose |  |
| `__add__` | method | `__add__(self, other: MassProperties) -> MassProperties` |  |  |
| `__init__` | method | `__init__(self, mass: float, cm: np.ndarray[tuple[Literal[3], Literal[1]], np.dtype[np.float32]] \| list \| tuple, inertia: np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float32]] \| list \| tuple) -> None<br>__init__(self, mass: float, cmass_local_pose: sapien.Pose, inertia: np.ndarray[Literal[3], np.dtype[np.float32]] \| list[float] \| tuple) -> None` | construct inertia from mass, center of mass, inertia at center of mass |  |

## SimSense pybind API (`sapien.simsense`)

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.simsense`
- Source files: `python/py_package/pysapien/simsense.pyi`

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.simsense.DepthSensorEngine` |  | SimSense depth engine low-level interface. |  |

## `sapien.simsense.DepthSensorEngine`

- Use: SimSense depth engine low-level interface.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `compute` | method | `compute(self, left_array: np.ndarray[np.uint8], right_array: np.ndarray[np.uint8], bbox: bool=False, bbox_start_x: int=0, bbox_start_y: int=0, bbox_width: int=0, bbox_height: int=0) -> None<br>compute(self, left_cuda: sapien.CudaArray, right_cuda: sapien.CudaArray, bbox: bool=False, bbox_start_x: int=0, bbox_start_y: int=0, bbox_width: int=0, bbox_height: int=0) -> None` |  |  |
| `get_cuda` | method | `get_cuda(self) -> sapien.CudaArray` |  |  |
| `get_ndarray` | method | `get_ndarray(self) -> np.ndarray[Any, np.dtype[np.float32]]` |  |  |
| `get_point_cloud_cuda` | method | `get_point_cloud_cuda(self) -> sapien.CudaArray` |  |  |
| `get_point_cloud_ndarray` | method | `get_point_cloud_ndarray(self) -> np.ndarray[Any, np.dtype[np.float32]]` |  |  |
| `get_rgb_point_cloud_cuda` | method | `get_rgb_point_cloud_cuda(self, arg0: sapien.CudaArray) -> sapien.CudaArray` |  |  |
| `get_rgb_point_cloud_ndarray` | method | `get_rgb_point_cloud_ndarray(self, arg0: sapien.CudaArray) -> np.ndarray[Any, np.dtype[np.float32]]` |  |  |
| `set_census_window_size` | method | `set_census_window_size(self, arg0: int, arg1: int) -> None` |  |  |
| `set_ir_noise_parameters` | method | `set_ir_noise_parameters(self, arg0: float, arg1: float, arg2: float, arg3: float) -> None` |  |  |
| `set_lr_max_diff` | method | `set_lr_max_diff(self, arg0: int) -> None` |  |  |
| `set_matching_block_size` | method | `set_matching_block_size(self, arg0: int, arg1: int) -> None` |  |  |
| `set_penalties` | method | `set_penalties(self, arg0: int, arg1: int) -> None` |  |  |
| `set_uniqueness_ratio` | method | `set_uniqueness_ratio(self, arg0: int) -> None` |  |  |
| `__init__` | method | `__init__(self, arg0: int, arg1: int, arg2: int, arg3: int, arg4: float, arg5: float, arg6: float, arg7: float, arg8: int, arg9: float, arg10: float, arg11: float, arg12: float, arg13: bool, arg14: int, arg15: int, arg16: int, arg17: int, arg18: int, arg19: int, arg20: int, arg21: int, arg22: int, arg23: int, arg24: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple, arg25: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple, arg26: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple, arg27: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple, arg28: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple, arg29: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple, arg30: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple, arg31: float, arg32: float, arg33: float, arg34: bool, arg35: float, arg36: float, arg37: float, arg38: float, arg39: float) -> None` |  |  |

## StereoDepthSensor API

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.sensor.stereodepth`
- Source files: `python/py_package/sensor/stereodepth.py`

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.sensor.stereodepth.StereoDepthSensor` |  | This class simulates an active stereo depth sensor. It has one RGB camera and two infrared camera. Depth is computed via semi-global block matching. Refer to StereoDep... |  |
| `sapien.sensor.stereodepth.StereoDepthSensorConfig` |  | An instance of this class is required to initialize StereoDepthSensor. |  |

## `sapien.sensor.stereodepth.StereoDepthSensor`

- Use: This class simulates an active stereo depth sensor. It has one RGB camera and two infrared camera. Depth is computed via semi-global block matching. Refer to StereoDep...
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `compute_depth` | method | `compute_depth(self, bbox_start: tuple=None, bbox_size: tuple=None)` |  |  |
| `get_config` | method | `get_config(self)` |  |  |
| `get_depth` | method | `get_depth(self)` | Note: Returned depth map will be of the same resolution and frame of RGB camera. |  |
| `get_depth_cuda` | method | `get_depth_cuda(self)` | Note: Returned depth map will be of the same resolution and frame of RGB camera. |  |
| `get_ir` | method | `get_ir(self)` | Note: Noise simulation won't be reflected here. |  |
| `get_pointcloud` | method | `get_pointcloud(self, with_rgb: bool=False)` | Note: Returned point cloud is from RGB camera's with x rightward, y downward, z forward. |  |
| `get_pointcloud_cuda` | method | `get_pointcloud_cuda(self, with_rgb: bool=False)` | Note: Returned point cloud is from RGB camera's with x rightward, y downward, z forward. |  |
| `get_pose` | method | `get_pose(self)` |  |  |
| `get_rgb` | method | `get_rgb(self)` |  |  |
| `get_rgba_cuda` | method | `get_rgba_cuda(self)` |  |  |
| `set_census_window_size` | method | `set_census_window_size(self, census_width: int, census_height: int)` | :param census_width: Width of the center-symmetric census transform window. This must be an odd number. :param census_height: Height of the center-symmetric census tra... |  |
| `set_ir_noise` | method | `set_ir_noise(self, ir_speckle_noise: float, ir_thermal_noise: float)` | :param ir_speckle_noise: Scale for simulating infrared speckle noise. Set to 0 will disable noise simulation. :param ir_thermal_noise: Scale for simulating infrared th... |  |
| `set_local_pose` | method | `set_local_pose(self, pose: Pose)` | Set local pose of the sensor relative to mounted actor. |  |
| `set_lr_max_diff` | method | `set_lr_max_diff(self, lr_max_diff: int)` | :param lr_max_diff: Maximum allowed difference in the left-right consistency check. Set it to 255 to disable the check. |  |
| `set_matching_block_size` | method | `set_matching_block_size(self, block_width: int, block_height: int)` | :param block_width: Width of the matched block. This must be an odd number. :param block_height: Height of the matched block. This must be an odd number. |  |
| `set_penalties` | method | `set_penalties(self, p1_penalty: int, p2_penalty: int)` | :param p1_penalty: P1 penalty for semi-global matching algorithm. :param p2_penalty: P2 penalty for semi-global matching algorithm. |  |
| `set_uniqueness_ratio` | method | `set_uniqueness_ratio(self, uniqueness_ratio: int)` | :param uniqueness_ratio: Margin in percentage by which the minimum computed cost should win the second best (not considering best match's adjacent pixels) cost to cons... |  |
| `take_picture` | method | `take_picture(self, infrared_only: bool=False)` | Note: We expect one scene.update_render() call before calling take_picture(). :param infrared_only: If true, only take infrared pictures without taking RGB picture. |  |
| `__init__` | method | `__init__(self, config: StereoDepthSensorConfig, mount_entity: Entity, pose: Optional[Pose]=None)` | :param config: configuration of the sensor. :param mount_entity: entity that the sensor is mounted to. :param pose: local pose relative to the mounted entity. If not g... |  |
| `_create_cameras` | method | `_create_cameras(self)` | Call create cameras. |  |
| `_create_light` | method | `_create_light(self)` | Call create light. |  |
| `_ir_mode` | method | `_ir_mode(self)` |  |  |
| `_normal_mode` | method | `_normal_mode(self)` |  |  |

## `sapien.sensor.stereodepth.StereoDepthSensorConfig`

- Use: An instance of this class is required to initialize StereoDepthSensor.
- Bases: `-`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `SUPPORTED_MODELS` | `` |  |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__init__` | method | `__init__(self, model: str='D435')` | :param model: sensor model, one of: D415, D435 |  |

## Sensor base API

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.sensor.sensor_base`
- Source files: `python/py_package/sensor/sensor_base.py`

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.sensor.sensor_base.SensorEntity` |  | SAPIEN API object. |  |

## `sapien.sensor.sensor_base.SensorEntity`

- Use: SAPIEN API object.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `__init__` | method | `__init__(self)` |  |  |

## ActiveLightSensor API

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.sensor.activelight`
- Source files: `python/py_package/sensor/activelight.py`

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.sensor.activelight.ActiveLightSensor` | `SensorEntity` | Active-light depth sensor wrapper. |  |

## `sapien.sensor.activelight.ActiveLightSensor`

- Use: Active-light depth sensor wrapper.
- Bases: `SensorEntity`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `clear_cache` | method | `clear_cache(self)` | Call clear cache. |  |
| `get_depth` | method | `get_depth(self)` |  |  |
| `get_ir` | method | `get_ir(self)` |  |  |
| `get_pointcloud` | method | `get_pointcloud(self, frame='camera', with_rgb=False)` |  |  |
| `get_pose` | method | `get_pose(self)` |  |  |
| `get_rgb` | method | `get_rgb(self)` |  |  |
| `set_pose` | method | `set_pose(self, pose: Pose)` |  |  |
| `take_picture` | method | `take_picture(self)` | Note: we expect one scene.update_render() call before calling take_picture(). |  |
| `__init__` | method | `__init__(self, sensor_name: str, renderer: Renderer, scene: Scene, sensor_type: Optional[str]='d415', rgb_resolution: Tuple[int, int]=None, ir_resolution: Tuple[int, int]=None, rgb_intrinsic: Optional[np.ndarray]=None, ir_intrinsic: Optional[np.ndarray]=None, trans_pose_l: Optional[Pose]=None, trans_pose_r: Optional[Pose]=None, light_pattern: Optional[str]=None, max_depth: float=10.0, min_depth: float=0.2, ir_ambient_strength: float=0.002, ir_light_dim_factor: float=0.05)` | :param sensor_name: Name of the sensor. :param renderer: Renderer used in the scene. :param scene: Scene that the sensor is attached to. :param sensor_type: If this is... |  |
| `_create_cameras` | method | `_create_cameras(self)` | Call create cameras. |  |
| `_depth2pts_np` | staticmethod | `_depth2pts_np(depth_map, cam_intrinsic, cam_extrinsic=np.eye(4))` |  |  |
| `_fetch` | method | `_fetch(self, mod)` |  |  |
| `_float2uint8` | staticmethod | `_float2uint8(x)` |  |  |
| `_get_pixel_grids_np` | staticmethod | `_get_pixel_grids_np(height, width)` | Call get pixel grids np. |  |
| `_ir_mode` | method | `_ir_mode(self)` |  |  |
| `_normal_mode` | method | `_normal_mode(self)` |  |  |
| `_pose2cv2ex` | staticmethod | `_pose2cv2ex(pose)` |  |  |
| `_set_sensor_parameters` | method | `_set_sensor_parameters(self, sensor_type)` | Call set sensor parameters. |  |

## SimSenseComponent API

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.sensor.simsense_component`
- Source files: `python/py_package/sensor/simsense_component.py`

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.sensor.simsense_component.SimSenseComponent` | `Component` | SimSense render/compute component wrapper. |  |

## `sapien.sensor.simsense_component.SimSenseComponent`

- Use: SimSense render/compute component wrapper.
- Bases: `Component`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `compute` | method | `compute(self, left: Union[np.ndarray, CudaArray], right: Union[np.ndarray, CudaArray], bbox_start: tuple=None, bbox_size: tuple=None) -> None` |  |  |
| `get_cuda` | method | `get_cuda(self) -> CudaArray` |  |  |
| `get_ndarray` | method | `get_ndarray(self) -> np.ndarray` |  |  |
| `get_point_cloud_cuda` | method | `get_point_cloud_cuda(self) -> CudaArray` |  |  |
| `get_point_cloud_ndarray` | method | `get_point_cloud_ndarray(self) -> np.ndarray` |  |  |
| `get_rgb_point_cloud_cuda` | method | `get_rgb_point_cloud_cuda(self, rgba_cuda: CudaArray) -> CudaArray` |  |  |
| `get_rgb_point_cloud_ndarray` | method | `get_rgb_point_cloud_ndarray(self, rgba_cuda: CudaArray) -> np.ndarray` |  |  |
| `on_add_to_scene` | method | `on_add_to_scene(self, scene)` | Call on add to scene. |  |
| `on_remove_from_scene` | method | `on_remove_from_scene(self, scene)` | Call on remove from scene. |  |
| `__init__` | method | `__init__(self, rgb_resolution: tuple, ir_resolution: tuple, rgb_intrinsic: np.ndarray, ir_intrinsic: np.ndarray, trans_pose_l: Pose, trans_pose_r: Pose, min_depth: float, max_depth: float, ir_noise_seed: int, ir_speckle_noise: float, ir_thermal_noise: float, rectified: bool, census_width: int, census_height: int, max_disp: int, block_width: int, block_height: int, p1_penalty: int, p2_penalty: int, uniqueness_ratio: int, lr_max_diff: int, median_filter_size: int, depth_dilation: bool)` |  |  |
| `_get_registration_mat` | staticmethod | `_get_registration_mat(ir_size, ir_intrinsic, rgb_intrinsic, ir2rgb)` | Call get registration mat. |  |
| `_pose2cv2ex` | staticmethod | `_pose2cv2ex(pose)` |  |  |

## Asset helpers

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.asset`
- Source files: `python/py_package/asset/__init__.py`

## Functions

| API | Signature | Use | Notes |
|---|---|---|---|
| `sapien.asset.create_dome_envmap` | `create_dome_envmap(filename='sapien_dome.ktx', sky_color=[0.6, 0.6, 0.6], ground_color=[0.2, 0.2, 0.2], blend=0.3, resolution=256)` | Create a dome envmap. | Use a project-relative or caller-provided output path. |
| `sapien.asset.download_partnet_mobility` | `download_partnet_mobility(model_id, token=None, directory=None)` | Call download partnet mobility. |  |