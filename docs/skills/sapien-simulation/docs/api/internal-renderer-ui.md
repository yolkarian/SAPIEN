# Internal renderer / UI API (`sapien.internal_renderer`)

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.internal_renderer`
- Source files: `python/py_package/pysapien/internal_renderer.pyi`
- Notes: Low-level renderer and ImGui widgets used by viewer/tools; prefer `sapien.render` for simulation code unless extending viewer/UI.

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.internal_renderer.Context` |  | SAPIEN API 对象。 |  |
| `sapien.internal_renderer.Cubemap` |  | SAPIEN API 对象。 |  |
| `sapien.internal_renderer.LineSet` | `PrimitiveSet` | SAPIEN API 对象。 |  |
| `sapien.internal_renderer.LineSetObject` | `Node` | SAPIEN API 对象。 |  |
| `sapien.internal_renderer.Material` |  | SAPIEN API 对象。 |  |
| `sapien.internal_renderer.Mesh` |  | SAPIEN API 对象。 |  |
| `sapien.internal_renderer.Model` |  | SAPIEN API 对象。 |  |
| `sapien.internal_renderer.Node` |  | SAPIEN API 对象。 |  |
| `sapien.internal_renderer.Object` | `Node` | SAPIEN API 对象。 |  |
| `sapien.internal_renderer.PointSet` | `PrimitiveSet` | SAPIEN API 对象。 |  |
| `sapien.internal_renderer.PointSetObject` | `Node` | SAPIEN API 对象。 |  |
| `sapien.internal_renderer.PrimitiveSet` |  | SAPIEN API 对象。 |  |
| `sapien.internal_renderer.Renderer` |  | SAPIEN 渲染 API 对象。 |  |
| `sapien.internal_renderer.Scene` |  | 实体/系统容器；高层 wrapper 还提供 builder、灯光、相机、地面等便捷函数。 |  |
| `sapien.internal_renderer.Shape` |  | SAPIEN API 对象。 |  |
| `sapien.internal_renderer.Texture` |  | SAPIEN API 对象。 |  |
| `sapien.internal_renderer.UIButton` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UICheckbox` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIConditional` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIDisplayText` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIDummy` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIDuration` |  | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIFileChooser` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIGizmo` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIInputFloat` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIInputFloat2` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIInputFloat3` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIInputFloat4` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIInputInt` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIInputInt2` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIInputInt3` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIInputInt4` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIInputText` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIInputTextMultiline` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIKeyframe` |  | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIKeyframeEditor` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIOptions` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIPicture` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIPopup` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UISameLine` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UISection` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UISelectable` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UISliderAngle` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UISliderFloat` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UITreeNode` | `UIWidget` | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIWidget` |  | internal_renderer ImGui UI widget。 |  |
| `sapien.internal_renderer.UIWindow` | `UIWidget` | internal_renderer ImGui UI widget。 |  |

## `sapien.internal_renderer.Context`

- Use: SAPIEN API 对象。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `create_box_mesh` | method | `create_box_mesh(self) -> Mesh` | 创建 box mesh。 |  |
| `create_brdf_lut` | method | `create_brdf_lut(self, size: int=128) -> Texture` | Generate BRDF LUT texture, see https://learnopengl.com/PBR/IBL/Specular-IBL |  |
| `create_capsule_mesh` | method | `create_capsule_mesh(self, radius: float, half_length: float, segments: int=32, half_rings: int=8) -> Mesh` | 创建 capsule mesh。 |  |
| `create_cone_mesh` | method | `create_cone_mesh(self, segments: int=32) -> Mesh` | 创建 cone mesh。 |  |
| `create_cubemap_from_files` | method | `create_cubemap_from_files(self, filenames: Annotated[list[str], FixedSize(6)], mipmap_levels: int) -> Cubemap` | Load cube map, its mipmaps are generated based on roughness, details see https://learnopengl.com/PBR/IBL/Specular-IBL |  |
| `create_cylinder_mesh` | method | `create_cylinder_mesh(self, segments: int=32) -> Mesh` | 创建 cylinder mesh。 |  |
| `create_line_set` | method | `create_line_set(self, vertices: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple, colors: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> PointSet` | 创建 line set。 |  |
| `create_material` | method | `create_material(self, emission: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple, base_color: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple, specular: float, roughness: float, metallic: float, transmission: float=0.0, ior: float=1.0099999904632568) -> Material` | 创建 material。 |  |
| `create_mesh_from_array` | method | `create_mesh_from_array(self, vertices: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple, indices: np.ndarray[np.uint32], normals: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple=..., uvs: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple=...) -> Mesh` | 创建 mesh from array。 |  |
| `create_model` | method | `create_model(self, meshes: list[Mesh], materials: list[Material]) -> Model` | 创建 model。 |  |
| `create_model_from_file` | method | `create_model_from_file(self, filename: str) -> Model` | 创建 model from file。 |  |
| `create_point_set` | method | `create_point_set(self, vertices: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple, colors: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> LineSet` | 创建 point set。 |  |
| `create_texture_from_file` | method | `create_texture_from_file(self, filename: str, mipmap_levels: int, filter: str='linear', address_mode: str='repeat') -> Texture` | 创建 texture from file。 |  |
| `create_uvsphere_mesh` | method | `create_uvsphere_mesh(self, segments: int=32, half_rings: int=16) -> Mesh` | 创建 uvsphere mesh。 |  |

## `sapien.internal_renderer.Cubemap`

- Use: SAPIEN API 对象。
- Bases: `-`

## `sapien.internal_renderer.LineSet`

- Use: SAPIEN API 对象。
- Bases: `PrimitiveSet`

## `sapien.internal_renderer.LineSetObject`

- Use: SAPIEN API 对象。
- Bases: `Node`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `line_width` | `float` | 属性：line width。 |  |

## `sapien.internal_renderer.Material`

- Use: SAPIEN API 对象。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `set_base_color` | method | `set_base_color(self, rgba: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> None` | 设置 base color。 |  |
| `set_emission` | method | `set_emission(self, emission: float) -> None` | 设置 emission。 |  |
| `set_metallic` | method | `set_metallic(self, metallic: float) -> None` | 设置 metallic。 |  |
| `set_roughness` | method | `set_roughness(self, roughness: float) -> None` | 设置 roughness。 |  |
| `set_specular` | method | `set_specular(self, specular: float) -> None` | 设置 specular。 |  |
| `set_textures` | method | `set_textures(self, base_color: Texture \| None=None, roughness: Texture \| None=None, normal: Texture \| None=None, metallic: Texture \| None=None, emission: Texture \| None=None, transmission: Texture \| None=None) -> None` | 设置 textures。 |  |
| `set_transmission` | method | `set_transmission(self, transmission: float) -> None` | 设置 transmission。 |  |

## `sapien.internal_renderer.Mesh`

- Use: SAPIEN API 对象。
- Bases: `-`

## `sapien.internal_renderer.Model`

- Use: SAPIEN API 对象。
- Bases: `-`

## `sapien.internal_renderer.Node`

- Use: SAPIEN API 对象。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `children` | property | `children(self) -> list[Node]` | 属性：children。 |  |
| `position` | property | `position(self) -> np.ndarray[Any, np.dtype[np.float32]]` | 属性：position。 |  |
| `rotation` | property | `rotation(self) -> np.ndarray[Any, np.dtype[np.float32]]` | 属性：rotation。 |  |
| `scale` | property | `scale(self) -> np.ndarray[Any, np.dtype[np.float32]]` | 属性：scale。 |  |
| `set_position` | method | `set_position(self, position: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> None` | 设置 position。 |  |
| `set_rotation` | method | `set_rotation(self, quat: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> None` | 设置 rotation。 |  |
| `set_scale` | method | `set_scale(self, scale: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> None` | 设置 scale。 |  |

## `sapien.internal_renderer.Object`

- Use: SAPIEN API 对象。
- Bases: `Node`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `cast_shadow` | `bool` | 属性：cast shadow。 |  |
| `shading_mode` | `int` | 属性：shading mode。 |  |
| `transparency` | `float` | 属性：transparency。 |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_segmentation` | method | `get_segmentation(self) -> np.ndarray[np.uint32]` | 读取 segmentation。 |  |
| `model` | property | `model(self) -> Model` | 属性：model。 |  |
| `set_segmentation` | method | `set_segmentation(self, arg0: np.ndarray[np.uint32]) -> None` | 设置 segmentation。 |  |

## `sapien.internal_renderer.PointSet`

- Use: SAPIEN API 对象。
- Bases: `PrimitiveSet`

## `sapien.internal_renderer.PointSetObject`

- Use: SAPIEN API 对象。
- Bases: `Node`

## `sapien.internal_renderer.PrimitiveSet`

- Use: SAPIEN API 对象。
- Bases: `-`

## `sapien.internal_renderer.Renderer`

- Use: SAPIEN 渲染 API 对象。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `set_custom_cubemap` | method | `set_custom_cubemap(self, name: str, texture: Cubemap) -> None` | 设置 custom cubemap。 |  |
| `set_custom_property` | method | `set_custom_property(self, name: str, value: float) -> None<br>set_custom_property(self, name: str, value: int) -> None` | 设置 custom property。 |  |
| `set_custom_texture` | method | `set_custom_texture(self, name: str, texture: Texture) -> None` | 设置 custom texture。 |  |

## `sapien.internal_renderer.Scene`

- Use: 实体/系统容器；高层 wrapper 还提供 builder、灯光、相机、地面等便捷函数。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `add_line_set` | method | `add_line_set(self, line_set: PointSet, parent: Node \| None=None) -> LineSetObject` | 添加/创建 line set。 |  |
| `add_node` | method | `add_node(self, parent: Node \| None=None) -> Node` | 添加/创建 node。 |  |
| `add_object` | method | `add_object(self, model: Model, parent: Node \| None=None) -> Object` | 添加/创建 object。 |  |
| `add_point_set` | method | `add_point_set(self, point_set: LineSet, parent: Node \| None=None) -> PointSetObject` | 添加/创建 point set。 |  |
| `force_rebuild` | method | `force_rebuild(self) -> None` | 调用 force rebuild。 |  |
| `force_update` | method | `force_update(self) -> None` | 调用 force update。 |  |
| `remove_node` | method | `remove_node(self, node: Node) -> None` | 移除 node。 |  |
| `set_ambient_light` | method | `set_ambient_light(self, arg0: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> None` | 设置 ambient light。 |  |
| `set_cubemap` | method | `set_cubemap(self, arg0: Cubemap) -> None` | 设置 cubemap。 |  |

## `sapien.internal_renderer.Shape`

- Use: SAPIEN API 对象。
- Bases: `-`

## `sapien.internal_renderer.Texture`

- Use: SAPIEN API 对象。
- Bases: `-`

## `sapien.internal_renderer.UIButton`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Callback` | method | `Callback(self, func: Callable[[UIButton], None]) -> UIButton` | 调用 Callback。 |  |
| `Id` | method | `Id(self, id: str) -> UIButton` | 调用 Id。 |  |
| `Label` | method | `Label(self, label: str) -> UIButton` | 调用 Label。 |  |
| `Width` | method | `Width(self, width: float) -> UIButton` | 调用 Width。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UICheckbox`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UICheckbox<br>Bind(self, arg0: Any, arg1: int) -> UICheckbox` | 调用 Bind。 |  |
| `Callback` | method | `Callback(self, func: Callable[[UICheckbox], None]) -> UICheckbox` | 调用 Callback。 |  |
| `Checked` | method | `Checked(self, checked: bool) -> UICheckbox` | 调用 Checked。 |  |
| `Id` | method | `Id(self, id: str) -> UICheckbox` | 调用 Id。 |  |
| `Label` | method | `Label(self, label: str) -> UICheckbox` | 调用 Label。 |  |
| `checked` | property | `checked(self) -> bool` | 属性：checked。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIConditional`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIConditional<br>Bind(self, arg0: Callable[[], bool]) -> UIConditional` | 调用 Bind。 |  |
| `append` | method | `append(self, *args) -> UIConditional` | 调用 append。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIDisplayText`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIDisplayText<br>Bind(self, arg0: Callable[[], str]) -> UIDisplayText` | 调用 Bind。 |  |
| `Text` | method | `Text(self, text: str) -> UIDisplayText` | 调用 Text。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIDummy`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Height` | method | `Height(self, arg0: float) -> UIDummy` | 调用 Height。 |  |
| `Width` | method | `Width(self, arg0: float) -> UIDummy` | 调用 Width。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIDuration`

- Use: internal_renderer ImGui UI widget。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `keyframe0` | method | `keyframe0(self) -> UIKeyframe` | 调用 keyframe0。 |  |
| `keyframe1` | method | `keyframe1(self) -> UIKeyframe` | 调用 keyframe1。 |  |
| `name` | method | `name(self) -> str` | 调用 name。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIFileChooser`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Callback` | method | `Callback(self, func: Callable[[UIFileChooser, str, str], None]) -> UIFileChooser` | 调用 Callback。 |  |
| `Filter` | method | `Filter(self, filter: str) -> UIFileChooser` | 调用 Filter。 |  |
| `Id` | method | `Id(self, id: str) -> UIFileChooser` | 调用 Id。 |  |
| `Label` | method | `Label(self, label: str) -> UIFileChooser` | 调用 Label。 |  |
| `Path` | method | `Path(self, path: str) -> UIFileChooser` | 调用 Path。 |  |
| `Title` | method | `Title(self, title: str) -> UIFileChooser` | 调用 Title。 |  |
| `close` | method | `close(self) -> None` | 调用 close。 |  |
| `open` | method | `open(self) -> None` | 调用 open。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIGizmo`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIGizmo` | 调用 Bind。 |  |
| `CameraMatrices` | method | `CameraMatrices(self, arg0: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple, arg1: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> None` | 调用 CameraMatrices。 |  |
| `Matrix` | method | `Matrix(self, matrix: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> UIGizmo` | 调用 Matrix。 |  |
| `matrix` | property | `matrix(self) -> np.ndarray[Any, np.dtype[np.float32]]` | 属性：matrix。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIInputFloat`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIInputFloat` | 调用 Bind。 |  |
| `Callback` | method | `Callback(self, func: Callable[[UIInputFloat], None]) -> UIInputFloat` | 调用 Callback。 |  |
| `Id` | method | `Id(self, id: str) -> UIInputFloat` | 调用 Id。 |  |
| `Label` | method | `Label(self, label: str) -> UIInputFloat` | 调用 Label。 |  |
| `ReadOnly` | method | `ReadOnly(self, read_only: bool) -> UIInputFloat` | 调用 ReadOnly。 |  |
| `Value` | method | `Value(self, value: float) -> UIInputFloat` | 调用 Value。 |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UIInputFloat` | 调用 WidthRatio。 |  |
| `value` | property | `value(self) -> float` | 属性：value。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIInputFloat2`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIInputFloat2` | 调用 Bind。 |  |
| `Callback` | method | `Callback(self, func: Callable[[UIInputFloat2], None]) -> UIInputFloat2` | 调用 Callback。 |  |
| `Id` | method | `Id(self, id: str) -> UIInputFloat2` | 调用 Id。 |  |
| `Label` | method | `Label(self, label: str) -> UIInputFloat2` | 调用 Label。 |  |
| `ReadOnly` | method | `ReadOnly(self, read_only: bool) -> UIInputFloat2` | 调用 ReadOnly。 |  |
| `Value` | method | `Value(self, value: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> UIInputFloat2` | 调用 Value。 |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UIInputFloat2` | 调用 WidthRatio。 |  |
| `value` | property | `value(self) -> np.ndarray[Any, np.dtype[np.float32]]` | 属性：value。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIInputFloat3`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIInputFloat3` | 调用 Bind。 |  |
| `Callback` | method | `Callback(self, func: Callable[[UIInputFloat3], None]) -> UIInputFloat3` | 调用 Callback。 |  |
| `Id` | method | `Id(self, id: str) -> UIInputFloat3` | 调用 Id。 |  |
| `Label` | method | `Label(self, label: str) -> UIInputFloat3` | 调用 Label。 |  |
| `ReadOnly` | method | `ReadOnly(self, read_only: bool) -> UIInputFloat3` | 调用 ReadOnly。 |  |
| `Value` | method | `Value(self, value: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> UIInputFloat3` | 调用 Value。 |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UIInputFloat3` | 调用 WidthRatio。 |  |
| `value` | property | `value(self) -> np.ndarray[Any, np.dtype[np.float32]]` | 属性：value。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIInputFloat4`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIInputFloat4` | 调用 Bind。 |  |
| `Callback` | method | `Callback(self, func: Callable[[UIInputFloat4], None]) -> UIInputFloat4` | 调用 Callback。 |  |
| `Id` | method | `Id(self, id: str) -> UIInputFloat4` | 调用 Id。 |  |
| `Label` | method | `Label(self, label: str) -> UIInputFloat4` | 调用 Label。 |  |
| `ReadOnly` | method | `ReadOnly(self, read_only: bool) -> UIInputFloat4` | 调用 ReadOnly。 |  |
| `Value` | method | `Value(self, value: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> UIInputFloat4` | 调用 Value。 |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UIInputFloat4` | 调用 WidthRatio。 |  |
| `value` | property | `value(self) -> np.ndarray[Any, np.dtype[np.float32]]` | 属性：value。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIInputInt`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIInputInt` | 调用 Bind。 |  |
| `Callback` | method | `Callback(self, func: Callable[[UIInputInt], None]) -> UIInputInt` | 调用 Callback。 |  |
| `Id` | method | `Id(self, id: str) -> UIInputInt` | 调用 Id。 |  |
| `Label` | method | `Label(self, label: str) -> UIInputInt` | 调用 Label。 |  |
| `ReadOnly` | method | `ReadOnly(self, read_only: bool) -> UIInputInt` | 调用 ReadOnly。 |  |
| `Value` | method | `Value(self, value: int) -> UIInputInt` | 调用 Value。 |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UIInputInt` | 调用 WidthRatio。 |  |
| `value` | property | `value(self) -> int` | 属性：value。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIInputInt2`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIInputInt2` | 调用 Bind。 |  |
| `Callback` | method | `Callback(self, func: Callable[[UIInputInt2], None]) -> UIInputInt2` | 调用 Callback。 |  |
| `Id` | method | `Id(self, id: str) -> UIInputInt2` | 调用 Id。 |  |
| `Label` | method | `Label(self, label: str) -> UIInputInt2` | 调用 Label。 |  |
| `ReadOnly` | method | `ReadOnly(self, read_only: bool) -> UIInputInt2` | 调用 ReadOnly。 |  |
| `Value` | method | `Value(self, value: np.ndarray[Any, np.dtype[np.int32]] \| list \| tuple) -> UIInputInt2` | 调用 Value。 |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UIInputInt2` | 调用 WidthRatio。 |  |
| `value` | property | `value(self) -> np.ndarray[Any, np.dtype[np.int32]]` | 属性：value。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIInputInt3`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIInputInt3` | 调用 Bind。 |  |
| `Callback` | method | `Callback(self, func: Callable[[UIInputInt3], None]) -> UIInputInt3` | 调用 Callback。 |  |
| `Id` | method | `Id(self, id: str) -> UIInputInt3` | 调用 Id。 |  |
| `Label` | method | `Label(self, label: str) -> UIInputInt3` | 调用 Label。 |  |
| `ReadOnly` | method | `ReadOnly(self, read_only: bool) -> UIInputInt3` | 调用 ReadOnly。 |  |
| `Value` | method | `Value(self, value: np.ndarray[Any, np.dtype[np.int32]] \| list \| tuple) -> UIInputInt3` | 调用 Value。 |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UIInputInt3` | 调用 WidthRatio。 |  |
| `value` | property | `value(self) -> np.ndarray[Any, np.dtype[np.int32]]` | 属性：value。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIInputInt4`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIInputInt4` | 调用 Bind。 |  |
| `Callback` | method | `Callback(self, func: Callable[[UIInputInt4], None]) -> UIInputInt4` | 调用 Callback。 |  |
| `Id` | method | `Id(self, id: str) -> UIInputInt4` | 调用 Id。 |  |
| `Label` | method | `Label(self, label: str) -> UIInputInt4` | 调用 Label。 |  |
| `ReadOnly` | method | `ReadOnly(self, read_only: bool) -> UIInputInt4` | 调用 ReadOnly。 |  |
| `Value` | method | `Value(self, value: np.ndarray[Any, np.dtype[np.int32]] \| list \| tuple) -> UIInputInt4` | 调用 Value。 |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UIInputInt4` | 调用 WidthRatio。 |  |
| `value` | property | `value(self) -> np.ndarray[Any, np.dtype[np.int32]]` | 属性：value。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIInputText`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Callback` | method | `Callback(self, func: Callable[[UIInputText], None]) -> UIInputText` | 调用 Callback。 |  |
| `Id` | method | `Id(self, id: str) -> UIInputText` | 调用 Id。 |  |
| `Label` | method | `Label(self, label: str) -> UIInputText` | 调用 Label。 |  |
| `ReadOnly` | method | `ReadOnly(self, read_only: bool) -> UIInputText` | 调用 ReadOnly。 |  |
| `Size` | method | `Size(self, size: int) -> UIInputText` | 调用 Size。 |  |
| `Value` | method | `Value(self, value: str) -> UIInputText` | 调用 Value。 |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UIInputText` | 调用 WidthRatio。 |  |
| `value` | property | `value(self) -> str` | 属性：value。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIInputTextMultiline`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Callback` | method | `Callback(self, func: Callable[[UIInputTextMultiline], None]) -> UIInputTextMultiline` | 调用 Callback。 |  |
| `Id` | method | `Id(self, id: str) -> UIInputTextMultiline` | 调用 Id。 |  |
| `Label` | method | `Label(self, label: str) -> UIInputTextMultiline` | 调用 Label。 |  |
| `ReadOnly` | method | `ReadOnly(self, read_only: bool) -> UIInputTextMultiline` | 调用 ReadOnly。 |  |
| `Size` | method | `Size(self, size: int) -> UIInputTextMultiline` | 调用 Size。 |  |
| `Value` | method | `Value(self, value: str) -> UIInputTextMultiline` | 调用 Value。 |  |
| `value` | property | `value(self) -> str` | 属性：value。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIKeyframe`

- Use: internal_renderer ImGui UI widget。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `frame` | method | `frame(self) -> int` | 调用 frame。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIKeyframeEditor`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `AddDurationCallback` | method | `AddDurationCallback(self, func: Callable[[UIKeyframe, UIKeyframe], None]) -> UIKeyframeEditor` | 调用 AddDurationCallback。 |  |
| `AddKeyframeCallback` | method | `AddKeyframeCallback(self, func: Callable[[int], None]) -> UIKeyframeEditor` | 调用 AddKeyframeCallback。 |  |
| `BindCurrentFrame` | method | `BindCurrentFrame(self, arg0: Any, arg1: str) -> UIKeyframeEditor` | 调用 BindCurrentFrame。 |  |
| `BindTotalFrames` | method | `BindTotalFrames(self, arg0: Any, arg1: str) -> UIKeyframeEditor` | 调用 BindTotalFrames。 |  |
| `DoubleClickDurationCallback` | method | `DoubleClickDurationCallback(self, func: Callable[[UIDuration], None]) -> UIKeyframeEditor` | 调用 DoubleClickDurationCallback。 |  |
| `DoubleClickKeyframeCallback` | method | `DoubleClickKeyframeCallback(self, func: Callable[[UIKeyframe], None]) -> UIKeyframeEditor` | 调用 DoubleClickKeyframeCallback。 |  |
| `MoveKeyframeCallback` | method | `MoveKeyframeCallback(self, func: Callable[[UIKeyframe, int], None]) -> UIKeyframeEditor` | 调用 MoveKeyframeCallback。 |  |
| `add_duration` | method | `add_duration(self, duration: UIDuration) -> None` | 添加/创建 duration。 |  |
| `add_keyframe` | method | `add_keyframe(self, keyframe: UIKeyframe) -> None` | 添加/创建 keyframe。 |  |
| `append` | method | `append(self, *args) -> UIKeyframeEditor` | 调用 append。 |  |
| `get_durations` | method | `get_durations(self) -> list[UIDuration]` | 读取 durations。 |  |
| `get_keyframes` | method | `get_keyframes(self) -> list[UIKeyframe]` | 读取 keyframes。 |  |
| `remove_duration` | method | `remove_duration(self, duration: UIDuration) -> None` | 移除 duration。 |  |
| `remove_keyframe` | method | `remove_keyframe(self, keyframe: UIKeyframe) -> None` | 移除 keyframe。 |  |
| `set_state` | method | `set_state(self, keyframes: list[UIKeyframe], durations: list[UIDuration]) -> None` | 设置 state。 |  |
| `__init__` | method | `__init__(self, content_scale: float) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIOptions`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `BindIndex` | method | `BindIndex(self, arg0: Any, arg1: str) -> UIOptions` | 调用 BindIndex。 |  |
| `BindItems` | method | `BindItems(self, arg0: Any, arg1: str) -> UIOptions` | 调用 BindItems。 |  |
| `Callback` | method | `Callback(self, func: Callable[[UIOptions], None]) -> UIOptions` | 调用 Callback。 |  |
| `Id` | method | `Id(self, id: str) -> UIOptions` | 调用 Id。 |  |
| `Index` | method | `Index(self, index: int) -> UIOptions` | 调用 Index。 |  |
| `Items` | method | `Items(self, items: list[str]) -> UIOptions` | 调用 Items。 |  |
| `Label` | method | `Label(self, label: str) -> UIOptions` | 调用 Label。 |  |
| `Style` | method | `Style(self, style: str) -> UIOptions` | 调用 Style。 |  |
| `index` | property | `index(self) -> int` | 属性：index。 |  |
| `value` | property | `value(self) -> str` | 属性：value。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIPicture`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Clear` | method | `Clear(self) -> UIPicture` | 调用 Clear。 |  |
| `Picture` | method | `Picture(self, renderer: Renderer, name: str) -> UIPicture` | 调用 Picture。 |  |
| `Size` | method | `Size(self, x: float, y: float) -> UIPicture` | 调用 Size。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIPopup`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `EscCallback` | method | `EscCallback(self, func: Callable[[], None]) -> UIPopup` | 调用 EscCallback。 |  |
| `Id` | method | `Id(self, id: str) -> UIPopup` | 调用 Id。 |  |
| `Label` | method | `Label(self, label: str) -> UIPopup` | 调用 Label。 |  |
| `append` | method | `append(self, *args) -> UIPopup` | 调用 append。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UISameLine`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Offset` | method | `Offset(self, offset: float) -> UISameLine` | 调用 Offset。 |  |
| `Spacing` | method | `Spacing(self, spacing: float) -> UISameLine` | 调用 Spacing。 |  |
| `append` | method | `append(self, *args) -> UISameLine` | 调用 append。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UISection`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Expanded` | method | `Expanded(self, expanded: bool) -> UISection` | 调用 Expanded。 |  |
| `Id` | method | `Id(self, id: str) -> UISection` | 调用 Id。 |  |
| `Label` | method | `Label(self, label: str) -> UISection` | 调用 Label。 |  |
| `append` | method | `append(self, *args) -> UISection` | 调用 append。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UISelectable`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Callback` | method | `Callback(self, func: Callable[[UISelectable], None]) -> UISelectable` | 调用 Callback。 |  |
| `Id` | method | `Id(self, id: str) -> UISelectable` | 调用 Id。 |  |
| `Label` | method | `Label(self, label: str) -> UISelectable` | 调用 Label。 |  |
| `Selected` | method | `Selected(self, selected: bool) -> UISelectable` | 调用 Selected。 |  |
| `value` | property | `value(self) -> bool` | 属性：value。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UISliderAngle`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UISliderAngle` | 调用 Bind。 |  |
| `Callback` | method | `Callback(self, func: Callable[[UISliderAngle], None]) -> UISliderAngle` | 调用 Callback。 |  |
| `Id` | method | `Id(self, id: str) -> UISliderAngle` | 调用 Id。 |  |
| `Label` | method | `Label(self, label: str) -> UISliderAngle` | 调用 Label。 |  |
| `Max` | method | `Max(self, max: float) -> UISliderAngle` | 调用 Max。 |  |
| `Min` | method | `Min(self, min: float) -> UISliderAngle` | 调用 Min。 |  |
| `Value` | method | `Value(self, value: float) -> UISliderAngle` | 调用 Value。 |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UISliderAngle` | 调用 WidthRatio。 |  |
| `value` | property | `value(self) -> float` | 属性：value。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UISliderFloat`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UISliderFloat` | 调用 Bind。 |  |
| `Callback` | method | `Callback(self, func: Callable[[UISliderFloat], None]) -> UISliderFloat` | 调用 Callback。 |  |
| `Id` | method | `Id(self, id: str) -> UISliderFloat` | 调用 Id。 |  |
| `Label` | method | `Label(self, label: str) -> UISliderFloat` | 调用 Label。 |  |
| `Max` | method | `Max(self, max: float) -> UISliderFloat` | 调用 Max。 |  |
| `Min` | method | `Min(self, min: float) -> UISliderFloat` | 调用 Min。 |  |
| `Value` | method | `Value(self, value: float) -> UISliderFloat` | 调用 Value。 |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UISliderFloat` | 调用 WidthRatio。 |  |
| `value` | property | `value(self) -> float` | 属性：value。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UITreeNode`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Id` | method | `Id(self, id: str) -> UITreeNode` | 调用 Id。 |  |
| `Label` | method | `Label(self, label: str) -> UITreeNode` | 调用 Label。 |  |
| `append` | method | `append(self, *args) -> UITreeNode` | 调用 append。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |

## `sapien.internal_renderer.UIWidget`

- Use: internal_renderer ImGui UI widget。
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_children` | method | `get_children(self) -> list[UIWidget]` | 读取 children。 |  |
| `remove` | method | `remove(self) -> None` | 调用 remove。 |  |
| `remove_children` | method | `remove_children(self) -> None` | 移除 children。 |  |

## `sapien.internal_renderer.UIWindow`

- Use: internal_renderer ImGui UI widget。
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Id` | method | `Id(self, id: str) -> UIWindow` | 调用 Id。 |  |
| `Label` | method | `Label(self, label: str) -> UIWindow` | 调用 Label。 |  |
| `Pos` | method | `Pos(self, x: float, y: float) -> UIWindow` | 调用 Pos。 |  |
| `Size` | method | `Size(self, x: float, y: float) -> UIWindow` | 调用 Size。 |  |
| `append` | method | `append(self, *args) -> UIWindow` | 调用 append。 |  |
| `__init__` | method | `__init__(self) -> None` | Python special method。 |  |
