# Internal renderer / UI API (`sapien.internal_renderer`)

Agent-facing compact API table. Source of truth is checked-in `.pyi`/wrapper source; verify pybind/C++ when behavior matters.

- Preferred import prefix: `sapien.internal_renderer`
- Source files: `python/py_package/pysapien/internal_renderer.pyi`
- Notes: Low-level renderer and ImGui widgets used by viewer/tools; prefer `sapien.render` for simulation code unless extending viewer/UI.

## Class index

| Class | Bases | Use | Notes |
|---|---|---|---|
| `sapien.internal_renderer.Context` |  | SAPIEN API object. |  |
| `sapien.internal_renderer.Cubemap` |  | SAPIEN API object. |  |
| `sapien.internal_renderer.LineSet` | `PrimitiveSet` | SAPIEN API object. |  |
| `sapien.internal_renderer.LineSetObject` | `Node` | SAPIEN API object. |  |
| `sapien.internal_renderer.Material` |  | SAPIEN API object. |  |
| `sapien.internal_renderer.Mesh` |  | SAPIEN API object. |  |
| `sapien.internal_renderer.Model` |  | SAPIEN API object. |  |
| `sapien.internal_renderer.Node` |  | SAPIEN API object. |  |
| `sapien.internal_renderer.Object` | `Node` | SAPIEN API object. |  |
| `sapien.internal_renderer.PointSet` | `PrimitiveSet` | SAPIEN API object. |  |
| `sapien.internal_renderer.PointSetObject` | `Node` | SAPIEN API object. |  |
| `sapien.internal_renderer.PrimitiveSet` |  | SAPIEN API object. |  |
| `sapien.internal_renderer.Renderer` |  | SAPIEN render API object. |  |
| `sapien.internal_renderer.Scene` |  | Entity/system container; high-level wrapper also provides builder, lights, camera, ground, and other convenience functions. |  |
| `sapien.internal_renderer.Shape` |  | SAPIEN API object. |  |
| `sapien.internal_renderer.Texture` |  | SAPIEN API object. |  |
| `sapien.internal_renderer.UIButton` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UICheckbox` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIConditional` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIDisplayText` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIDummy` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIDuration` |  | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIFileChooser` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIGizmo` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIInputFloat` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIInputFloat2` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIInputFloat3` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIInputFloat4` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIInputInt` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIInputInt2` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIInputInt3` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIInputInt4` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIInputText` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIInputTextMultiline` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIKeyframe` |  | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIKeyframeEditor` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIOptions` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIPicture` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIPopup` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UISameLine` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UISection` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UISelectable` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UISliderAngle` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UISliderFloat` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UITreeNode` | `UIWidget` | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIWidget` |  | internal_renderer ImGui UI widget |  |
| `sapien.internal_renderer.UIWindow` | `UIWidget` | internal_renderer ImGui UI widget |  |

## `sapien.internal_renderer.Context`

- Use: SAPIEN API object.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `create_box_mesh` | method | `create_box_mesh(self) -> Mesh` | Create a box mesh. |  |
| `create_brdf_lut` | method | `create_brdf_lut(self, size: int=128) -> Texture` | Generate BRDF LUT texture, see https://learnopengl.com/PBR/IBL/Specular-IBL |  |
| `create_capsule_mesh` | method | `create_capsule_mesh(self, radius: float, half_length: float, segments: int=32, half_rings: int=8) -> Mesh` | Create a capsule mesh. |  |
| `create_cone_mesh` | method | `create_cone_mesh(self, segments: int=32) -> Mesh` | Create a cone mesh. |  |
| `create_cubemap_from_files` | method | `create_cubemap_from_files(self, filenames: Annotated[list[str], FixedSize(6)], mipmap_levels: int) -> Cubemap` | Load cube map, its mipmaps are generated based on roughness, details see https://learnopengl.com/PBR/IBL/Specular-IBL |  |
| `create_cylinder_mesh` | method | `create_cylinder_mesh(self, segments: int=32) -> Mesh` | Create a cylinder mesh. |  |
| `create_line_set` | method | `create_line_set(self, vertices: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple, colors: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> PointSet` | Create a line set. |  |
| `create_material` | method | `create_material(self, emission: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple, base_color: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple, specular: float, roughness: float, metallic: float, transmission: float=0.0, ior: float=1.0099999904632568) -> Material` | Create a material. |  |
| `create_mesh_from_array` | method | `create_mesh_from_array(self, vertices: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple, indices: np.ndarray[np.uint32], normals: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple=..., uvs: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple=...) -> Mesh` | Create a mesh from an array. |  |
| `create_model` | method | `create_model(self, meshes: list[Mesh], materials: list[Material]) -> Model` | Create a model. |  |
| `create_model_from_file` | method | `create_model_from_file(self, filename: str) -> Model` | Create a model from a file. |  |
| `create_point_set` | method | `create_point_set(self, vertices: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple, colors: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> LineSet` | Create a point set. |  |
| `create_texture_from_file` | method | `create_texture_from_file(self, filename: str, mipmap_levels: int, filter: str='linear', address_mode: str='repeat') -> Texture` | Create a texture from a file. |  |
| `create_uvsphere_mesh` | method | `create_uvsphere_mesh(self, segments: int=32, half_rings: int=16) -> Mesh` | Create a UV-sphere mesh. |  |

## `sapien.internal_renderer.Cubemap`

- Use: SAPIEN API object.
- Bases: `-`

## `sapien.internal_renderer.LineSet`

- Use: SAPIEN API object.
- Bases: `PrimitiveSet`

## `sapien.internal_renderer.LineSetObject`

- Use: SAPIEN API object.
- Bases: `Node`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `line_width` | `float` |  |  |

## `sapien.internal_renderer.Material`

- Use: SAPIEN API object.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `set_base_color` | method | `set_base_color(self, rgba: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> None` |  |  |
| `set_emission` | method | `set_emission(self, emission: float) -> None` |  |  |
| `set_metallic` | method | `set_metallic(self, metallic: float) -> None` |  |  |
| `set_roughness` | method | `set_roughness(self, roughness: float) -> None` |  |  |
| `set_specular` | method | `set_specular(self, specular: float) -> None` |  |  |
| `set_textures` | method | `set_textures(self, base_color: Texture \| None=None, roughness: Texture \| None=None, normal: Texture \| None=None, metallic: Texture \| None=None, emission: Texture \| None=None, transmission: Texture \| None=None) -> None` |  |  |
| `set_transmission` | method | `set_transmission(self, transmission: float) -> None` |  |  |

## `sapien.internal_renderer.Mesh`

- Use: SAPIEN API object.
- Bases: `-`

## `sapien.internal_renderer.Model`

- Use: SAPIEN API object.
- Bases: `-`

## `sapien.internal_renderer.Node`

- Use: SAPIEN API object.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `children` | property | `children(self) -> list[Node]` |  |  |
| `position` | property | `position(self) -> np.ndarray[Any, np.dtype[np.float32]]` |  |  |
| `rotation` | property | `rotation(self) -> np.ndarray[Any, np.dtype[np.float32]]` |  |  |
| `scale` | property | `scale(self) -> np.ndarray[Any, np.dtype[np.float32]]` |  |  |
| `set_position` | method | `set_position(self, position: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> None` |  |  |
| `set_rotation` | method | `set_rotation(self, quat: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> None` |  |  |
| `set_scale` | method | `set_scale(self, scale: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> None` |  |  |

## `sapien.internal_renderer.Object`

- Use: SAPIEN API object.
- Bases: `Node`

### Attributes/properties declared as fields

| Member | Type | Use | Notes |
|---|---|---|---|
| `cast_shadow` | `bool` |  |  |
| `shading_mode` | `int` |  |  |
| `transparency` | `float` |  |  |

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_segmentation` | method | `get_segmentation(self) -> np.ndarray[np.uint32]` |  |  |
| `model` | property | `model(self) -> Model` |  |  |
| `set_segmentation` | method | `set_segmentation(self, arg0: np.ndarray[np.uint32]) -> None` |  |  |

## `sapien.internal_renderer.PointSet`

- Use: SAPIEN API object.
- Bases: `PrimitiveSet`

## `sapien.internal_renderer.PointSetObject`

- Use: SAPIEN API object.
- Bases: `Node`

## `sapien.internal_renderer.PrimitiveSet`

- Use: SAPIEN API object.
- Bases: `-`

## `sapien.internal_renderer.Renderer`

- Use: SAPIEN render API object.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `set_custom_cubemap` | method | `set_custom_cubemap(self, name: str, texture: Cubemap) -> None` |  |  |
| `set_custom_property` | method | `set_custom_property(self, name: str, value: float) -> None<br>set_custom_property(self, name: str, value: int) -> None` |  |  |
| `set_custom_texture` | method | `set_custom_texture(self, name: str, texture: Texture) -> None` |  |  |

## `sapien.internal_renderer.Scene`

- Use: Entity/system container; high-level wrapper also provides builder, lights, camera, ground, and other convenience functions.
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `add_line_set` | method | `add_line_set(self, line_set: PointSet, parent: Node \| None=None) -> LineSetObject` |  |  |
| `add_node` | method | `add_node(self, parent: Node \| None=None) -> Node` |  |  |
| `add_object` | method | `add_object(self, model: Model, parent: Node \| None=None) -> Object` |  |  |
| `add_point_set` | method | `add_point_set(self, point_set: LineSet, parent: Node \| None=None) -> PointSetObject` |  |  |
| `force_rebuild` | method | `force_rebuild(self) -> None` |  |  |
| `force_update` | method | `force_update(self) -> None` |  |  |
| `remove_node` | method | `remove_node(self, node: Node) -> None` |  |  |
| `set_ambient_light` | method | `set_ambient_light(self, arg0: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> None` |  |  |
| `set_cubemap` | method | `set_cubemap(self, arg0: Cubemap) -> None` |  |  |

## `sapien.internal_renderer.Shape`

- Use: SAPIEN API object.
- Bases: `-`

## `sapien.internal_renderer.Texture`

- Use: SAPIEN API object.
- Bases: `-`

## `sapien.internal_renderer.UIButton`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Callback` | method | `Callback(self, func: Callable[[UIButton], None]) -> UIButton` |  |  |
| `Id` | method | `Id(self, id: str) -> UIButton` |  |  |
| `Label` | method | `Label(self, label: str) -> UIButton` |  |  |
| `Width` | method | `Width(self, width: float) -> UIButton` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UICheckbox`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UICheckbox<br>Bind(self, arg0: Any, arg1: int) -> UICheckbox` |  |  |
| `Callback` | method | `Callback(self, func: Callable[[UICheckbox], None]) -> UICheckbox` |  |  |
| `Checked` | method | `Checked(self, checked: bool) -> UICheckbox` |  |  |
| `Id` | method | `Id(self, id: str) -> UICheckbox` |  |  |
| `Label` | method | `Label(self, label: str) -> UICheckbox` |  |  |
| `checked` | property | `checked(self) -> bool` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UIConditional`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIConditional<br>Bind(self, arg0: Callable[[], bool]) -> UIConditional` |  |  |
| `append` | method | `append(self, *args) -> UIConditional` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UIDisplayText`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIDisplayText<br>Bind(self, arg0: Callable[[], str]) -> UIDisplayText` |  |  |
| `Text` | method | `Text(self, text: str) -> UIDisplayText` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UIDummy`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Height` | method | `Height(self, arg0: float) -> UIDummy` |  |  |
| `Width` | method | `Width(self, arg0: float) -> UIDummy` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UIDuration`

- Use: internal_renderer ImGui UI widget
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `keyframe0` | method | `keyframe0(self) -> UIKeyframe` |  |  |
| `keyframe1` | method | `keyframe1(self) -> UIKeyframe` |  |  |
| `name` | method | `name(self) -> str` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UIFileChooser`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Callback` | method | `Callback(self, func: Callable[[UIFileChooser, str, str], None]) -> UIFileChooser` |  |  |
| `Filter` | method | `Filter(self, filter: str) -> UIFileChooser` |  |  |
| `Id` | method | `Id(self, id: str) -> UIFileChooser` |  |  |
| `Label` | method | `Label(self, label: str) -> UIFileChooser` |  |  |
| `Path` | method | `Path(self, path: str) -> UIFileChooser` |  |  |
| `Title` | method | `Title(self, title: str) -> UIFileChooser` |  |  |
| `close` | method | `close(self) -> None` |  |  |
| `open` | method | `open(self) -> None` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UIGizmo`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIGizmo` |  |  |
| `CameraMatrices` | method | `CameraMatrices(self, arg0: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple, arg1: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> None` |  |  |
| `Matrix` | method | `Matrix(self, matrix: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> UIGizmo` |  |  |
| `matrix` | property | `matrix(self) -> np.ndarray[Any, np.dtype[np.float32]]` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UIInputFloat`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIInputFloat` |  |  |
| `Callback` | method | `Callback(self, func: Callable[[UIInputFloat], None]) -> UIInputFloat` |  |  |
| `Id` | method | `Id(self, id: str) -> UIInputFloat` |  |  |
| `Label` | method | `Label(self, label: str) -> UIInputFloat` |  |  |
| `ReadOnly` | method | `ReadOnly(self, read_only: bool) -> UIInputFloat` |  |  |
| `Value` | method | `Value(self, value: float) -> UIInputFloat` |  |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UIInputFloat` |  |  |
| `value` | property | `value(self) -> float` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UIInputFloat2`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIInputFloat2` |  |  |
| `Callback` | method | `Callback(self, func: Callable[[UIInputFloat2], None]) -> UIInputFloat2` |  |  |
| `Id` | method | `Id(self, id: str) -> UIInputFloat2` |  |  |
| `Label` | method | `Label(self, label: str) -> UIInputFloat2` |  |  |
| `ReadOnly` | method | `ReadOnly(self, read_only: bool) -> UIInputFloat2` |  |  |
| `Value` | method | `Value(self, value: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> UIInputFloat2` |  |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UIInputFloat2` |  |  |
| `value` | property | `value(self) -> np.ndarray[Any, np.dtype[np.float32]]` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UIInputFloat3`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIInputFloat3` |  |  |
| `Callback` | method | `Callback(self, func: Callable[[UIInputFloat3], None]) -> UIInputFloat3` |  |  |
| `Id` | method | `Id(self, id: str) -> UIInputFloat3` |  |  |
| `Label` | method | `Label(self, label: str) -> UIInputFloat3` |  |  |
| `ReadOnly` | method | `ReadOnly(self, read_only: bool) -> UIInputFloat3` |  |  |
| `Value` | method | `Value(self, value: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> UIInputFloat3` |  |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UIInputFloat3` |  |  |
| `value` | property | `value(self) -> np.ndarray[Any, np.dtype[np.float32]]` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UIInputFloat4`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIInputFloat4` |  |  |
| `Callback` | method | `Callback(self, func: Callable[[UIInputFloat4], None]) -> UIInputFloat4` |  |  |
| `Id` | method | `Id(self, id: str) -> UIInputFloat4` |  |  |
| `Label` | method | `Label(self, label: str) -> UIInputFloat4` |  |  |
| `ReadOnly` | method | `ReadOnly(self, read_only: bool) -> UIInputFloat4` |  |  |
| `Value` | method | `Value(self, value: np.ndarray[Any, np.dtype[np.float32]] \| list \| tuple) -> UIInputFloat4` |  |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UIInputFloat4` |  |  |
| `value` | property | `value(self) -> np.ndarray[Any, np.dtype[np.float32]]` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UIInputInt`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIInputInt` |  |  |
| `Callback` | method | `Callback(self, func: Callable[[UIInputInt], None]) -> UIInputInt` |  |  |
| `Id` | method | `Id(self, id: str) -> UIInputInt` |  |  |
| `Label` | method | `Label(self, label: str) -> UIInputInt` |  |  |
| `ReadOnly` | method | `ReadOnly(self, read_only: bool) -> UIInputInt` |  |  |
| `Value` | method | `Value(self, value: int) -> UIInputInt` |  |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UIInputInt` |  |  |
| `value` | property | `value(self) -> int` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UIInputInt2`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIInputInt2` |  |  |
| `Callback` | method | `Callback(self, func: Callable[[UIInputInt2], None]) -> UIInputInt2` |  |  |
| `Id` | method | `Id(self, id: str) -> UIInputInt2` |  |  |
| `Label` | method | `Label(self, label: str) -> UIInputInt2` |  |  |
| `ReadOnly` | method | `ReadOnly(self, read_only: bool) -> UIInputInt2` |  |  |
| `Value` | method | `Value(self, value: np.ndarray[Any, np.dtype[np.int32]] \| list \| tuple) -> UIInputInt2` |  |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UIInputInt2` |  |  |
| `value` | property | `value(self) -> np.ndarray[Any, np.dtype[np.int32]]` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UIInputInt3`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIInputInt3` |  |  |
| `Callback` | method | `Callback(self, func: Callable[[UIInputInt3], None]) -> UIInputInt3` |  |  |
| `Id` | method | `Id(self, id: str) -> UIInputInt3` |  |  |
| `Label` | method | `Label(self, label: str) -> UIInputInt3` |  |  |
| `ReadOnly` | method | `ReadOnly(self, read_only: bool) -> UIInputInt3` |  |  |
| `Value` | method | `Value(self, value: np.ndarray[Any, np.dtype[np.int32]] \| list \| tuple) -> UIInputInt3` |  |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UIInputInt3` |  |  |
| `value` | property | `value(self) -> np.ndarray[Any, np.dtype[np.int32]]` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UIInputInt4`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UIInputInt4` |  |  |
| `Callback` | method | `Callback(self, func: Callable[[UIInputInt4], None]) -> UIInputInt4` |  |  |
| `Id` | method | `Id(self, id: str) -> UIInputInt4` |  |  |
| `Label` | method | `Label(self, label: str) -> UIInputInt4` |  |  |
| `ReadOnly` | method | `ReadOnly(self, read_only: bool) -> UIInputInt4` |  |  |
| `Value` | method | `Value(self, value: np.ndarray[Any, np.dtype[np.int32]] \| list \| tuple) -> UIInputInt4` |  |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UIInputInt4` |  |  |
| `value` | property | `value(self) -> np.ndarray[Any, np.dtype[np.int32]]` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UIInputText`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Callback` | method | `Callback(self, func: Callable[[UIInputText], None]) -> UIInputText` |  |  |
| `Id` | method | `Id(self, id: str) -> UIInputText` |  |  |
| `Label` | method | `Label(self, label: str) -> UIInputText` |  |  |
| `ReadOnly` | method | `ReadOnly(self, read_only: bool) -> UIInputText` |  |  |
| `Size` | method | `Size(self, size: int) -> UIInputText` |  |  |
| `Value` | method | `Value(self, value: str) -> UIInputText` |  |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UIInputText` |  |  |
| `value` | property | `value(self) -> str` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UIInputTextMultiline`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Callback` | method | `Callback(self, func: Callable[[UIInputTextMultiline], None]) -> UIInputTextMultiline` |  |  |
| `Id` | method | `Id(self, id: str) -> UIInputTextMultiline` |  |  |
| `Label` | method | `Label(self, label: str) -> UIInputTextMultiline` |  |  |
| `ReadOnly` | method | `ReadOnly(self, read_only: bool) -> UIInputTextMultiline` |  |  |
| `Size` | method | `Size(self, size: int) -> UIInputTextMultiline` |  |  |
| `Value` | method | `Value(self, value: str) -> UIInputTextMultiline` |  |  |
| `value` | property | `value(self) -> str` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UIKeyframe`

- Use: internal_renderer ImGui UI widget
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `frame` | method | `frame(self) -> int` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UIKeyframeEditor`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `AddDurationCallback` | method | `AddDurationCallback(self, func: Callable[[UIKeyframe, UIKeyframe], None]) -> UIKeyframeEditor` |  |  |
| `AddKeyframeCallback` | method | `AddKeyframeCallback(self, func: Callable[[int], None]) -> UIKeyframeEditor` |  |  |
| `BindCurrentFrame` | method | `BindCurrentFrame(self, arg0: Any, arg1: str) -> UIKeyframeEditor` |  |  |
| `BindTotalFrames` | method | `BindTotalFrames(self, arg0: Any, arg1: str) -> UIKeyframeEditor` |  |  |
| `DoubleClickDurationCallback` | method | `DoubleClickDurationCallback(self, func: Callable[[UIDuration], None]) -> UIKeyframeEditor` |  |  |
| `DoubleClickKeyframeCallback` | method | `DoubleClickKeyframeCallback(self, func: Callable[[UIKeyframe], None]) -> UIKeyframeEditor` |  |  |
| `MoveKeyframeCallback` | method | `MoveKeyframeCallback(self, func: Callable[[UIKeyframe, int], None]) -> UIKeyframeEditor` |  |  |
| `add_duration` | method | `add_duration(self, duration: UIDuration) -> None` |  |  |
| `add_keyframe` | method | `add_keyframe(self, keyframe: UIKeyframe) -> None` |  |  |
| `append` | method | `append(self, *args) -> UIKeyframeEditor` |  |  |
| `get_durations` | method | `get_durations(self) -> list[UIDuration]` |  |  |
| `get_keyframes` | method | `get_keyframes(self) -> list[UIKeyframe]` |  |  |
| `remove_duration` | method | `remove_duration(self, duration: UIDuration) -> None` |  |  |
| `remove_keyframe` | method | `remove_keyframe(self, keyframe: UIKeyframe) -> None` |  |  |
| `set_state` | method | `set_state(self, keyframes: list[UIKeyframe], durations: list[UIDuration]) -> None` |  |  |
| `__init__` | method | `__init__(self, content_scale: float) -> None` |  |  |

## `sapien.internal_renderer.UIOptions`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `BindIndex` | method | `BindIndex(self, arg0: Any, arg1: str) -> UIOptions` |  |  |
| `BindItems` | method | `BindItems(self, arg0: Any, arg1: str) -> UIOptions` |  |  |
| `Callback` | method | `Callback(self, func: Callable[[UIOptions], None]) -> UIOptions` |  |  |
| `Id` | method | `Id(self, id: str) -> UIOptions` |  |  |
| `Index` | method | `Index(self, index: int) -> UIOptions` |  |  |
| `Items` | method | `Items(self, items: list[str]) -> UIOptions` |  |  |
| `Label` | method | `Label(self, label: str) -> UIOptions` |  |  |
| `Style` | method | `Style(self, style: str) -> UIOptions` |  |  |
| `index` | property | `index(self) -> int` |  |  |
| `value` | property | `value(self) -> str` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UIPicture`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Clear` | method | `Clear(self) -> UIPicture` |  |  |
| `Picture` | method | `Picture(self, renderer: Renderer, name: str) -> UIPicture` |  |  |
| `Size` | method | `Size(self, x: float, y: float) -> UIPicture` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UIPopup`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `EscCallback` | method | `EscCallback(self, func: Callable[[], None]) -> UIPopup` |  |  |
| `Id` | method | `Id(self, id: str) -> UIPopup` |  |  |
| `Label` | method | `Label(self, label: str) -> UIPopup` |  |  |
| `append` | method | `append(self, *args) -> UIPopup` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UISameLine`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Offset` | method | `Offset(self, offset: float) -> UISameLine` |  |  |
| `Spacing` | method | `Spacing(self, spacing: float) -> UISameLine` |  |  |
| `append` | method | `append(self, *args) -> UISameLine` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UISection`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Expanded` | method | `Expanded(self, expanded: bool) -> UISection` |  |  |
| `Id` | method | `Id(self, id: str) -> UISection` |  |  |
| `Label` | method | `Label(self, label: str) -> UISection` |  |  |
| `append` | method | `append(self, *args) -> UISection` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UISelectable`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Callback` | method | `Callback(self, func: Callable[[UISelectable], None]) -> UISelectable` |  |  |
| `Id` | method | `Id(self, id: str) -> UISelectable` |  |  |
| `Label` | method | `Label(self, label: str) -> UISelectable` |  |  |
| `Selected` | method | `Selected(self, selected: bool) -> UISelectable` |  |  |
| `value` | property | `value(self) -> bool` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UISliderAngle`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UISliderAngle` |  |  |
| `Callback` | method | `Callback(self, func: Callable[[UISliderAngle], None]) -> UISliderAngle` |  |  |
| `Id` | method | `Id(self, id: str) -> UISliderAngle` |  |  |
| `Label` | method | `Label(self, label: str) -> UISliderAngle` |  |  |
| `Max` | method | `Max(self, max: float) -> UISliderAngle` |  |  |
| `Min` | method | `Min(self, min: float) -> UISliderAngle` |  |  |
| `Value` | method | `Value(self, value: float) -> UISliderAngle` |  |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UISliderAngle` |  |  |
| `value` | property | `value(self) -> float` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UISliderFloat`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Bind` | method | `Bind(self, arg0: Any, arg1: str) -> UISliderFloat` |  |  |
| `Callback` | method | `Callback(self, func: Callable[[UISliderFloat], None]) -> UISliderFloat` |  |  |
| `Id` | method | `Id(self, id: str) -> UISliderFloat` |  |  |
| `Label` | method | `Label(self, label: str) -> UISliderFloat` |  |  |
| `Max` | method | `Max(self, max: float) -> UISliderFloat` |  |  |
| `Min` | method | `Min(self, min: float) -> UISliderFloat` |  |  |
| `Value` | method | `Value(self, value: float) -> UISliderFloat` |  |  |
| `WidthRatio` | method | `WidthRatio(self, width: float) -> UISliderFloat` |  |  |
| `value` | property | `value(self) -> float` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UITreeNode`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Id` | method | `Id(self, id: str) -> UITreeNode` |  |  |
| `Label` | method | `Label(self, label: str) -> UITreeNode` |  |  |
| `append` | method | `append(self, *args) -> UITreeNode` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |

## `sapien.internal_renderer.UIWidget`

- Use: internal_renderer ImGui UI widget
- Bases: `-`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `get_children` | method | `get_children(self) -> list[UIWidget]` |  |  |
| `remove` | method | `remove(self) -> None` |  |  |
| `remove_children` | method | `remove_children(self) -> None` |  |  |

## `sapien.internal_renderer.UIWindow`

- Use: internal_renderer ImGui UI widget
- Bases: `UIWidget`

### Methods/properties

| Member | Kind | Signature | Use | Notes |
|---|---|---|---|---|
| `Id` | method | `Id(self, id: str) -> UIWindow` |  |  |
| `Label` | method | `Label(self, label: str) -> UIWindow` |  |  |
| `Pos` | method | `Pos(self, x: float, y: float) -> UIWindow` |  |  |
| `Size` | method | `Size(self, x: float, y: float) -> UIWindow` |  |  |
| `append` | method | `append(self, *args) -> UIWindow` |  |  |
| `__init__` | method | `__init__(self) -> None` |  |  |