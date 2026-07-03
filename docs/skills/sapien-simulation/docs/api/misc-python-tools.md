# Misc Python Tool APIs

Agent-facing table for non-core Python helper modules. These are useful for assets, URDF conversion/export, and debugging; normal simulation code usually uses `scene-builder-loader.md`, `physx.md`, and `render.md` first.

## `sapien.show_anything`

- Source: `python/py_package/show_anything.py`
- Use: viewer utility / CLI helpers

| API | Signature | Use | Notes |
|---|---|---|---|
| `sapien.show_anything.is_mesh_file` | `is_mesh_file(anything)` | predicate/helper. | internal/helper; verify call sites before relying on stability |
| `sapien.show_anything.is_pcd_file` | `is_pcd_file(anything)` | predicate/helper. | internal/helper; verify call sites before relying on stability |
| `sapien.show_anything.is_urdf_file` | `is_urdf_file(anything)` | predicate/helper. | internal/helper; verify call sites before relying on stability |
| `sapien.show_anything.show_anything` | `show_anything(*args, update_camera=True, loop=True)` | display/inspect helper. | internal/helper; verify call sites before relying on stability |

| Class | Bases | Public methods/properties | Use | Notes |
|---|---|---|---|---|
| `sapien.show_anything.AnythingViewer` | `` | `__init__, add_mesh_file, add_urdf_file, add_pcd_file, get_scene_aabb, focus_scene` | helper class. | not primary simulation API |

## `sapien.wrapper.urdf_exporter`

- Source: `python/py_package/wrapper/urdf_exporter.py`
- Use: export articulation kinematic chains to URDF/XML

| API | Signature | Use | Notes |
|---|---|---|---|
| `sapien.wrapper.urdf_exporter.export_link` | `export_link(link: PhysxArticulationLinkComponent)` | URDF/XML export helper. | internal/helper; verify call sites before relying on stability |
| `sapien.wrapper.urdf_exporter.export_joint` | `export_joint(joint: PhysxArticulationJoint)` | URDF/XML export helper. | internal/helper; verify call sites before relying on stability |
| `sapien.wrapper.urdf_exporter.export_kinematic_chain_xml` | `export_kinematic_chain_xml(articulation)` | URDF/XML export helper. | internal/helper; verify call sites before relying on stability |
| `sapien.wrapper.urdf_exporter.export_kinematic_chain_urdf` | `export_kinematic_chain_urdf(articulation, force_fix_root=False)` | URDF/XML export helper. | internal/helper; verify call sites before relying on stability |

## `sapien.wrapper.coacd`

- Source: `python/py_package/wrapper/coacd.py`
- Use: COACD convex decomposition helper

| API | Signature | Use | Notes |
|---|---|---|---|
| `sapien.wrapper.coacd.get_file_md5` | `get_file_md5(filename)` | read/compute helper. | internal/helper; verify call sites before relying on stability |
| `sapien.wrapper.coacd.do_coacd` | `do_coacd(filename, threshold=0.05, max_convex_hull=-1, preprocess_mode='auto', preprocess_resolution=30, resolution=2000, mcts_nodes=20, mcts_iterations=150, mcts_max_depth=3, pca=False, merge=True, seed=0, verbose=False)` | utility API. | internal/helper; verify call sites before relying on stability |

## `sapien.wrapper.geometry.usd`

- Source: `python/py_package/wrapper/geometry/usd.py`
- Use: USD-to-GLB conversion via Blender

| API | Signature | Use | Notes |
|---|---|---|---|
| `sapien.wrapper.geometry.usd.find_blender` | `find_blender()` | utility API. | internal/helper; verify call sites before relying on stability |
| `sapien.wrapper.geometry.usd.convert_usd_to_glb` | `convert_usd_to_glb(usd_file: str, glb_file: str)` | asset conversion helper. | internal/helper; verify call sites before relying on stability |

## `sapien.wrapper.geometry.cache`

- Source: `python/py_package/wrapper/geometry/cache.py`
- Use: hash/cache helpers for geometry conversion

| API | Signature | Use | Notes |
|---|---|---|---|
| `sapien.wrapper.geometry.cache.get_file_md5` | `get_file_md5(files: Union[str, List[str]])` | read/compute helper. | internal/helper; verify call sites before relying on stability |
| `sapien.wrapper.geometry.cache.file_exists` | `file_exists(files: Union[str, List[str]])` | utility API. | internal/helper; verify call sites before relying on stability |
| `sapien.wrapper.geometry.cache.serialize` | `serialize(input_file, input_md5, output_file, output_md5, param_md5)` | utility API. | internal/helper; verify call sites before relying on stability |
| `sapien.wrapper.geometry.cache.read_hash_file` | `read_hash_file(filename)` | utility API. | internal/helper; verify call sites before relying on stability |
| `sapien.wrapper.geometry.cache.write_hash_file` | `write_hash_file(filename, input_file, input_md5, output_file, output_md5, param_md5)` | utility API. | internal/helper; verify call sites before relying on stability |
| `sapien.wrapper.geometry.cache.check_hash` | `check_hash(hash_file, input_file, input_md5, output_file, output_md5, param_md5)` | utility API. | internal/helper; verify call sites before relying on stability |
| `sapien.wrapper.geometry.cache.cached` | `cached(checksum_suffix)` | utility API. | internal/helper; verify call sites before relying on stability |

## `sapien.wrapper.urchin.utils`

- Source: `python/py_package/wrapper/urchin/utils.py`
- Use: vendored URDF math/path/mesh utilities

| API | Signature | Use | Notes |
|---|---|---|---|
| `sapien.wrapper.urchin.utils.rpy_to_matrix` | `rpy_to_matrix(coords)` | Convert roll-pitch-yaw coordinates to a 3x3 homogenous rotation matrix. The roll-pitch-yaw axes in a typical URDF are defined as a rotati... | internal/helper; verify call sites before relying on stability |
| `sapien.wrapper.urchin.utils.matrix_to_rpy` | `matrix_to_rpy(R, solution=1)` | Convert a 3x3 transform matrix to roll-pitch-yaw coordinates. The roll-pitchRyaw axes in a typical URDF are defined as a rotation of ``r`... | internal/helper; verify call sites before relying on stability |
| `sapien.wrapper.urchin.utils.matrix_to_xyz_rpy` | `matrix_to_xyz_rpy(matrix)` | Convert a 4x4 homogenous matrix to xyzrpy coordinates. Parameters ---------- matrix : (4,4) float The homogenous transform matrix. Return... | internal/helper; verify call sites before relying on stability |
| `sapien.wrapper.urchin.utils.xyz_rpy_to_matrix` | `xyz_rpy_to_matrix(xyz_rpy)` | Convert xyz_rpy coordinates to a 4x4 homogenous matrix. Parameters ---------- xyz_rpy : (6,) float The xyz_rpy vector. Returns ------- ma... | internal/helper; verify call sites before relying on stability |
| `sapien.wrapper.urchin.utils.parse_origin` | `parse_origin(node)` | Find the ``origin`` subelement of an XML node and convert it into a 4x4 homogenous transformation matrix. Parameters ---------- node : :c... | internal/helper; verify call sites before relying on stability |
| `sapien.wrapper.urchin.utils.unparse_origin` | `unparse_origin(matrix)` | Turn a 4x4 homogenous matrix into an ``origin`` XML node. Parameters ---------- matrix : (4,4) float The 4x4 homogneous transform matrix ... | internal/helper; verify call sites before relying on stability |
| `sapien.wrapper.urchin.utils.get_filename` | `get_filename(base_path, file_path, makedirs=False)` | Formats a file path correctly for URDF loading. Parameters ---------- base_path : str The base path to the URDF's folder. file_path : str... | internal/helper; verify call sites before relying on stability |
| `sapien.wrapper.urchin.utils.load_meshes` | `load_meshes(filename)` | Loads triangular meshes from a file. Parameters ---------- filename : str Path to the mesh file. Returns ------- meshes : list of :class:... | internal/helper; verify call sites before relying on stability |
| `sapien.wrapper.urchin.utils.configure_origin` | `configure_origin(value)` | Convert a value into a 4x4 transform matrix. Parameters ---------- value : None, (6,) float, or (4,4) float The value to turn into the ma... | internal/helper; verify call sites before relying on stability |

## `sapien.wrapper.urchin.urdf`

- Source: `python/py_package/wrapper/urchin/urdf.py`
- Use: vendored URDF object model used by URDFLoader

| Class | Bases | Public methods/properties | Use | Notes |
|---|---|---|---|---|
| `sapien.wrapper.urchin.urdf.URDFType` | `object` | `__init__` | Abstract base class for all URDF types. This has useful class methods for automatic parsing/unparsing of XML trees. T... | not primary simulation API |
| `sapien.wrapper.urchin.urdf.URDFTypeWithMesh` | `URDFType` | `` | vendored URDF data/model class. | not primary simulation API |
| `sapien.wrapper.urchin.urdf.Box` | `URDFType` | `__init__, size, size, meshes, copy` | A rectangular prism whose center is at the local origin. Parameters ---------- size : (3,) float The length, width, a... | not primary simulation API |
| `sapien.wrapper.urchin.urdf.Cylinder` | `URDFType` | `__init__, radius, radius, length, length, meshes, copy` | A cylinder whose center is at the local origin. Parameters ---------- radius : float The radius of the cylinder in me... | not primary simulation API |
| `sapien.wrapper.urchin.urdf.Capsule` | `URDFType` | `__init__, radius, radius, length, length, meshes, copy` | A capsule whose center is at the local origin. Parameters ---------- radius : float The radius of the capsule in mete... | not primary simulation API |
| `sapien.wrapper.urchin.urdf.Sphere` | `URDFType` | `__init__, radius, radius, meshes, copy` | A sphere whose center is at the local origin. Parameters ---------- radius : float The radius of the sphere in meters. | not primary simulation API |
| `sapien.wrapper.urchin.urdf.Mesh` | `URDFTypeWithMesh` | `__init__, filename, filename, scale, scale, meshes, meshes, copy` | A triangular mesh object. Parameters ---------- filename : str The path to the mesh that contains this object. This c... | not primary simulation API |
| `sapien.wrapper.urchin.urdf.Geometry` | `URDFTypeWithMesh` | `__init__, box, box, cylinder, cylinder, capsule, capsule, sphere, sphere, mesh, mesh, geometry, meshes, copy` | A wrapper for all geometry types. Only one of the following values can be set, all others should be set to ``None``. ... | not primary simulation API |
| `sapien.wrapper.urchin.urdf.Texture` | `URDFType` | `__init__, filename, filename, copy` | An image-based texture. Parameters ---------- filename : str The path to the image that contains this texture. This c... | not primary simulation API |
| `sapien.wrapper.urchin.urdf.Material` | `URDFType` | `__init__, name, name, color, color, texture, texture, copy` | A material for some geometry. Parameters ---------- name : str The name of the material. color : (4,) float, optional... | not primary simulation API |
| `sapien.wrapper.urchin.urdf.SDF` | `URDFType` | `__init__, resolution, resolution, spacing, spacing, subgrid_size, subgrid_size, num_threads_for_construction, num_threads_for_construction, bits_per_subgrid_pixel, bits_per_subgrid_pixel, narrow_band_thickness, narrow_band_thickness, margin, margin, enable_remeshing, enable_remeshing, triangle_count_reduction_factor ...` | SAPIEN-specific SDF collision settings for a mesh collision. | not primary simulation API |
| `sapien.wrapper.urchin.urdf.Collision` | `URDFTypeWithMesh` | `__init__, geometry, geometry, name, name, origin, origin, sdf, sdf, copy` | Collision properties of a link. Parameters ---------- geometry : :class:`.Geometry` The geometry of the element name ... | not primary simulation API |
| `sapien.wrapper.urchin.urdf.Visual` | `URDFTypeWithMesh` | `__init__, geometry, geometry, name, name, origin, origin, material, material, copy` | Visual properties of a link. Parameters ---------- geometry : :class:`.Geometry` The geometry of the element name : s... | not primary simulation API |
| `sapien.wrapper.urchin.urdf.Inertial` | `URDFType` | `__init__, mass, mass, inertia, inertia, origin, origin, copy` | The inertial properties of a link. Parameters ---------- mass : float The mass of the link in kilograms. inertia : (3... | not primary simulation API |
| `sapien.wrapper.urchin.urdf.JointCalibration` | `URDFType` | `__init__, rising, rising, falling, falling, copy` | The reference positions of the joint. Parameters ---------- rising : float, optional When the joint moves in a positi... | not primary simulation API |
| `sapien.wrapper.urchin.urdf.JointDynamics` | `URDFType` | `__init__, damping, damping, friction, friction, copy` | The dynamic properties of the joint. Parameters ---------- damping : float The damping value of the joint (Ns/m for p... | not primary simulation API |
| `sapien.wrapper.urchin.urdf.JointLimit` | `URDFType` | `__init__, effort, effort, velocity, velocity, lower, lower, upper, upper, copy` | The limits of the joint. Parameters ---------- effort : float The maximum joint effort (N for prismatic joints, Nm fo... | not primary simulation API |
| `sapien.wrapper.urchin.urdf.JointMimic` | `URDFType` | `__init__, joint, joint, multiplier, multiplier, offset, offset, copy` | A mimicry tag for a joint, which forces its configuration to mimic another joint's. This joint's configuration value ... | not primary simulation API |
| `sapien.wrapper.urchin.urdf.SafetyController` | `URDFType` | `__init__, soft_lower_limit, soft_lower_limit, soft_upper_limit, soft_upper_limit, k_position, k_position, k_velocity, k_velocity, copy` | A controller for joint movement safety. Parameters ---------- k_velocity : float An attribute specifying the relation... | not primary simulation API |
| `sapien.wrapper.urchin.urdf.Actuator` | `URDFType` | `__init__, name, name, mechanicalReduction, mechanicalReduction, hardwareInterfaces, hardwareInterfaces, copy` | An actuator. Parameters ---------- name : str The name of this actuator. mechanicalReduction : str, optional A specif... | not primary simulation API |
| `sapien.wrapper.urchin.urdf.TransmissionJoint` | `URDFType` | `__init__, name, name, hardwareInterfaces, hardwareInterfaces, copy` | A transmission joint specification. Parameters ---------- name : str The name of this actuator. hardwareInterfaces : ... | not primary simulation API |
| `sapien.wrapper.urchin.urdf.Transmission` | `URDFType` | `__init__, name, name, trans_type, trans_type, joints, joints, actuators, actuators, copy` | An element that describes the relationship between an actuator and a joint. Parameters ---------- name : str The name... | not primary simulation API |
| `sapien.wrapper.urchin.urdf.Joint` | `URDFType` | `__init__, name, name, joint_type, joint_type, parent, parent, child, child, axis, axis, origin, origin, limit, limit, dynamics, dynamics, safety_controller ...` | A connection between two links. There are several types of joints, including: - ``fixed`` - a joint that cannot move.... | not primary simulation API |
| `sapien.wrapper.urchin.urdf.Link` | `URDFTypeWithMesh` | `__init__, name, name, inertial, inertial, visuals, visuals, collisions, collisions, collision_mesh, copy` | A link of a rigid object. Parameters ---------- name : str The name of the link. inertial : :class:`.Inertial`, optio... | not primary simulation API |
| `sapien.wrapper.urchin.urdf.URDF` | `URDFTypeWithMesh` | `__init__, name, name, links, link_map, joints, joint_map, transmissions, transmission_map, materials, material_map, other_xml, other_xml, actuated_joints, actuated_joint_names, cfg_to_vector, base_link, end_links ...` | The top-level URDF specification. The URDF encapsulates an articulated object, such as a robot or a gripper. It is ma... | not primary simulation API |
