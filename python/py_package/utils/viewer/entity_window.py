from .plugin import Plugin, copy_to_clipboard
from sapien import internal_renderer as R
import sapien
import numpy as np


def _height_field_collision_visual_mesh(shape) -> tuple[np.ndarray, np.ndarray]:
    """Create a render mesh in the PhysX heightfield local frame."""
    height_field = np.ascontiguousarray(shape.height_field, dtype=np.int16)
    rows, columns = height_field.shape
    row_grid, column_grid = np.meshgrid(
        np.arange(rows, dtype=np.float32),
        np.arange(columns, dtype=np.float32),
        indexing="ij",
    )

    vertices = np.empty((rows * columns, 3), dtype=np.float32)
    vertices[:, 0] = (row_grid * shape.row_scale).reshape(-1)
    vertices[:, 1] = (height_field.astype(np.float32) * shape.height_scale).reshape(-1)
    vertices[:, 2] = (column_grid * shape.column_scale).reshape(-1)

    base = (
        np.arange(rows - 1, dtype=np.uint32)[:, None] * columns
        + np.arange(columns - 1, dtype=np.uint32)[None, :]
    ).reshape(-1)
    triangles = np.empty((2 * base.size, 3), dtype=np.uint32)
    # Match PhysX PxHeightFieldSample::clearTessFlag(): split each cell along
    # the top-left -> bottom-right diagonal. Winding is chosen for +local-y
    # normals because PhysX heightfields store height on the local y axis.
    triangles[0::2] = np.stack([base, base + columns + 1, base + columns], axis=1)
    triangles[1::2] = np.stack([base, base + 1, base + columns + 1], axis=1)
    return vertices, triangles


class EntityWindow(Plugin):
    def __init__(self):
        self.reset()

    def reset(self):
        self.ui_window = None

    def close(self):
        self.reset()

    @property
    def selected_entity(self):
        return self.viewer.selected_entity

    def set_actor_pose(self, pose: sapien.Pose) -> None:
        try:
            entity = self.selected_entity
            system = self.viewer._physx_gpu_system
            if system is not None and self.viewer._entity_uses_physx_gpu(entity):
                rigid = entity.find_component_by_type(
                    sapien.physx.PhysxRigidDynamicComponent
                )
                if rigid is not None:
                    self.viewer.queue_gpu_rigid_dynamic_pose(rigid, pose)
                    return
                link = entity.find_component_by_type(
                    sapien.physx.PhysxArticulationLinkComponent
                )
                if link is not None:
                    if link.is_root:
                        self.viewer.queue_gpu_articulation_root_pose(
                            link.articulation, pose
                        )
                    return
            entity.pose = pose
        except AttributeError:
            pass

    def enable_collision_visual(self, entity=None):
        if entity is None:
            entity = self.selected_entity
        if entity is None:
            return

        for c in entity.components:
            if isinstance(c, sapien.render.RenderBodyComponent):
                if c.name == "Collision":
                    return

                c.disable()

        new_visual = sapien.render.RenderBodyComponent()
        new_visual.disable_render_id()  # avoid it interfere with visual id counting

        red_mat = sapien.render.RenderMaterial(base_color=[1, 0, 0, 1])
        green_mat = sapien.render.RenderMaterial(base_color=[0, 1, 0, 1])
        blue_mat = sapien.render.RenderMaterial(base_color=[0, 0, 1, 1])
        new_visual.name = "Collision"

        for c in entity.components:
            if isinstance(c, sapien.physx.PhysxRigidBaseComponent):
                for s in c.collision_shapes:
                    if isinstance(s, sapien.physx.PhysxCollisionShapeSphere):
                        vs = sapien.render.RenderShapeSphere(s.radius, blue_mat)

                    elif isinstance(s, sapien.physx.PhysxCollisionShapeBox):
                        vs = sapien.render.RenderShapeBox(s.half_size, blue_mat)

                    elif isinstance(s, sapien.physx.PhysxCollisionShapeCapsule):
                        vs = sapien.render.RenderShapeCapsule(
                            s.radius, s.half_length, blue_mat
                        )

                    elif isinstance(s, sapien.physx.PhysxCollisionShapeConvexMesh):
                        vs = sapien.render.RenderShapeTriangleMesh(
                            s.vertices,
                            s.triangles,
                            np.zeros((0, 3)),
                            np.zeros((0, 2)),
                            green_mat,
                        )
                        vs.scale = s.scale

                    elif isinstance(s, sapien.physx.PhysxCollisionShapeTriangleMesh):
                        vs = sapien.render.RenderShapeTriangleMesh(
                            s.vertices,
                            s.triangles,
                            np.zeros((0, 3)),
                            np.zeros((0, 2)),
                            red_mat,
                        )
                        vs.scale = s.scale

                    elif isinstance(s, sapien.physx.PhysxCollisionShapeHeightField):
                        vertices, triangles = _height_field_collision_visual_mesh(s)
                        vs = sapien.render.RenderShapeTriangleMesh(
                            vertices,
                            triangles,
                            np.zeros((0, 3)),
                            np.zeros((0, 2)),
                            red_mat,
                        )

                    elif isinstance(s, sapien.physx.PhysxCollisionShapePlane):
                        vs = sapien.render.RenderShapePlane([1, 1e4, 1e4], blue_mat)

                    elif isinstance(s, sapien.physx.PhysxCollisionShapeCylinder):
                        vs = sapien.render.RenderShapeCylinder(
                            s.radius, s.half_length, green_mat
                        )

                    else:
                        raise Exception(
                            "invalid collision shape, this code should be unreachable."
                        )

                    vs.local_pose = s.local_pose

                    new_visual.attach(vs)

        entity.add_component(new_visual)
        new_visual.set_property("shadeFlat", 1)

    def disable_collision_visual(self, entity=None):
        if entity is None:
            entity = self.selected_entity
        if entity is None:
            return

        for c in entity.components:
            if isinstance(c, sapien.render.RenderBodyComponent):
                if c.name == "Collision":
                    entity.remove_component(c)
                    continue

                c.enable()

    def copy_pose(self, _):
        copy_to_clipboard(str(self.viewer.get_entity_viewer_pose(self.selected_entity)))

    def disconnect(self, c):
        self.viewer.select_entity(None)
        c.set_parent(None)

    def build(self):
        if self.viewer.render_scene is None:
            self.ui_window = None
            return

        if self.ui_window is None:
            self.ui_window = R.UIWindow().Pos(10, 10).Size(400, 400).Label("Entity")
        elif not self.ui_window.expanded:
            return
        else:
            self.ui_window.remove_children()

        if self.selected_entity is None:
            self.ui_window.append(R.UIDisplayText().Text("No actor/entity selected."))
            return

        pose = self.viewer.get_entity_viewer_pose(self.selected_entity)
        p = pose.p
        q = pose.q
        pose_read_only = False
        if self.viewer._entity_uses_physx_gpu(self.selected_entity):
            link = self.selected_entity.find_component_by_type(
                sapien.physx.PhysxArticulationLinkComponent
            )
            pose_read_only = link is not None and not link.is_root
        self.ui_window.append(
            R.UIDisplayText().Text("Name: {}".format(self.selected_entity.name)),
            R.UIDisplayText().Text("Id: {}".format(self.selected_entity.per_scene_id)),
            R.UIInputFloat3()
            .Label("pose.p")
            .Id("xyz")
            .Value(p)
            .ReadOnly(pose_read_only)
            .Callback(lambda v: self.set_actor_pose(sapien.Pose(v.value, q))),
            R.UIInputFloat4().Value(q).ReadOnly(True).Label("pose.q"),
            R.UIButton().Label("Copy Pose").Callback(self.copy_pose),
        )

        for cid, c in enumerate(self.selected_entity.components):
            name = f"({c.name})" if c.name else ""
            if isinstance(c, sapien.render.RenderBodyComponent):
                section = R.UISection().Label(f"Render Body {name}##{cid}")
                self.ui_window.append(section)
                for idx, s in enumerate(c.render_shapes):
                    if s.__class__.__name__ == "RenderShapePlane":
                        shape_info = (
                            R.UITreeNode()
                            .Label("Plane")
                            .Id("visual{}".format(idx))
                            .append(
                                R.UIDisplayText().Text(f"Name: {s.name}"),
                                R.UIDisplayText().Text(f"Id: {s.per_scene_id}"),
                                R.UIInputFloat3()
                                .Label("Scale")
                                .ReadOnly(True)
                                .Value(s.scale),
                            )
                        )
                    if s.__class__.__name__ == "RenderShapeBox":
                        shape_info = (
                            R.UITreeNode()
                            .Label("Box")
                            .Id("visual{}".format(idx))
                            .append(
                                R.UIDisplayText().Text(f"Name: {s.name}"),
                                R.UIDisplayText().Text(f"Id: {s.per_scene_id}"),
                                R.UIInputFloat3()
                                .Label("Half Size")
                                .ReadOnly(True)
                                .Value(s.half_size),
                            )
                        )
                    if s.__class__.__name__ == "RenderShapeSphere":
                        shape_info = (
                            R.UITreeNode()
                            .Label("Sphere")
                            .Id("visual{}".format(idx))
                            .append(
                                R.UIDisplayText().Text(f"Name: {s.name}"),
                                R.UIDisplayText().Text(f"Id: {s.per_scene_id}"),
                                R.UIInputFloat()
                                .Label("Radius")
                                .ReadOnly(True)
                                .Value(s.radius),
                            )
                        )
                    if s.__class__.__name__ == "RenderShapeCapsule":
                        shape_info = (
                            R.UITreeNode()
                            .Label("Capsule")
                            .Id("visual{}".format(idx))
                            .append(
                                R.UIDisplayText().Text(f"Name: {s.name}"),
                                R.UIDisplayText().Text(f"Id: {s.per_scene_id}"),
                                R.UIInputFloat()
                                .Label("Radius")
                                .ReadOnly(True)
                                .Value(s.radius),
                                R.UIInputFloat()
                                .Label("Half Length")
                                .ReadOnly(True)
                                .Value(s.half_length),
                            )
                        )
                    if s.__class__.__name__ == "RenderShapeCylinder":
                        shape_info = (
                            R.UITreeNode()
                            .Label("Cylinder")
                            .Id("visual{}".format(idx))
                            .append(
                                R.UIDisplayText().Text(f"Name: {s.name}"),
                                R.UIDisplayText().Text(f"Id: {s.per_scene_id}"),
                                R.UIInputFloat()
                                .Label("Radius")
                                .ReadOnly(True)
                                .Value(s.radius),
                                R.UIInputFloat()
                                .Label("Half Length")
                                .ReadOnly(True)
                                .Value(s.half_length),
                            )
                        )
                    if s.__class__.__name__ == "RenderShapeTriangleMesh":
                        shape_info = (
                            R.UITreeNode()
                            .Label("Triangle Mesh")
                            .Id("visual{}".format(idx))
                            .append(
                                R.UIDisplayText().Text(f"Name: {s.name}"),
                                R.UIDisplayText().Text(f"Id: {s.per_scene_id}"),
                            )
                            # .append(
                            #     R.UIInputFloat()
                            #     .Label("filename")
                            #     .ReadOnly(True)
                            #     .Value(s.filename),
                            #     R.UIInputFloat()
                            #     .Label("Scale")
                            #     .ReadOnly(True)
                            #     .Value(s.scale),
                            # )
                        )

                    section.append(shape_info)

            elif isinstance(c, sapien.physx.PhysxRigidBaseComponent):
                if isinstance(c, sapien.physx.PhysxRigidStaticComponent):
                    section = R.UISection().Label("Static Body")
                    self.ui_window.append(section)

                if isinstance(c, sapien.physx.PhysxRigidDynamicComponent) or isinstance(
                    c, sapien.physx.PhysxArticulationLinkComponent
                ):
                    section = R.UISection().Label(
                        "Dynamic Body"
                        if isinstance(c, sapien.physx.PhysxRigidDynamicComponent)
                        else "Articulation Link"
                    )
                    self.ui_window.append(section)
                    section.append(
                        R.UIInputFloat().Label("Mass").ReadOnly(True).Value(c.mass),
                    )
                    section.append(
                        R.UIInputFloat3()
                        .Label("Inertia")
                        .ReadOnly(True)
                        .Value(c.inertia),
                    )
                    section.append(
                        R.UICheckbox()
                        .Label("Auto Compute Mass")
                        .Checked(c.auto_compute_mass),
                    )

                if (
                    isinstance(c, sapien.physx.PhysxArticulationLinkComponent)
                    and c.parent
                ):
                    section.append(
                        R.UISameLine().append(
                            R.UIButton()
                            .Label("Select Parent")
                            .Callback(
                                (
                                    lambda c: lambda _: self.viewer.select_entity(
                                        c.parent.entity if c.parent else None
                                    )
                                )(c)
                            ),
                            R.UIButton()
                            .Label("Disconnect")
                            .Callback((lambda c: lambda _: self.disconnect(c))(c)),
                        )
                    )

                for idx, s in enumerate(c.collision_shapes):
                    if s.__class__.__name__ == "PhysxCollisionShapePlane":
                        shape_info = (
                            R.UITreeNode().Label("Plane").Id("collision{}".format(idx))
                        )
                    if s.__class__.__name__ == "PhysxCollisionShapeBox":
                        shape_info = (
                            R.UITreeNode()
                            .Label("Box")
                            .Id("collision{}".format(idx))
                            .append(
                                R.UIInputFloat3()
                                .Label("Half Size")
                                .ReadOnly(True)
                                .Value(s.half_size),
                                R.UIDisplayText().Text(f"Density: {s.density}"),
                            )
                        )
                    if s.__class__.__name__ == "PhysxCollisionShapeCapsule":
                        shape_info = (
                            R.UITreeNode()
                            .Label("Capsule")
                            .Id("collision{}".format(idx))
                            .append(
                                R.UIInputFloat()
                                .Label("Radius")
                                .ReadOnly(True)
                                .Value(s.radius),
                                R.UIInputFloat()
                                .Label("Half Length")
                                .ReadOnly(True)
                                .Value(s.half_length),
                                R.UIDisplayText().Text(f"Density: {s.density}"),
                            )
                        )
                    if s.__class__.__name__ == "PhysxCollisionShapeCylinder":
                        shape_info = (
                            R.UITreeNode()
                            .Label("Cylinder")
                            .Id("collision{}".format(idx))
                            .append(
                                R.UIInputFloat()
                                .Label("Radius")
                                .ReadOnly(True)
                                .Value(s.radius),
                                R.UIInputFloat()
                                .Label("Half Length")
                                .ReadOnly(True)
                                .Value(s.half_length),
                                R.UIDisplayText().Text(f"Density: {s.density}"),
                            )
                        )
                    if s.__class__.__name__ == "PhysxCollisionShapeSphere":
                        shape_info = (
                            R.UITreeNode()
                            .Label("Sphere")
                            .Id("collision{}".format(idx))
                            .append(
                                R.UIInputFloat()
                                .Label("Radius")
                                .ReadOnly(True)
                                .Value(s.radius),
                                R.UIDisplayText().Text(f"Density: {s.density}"),
                            )
                        )
                    if s.__class__.__name__ == "PhysxCollisionShapeConvexMesh":
                        shape_info = (
                            R.UITreeNode()
                            .Label("Convex Mesh")
                            .Id("collision{}".format(idx))
                            .append(
                                R.UIDisplayText().Text(f"Density: {s.density}"),
                            )
                        )
                    if s.__class__.__name__ == "PhysxCollisionShapeTriangleMesh":
                        shape_info = (
                            R.UITreeNode()
                            .Label("Triangle Mesh")
                            .Id("collision{}".format(idx))
                        )
                    if s.__class__.__name__ == "PhysxCollisionShapeHeightField":
                        height_field = s.height_field
                        shape_info = (
                            R.UITreeNode()
                            .Label("Height Field")
                            .Id("collision{}".format(idx))
                            .append(
                                R.UIDisplayText().Text(
                                    "Samples: {} x {}".format(*height_field.shape)
                                ),
                                R.UIDisplayText().Text(
                                    "Row scale: {:.3g}".format(s.row_scale)
                                ),
                                R.UIDisplayText().Text(
                                    "Column scale: {:.3g}".format(s.column_scale)
                                ),
                                R.UIDisplayText().Text(
                                    "Height scale: {:.3g}".format(s.height_scale)
                                ),
                                R.UIDisplayText().Text(
                                    "Height range: {} to {}".format(
                                        int(height_field.min()), int(height_field.max())
                                    )
                                ),
                            )
                        )

                    s: sapien.physx.PhysxCollisionShape
                    c0, c1, c2, c3 = s.collision_groups
                    mat = s.physical_material

                    shape_info.append(
                        R.UIDisplayText().Text(
                            "Contact offset: {:.3g}".format(s.contact_offset)
                        ),
                        R.UIDisplayText().Text(
                            "Rest offset: {:.3g}".format(s.rest_offset)
                        ),
                        R.UIDisplayText().Text(
                            "Patch radius: {:.3g}".format(s.patch_radius)
                        ),
                        R.UIDisplayText().Text(
                            "Min path radius: {:.3g}".format(s.min_patch_radius)
                        ),
                        # R.UICheckbox().Label("Is trigger").Checked(s.is_trigger),
                        R.UIDisplayText().Text(
                            "Static friction: {:.3g}".format(mat.get_static_friction())
                        ),
                        R.UIDisplayText().Text(
                            "Dynamic friction: {:.3g}".format(
                                mat.get_dynamic_friction()
                            )
                        ),
                        R.UIDisplayText().Text(
                            "Restitution: {:.3g}".format(mat.get_restitution())
                        ),
                        R.UIDisplayText().Text("Collision groups:"),
                        R.UIDisplayText().Text("  0x{:08x}  0x{:08x}".format(c0, c1)),
                        R.UIDisplayText().Text("  0x{:08x}  0x{:08x}".format(c2, c3)),
                    )

                    section.append(shape_info)

                # only supported in single scene mode
                if len(self.viewer.scenes) == 1:
                    section.append(
                        R.UISameLine().append(
                            R.UIButton()
                            .Label("Show")
                            .Callback(lambda _: self.enable_collision_visual()),
                            R.UIButton()
                            .Label("Hide")
                            .Callback(lambda _: self.disable_collision_visual()),
                            R.UIDisplayText().Text("Collision"),
                        )
                    )

            elif isinstance(c, sapien.physx.PhysxDriveComponent):
                c: sapien.physx.PhysxDriveComponent
                section = R.UISection().Label("6D Drive")
                self.ui_window.append(section)
                section.append(
                    R.UIButton()
                    .Label("Select Parent")
                    .Callback(
                        (
                            lambda c: lambda _: self.viewer.select_entity(
                                c.parent.entity if c.parent else None
                            )
                        )(c)
                    ),
                )
            else:
                section = R.UISection().Label(c.__class__.__name__)
                self.ui_window.append(section)
                section.append(
                    R.UIDisplayText().Text("UI for this component is not implemneted")
                )

    def get_ui_windows(self):
        self.build()
        if self.ui_window:
            return [self.ui_window]
        return []

    def notify_scene_change(self):
        self.reset()
