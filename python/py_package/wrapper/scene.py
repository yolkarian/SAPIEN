from __future__ import annotations

from typing import Optional, Sequence, TypeVar, Union
from warnings import warn

import numpy as np

from .. import pysapien as sapien
from ..pysapien import Scene as _Scene
from ..pysapien.physx import PhysxSceneConfig as SceneConfig
from ..pysapien.render import RenderCameraComponent, RenderCubemap


_DEFAULT_GROUND_TEXTURES: dict[
    tuple[int, int, int], sapien.render.RenderTexture2D
] = {}


def _validate_height_field(height_field) -> np.ndarray:
    samples = np.asarray(height_field)
    if samples.ndim != 2:
        raise ValueError("height_field must be a 2D array")
    if samples.shape[0] < 2 or samples.shape[1] < 2:
        raise ValueError("height_field must have at least two rows and two columns")
    if samples.min() < np.iinfo(np.int16).min or samples.max() > np.iinfo(np.int16).max:
        raise ValueError("height_field values must fit in int16")
    return np.ascontiguousarray(samples, dtype=np.int16)


def _default_ground_render_material(
    full_size: tuple[float, float],
) -> sapien.render.RenderMaterial:
    """Create the subtle checker material used by default terrain visuals."""
    tile_size = 0.5
    tile_counts = np.maximum(np.ceil(np.asarray(full_size) / tile_size), 2).astype(int)
    tile_counts = np.minimum(tile_counts + tile_counts % 2, 256)

    pixels_per_tile = max(2, min(8, 512 // int(tile_counts.max())))
    texture_width = int(tile_counts[0] * pixels_per_tile)
    texture_height = int(tile_counts[1] * pixels_per_tile)
    texture_key = (texture_width, texture_height, pixels_per_tile)
    texture = _DEFAULT_GROUND_TEXTURES.get(texture_key)
    if texture is None:
        row_indices, column_indices = np.indices((texture_height, texture_width))
        checker = (
            row_indices // pixels_per_tile + column_indices // pixels_per_tile
        ) % 2
        colors = np.array(
            [[112, 119, 128, 255], [145, 151, 160, 255]], dtype=np.uint8
        )
        texture_data = np.ascontiguousarray(colors[checker])
        mipmap_levels = int(np.floor(np.log2(max(texture_width, texture_height)))) + 1
        texture = sapien.render.RenderTexture2D(
            texture_data,
            "R8G8B8A8Unorm",
            mipmap_levels=mipmap_levels,
            filter_mode="linear",
            address_mode="repeat",
            srgb=True,
        )
        _DEFAULT_GROUND_TEXTURES[texture_key] = texture

    material = sapien.render.RenderMaterial(
        base_color=[1.0, 1.0, 1.0, 1.0],
        specular=0.5,
        roughness=0.6,
    )
    material.base_color_texture = texture
    return material


def _height_field_render_material(
    material: sapien.render.RenderMaterial | Sequence[float] | None,
    full_size: tuple[float, float],
) -> sapien.render.RenderMaterial:
    if material is None:
        return _default_ground_render_material(full_size)
    if isinstance(material, sapien.render.RenderMaterial):
        return material
    return sapien.render.RenderMaterial(base_color=(*material[:3], 1))


def _height_field_render_mesh(
    height_field: np.ndarray, row_scale: float, column_scale: float, height_scale: float
):
    rows, columns = height_field.shape
    row_grid, column_grid = np.meshgrid(
        np.arange(rows, dtype=np.float32),
        np.arange(columns, dtype=np.float32),
        indexing="ij",
    )

    vertices = np.empty((rows * columns, 3), dtype=np.float32)
    vertices[:, 0] = (row_grid * row_scale).reshape(-1)
    vertices[:, 1] = (column_grid * column_scale).reshape(-1)
    vertices[:, 2] = (height_field.astype(np.float32) * height_scale).reshape(-1)

    base = (
        np.arange(rows - 1, dtype=np.uint32)[:, None] * columns
        + np.arange(columns - 1, dtype=np.uint32)[None, :]
    ).reshape(-1)
    triangles = np.empty((2 * base.size, 3), dtype=np.uint32)
    # PhysX height fields with PxHeightFieldSample::clearTessFlag() split each
    # cell along the top-left -> bottom-right diagonal. Match that split so the
    # render mesh is geometrically identical to the collision surface.
    triangles[0::2] = np.stack([base, base + columns, base + columns + 1], axis=1)
    triangles[1::2] = np.stack([base, base + columns + 1, base + 1], axis=1)

    face_vertices = vertices[triangles]
    face_normals = np.cross(
        face_vertices[:, 1] - face_vertices[:, 0],
        face_vertices[:, 2] - face_vertices[:, 0],
    )
    face_norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = np.divide(
        face_normals,
        np.maximum(face_norms, 1e-12),
        out=np.zeros_like(face_normals),
    )
    normals = np.zeros_like(vertices)
    np.add.at(normals, triangles.reshape(-1), np.repeat(face_normals, 3, axis=0))
    normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, np.maximum(normal_norms, 1e-12), out=normals)
    normals[normal_norms.reshape(-1) <= 1e-12] = [0, 0, 1]

    uvs = np.empty((rows * columns, 2), dtype=np.float32)
    uvs[:, 0] = (row_grid / max(rows - 1, 1)).reshape(-1)
    uvs[:, 1] = (column_grid / max(columns - 1, 1)).reshape(-1)
    return vertices, triangles, normals.astype(np.float32), uvs


class Widget:
    def __init__(self):
        warn(
            "Inheriting from sapien.Widget is not required",
            DeprecationWarning,
            stacklevel=2,
        )


class Scene(_Scene):
    def __init__(self, systems=None):
        if systems is None:
            super().__init__(
                [sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem()]
            )
        else:
            super().__init__(systems)

    @property
    def timestep(self):
        return self.physx_system.timestep

    @timestep.setter
    def timestep(self, timestep):
        self.physx_system.timestep = timestep

    def set_timestep(self, timestep):
        self.timestep = timestep

    def get_timestep(self):
        return self.timestep

    def create_actor_builder(self):
        from .actor_builder import ActorBuilder

        return ActorBuilder().set_scene(self)

    def create_articulation_builder(self):
        from .articulation_builder import ArticulationBuilder

        return ArticulationBuilder().set_scene(self)

    def create_urdf_loader(self):
        from .urdf_loader import URDFLoader

        loader = URDFLoader()
        loader.set_scene(self)
        return loader

    def create_physical_material(
        self, static_friction: float, dynamic_friction: float, restitution: float
    ):
        return sapien.physx.PhysxMaterial(
            static_friction, dynamic_friction, restitution
        )

    def remove_actor(self, actor):
        self.remove_entity(actor)
        pass

    def remove_articulation(self, articulation):
        entities = [l.entity for l in articulation.links]
        for e in entities:
            self.remove_entity(e)

    # TODO: find actor by id
    def add_camera(
        self, name, width: int, height: int, fovy: float, near: float, far: float
    ) -> RenderCameraComponent:
        camera_mount = sapien.Entity()
        camera = RenderCameraComponent(width, height)
        camera.set_fovy(fovy, compute_x=True)
        camera.near = near
        camera.far = far
        camera_mount.add_component(camera)
        self.add_entity(camera_mount)
        camera_mount.name = name
        camera.name = name

        return camera

    def add_mounted_camera(
        self, name, mount, pose, width, height, fovy, near, far
    ) -> RenderCameraComponent:
        camera = RenderCameraComponent(width, height)
        camera.set_fovy(fovy, compute_x=True)
        camera.near = near
        camera.far = far
        mount.add_component(camera)
        camera.local_pose = pose
        camera.name = name

        return camera

    def remove_camera(self, camera):
        self.remove_entity(camera.entity)

    def get_cameras(self):
        return self.render_system.cameras

    def get_mounted_cameras(self):
        return self.get_cameras()

    def step(self):
        self.physx_system.step()

    def update_render(self):
        self.render_system.step()

    def add_ground(
        self,
        altitude,
        render=True,
        material=None,
        render_material=None,
        render_half_size=[10, 10],
    ):
        from .actor_builder import ActorBuilder

        builder = self.create_actor_builder()
        if render:
            if render_material is None:
                render_material = _default_ground_render_material(
                    (2.0 * render_half_size[0], 2.0 * render_half_size[1])
                )
            builder.add_plane_visual(
                sapien.Pose(p=[0, 0, altitude], q=[0.7071068, 0, -0.7071068, 0]),
                [10, *render_half_size],
                render_material,
                "",
            )

        builder.add_plane_collision(
            sapien.Pose(p=[0, 0, altitude], q=[0.7071068, 0, -0.7071068, 0]),
            material,
        )
        builder.set_physx_body_type("static")
        ground = builder.build()
        ground.name = "ground"
        return ground

    def add_heightfield(
        self,
        height_field,
        row_scale: float,
        column_scale: float | None = None,
        height_scale: float = 1.0,
        pose=None,
        render=True,
        material=None,
        render_material=None,
        name="heightfield",
    ):
        """Add a static z-up PhysX height field with an optional render mesh.

        ``height_field`` is a 2D int16-compatible array. Public scene coordinates
        are z-up: rows map to +x, columns map to +y, and sample values map to +z
        after applying ``height_scale``. For ``PhysxGpuSystem`` / Direct GPU API
        workflows, add the height field before calling ``gpu_init()``.
        """
        height_field = _validate_height_field(height_field)
        row_scale = float(row_scale)
        column_scale = row_scale if column_scale is None else float(column_scale)
        height_scale = float(height_scale)
        if row_scale <= 0 or column_scale <= 0 or height_scale <= 0:
            raise ValueError("height field scales must be positive")
        if pose is None:
            pose = sapien.Pose()
        if material is None:
            material = sapien.physx.get_default_material()

        rows, columns = height_field.shape
        # PhysX height fields use local x/z as the grid plane and local y as
        # height. Rotate local +y to SAPIEN +z. A pure rotation maps local +z to
        # -y, so reverse columns and offset by the terrain width to preserve the
        # public +y column direction.
        collision_height_field = np.ascontiguousarray(height_field[:, ::-1], dtype=np.int16)
        collision_pose = sapien.Pose(
            p=[0, (columns - 1) * column_scale, 0],
            q=[0.7071068, 0.7071068, 0, 0],
        )

        entity = sapien.Entity()
        entity.name = name
        entity.set_pose(pose)

        body = sapien.physx.PhysxRigidStaticComponent()
        body.name = name
        collision_shape = sapien.physx.PhysxCollisionShapeHeightField(
            collision_height_field,
            row_scale,
            column_scale,
            height_scale,
            material,
        )
        collision_shape.local_pose = collision_pose
        body.attach(collision_shape)
        entity.add_component(body)

        if render:
            vertices, triangles, normals, uvs = _height_field_render_mesh(
                height_field, row_scale, column_scale, height_scale
            )
            render_body = sapien.render.RenderBodyComponent()
            render_body.name = name
            render_shape = sapien.render.RenderShapeTriangleMesh(
                vertices,
                triangles,
                normals,
                uvs,
                _height_field_render_material(
                    render_material,
                    ((rows - 1) * row_scale, (columns - 1) * column_scale),
                ),
            )
            render_shape.local_pose = sapien.Pose()
            render_shape.name = name
            render_body.attach(render_shape)
            entity.add_component(render_body)

        self.add_entity(entity)
        return entity

    def get_contacts(self):
        return self.physx_system.get_contacts()

    def set_environment_id(self, env_id: int, allow_duplicate: bool = False):
        """Set the PhysX GPU broadphase environment ID for this scene.

        In GPU mode, all SAPIEN scenes share one PhysX scene. The environment ID
        is used by the GPU broadphase to avoid cross-environment broadphase pairs.
        Actors with the same envId collide. ``-1``/``0xFFFFFFFF`` means
        "shared" and collides with all environments.

        Must be called BEFORE adding actors/articulations to the scene.
        Non-shared env IDs must be unique by default. Pass
        ``allow_duplicate=True`` to intentionally share a non-shared env ID.
        """
        if not hasattr(self.physx_system, "set_scene_environment_id"):
            raise RuntimeError("environment_id is only available for PhysxGpuSystem")
        self.physx_system.set_scene_environment_id(
            self, env_id, allow_duplicate=allow_duplicate
        )

    def get_environment_id(self) -> int | None:
        """Return the already assigned PhysX GPU broadphase environment ID.

        Returns ``None`` if no environment ID has been assigned yet. This method
        has no side effects; use :meth:`get_or_assign_environment_id` to lazily
        assign a unique ID.
        """
        if not hasattr(self.physx_system, "get_assigned_scene_environment_id"):
            raise RuntimeError("environment_id is only available for PhysxGpuSystem")
        return self.physx_system.get_assigned_scene_environment_id(self)

    def get_or_assign_environment_id(self) -> int:
        """Return this scene's environment ID, assigning a unique one if needed."""
        if not hasattr(self.physx_system, "get_or_assign_scene_environment_id"):
            raise RuntimeError("environment_id is only available for PhysxGpuSystem")
        return self.physx_system.get_or_assign_scene_environment_id(self)

    environment_id = property(
        get_environment_id,
        set_environment_id,
        doc=(
            "Already assigned PhysX GPU broadphase environment ID. Reading this "
            "property has no side effects and returns None until an ID is assigned "
            "explicitly or via get_or_assign_environment_id()."
        ),
    )

    def get_all_actors(self):
        return [
            c.entity
            for c in self.physx_system.rigid_dynamic_components
            + self.physx_system.rigid_static_components
        ]

    def get_all_articulations(self):
        return [
            c.articulation
            for c in self.physx_system.articulation_link_components
            if c.is_root
        ]

    def create_drive(
        self,
        body0: Optional[Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent]],
        pose0: sapien.Pose,
        body1: Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent],
        pose1: sapien.Pose,
    ):
        if body0 is None:
            c0 = None
        elif isinstance(body0, sapien.Entity):
            c0 = next(
                c
                for c in body0.components
                if isinstance(c, sapien.physx.PhysxRigidBaseComponent)
            )
        else:
            c0 = body0

        assert body1 is not None
        if isinstance(body1, sapien.Entity):
            e1 = body1
            c1 = next(
                c
                for c in body1.components
                if isinstance(c, sapien.physx.PhysxRigidBaseComponent)
            )
        else:
            e1 = body1.entity
            c1 = body1

        drive = sapien.physx.PhysxDriveComponent(c1)
        drive.parent = c0
        drive.pose_in_child = pose1
        drive.pose_in_parent = pose0
        e1.add_component(drive)
        return drive

    def create_connection(
        self,
        body0: Optional[Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent]],
        pose0: sapien.Pose,
        body1: Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent],
        pose1: sapien.Pose,
    ):
        if body0 is None:
            c0 = None
        elif isinstance(body0, sapien.Entity):
            c0 = next(
                c
                for c in body0.components
                if isinstance(c, sapien.physx.PhysxRigidBaseComponent)
            )
        else:
            c0 = body0

        assert body1 is not None
        if isinstance(body1, sapien.Entity):
            e1 = body1
            c1 = next(
                c
                for c in body1.components
                if isinstance(c, sapien.physx.PhysxRigidBaseComponent)
            )
        else:
            e1 = body1.entity
            c1 = body1

        connection = sapien.physx.PhysxDistanceJointComponent(c1)
        connection.parent = c0
        connection.pose_in_child = pose1
        connection.pose_in_parent = pose0
        e1.add_component(connection)
        connection.set_limit(0, 0)
        return connection

    def create_gear(
        self,
        body0: Optional[Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent]],
        pose0: sapien.Pose,
        body1: Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent],
        pose1: sapien.Pose,
    ):
        if body0 is None:
            c0 = None
        elif isinstance(body0, sapien.Entity):
            c0 = next(
                c
                for c in body0.components
                if isinstance(c, sapien.physx.PhysxRigidBaseComponent)
            )
        else:
            c0 = body0

        assert body1 is not None
        if isinstance(body1, sapien.Entity):
            e1 = body1
            c1 = next(
                c
                for c in body1.components
                if isinstance(c, sapien.physx.PhysxRigidBaseComponent)
            )
        else:
            e1 = body1.entity
            c1 = body1

        gear = sapien.physx.PhysxGearComponent(c1)
        gear.parent = c0
        gear.pose_in_child = pose1
        gear.pose_in_parent = pose0
        e1.add_component(gear)
        return gear

    @property
    def render_id_to_visual_name(self):
        # TODO
        return

    @property
    def ambient_light(self):
        return self.render_system.ambient_light

    @ambient_light.setter
    def ambient_light(self, color):
        self.render_system.ambient_light = color

    def set_ambient_light(self, color):
        self.ambient_light = color

    def add_point_light(
        self,
        position,
        color,
        shadow=False,
        shadow_near=0.1,
        shadow_far=10.0,
        shadow_map_size=2048,
    ):
        entity = sapien.Entity()
        light = sapien.render.RenderPointLightComponent()
        entity.add_component(light)
        light.color = color
        light.shadow = shadow
        light.shadow_near = shadow_near
        light.shadow_far = shadow_far
        light.shadow_map_size = shadow_map_size
        light.pose = sapien.Pose(position)
        self.add_entity(entity)
        return light

    def add_directional_light(
        self,
        direction,
        color,
        shadow=False,
        position=[0, 0, 0],
        shadow_scale=10.0,
        shadow_near=-10.0,
        shadow_far=10.0,
        shadow_map_size=2048,
    ):
        entity = sapien.Entity()
        light = sapien.render.RenderDirectionalLightComponent()
        entity.add_component(light)
        light.color = color
        light.shadow = shadow
        light.shadow_near = shadow_near
        light.shadow_far = shadow_far
        light.shadow_half_size = shadow_scale
        light.shadow_map_size = shadow_map_size
        light.pose = sapien.Pose(
            position, sapien.math.shortest_rotation([1, 0, 0], direction)
        )
        self.add_entity(entity)
        return light

    def add_spot_light(
        self,
        position,
        direction,
        inner_fov: float,
        outer_fov: float,
        color,
        shadow=False,
        shadow_near=0.1,
        shadow_far=10.0,
        shadow_map_size=2048,
    ):
        entity = sapien.Entity()
        light = sapien.render.RenderSpotLightComponent()
        entity.add_component(light)
        light.color = color
        light.shadow = shadow
        light.shadow_near = shadow_near
        light.shadow_far = shadow_far
        light.shadow_map_size = shadow_map_size
        light.inner_fov = inner_fov
        light.outer_fov = outer_fov
        light.pose = sapien.Pose(
            position, sapien.math.shortest_rotation([1, 0, 0], direction)
        )
        self.add_entity(entity)
        return light

    def add_area_light_for_ray_tracing(
        self, pose: sapien.Pose, color, half_width: float, half_height: float
    ):
        entity = sapien.Entity()
        light = sapien.render.RenderParallelogramLightComponent()
        entity.add_component(light)
        light.set_shape(half_width, half_height)
        light.color = color
        light.pose = pose
        self.add_entity(entity)
        return light

    # TODO: textured light

    def remove_light(self, light):
        self.remove_entity(light.entity)

    def set_environment_map(self, cubemap: str | RenderCubemap):
        if isinstance(cubemap, str):
            self.render_system.cubemap = sapien.render.RenderCubemap(cubemap)
        else:
            self.render_system.cubemap = cubemap

    def set_environment_map_from_files(
        self, px: str, nx: str, py: str, ny: str, pz: str, nz: str
    ):
        self.render_system.cubemap = sapien.render.RenderCubemap(px, nx, py, ny, pz, nz)

    # TODO particle entity, deformable entity

    def create_viewer(self):
        from sapien.utils import Viewer

        viewer = Viewer()
        viewer.set_scene(self)

        return viewer

    def __del__(self):
        self.clear()
