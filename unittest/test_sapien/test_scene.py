import unittest

import numpy as np
import sapien
from common import pose_equal, rand_pose, rand_size


class TestScene(unittest.TestCase):
    def test_create_scene(self):
        scene = sapien.Scene()
        self.assertTrue(isinstance(scene.physx_system, sapien.physx.PhysxCpuSystem))
        self.assertTrue(isinstance(scene.render_system, sapien.render.RenderSystem))
        scene = sapien.Scene([])
        with self.assertRaises(RuntimeError):
            scene.physx_system
        with self.assertRaises(RuntimeError):
            scene.render_system

    def test_add_system(self):
        scene = sapien.Scene([])
        system = sapien.physx.PhysxCpuSystem()
        scene.add_system(system)
        self.assertEqual(scene.get_system("physx"), system)

    def test_default_ground_render_material(self):
        scene = sapien.Scene()
        ground = scene.add_ground(0.0, render_half_size=[2.0, 3.0])
        render_body = ground.find_component_by_type(sapien.render.RenderBodyComponent)
        material = render_body.render_shapes[0].material

        self.assertIsNotNone(material.base_color_texture)
        self.assertGreater(material.base_color_texture.mipmap_levels, 1)
        self.assertTrue(np.allclose(material.base_color, [1.0, 1.0, 1.0, 1.0]))
        self.assertAlmostEqual(material.specular, 0.5)
        self.assertAlmostEqual(material.roughness, 0.6)

        second_ground = scene.add_ground(-1.0, render_half_size=[2.0, 3.0])
        second_render_body = second_ground.find_component_by_type(
            sapien.render.RenderBodyComponent
        )
        second_material = second_render_body.render_shapes[0].material
        self.assertEqual(
            second_material.base_color_texture, material.base_color_texture
        )

    def test_add_heightfield(self):
        scene = sapien.Scene()
        height_field = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int16)
        terrain = scene.add_heightfield(
            height_field,
            row_scale=0.1,
            column_scale=0.2,
            height_scale=0.01,
            render=True,
            name="terrain",
        )
        self.assertEqual(terrain.name, "terrain")
        body = terrain.find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
        self.assertIsNotNone(body)
        self.assertEqual(len(body.collision_shapes), 1)
        self.assertIsInstance(
            body.collision_shapes[0], sapien.physx.PhysxCollisionShapeHeightField
        )
        render_body = terrain.find_component_by_type(sapien.render.RenderBodyComponent)
        self.assertIsNotNone(render_body)
        self.assertEqual(len(render_body.render_shapes), 1)

        render_part = render_body.render_shapes[0].parts[0]
        self.assertIsNotNone(render_part.material.base_color_texture)
        self.assertAlmostEqual(render_part.material.specular, 0.5)
        self.assertAlmostEqual(render_part.material.roughness, 0.6)
        np.testing.assert_allclose(
            render_part.vertices,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.2, 0.01],
                    [0.0, 0.4, 0.02],
                    [0.1, 0.0, 0.03],
                    [0.1, 0.2, 0.04],
                    [0.1, 0.4, 0.05],
                ],
                dtype=np.float32,
            ),
        )
        # Keep the render mesh split aligned with PhysX heightfield tessellation.
        np.testing.assert_array_equal(
            render_part.triangles,
            np.array(
                [
                    [0, 3, 4],
                    [0, 4, 1],
                    [1, 4, 5],
                    [1, 5, 2],
                ],
                dtype=np.uint32,
            ),
        )

    def test_viewer_camera_setters_update_window_pose(self):
        from sapien.utils.viewer.camera_control import FPSCameraController
        from sapien.utils.viewer.viewer import Viewer

        class DummyWindow:
            def __init__(self):
                self.pose = sapien.Pose([1, 2, 3], [1, 0, 0, 0])

            def get_camera_pose(self):
                return self.pose

            def set_camera_pose(self, pose):
                self.pose = pose

        viewer = Viewer.__new__(Viewer)
        viewer.window = DummyWindow()
        viewer.plugins = []
        viewer.render_updated = False

        viewer.set_camera_xyz(4, 5, 6)
        np.testing.assert_allclose(viewer.window.pose.p, [4, 5, 6])
        np.testing.assert_allclose(viewer.window.pose.q, [1, 0, 0, 0])
        self.assertTrue(viewer.render_updated)

        viewer.render_updated = False
        viewer.set_camera_rpy(0.1, -0.2, 0.3)
        expected = FPSCameraController()
        expected.setXYZ(4, 5, 6)
        expected.setRPY(0.1, -0.2, 0.3)
        self.assertTrue(pose_equal(viewer.window.pose, expected.pose))
        self.assertTrue(viewer.render_updated)

    def test_viewer_collision_visual_heightfield(self):
        from sapien.utils.viewer.entity_window import EntityWindow

        class DummyViewer:
            def __init__(self):
                self.notified = False
                self.selected_entity = None

            def notify_render_update(self):
                self.notified = True

        scene = sapien.Scene()
        height_field = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int16)
        terrain = scene.add_heightfield(
            height_field,
            row_scale=0.1,
            column_scale=0.2,
            height_scale=0.01,
            render=False,
        )
        body = terrain.find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
        collision_shape = body.collision_shapes[0]

        plugin = EntityWindow()
        viewer = DummyViewer()
        plugin.viewer = viewer
        plugin.enable_collision_visual(terrain)

        self.assertTrue(viewer.notified)
        collision_bodies = [
            component
            for component in terrain.components
            if isinstance(component, sapien.render.RenderBodyComponent)
            and component.name == "Collision"
        ]
        self.assertEqual(len(collision_bodies), 1)
        self.assertEqual(len(collision_bodies[0].render_shapes), 1)

        render_shape = collision_bodies[0].render_shapes[0]
        self.assertTrue(pose_equal(render_shape.local_pose, collision_shape.local_pose))
        render_part = render_shape.parts[0]
        np.testing.assert_allclose(
            render_part.vertices,
            np.array(
                [
                    [0.0, 0.02, 0.0],
                    [0.0, 0.01, 0.2],
                    [0.0, 0.0, 0.4],
                    [0.1, 0.05, 0.0],
                    [0.1, 0.04, 0.2],
                    [0.1, 0.03, 0.4],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            render_part.triangles,
            np.array(
                [
                    [0, 4, 3],
                    [0, 1, 4],
                    [1, 5, 4],
                    [1, 2, 5],
                ],
                dtype=np.uint32,
            ),
        )

    def test_clear(self):
        scene = sapien.Scene()
        scene.add_entity(sapien.Entity())
        scene.add_entity(sapien.Entity())
        scene.clear()
        self.assertEqual(scene.entities, [])

    def test_pack_unpack(self):
        p0 = rand_pose()
        p1 = rand_pose()
        e0 = sapien.Entity()
        e1 = sapien.Entity()

        e0.set_pose(p0)
        e1.set_pose(p1)

        scene = sapien.Scene()
        scene.add_entity(e0)
        scene.add_entity(e1)

        data = scene.pack_poses()
        e0.set_pose(sapien.Pose())
        e1.set_pose(sapien.Pose())
        scene.unpack_poses(data)

        self.assertTrue(pose_equal(e0.pose, p0))
        self.assertTrue(pose_equal(e1.pose, p1))

    def test_actor_builder(self):
        scene = sapien.Scene()
        builder = scene.create_actor_builder()

        poses = [rand_pose() for _ in range(10)]
        sizes = [rand_size() for _ in range(10)]
        mats = [sapien.physx.PhysxMaterial(*np.random.rand(3)) for _ in range(10)]
        densities = [np.random.rand() * 1000 for _ in range(10)]
        patch_radii = [np.random.rand() for _ in range(10)]
        min_patch_radii = [np.random.rand() for _ in range(10)]

        idx = 0
        builder.add_box_collision(
            pose=poses[idx],
            half_size=sizes[idx],
            material=mats[idx],
            density=densities[idx],
            patch_radius=patch_radii[idx],
            min_patch_radius=min_patch_radii[idx],
            is_trigger=False,
        )
        idx += 1

        builder.add_sphere_collision(
            pose=poses[idx],
            radius=sizes[idx][0],
            material=mats[idx],
            density=densities[idx],
            patch_radius=patch_radii[idx],
            min_patch_radius=min_patch_radii[idx],
            is_trigger=False,
        )
        idx += 1

        builder.add_capsule_collision(
            pose=poses[idx],
            radius=sizes[idx][1],
            half_length=sizes[idx][0],
            material=mats[idx],
            density=densities[idx],
            patch_radius=patch_radii[idx],
            min_patch_radius=min_patch_radii[idx],
            is_trigger=False,
        )
        idx += 1

        builder.add_convex_collision_from_file(
            filename="assets/cone.stl",
            pose=poses[idx],
            scale=sizes[idx],
            material=mats[idx],
            density=densities[idx],
            patch_radius=patch_radii[idx],
            min_patch_radius=min_patch_radii[idx],
            is_trigger=False,
        )
        idx += 1

        nonconvex_sdf = sapien.physx.PhysxSDFConfig()
        nonconvex_sdf.resolution = 96
        nonconvex_sdf.bits_per_subgrid_pixel = 32
        nonconvex_sdf.narrow_band_thickness = 0.02

        builder.add_nonconvex_collision_from_file(
            filename="assets/torus.stl",
            pose=poses[idx],
            scale=sizes[idx],
            material=mats[idx],
            patch_radius=patch_radii[idx],
            min_patch_radius=min_patch_radii[idx],
            is_trigger=False,
            sdf_config=nonconvex_sdf,
        )

        collisions = builder.collision_records
        self.assertEqual(len(collisions), 5)
        for i, c in enumerate(collisions):
            self.assertTrue(np.allclose(c.pose.p, poses[i].p))
            self.assertTrue(np.allclose(c.pose.q, poses[i].q))
            self.assertEqual(c.material, mats[i])
            if c.type != "nonconvex_mesh":
                self.assertTrue(np.allclose(c.density, densities[i]))

        self.assertEqual(collisions[0].type, "box")
        self.assertEqual(collisions[1].type, "sphere")
        self.assertEqual(collisions[2].type, "capsule")
        self.assertEqual(collisions[3].type, "convex_mesh")
        self.assertEqual(collisions[4].type, "nonconvex_mesh")

        self.assertTrue(np.allclose(collisions[0].scale, sizes[0]))
        self.assertTrue(np.allclose(collisions[1].radius, sizes[1][0]))
        self.assertTrue(np.allclose(collisions[2].radius, sizes[2][1]))
        self.assertTrue(np.allclose(collisions[2].length, sizes[2][0]))
        self.assertTrue(np.allclose(collisions[3].scale, sizes[3]))
        self.assertTrue(np.allclose(collisions[4].scale, sizes[4]))
        self.assertIs(collisions[4].sdf_config, nonconvex_sdf)
        self.assertEqual(collisions[4].sdf_config.resolution, 96)
        self.assertEqual(collisions[4].sdf_config.bits_per_subgrid_pixel, 32)
        self.assertAlmostEqual(collisions[4].sdf_config.narrow_band_thickness, 0.02)

        body = builder.build_kinematic().find_component_by_type(
            sapien.physx.PhysxRigidBaseComponent
        )
        self.assertTrue(body.kinematic)
        self.assertEqual(
            tuple(c.__class__ for c in body.get_collision_shapes()),
            (
                sapien.physx.PhysxCollisionShapeBox,
                sapien.physx.PhysxCollisionShapeSphere,
                sapien.physx.PhysxCollisionShapeCapsule,
                sapien.physx.PhysxCollisionShapeConvexMesh,
                sapien.physx.PhysxCollisionShapeTriangleMesh,
            ),
        )

        body = builder.build_static().find_component_by_type(
            sapien.physx.PhysxRigidBaseComponent
        )
        self.assertTrue(isinstance(body, sapien.physx.PhysxRigidStaticComponent))
        self.assertEqual(
            tuple(c.__class__ for c in body.get_collision_shapes()),
            (
                sapien.physx.PhysxCollisionShapeBox,
                sapien.physx.PhysxCollisionShapeSphere,
                sapien.physx.PhysxCollisionShapeCapsule,
                sapien.physx.PhysxCollisionShapeConvexMesh,
                sapien.physx.PhysxCollisionShapeTriangleMesh,
            ),
        )

        del builder.collision_records[4]
        builder.physx_body_type = "dynamic"
        body = builder.build().find_component_by_type(
            sapien.physx.PhysxRigidBaseComponent
        )
        self.assertFalse(body.kinematic)
        self.assertEqual(
            tuple(c.__class__ for c in body.get_collision_shapes()),
            (
                sapien.physx.PhysxCollisionShapeBox,
                sapien.physx.PhysxCollisionShapeSphere,
                sapien.physx.PhysxCollisionShapeCapsule,
                sapien.physx.PhysxCollisionShapeConvexMesh,
            ),
        )

        # TODO: check details of the built shapes
