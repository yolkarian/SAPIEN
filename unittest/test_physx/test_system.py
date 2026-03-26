import unittest
import pickle
import numpy as np
from common import pose_equal

import sapien


class TestSystem(unittest.TestCase):
    def test_sdf_config_extended(self):
        original = sapien.physx.get_sdf_config()
        self.addCleanup(sapien.physx.set_sdf_config, original)

        sdf = sapien.physx.PhysxSDFConfig()
        self.assertEqual(sdf.resolution, 0)
        self.assertEqual(sdf.bits_per_subgrid_pixel, 16)
        self.assertAlmostEqual(sdf.narrow_band_thickness, 0.01)
        self.assertAlmostEqual(sdf.margin, 0.0)
        self.assertFalse(sdf.enable_remeshing)
        self.assertAlmostEqual(sdf.triangle_count_reduction_factor, 1.0)

        sdf.spacing = 0.0025
        sdf.subgrid_size = 8
        sdf.num_threads_for_construction = 7
        sdf.resolution = 96
        sdf.bits_per_subgrid_pixel = 32
        sdf.narrow_band_thickness = 0.025
        sdf.margin = 0.001
        sdf.enable_remeshing = True
        sdf.triangle_count_reduction_factor = 0.35

        self.assertEqual(sdf.subgridSize, 8)
        self.assertEqual(sdf.bitsPerSubgridPixel, 32)
        self.assertAlmostEqual(sdf.narrowBandThickness, 0.025)
        self.assertTrue(sdf.enableRemeshing)
        self.assertAlmostEqual(sdf.triangleCountReductionFactor, 0.35)

        sdf.bitsPerSubgridPixel = 8
        sdf.narrowBandThickness = 0.015
        sdf.enableRemeshing = False
        sdf.triangleCountReductionFactor = 0.5

        self.assertEqual(sdf.bits_per_subgrid_pixel, 8)
        self.assertAlmostEqual(sdf.narrow_band_thickness, 0.015)
        self.assertFalse(sdf.enable_remeshing)
        self.assertAlmostEqual(sdf.triangle_count_reduction_factor, 0.5)

        roundtrip = pickle.loads(pickle.dumps(sdf))
        self.assertAlmostEqual(roundtrip.spacing, 0.0025)
        self.assertEqual(roundtrip.subgrid_size, 8)
        self.assertEqual(roundtrip.num_threads_for_construction, 7)
        self.assertEqual(roundtrip.resolution, 96)
        self.assertEqual(roundtrip.bits_per_subgrid_pixel, 8)
        self.assertAlmostEqual(roundtrip.narrow_band_thickness, 0.015)
        self.assertAlmostEqual(roundtrip.margin, 0.001)
        self.assertFalse(roundtrip.enable_remeshing)
        self.assertAlmostEqual(roundtrip.triangle_count_reduction_factor, 0.5)

        sapien.physx.set_sdf_config(sdf)
        current_sdf = sapien.physx.get_sdf_config()
        self.assertAlmostEqual(current_sdf.spacing, 0.0025)
        self.assertEqual(current_sdf.subgrid_size, 8)
        self.assertEqual(current_sdf.num_threads_for_construction, 7)
        self.assertEqual(current_sdf.resolution, 96)
        self.assertEqual(current_sdf.bits_per_subgrid_pixel, 8)
        self.assertAlmostEqual(current_sdf.narrow_band_thickness, 0.015)
        self.assertAlmostEqual(current_sdf.margin, 0.001)
        self.assertFalse(current_sdf.enable_remeshing)
        self.assertAlmostEqual(current_sdf.triangle_count_reduction_factor, 0.5)

        sapien.physx.set_sdf_config(
            spacing=0.01,
            subgrid_size=6,
            num_threads_for_construction=4,
            resolution=128,
            bits_per_subgrid_pixel=32,
            narrow_band_thickness=0.02,
            margin=0.005,
            enable_remeshing=True,
            triangle_count_reduction_factor=0.25,
        )
        current_sdf = sapien.physx.get_sdf_config()
        self.assertAlmostEqual(current_sdf.spacing, 0.01)
        self.assertEqual(current_sdf.subgrid_size, 6)
        self.assertEqual(current_sdf.num_threads_for_construction, 4)
        self.assertEqual(current_sdf.resolution, 128)
        self.assertEqual(current_sdf.bits_per_subgrid_pixel, 32)
        self.assertAlmostEqual(current_sdf.narrow_band_thickness, 0.02)
        self.assertAlmostEqual(current_sdf.margin, 0.005)
        self.assertTrue(current_sdf.enable_remeshing)
        self.assertAlmostEqual(current_sdf.triangle_count_reduction_factor, 0.25)

    def test_timestep(self):
        system = sapien.physx.PhysxCpuSystem()
        system.timestep = 1 / 240
        self.assertAlmostEqual(system.get_timestep(), 1 / 240)
        system.set_timestep(1 / 260)
        self.assertAlmostEqual(system.timestep, 1 / 260)

    def test_config(self):
        config = sapien.physx.PhysxSceneConfig()
        config.gravity = [0, 0, -1]
        config.bounce_threshold = 1.0
        config.enable_pcm = True
        config.enable_tgs = True
        config.enable_ccd = True
        config.enable_enhanced_determinism = True
        config.enable_friction_every_iteration = False
        config.friction_offset_threshold = 0.012
        config.friction_correlation_distance = 0.007
        config.cpu_workers = 2

        sapien.physx.set_scene_config(config)
        system = sapien.physx.PhysxCpuSystem()
        scene = sapien.Scene([system])
        config = scene.physx_system.get_config()

        self.assertAlmostEqual(tuple(config.gravity), (0, 0, -1))
        self.assertAlmostEqual(config.bounce_threshold, 1.0)
        self.assertEqual(config.enable_pcm, True)
        self.assertEqual(config.enable_tgs, True)
        self.assertEqual(config.enable_ccd, True)
        self.assertEqual(config.enable_enhanced_determinism, True)
        self.assertEqual(config.enable_friction_every_iteration, False)
        self.assertAlmostEqual(config.friction_offset_threshold, 0.012)
        self.assertAlmostEqual(config.friction_correlation_distance, 0.007)
        self.assertEqual(config.cpu_workers, 2)

        sapien.physx.set_scene_config(
            gravity=[0, 0, -2],
            bounce_threshold=2.0,
            enable_pcm=False,
            enable_tgs=False,
            enable_ccd=False,
            enable_enhanced_determinism=False,
            enable_friction_every_iteration=True,
        )
        system = sapien.physx.PhysxCpuSystem()
        scene = sapien.Scene([system])
        config = scene.physx_system.get_config()

        self.assertAlmostEqual(tuple(config.gravity), (0, 0, -2))
        self.assertAlmostEqual(config.bounce_threshold, 2.0)
        self.assertEqual(config.enable_pcm, False)
        self.assertEqual(config.enable_tgs, False)
        self.assertEqual(config.enable_ccd, False)
        self.assertEqual(config.enable_enhanced_determinism, False)
        self.assertEqual(config.enable_friction_every_iteration, True)

        config = sapien.physx.PhysxShapeConfig()
        config.contact_offset = 0.02
        config.rest_offset = 0.001
        sapien.physx.set_shape_config(config)
        self.assertAlmostEqual(sapien.physx.get_shape_config().contact_offset, 0.02)
        self.assertAlmostEqual(sapien.physx.get_shape_config().rest_offset, 0.001)

        sapien.physx.set_shape_config(contact_offset=0.01, rest_offset=0)
        self.assertAlmostEqual(sapien.physx.get_shape_config().contact_offset, 0.01)
        self.assertAlmostEqual(sapien.physx.get_shape_config().rest_offset, 0)

        sdf = sapien.physx.PhysxSDFConfig()
        sdf.spacing = 0.0025
        sdf.subgrid_size = 8
        sdf.num_threads_for_construction = 7
        self.assertEqual(sdf.subgridSize, 8)
        sapien.physx.set_sdf_config(sdf)

        current_sdf = sapien.physx.get_sdf_config()
        self.assertAlmostEqual(current_sdf.spacing, 0.0025)
        self.assertEqual(current_sdf.subgrid_size, 8)
        self.assertEqual(current_sdf.subgridSize, 8)
        self.assertEqual(current_sdf.num_threads_for_construction, 7)

        sapien.physx.set_sdf_config(
            spacing=0.01,
            subgrid_size=6,
            num_threads_for_construction=4,
        )
        current_sdf = sapien.physx.get_sdf_config()
        self.assertAlmostEqual(current_sdf.spacing, 0.01)
        self.assertEqual(current_sdf.subgrid_size, 6)
        self.assertEqual(current_sdf.num_threads_for_construction, 4)

        config = sapien.physx.PhysxBodyConfig()
        config.sleep_threshold = 0.001
        config.solver_position_iterations = 25
        config.solver_velocity_iterations = 2
        sapien.physx.set_body_config(config)
        self.assertAlmostEqual(sapien.physx.get_body_config().sleep_threshold, 0.001)
        self.assertEqual(sapien.physx.get_body_config().solver_position_iterations, 25)
        self.assertEqual(sapien.physx.get_body_config().solver_velocity_iterations, 2)

        sapien.physx.set_body_config(
            sleep_threshold=0.005,
            solver_position_iterations=10,
            solver_velocity_iterations=1,
        )
        self.assertAlmostEqual(sapien.physx.get_body_config().sleep_threshold, 0.005)
        self.assertEqual(sapien.physx.get_body_config().solver_position_iterations, 10)
        self.assertEqual(sapien.physx.get_body_config().solver_velocity_iterations, 1)

    def test_cpu_system(self):
        system = sapien.physx.PhysxCpuSystem()
        scene = sapien.Scene([system])
        mat = sapien.physx.PhysxMaterial(0.2, 0.1, 0.05)

        b0 = sapien.physx.PhysxCollisionShapeBox([0.1, 0.2, 0.3], mat)
        c0 = sapien.physx.PhysxRigidDynamicComponent()
        c0.attach(b0)
        e0 = sapien.Entity().add_component(c0)
        scene.add_entity(e0)

        b1 = sapien.physx.PhysxCollisionShapeBox([0.1, 0.2, 0.3], mat)
        c1 = sapien.physx.PhysxRigidDynamicComponent()
        c1.kinematic = True
        c1.attach(b1)
        e1 = sapien.Entity().add_component(c1)
        e1.set_pose(sapien.Pose([0.201, 0, 0]))
        scene.add_entity(e1)

        # test contact
        system.step()
        contacts = system.get_contacts()
        self.assertEqual(len(contacts), 1)
        self.assertEqual(set(contacts[0].bodies), set([c0, c1]))
        self.assertEqual(set(contacts[0].shapes), set([b0, b1]))
        for p in contacts[0].points:
            self.assertAlmostEqual(p.separation, 0.001)

        # test pack unpack
        c1.kinematic_target = sapien.Pose([0, 0, 1])
        p0 = e0.pose
        p1 = e1.pose
        v0 = c0.linear_velocity
        data = system.pack()
        system.step()
        self.assertFalse(pose_equal(e0.pose, p0))
        self.assertFalse(pose_equal(e1.pose, p1))
        self.assertFalse(np.allclose(c0.linear_velocity, v0, atol=1e-5))
        data = system.unpack(data)
        self.assertTrue(pose_equal(e0.pose, p0))
        self.assertTrue(pose_equal(e1.pose, p1))
        self.assertTrue(np.allclose(c0.linear_velocity, v0, atol=1e-5))

    def test_raycast(self):
        system = sapien.physx.PhysxCpuSystem()
        scene = sapien.Scene([system])
        mat = sapien.physx.PhysxMaterial(0.2, 0.1, 0.05)
        b0 = sapien.physx.PhysxCollisionShapePlane(mat)
        c0 = sapien.physx.PhysxRigidStaticComponent()
        b0.set_local_pose(sapien.Pose(p=[0, 0, -1.0], q=[0.7071068, 0, -0.7071068, 0]))
        c0.attach(b0)
        e0 = sapien.Entity().add_component(c0)
        scene.add_entity(e0)

        res = system.raycast([0, 0, 1], [0.70710678, 0.0, -0.70710678], 1000)
        self.assertIsNotNone(res)
        self.assertTrue(np.allclose(res.normal, [0, 0, 1], atol=1e-5))
        self.assertEqual(res.component, c0)
        self.assertTrue(np.allclose(res.distance, 8**0.5))
        self.assertTrue(np.allclose(res.position, [2, 0, -1], atol=1e-5))
