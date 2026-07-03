"""Tests for PhysX GPU broadphase environment ID API."""

import gc
import pickle
import unittest

import sapien


PX_INVALID_U32 = 0xFFFFFFFF


class TestEnvironmentIDConfig(unittest.TestCase):
    def test_scene_config_gpu_broadphase_bits(self):
        original = sapien.physx.get_scene_config()
        self.addCleanup(sapien.physx.set_scene_config, original)

        config = sapien.physx.PhysxSceneConfig()
        self.assertEqual(config.gpu_broadphase_nb_bits_env_id_x, 0)
        self.assertEqual(config.gpu_broadphase_nb_bits_env_id_y, 0)
        self.assertEqual(config.gpu_broadphase_nb_bits_env_id_z, 0)
        self.assertEqual(config.gpu_broadphase_env_id_bits, 0)

        config.gpu_broadphase_env_id_bits = 8
        self.assertEqual(config.gpu_broadphase_nb_bits_env_id_x, 8)
        self.assertEqual(config.gpu_broadphase_nb_bits_env_id_y, 8)
        self.assertEqual(config.gpu_broadphase_nb_bits_env_id_z, 8)
        self.assertEqual(config.gpu_broadphase_env_id_bits, 8)

        config.gpu_broadphase_env_id_bits = -1
        self.assertEqual(config.gpu_broadphase_env_id_bits, 0)

        with self.assertRaises(RuntimeError):
            config.gpu_broadphase_env_id_bits = 17

        config.gpu_broadphase_nb_bits_env_id_x = 4
        config.gpu_broadphase_nb_bits_env_id_y = 4
        config.gpu_broadphase_nb_bits_env_id_z = 8
        sapien.physx.set_scene_config(config)
        retrieved = sapien.physx.get_scene_config()
        self.assertEqual(retrieved.gpu_broadphase_nb_bits_env_id_x, 4)
        self.assertEqual(retrieved.gpu_broadphase_nb_bits_env_id_y, 4)
        self.assertEqual(retrieved.gpu_broadphase_nb_bits_env_id_z, 8)
        self.assertEqual(retrieved.gpu_broadphase_env_id_bits, 8)

        roundtrip = pickle.loads(pickle.dumps(config))
        self.assertEqual(roundtrip.gpu_broadphase_nb_bits_env_id_x, 4)
        self.assertEqual(roundtrip.gpu_broadphase_nb_bits_env_id_y, 4)
        self.assertEqual(roundtrip.gpu_broadphase_nb_bits_env_id_z, 8)

    def test_scene_wrapper_rejects_cpu_system(self):
        system = sapien.physx.PhysxCpuSystem()
        scene = sapien.Scene([system])

        with self.assertRaisesRegex(RuntimeError, "PhysxGpuSystem"):
            scene.get_environment_id()
        with self.assertRaisesRegex(RuntimeError, "PhysxGpuSystem"):
            scene.set_environment_id(0)


class TestEnvironmentIDGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gc.collect()
        try:
            sapien.physx.enable_gpu()
        except Exception as exc:
            raise unittest.SkipTest(f"GPU PhysX not available: {exc}")

    def setUp(self):
        config = sapien.physx.PhysxSceneConfig()
        config.gpu_broadphase_env_id_bits = 4
        sapien.physx.set_scene_config(config)

    def tearDown(self):
        sapien.physx.set_scene_config(sapien.physx.PhysxSceneConfig())

    def test_gpu_system_auto_assigns_scene_environment_ids(self):
        system = sapien.physx.PhysxGpuSystem()
        scene0 = sapien.Scene([system])
        scene1 = sapien.Scene([system])
        shared_scene = sapien.Scene([system])
        scene2 = sapien.Scene([system])

        self.assertIsNone(scene0.environment_id)
        self.assertEqual(scene0.get_or_assign_environment_id(), 0)
        self.assertEqual(system.get_or_assign_scene_environment_id(scene1), 1)
        self.assertEqual(scene0.environment_id, 0)
        self.assertEqual(scene1.get_environment_id(), 1)

        shared_scene.set_environment_id(-1)
        self.assertEqual(shared_scene.get_environment_id(), PX_INVALID_U32)

        self.assertEqual(system.get_scene_environment_id(scene2), 2)

    def test_gpu_system_manual_scene_environment_id(self):
        system = sapien.physx.PhysxGpuSystem()
        scene = sapien.Scene([system])

        scene.set_environment_id(42)
        self.assertEqual(scene.get_environment_id(), 42)

        scene.set_environment_id(PX_INVALID_U32)
        self.assertEqual(scene.get_environment_id(), PX_INVALID_U32)

        scene.set_environment_id(-1)
        self.assertEqual(scene.get_environment_id(), PX_INVALID_U32)

    def test_shared_environment_id_sets_render_shared_flag(self):
        system = sapien.physx.PhysxGpuSystem()
        try:
            render_system = sapien.render.RenderSystem(system.device)
        except Exception as exc:
            raise unittest.SkipTest(f"RenderSystem not available: {exc}")
        scene = sapien.Scene([system, render_system])

        self.assertFalse(render_system.batched_render_shared)
        scene.set_environment_id(-1)
        self.assertTrue(render_system.batched_render_shared)
        self.assertEqual(scene.get_or_assign_environment_id(), PX_INVALID_U32)
        self.assertTrue(render_system.batched_render_shared)
        scene.set_environment_id(3)
        self.assertFalse(render_system.batched_render_shared)

        system.set_scene_environment_id(scene, PX_INVALID_U32)
        self.assertTrue(render_system.batched_render_shared)
        system.set_scene_environment_ids([(scene, 5)], allow_duplicate=True)
        self.assertFalse(render_system.batched_render_shared)

    def test_gpu_system_rejects_duplicate_scene_environment_ids_by_default(self):
        system = sapien.physx.PhysxGpuSystem()
        scene0 = sapien.Scene([system])
        scene1 = sapien.Scene([system])
        shared0 = sapien.Scene([system])
        shared1 = sapien.Scene([system])

        scene0.set_environment_id(12)
        with self.assertRaisesRegex(RuntimeError, "already used"):
            scene1.set_environment_id(12)

        scene1.set_environment_id(12, allow_duplicate=True)
        self.assertEqual(scene1.get_environment_id(), 12)

        shared0.set_environment_id(-1)
        shared1.set_environment_id(-1)
        self.assertEqual(shared0.get_environment_id(), PX_INVALID_U32)
        self.assertEqual(shared1.get_environment_id(), PX_INVALID_U32)

    def test_gpu_system_rejects_duplicate_bulk_environment_ids_by_default(self):
        system = sapien.physx.PhysxGpuSystem()
        scene0 = sapien.Scene([system])
        scene1 = sapien.Scene([system])

        with self.assertRaisesRegex(RuntimeError, "already used"):
            system.set_scene_environment_ids([(scene0, 3), (scene1, 3)])

        system = sapien.physx.PhysxGpuSystem()
        scene0 = sapien.Scene([system])
        scene1 = sapien.Scene([system])
        system.set_scene_environment_ids([(scene0, 3), (scene1, 3)], allow_duplicate=True)
        self.assertEqual(scene0.get_environment_id(), 3)
        self.assertEqual(scene1.get_environment_id(), 3)

    def test_gpu_system_ignores_expired_scene_environment_ids(self):
        system = sapien.physx.PhysxGpuSystem()
        scene0 = sapien.Scene([system])
        scene0.set_environment_id(5)
        del scene0
        gc.collect()

        scene1 = sapien.Scene([system])
        scene1.set_environment_id(5)
        self.assertEqual(scene1.get_environment_id(), 5)

    def test_gpu_system_rejects_invalid_scene_environment_ids(self):
        system = sapien.physx.PhysxGpuSystem()
        scene = sapien.Scene([system])

        for env_id in (-2, 1 << 24, 0xFFFFFFFE, 1 << 32):
            with self.subTest(env_id=env_id):
                with self.assertRaises(Exception):
                    scene.set_environment_id(env_id)

    def test_gpu_system_bulk_scene_environment_ids(self):
        system = sapien.physx.PhysxGpuSystem()
        scene0 = sapien.Scene([system])
        scene1 = sapien.Scene([system])
        shared_scene = sapien.Scene([system])

        system.set_scene_environment_ids([
            (scene0, 10),
            (scene1, 11),
            (shared_scene, -1),
        ])

        self.assertEqual(scene0.get_environment_id(), 10)
        self.assertEqual(scene1.get_environment_id(), 11)
        self.assertEqual(shared_scene.get_environment_id(), PX_INVALID_U32)

    def test_gpu_system_rejects_environment_id_change_after_body_add(self):
        system = sapien.physx.PhysxGpuSystem()
        scene = sapien.Scene([system])
        scene.set_environment_id(7)

        material = sapien.physx.PhysxMaterial(0.2, 0.1, 0.05)
        shape = sapien.physx.PhysxCollisionShapeBox([0.1, 0.2, 0.3], material)
        body = sapien.physx.PhysxRigidDynamicComponent()
        body.attach(shape)
        entity = sapien.Entity().add_component(body)
        scene.add_entity(entity)

        self.assertEqual(scene.get_environment_id(), 7)
        scene.set_environment_id(7)  # same value is allowed
        with self.assertRaises(RuntimeError):
            scene.set_environment_id(8)

    def test_gpu_actor_and_articulation_accept_auto_environment_id(self):
        system = sapien.physx.PhysxGpuSystem()
        scene = sapien.Scene([system])

        material = sapien.physx.PhysxMaterial(0.2, 0.1, 0.05)

        shape = sapien.physx.PhysxCollisionShapeBox([0.1, 0.2, 0.3], material)
        body = sapien.physx.PhysxRigidDynamicComponent()
        body.attach(shape)
        scene.add_entity(sapien.Entity().add_component(body))

        root = sapien.physx.PhysxArticulationLinkComponent()
        child = sapien.physx.PhysxArticulationLinkComponent(root)
        child.joint.set_type("revolute")
        child.joint.set_pose_in_parent(sapien.Pose([0.5, 0, 0]))
        child.joint.set_pose_in_child(sapien.Pose([0, 0, 0]))
        root.attach(sapien.physx.PhysxCollisionShapeBox([0.1, 0.1, 0.1], material))
        child.attach(sapien.physx.PhysxCollisionShapeBox([0.1, 0.1, 0.1], material))
        scene.add_entity(sapien.Entity().add_component(root))
        scene.add_entity(sapien.Entity().add_component(child))

        self.assertEqual(scene.get_environment_id(), 0)
        system.gpu_init()
        system.step()


if __name__ == "__main__":
    unittest.main()
