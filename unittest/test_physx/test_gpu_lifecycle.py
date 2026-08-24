import ctypes
import gc
import unittest

import numpy as np
import sapien


def _cuda_device_available() -> bool:
    try:
        lib = ctypes.CDLL("libcuda.so")
    except OSError:
        return False
    return lib.cuInit(0) == 0


class TestPhysxLifecycle(unittest.TestCase):
    """Job-scope lifecycle contract for the Python API.

    These tests deliberately shut the PhysX engine down and recreate it in the same
    interpreter. They must be runnable independently of FurnitureBench and policies.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not _cuda_device_available():
            raise unittest.SkipTest("no usable CUDA device")

    def tearDown(self) -> None:
        gc.collect()
        if sapien.physx.can_shutdown():
            sapien.physx.shutdown()

    def _enable_gpu(self) -> None:
        if not sapien.physx.is_gpu_enabled():
            sapien.physx.enable_gpu()

    def _make_gpu_box_drop(self):
        self._enable_gpu()
        config = sapien.physx.PhysxSceneConfig()
        config.num_scenes = 1
        sapien.physx.set_scene_config(config)
        system = sapien.physx.PhysxGpuSystem()
        scene = sapien.Scene([system])

        ground_builder = scene.create_actor_builder()
        ground_builder.add_box_collision(half_size=[1.0, 1.0, 0.05])
        ground_builder.set_physx_body_type("static")
        ground_builder.set_initial_pose(sapien.Pose([0.0, 0.0, -0.05]))
        ground = ground_builder.build(name="ground")

        box_builder = scene.create_actor_builder()
        box_builder.add_box_collision(half_size=[0.05, 0.05, 0.05], density=1000)
        box_builder.set_initial_pose(sapien.Pose([0.0, 0.0, 0.5]))
        box = box_builder.build(name="box")
        system.gpu_init()
        return system, scene, ground_builder, box_builder, ground, box

    def _close_gpu_box_drop(self, objects) -> None:
        system, scene, *_ = objects
        scene.close()
        system.close()
        self.assertTrue(scene.is_closed)
        self.assertTrue(system.is_closed)
        scene.close()
        system.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            system.step()

    def _shutdown_after_dropping_refs(self) -> None:
        gc.collect()
        self.assertTrue(sapien.physx.can_shutdown(), sapien.physx.get_live_resources())
        sapien.physx.shutdown()
        self.assertFalse(sapien.physx.is_gpu_enabled())

    def test_cpu_system_blocks_shutdown_until_closed(self) -> None:
        system = sapien.physx.PhysxCpuSystem()
        resources = sapien.physx.get_live_resources()
        self.assertEqual(resources["systems"], 1)
        self.assertEqual(resources["gpu_systems"], 0)
        self.assertFalse(sapien.physx.can_shutdown())
        with self.assertRaisesRegex(RuntimeError, "resources are still alive"):
            sapien.physx.shutdown()

        system.close()
        self.assertTrue(system.is_closed)
        system.close()
        del system
        gc.collect()
        self.assertTrue(sapien.physx.can_shutdown(), sapien.physx.get_live_resources())
        sapien.physx.shutdown()

    def test_live_material_blocks_shutdown_without_partial_reset(self) -> None:
        material = sapien.physx.PhysxMaterial(0.3, 0.3, 0.1)
        before = sapien.physx.get_live_resources()
        self.assertEqual(before["physx_objects"], 1)
        self.assertFalse(sapien.physx.can_shutdown())
        with self.assertRaisesRegex(RuntimeError, "resources are still alive"):
            sapien.physx.shutdown()
        # A failed preflight is side-effect-free: the engine and the material remain valid.
        self.assertAlmostEqual(material.static_friction, 0.3, places=6)
        self.assertTrue(sapien.physx.get_live_resources()["engine_exists"])

        del material
        gc.collect()
        self.assertTrue(sapien.physx.can_shutdown(), sapien.physx.get_live_resources())
        sapien.physx.shutdown()

    def test_scene_and_gpu_system_close_are_terminal_and_idempotent(self) -> None:
        objects = self._make_gpu_box_drop()
        self._close_gpu_box_drop(objects)
        del objects
        self._shutdown_after_dropping_refs()

    def test_live_cuda_capsule_blocks_gpu_system_close(self) -> None:
        objects = self._make_gpu_box_drop()
        system, scene = objects[0], objects[1]
        handle = system.cuda_rigid_dynamic_data
        with self.assertRaisesRegex(RuntimeError, "ownership is tracked"):
            _ = handle.__cuda_array_interface__
        capsule = handle.dlpack()
        del handle
        gc.collect()
        self.assertGreater(system.outstanding_cuda_view_count, 0)

        scene.close()
        with self.assertRaisesRegex(RuntimeError, "external CUDA array views"):
            system.close()

        del capsule
        gc.collect()
        self.assertEqual(system.outstanding_cuda_view_count, 0)
        self._close_gpu_box_drop(objects)
        del system, scene, objects
        self._shutdown_after_dropping_refs()

    def test_recreate_gpu_engine_after_job_scope_shutdown(self) -> None:
        final_heights = []
        for _ in range(5):
            objects = self._make_gpu_box_drop()
            system, box = objects[0], objects[-1]
            for _ in range(90):
                system.step()
            system.sync_poses_gpu_to_cpu()
            final_heights.append(float(box.pose.p[2]))
            self._close_gpu_box_drop(objects)
            del system, box, objects
            self._shutdown_after_dropping_refs()

        np.testing.assert_allclose(final_heights, final_heights[0], rtol=0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
