import gc
import unittest

import sapien


class TestRenderLifecycle(unittest.TestCase):
    def tearDown(self) -> None:
        gc.collect()
        if sapien.can_shutdown():
            sapien.shutdown()

    def _build_render_job(self):
        system = sapien.render.RenderSystem()
        physx_system = sapien.physx.PhysxCpuSystem()
        scene = sapien.Scene([physx_system, system])
        camera = scene.add_camera("camera", 32, 32, 1.0, 0.05, 10.0)
        group = sapien.render.RenderSystemGroup([system])
        camera_group = group.create_camera_group([camera], ["Color"])
        group.gpu_init()
        return system, physx_system, scene, group, camera_group

    def _close_render_job(self, objects) -> None:
        system, physx_system, scene, group, camera_group = objects
        group.close()
        scene.close()
        system.close()
        physx_system.close()
        self.assertTrue(group.is_closed)
        self.assertTrue(camera_group.is_closed)
        self.assertTrue(scene.is_closed)
        self.assertTrue(system.is_closed)
        self.assertTrue(physx_system.is_closed)
        group.close()
        scene.close()
        system.close()
        physx_system.close()

    def test_render_engine_can_recreate_after_shutdown(self) -> None:
        closed_jobs = []
        for _ in range(3):
            objects = self._build_render_job()
            self._close_render_job(objects)
            # Keep the closed system/group/scene objects alive across shutdown and the
            # next render-engine creation. close(), not Python GC, owns unregistration.
            closed_jobs.append(objects)
            gc.collect()
            self.assertTrue(sapien.render.can_shutdown(), sapien.render.get_live_resources())
            sapien.render.shutdown()
            self.assertFalse(sapien.render.get_live_resources()["engine_exists"])
        del closed_jobs
        gc.collect()

    def test_default_ground_texture_cache_is_library_owned(self) -> None:
        render_system = sapien.render.RenderSystem()
        physx_system = sapien.physx.PhysxCpuSystem()
        scene = sapien.Scene([physx_system, render_system])
        ground = scene.add_ground(0.0)
        scene.close()
        render_system.close()
        physx_system.close()
        del ground, scene, render_system, physx_system
        gc.collect()

        resources = sapien.render.get_live_resources()
        self.assertGreater(resources["default_ground_textures"], 0)
        self.assertTrue(sapien.can_shutdown(), sapien.get_live_resources())
        sapien.shutdown()
        self.assertEqual(sapien.render.get_live_resources()["default_ground_textures"], 0)

    def test_render_system_cuda_view_blocks_close(self) -> None:
        system = sapien.render.RenderSystem()
        scene = sapien.Scene([system])
        handle = system.cuda_object_transforms
        with self.assertRaisesRegex(RuntimeError, "ownership is tracked"):
            _ = handle.__cuda_array_interface__
        capsule = handle.dlpack()
        del handle
        gc.collect()
        self.assertGreater(system.outstanding_cuda_view_count, 0)

        scene.close()
        with self.assertRaisesRegex(RuntimeError, "cuda_object_transforms"):
            system.close()
        del capsule
        gc.collect()
        system.close()
        del scene, system
        gc.collect()
        self.assertTrue(sapien.can_shutdown(), sapien.get_live_resources())
        sapien.shutdown()

    def test_top_level_shutdown_closes_render_and_physx_runtime(self) -> None:
        render_system, physx_system, render_scene, render_group, render_camera_group = (
            self._build_render_job()
        )

        render_group.close()
        render_scene.close()
        render_system.close()
        physx_system.close()
        del render_group, render_camera_group, render_scene, render_system, physx_system
        gc.collect()

        self.assertTrue(sapien.can_shutdown(), sapien.get_live_resources())
        sapien.shutdown()
        resources = sapien.get_live_resources()
        self.assertFalse(resources["render"]["engine_exists"])
        self.assertFalse(resources["physx"]["engine_exists"])


if __name__ == "__main__":
    unittest.main()
