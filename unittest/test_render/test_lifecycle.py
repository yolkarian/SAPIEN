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
        scene = sapien.Scene([system])
        group = sapien.render.RenderSystemGroup([system])
        return system, scene, group

    def _close_render_job(self, objects) -> None:
        system, scene, group = objects
        group.close()
        scene.close()
        system.close()
        self.assertTrue(group.is_closed)
        self.assertTrue(scene.is_closed)
        self.assertTrue(system.is_closed)
        group.close()
        scene.close()
        system.close()

    def test_render_engine_can_recreate_after_shutdown(self) -> None:
        for _ in range(3):
            objects = self._build_render_job()
            self._close_render_job(objects)
            del objects
            gc.collect()
            self.assertTrue(sapien.render.can_shutdown(), sapien.render.get_live_resources())
            sapien.render.shutdown()
            self.assertFalse(sapien.render.get_live_resources()["engine_exists"])

    def test_top_level_shutdown_closes_render_and_physx_runtime(self) -> None:
        render_system, render_scene, render_group = self._build_render_job()
        physx_system = sapien.physx.PhysxCpuSystem()
        physx_scene = sapien.Scene([physx_system])

        render_group.close()
        render_scene.close()
        render_system.close()
        physx_scene.close()
        physx_system.close()
        del render_group, render_scene, render_system, physx_scene, physx_system
        gc.collect()

        self.assertTrue(sapien.can_shutdown(), sapien.get_live_resources())
        sapien.shutdown()
        resources = sapien.get_live_resources()
        self.assertFalse(resources["render"]["engine_exists"])
        self.assertFalse(resources["physx"]["engine_exists"])


if __name__ == "__main__":
    unittest.main()
