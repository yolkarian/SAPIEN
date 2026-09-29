"""Attached-light resource edits must affect subsequent captures."""

import unittest

import numpy as np
import sapien


class TestLightUpdates(unittest.TestCase):
    def test_directional_shadow_updates_after_capture(self) -> None:
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.1] * 3)
            scene.add_ground(0)
            builder = scene.create_actor_builder()
            builder.add_box_visual(half_size=[0.5] * 3, material=[0.8] * 3)
            builder.set_initial_pose(sapien.Pose([0, 0, 1]))
            builder.build_kinematic()
            light = sapien.render.RenderDirectionalLightComponent()
            light.color = [2] * 3
            light.shadow_near = 0.1
            light.shadow_far = 30
            entity = sapien.Entity()
            entity.add_component(light)
            entity.pose = sapien.Pose([-5, 0, 5], [0.9238795, 0, 0.3826834, 0])
            scene.add_entity(entity)
            camera = scene.add_camera("camera", 128, 128, 1.2, 0.1, 50)
            camera.entity.pose = sapien.Pose([0, 0, 6], [0.7071068, 0, 0.7071068, 0])
            scene.update_render()
            camera.take_picture()
            shaded = camera.get_picture("Color").copy()
            positions = camera.get_picture("Position")
            mask = (
                (np.abs(-positions[..., 2] - 6) < 0.05)
                & (positions[..., 1] > 0.7)
                & (positions[..., 1] < 1.7)
                & (np.abs(positions[..., 0]) < 0.4)
            )
            self.assertGreater(np.count_nonzero(mask), 10)
            light.shadow = False
            scene.update_render()
            camera.take_picture()
            unshaded = camera.get_picture("Color").copy()
            self.assertGreater(unshaded[..., :3][mask].mean(), shaded[..., :3][mask].mean() + 0.05)
            light.shadow = True
            light.shadow_map_size = 512
            scene.update_render()
            camera.take_picture()
            restored = camera.get_picture("Color").copy()
            self.assertLess(restored[..., :3][mask].mean(), unshaded[..., :3][mask].mean() - 0.05)
            # Resize without changing the light count or shadow enable state: a
            # scene-version bump alone only re-records commands in svulkan2.
            light.shadow_map_size = 32
            scene.update_render()
            camera.take_picture()
            coarse = camera.get_picture("Color")
            self.assertGreater(np.abs(coarse[..., :3] - restored[..., :3]).mean(), 1e-4)

    def test_texture_replacement_and_null_rejection(self) -> None:
        previous = sapien.render.get_camera_shader_dir()
        try:
            for shader in ("default", "rt"):
                with self.subTest(shader=shader):
                    sapien.render.set_camera_shader_dir(shader)
                    self._check_texture_replacement_and_null_rejection()
        finally:
            sapien.render.set_camera_shader_dir(previous)

    def _check_texture_replacement_and_null_rejection(self) -> None:
        with sapien.Scene() as scene:
            scene.set_ambient_light([0, 0, 0])
            scene.add_ground(0)
            bright = sapien.render.RenderTexture2D(np.ones((4, 4), dtype=np.float32), "R32Sfloat")
            dark = sapien.render.RenderTexture2D(np.zeros((4, 4), dtype=np.float32), "R32Sfloat")
            light = sapien.render.RenderTexturedLightComponent()
            light.texture = bright
            light.color = [10] * 3
            light.inner_fov = 1.0
            light.outer_fov = 1.5
            entity = sapien.Entity()
            entity.add_component(light)
            entity.pose = sapien.Pose([0, 0, 5], [0.7071068, 0, 0.7071068, 0])
            scene.add_entity(entity)
            camera = scene.add_camera("camera", 64, 64, 1.0, 0.1, 20)
            camera.entity.pose = entity.pose
            scene.update_render()
            camera.take_picture()
            before = camera.get_picture("Color")[..., :3].mean()
            light.texture = dark
            scene.update_render()
            camera.take_picture()
            after = camera.get_picture("Color")[..., :3].mean()
            self.assertGreater(before, after + 0.01)
            with self.assertRaisesRegex(RuntimeError, "non-null"):
                light.texture = None
            self.assertIs(light.texture, dark)
            group = sapien.render.RenderSystemGroup([scene.render_system])
            group.create_camera_group([camera], ["Color"])
            group.gpu_init()
            try:
                for field, value in (("texture", bright), ("shadow", False), ("shadow_map_size", 512)):
                    with self.subTest(field=field):
                        old = getattr(light, field)
                        with self.assertRaises(RuntimeError):
                            setattr(light, field, value)
                        self.assertEqual(getattr(light, field), old)
            finally:
                group.close()
