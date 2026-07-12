import unittest
import sapien
import numpy as np


class TestScene(unittest.TestCase):
    def test_empty(self):
        scene = sapien.Scene()
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        actor = scene.create_actor_builder().build_kinematic()
        actor.set_pose(sapien.Pose([-3, 0, 0.5]))
        cam = scene.add_mounted_camera("", actor, sapien.Pose(), 128, 128, 1, 0.01, 10)

        scene.update_render()
        cam.take_picture()
        color = cam.get_picture("Color")

    def test_texture_respects_base_color_and_alpha(self) -> None:
        sapien.render.set_camera_shader_dir("default")

        for alpha in (0.5, 1.0):
            with self.subTest(alpha=alpha):
                scene = sapien.Scene()
                scene.set_ambient_light([0.5, 0.5, 0.5])

                material = sapien.render.RenderMaterial(
                    base_color=[1.0, 0.02, 0.02, alpha],
                    roughness=0.8,
                    specular=0.0,
                )
                texture_data = np.full((2, 2, 4), 255, dtype=np.uint8)
                material.base_color_texture = sapien.render.RenderTexture2D(
                    texture_data, "R8G8B8A8Unorm", srgb=True
                )

                builder = scene.create_actor_builder()
                builder.add_box_visual(
                    half_size=[0.8, 0.8, 0.8], material=material
                )
                builder.build_kinematic()

                camera = scene.add_camera(
                    "camera", 64, 64, np.deg2rad(45), 0.05, 10
                )
                camera.pose = sapien.Pose([-3, 0, 0])
                camera.set_property("toneMapper", 1)

                scene.update_render()
                camera.take_picture()
                center = camera.get_picture("Color")[32, 32]

                self.assertGreater(center[0], center[1] + 0.1)
                self.assertAlmostEqual(center[3], alpha, delta=0.05)

    def test_rt_batched_shared_scene(self) -> None:
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(2)
        sapien.render.set_ray_tracing_path_depth(2)
        sapien.render.set_ray_tracing_denoiser("none")

        try:
            device = sapien.Device("cuda")

            main_render = sapien.render.RenderSystem(device)
            main_scene = sapien.Scene(
                [sapien.physx.PhysxCpuSystem(), main_render]
            )
            camera = main_scene.add_camera(
                "camera", 64, 64, np.deg2rad(45), 0.05, 10
            )
            camera.pose = sapien.Pose([-3, 0, 0])
            camera.set_property("toneMapper", 1)
            main_scene.update_render()

            shared_render = sapien.render.RenderSystem(device)
            shared_render.batched_render_shared = True
            shared_scene = sapien.Scene(
                [sapien.physx.PhysxCpuSystem(), shared_render]
            )
            builder = shared_scene.create_actor_builder()
            builder.add_box_visual(
                half_size=[0.5, 0.5, 0.5], material=[0.8, 0.1, 0.1]
            )
            builder.build_kinematic()
            shared_scene.add_point_light([-2, 0, 2], [10, 10, 10])
            shared_scene.update_render()

            group = sapien.render.RenderSystemGroup(
                [main_render, shared_render]
            )
            camera_group = group.create_camera_group([camera], ["Color"])
            camera_group.take_picture()
            self.assertEqual(
                camera_group.get_picture_cuda("Color").shape, [1, 64, 64, 4]
            )

            camera.take_picture()
            color = camera.get_picture("Color")
            self.assertGreater(float(np.max(color[..., :3])), 0.01)
        finally:
            sapien.render.set_camera_shader_dir("default")

    # def test_empty(self):
    #     scene = sapien.Scene()
    #     scene.add_ground(altitude=0)  # Add a ground
    #     actor_builder = scene.create_actor_builder()
    #     actor_builder.add_box_collision(half_size=[0.5, 0.5, 0.5])
    #     actor_builder.add_box_visual(half_size=[0.5, 0.5, 0.5], material=[1.0, 0.0, 0.0])
    #     box = actor_builder.build(name="box")
    #     box.set_pose(sapien.Pose(p=[0, 0, 0.5]))

    #     # Add some lights so that you can observe the scene
    #     scene.set_ambient_light([0.5, 0.5, 0.5])
    #     scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    #     actor = scene.create_actor_builder().build_kinematic()
    #     actor.set_pose(sapien.Pose([-3, 0, 0.5]))
    #     cam = scene.add_mounted_camera("", actor, sapien.Pose(), 128, 128, 1, 0.01, 10)

    #     scene.update_render()
    #     cam.take_picture()
    #     color = cam.get_picture("Color")

    #     Image.fromarray((color[..., :3].clip(0, 1) * 255).astype(np.uint8)).save(
    #         "sapien_offscreen.png"
    #     )
