import ctypes
import ctypes.util
import unittest

import numpy as np
import sapien


CUDA_MEMCPY_HOST_TO_DEVICE = 1


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

    def test_default_raster_style(self) -> None:
        sapien.render.set_camera_shader_dir("default")
        scene = sapien.Scene()
        scene.add_ground(0.0, render_half_size=[3.0, 3.0])

        builder = scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.5, 0.5, 0.5])
        box = builder.build_kinematic()
        box.pose = sapien.Pose([0.0, 0.0, 0.5])

        camera = scene.add_camera("camera", 96, 96, np.deg2rad(50), 0.05, 20)
        camera.pose = sapien.Pose([-3.0, 0.0, 0.7])
        camera.set_property("ambientOcclusionStrength", 0.65)
        camera.set_property("ambientOcclusionRadius", 0.3)

        scene.update_render()
        camera.take_picture()
        color = camera.get_picture("Color")

        self.assertGreater(float(np.max(color[..., :3])), 0.1)
        self.assertGreater(float(color[48, 48, 3]), 0.95)
        self.assertGreater(float(np.std(color[60:, :, :3])), 0.005)

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

    def test_camera_multiscene_selection_has_no_offsets(self) -> None:
        sapien.render.set_camera_shader_dir("default")
        owner_scene = sapien.Scene()
        owner_scene.set_ambient_light([0.5, 0.5, 0.5])
        camera = owner_scene.add_camera(
            "camera", 64, 64, np.deg2rad(45), 0.05, 10
        )
        camera.pose = sapien.Pose([-3.0, 0.0, 0.0])

        object_scene = sapien.Scene()
        object_scene.set_ambient_light([0.5, 0.5, 0.5])
        builder = object_scene.create_actor_builder()
        builder.add_box_visual(
            half_size=[0.5, 0.5, 0.5], material=[0.8, 0.05, 0.05]
        )
        builder.build_kinematic().pose = sapien.Pose()

        # Duplicate base scenes are deduplicated, and the object remains at the origin rather
        # than being moved into the legacy Viewer grid.
        camera.set_scenes([owner_scene, object_scene, object_scene])
        owner_scene.update_render()
        object_scene.update_render()
        camera.take_picture()
        center = camera.get_picture("Color")[32, 32]
        self.assertGreater(center[0] - center[1], 0.2)

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
            with self.assertRaisesRegex(RuntimeError, "GPU pose batch"):
                _ = camera._cuda_buffer
            camera_group.take_picture()
            self.assertEqual(
                camera_group.get_picture_cuda("Color").shape, [1, 64, 64, 4]
            )

            camera.take_picture()
            color = camera.get_picture("Color")
            self.assertGreater(float(np.max(color[..., :3])), 0.01)
        finally:
            sapien.render.set_camera_shader_dir("default")


class TestSceneGPU(unittest.TestCase):
    def test_rt_batched_gpu_pose_updates(self) -> None:
        # This is a required GPU test: CUDA or PhysX GPU initialization
        # failures must fail the test instead of being converted to skips.
        sapien.physx.enable_gpu()

        cuda_runtime_library = ctypes.util.find_library("cudart")
        if cuda_runtime_library is None:
            raise RuntimeError("CUDA runtime not available")
        cudart = ctypes.CDLL(cuda_runtime_library)
        cudart.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        cudart.cudaMemcpy.restype = ctypes.c_int

        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(4)
        sapien.render.set_ray_tracing_path_depth(2)
        sapien.render.set_ray_tracing_denoiser("none")

        try:
            device = sapien.Device("cuda")
            physx = sapien.physx.PhysxGpuSystem(device)
            physx.set_timestep(1 / 120)

            main_render = sapien.render.RenderSystem(device)
            main_scene = sapien.Scene([physx, main_render])
            main_scene.set_ambient_light([0.5, 0.5, 0.5])

            rig_builder = main_scene.create_actor_builder()
            rig_builder.initial_pose = sapien.Pose([-3, 0, 1])
            camera_rig = rig_builder.build_kinematic()
            camera = main_scene.add_mounted_camera(
                "camera",
                camera_rig,
                sapien.Pose(),
                64,
                64,
                np.deg2rad(45),
                0.05,
                20,
            )
            camera.set_property("toneMapper", 1)

            shared_render = sapien.render.RenderSystem(device)
            shared_render.batched_render_shared = True
            shared_scene = sapien.Scene([physx, shared_render])
            builder = shared_scene.create_actor_builder()
            builder.add_box_collision(half_size=[0.4, 0.4, 0.4])
            builder.add_box_visual(
                half_size=[0.4, 0.4, 0.4], material=[0.8, 0.05, 0.05]
            )
            builder.initial_pose = sapien.Pose([0, 0, 1])
            actor = builder.build()

            physx.gpu_init()
            camera_body = camera_rig.find_component_by_type(
                sapien.physx.PhysxRigidDynamicComponent
            )

            # RenderSystemGroup discovers sibling PhysX GPU pose indices for dynamic shapes and
            # mounted cameras after gpu_init(); no manual set_gpu_pose_batch_index calls are needed.
            group = sapien.render.RenderSystemGroup(
                [main_render, shared_render]
            )
            camera_group = group.create_camera_group([camera], ["Color"])

            class CudaPoseView:
                def __init__(self, shape, strides, typestr):
                    self.__cuda_array_interface__ = {
                        "shape": tuple(shape),
                        "strides": tuple(strides),
                        "typestr": typestr,
                        "data": (physx.cuda_rigid_body_data.ptr, False),
                        "version": 2,
                    }

            pose_buffer = physx.cuda_rigid_body_data
            with self.assertRaisesRegex(RuntimeError, "float32"):
                group.set_cuda_poses(
                    sapien.CudaArray(
                        CudaPoseView(
                            pose_buffer.shape,
                            [pose_buffer.shape[1] * 8, 8],
                            "<f8",
                        )
                    )
                )
            with self.assertRaisesRegex(RuntimeError, "at least 7 channels"):
                group.set_cuda_poses(
                    sapien.CudaArray(CudaPoseView([pose_buffer.shape[0], 6], [24, 4], "<f4"))
                )
            with self.assertRaisesRegex(RuntimeError, "outside the pose buffer"):
                group.set_cuda_poses(
                    sapien.CudaArray(CudaPoseView([0, 13], [52, 4], "<f4"))
                )

            group.set_cuda_poses(pose_buffer)

            def capture(*, fetch_physics: bool = True) -> np.ndarray:
                if fetch_physics:
                    physx.gpu_fetch_rigid_dynamic_data()
                group.update_render()
                camera_group.take_picture()
                camera.take_picture()
                return camera.get_picture("Color").copy()

            def red_centroid(image: np.ndarray) -> np.ndarray:
                mask = (image[..., 0] > image[..., 1] + 0.2) & (
                    image[..., 0] > 0.2
                )
                coordinates = np.argwhere(mask)
                self.assertGreater(coordinates.shape[0], 20)
                return coordinates.mean(axis=0)

            before = capture()

            # Move only the CUDA pose consumed by RenderSystemGroup. The CPU
            # entity pose intentionally remains unchanged, so the image shift
            # directly verifies the RT camera buffer update.
            camera_pose = np.array(
                [-3.0, 0.75, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
            )
            pose_buffer = physx.cuda_rigid_body_data
            camera_pose_ptr = (
                pose_buffer.ptr
                + camera_body.gpu_pose_index * pose_buffer.strides[0]
            )
            cuda_status = cudart.cudaMemcpy(
                ctypes.c_void_p(camera_pose_ptr),
                ctypes.c_void_p(camera_pose.ctypes.data),
                camera_pose.nbytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
            self.assertEqual(cuda_status, 0)

            shifted_camera = capture(fetch_physics=False)
            camera_image_shift = red_centroid(shifted_camera) - red_centroid(
                before
            )
            self.assertGreater(abs(float(camera_image_shift[1])), 8.0)
            self.assertLess(abs(float(camera_image_shift[0])), 3.0)

            for _ in range(180):
                physx.step()
            after = capture()

            before_center = before[32, 32]
            after_center = after[32, 32]
            self.assertGreater(before_center[0] - before_center[1], 0.3)
            self.assertLess(after_center[0] - after_center[1], 0.1)
            self.assertGreater(float(np.abs(before - after).mean()), 0.01)
            self.assertEqual(physx._sync_poses_gpu_to_cpu_count, 0)
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
