import ctypes
import ctypes.util
import gc
import unittest

import numpy as np
import sapien


CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_STREAM_NON_BLOCKING = 1


def _load_cudart() -> ctypes.CDLL:
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
    cudart.cudaDeviceSynchronize.restype = ctypes.c_int
    cudart.cudaStreamCreateWithFlags.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
    ]
    cudart.cudaStreamCreateWithFlags.restype = ctypes.c_int
    cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    cudart.cudaStreamSynchronize.restype = ctypes.c_int
    cudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    cudart.cudaStreamDestroy.restype = ctypes.c_int
    return cudart


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

    def test_cpu_viewer_update_and_reopen(self) -> None:
        from sapien.utils import Viewer

        scene = sapien.Scene()
        builder = scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.2, 0.2, 0.2])
        actor = builder.build_kinematic()

        for _ in range(2):
            viewer = Viewer(resolutions=(320, 240))
            try:
                viewer.set_scene(scene)
                viewer.update_render()
                viewer.render()
                segmentation = viewer.window.get_picture("Segmentation")
                self.assertGreater(
                    np.count_nonzero(segmentation[..., 0] == actor.per_scene_id),
                    0,
                )
                self.assertEqual(viewer.pose_transport, "cpu")
            finally:
                viewer.close()

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
            camera_group.set_pose_mode(camera, "cuda")
            group.gpu_init()
            # cuda-mode cameras expose their CUDA-owned camera buffer and a group
            # pose row seeded from the CPU pose at gpu_init().
            self.assertGreater(camera._cuda_buffer.shape[0], 0)
            self.assertEqual(camera_group.get_cuda_pose_index(camera), 0)
            camera_group.take_picture()
            self.assertEqual(
                camera_group.get_picture_cuda("Color").shape, [1, 64, 64, 4]
            )

            camera.take_picture()
            color = camera.get_picture("Color")
            self.assertGreater(float(np.max(color[..., :3])), 0.01)
        finally:
            sapien.render.set_camera_shader_dir("default")

    def test_free_camera_only_group_owns_and_releases_camera(self) -> None:
        device = sapien.Device("cuda")
        render_system = sapien.render.RenderSystem(device)
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), render_system])
        scene.set_ambient_light([0.5, 0.5, 0.5])
        builder = scene.create_actor_builder()
        builder.add_box_visual(
            half_size=[0.3, 0.3, 0.3], material=[0.8, 0.05, 0.05]
        )
        builder.build_kinematic()

        camera = scene.add_camera(
            "camera", 64, 64, np.deg2rad(45), 0.05, 10
        )
        camera.set_local_pose(sapien.Pose([-3.0, 0.0, 0.0]))
        group = sapien.render.RenderSystemGroup([render_system])
        camera_group = group.create_camera_group([camera], ["Color"])
        with self.assertRaisesRegex(RuntimeError, "already belongs"):
            group.create_camera_group([camera], ["Color"])
        camera_group.set_pose_mode(camera, "cuda")
        self.assertEqual(camera.pose_mode, "cuda")

        # A cuda-camera-only group needs no unrelated primary object pose source.
        group.gpu_init()
        with self.assertRaisesRegex(RuntimeError, "already initialized"):
            camera_group.set_pose_mode(camera, "cpu")
        camera_group.set_cuda_pose(camera, sapien.Pose([-3.0, 0.2, 0.0]))
        group.update_render()
        camera.take_picture()
        self.assertGreater(float(np.max(camera.get_picture("Color")[..., :3])), 0.01)

        # The CPU pose source is sealed while the group owns it; projection stays
        # CPU real-time in every mode.
        with self.assertRaisesRegex(RuntimeError, "sealed"):
            camera.set_gpu_pose_batch_index(0)
        version = camera.camera_state_version
        camera.set_fovx(np.deg2rad(50))
        self.assertGreater(camera.camera_state_version, version)
        group.update_render()
        camera.take_picture()
        self.assertGreater(float(np.max(camera.get_picture("Color")[..., :3])), 0.01)

        # The dirty projection upload must not override the CUDA pose: the inverse
        # view translation in the camera buffer stays at the CUDA row pose, not the
        # stale CPU pose [-3, 0, 0].
        cudart = _load_cudart()
        camera_buffer = np.empty(48, np.float32)
        self.assertEqual(
            cudart.cudaMemcpy(
                ctypes.c_void_p(camera_buffer.ctypes.data),
                ctypes.c_void_p(camera._cuda_buffer.ptr),
                camera_buffer.nbytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            ),
            0,
        )
        np.testing.assert_allclose(
            camera_buffer[44:47], [-3.0, 0.2, 0.0], rtol=0, atol=1e-6
        )
        camera.entity.set_pose(sapien.Pose([0.0, 0.1, 0.0]))
        with self.assertRaisesRegex(RuntimeError, "pose mode is 'cuda'"):
            scene.update_render()
        camera.entity.set_pose(sapien.Pose())
        scene.update_render()

        # Destroying the sole owner restores ordinary CPU-managed camera use.
        del camera_group
        del group
        gc.collect()
        camera.set_local_pose(sapien.Pose([-3.0, 0.0, 0.0]))
        scene.update_render()
        camera.take_picture()
        self.assertGreater(float(np.max(camera.get_picture("Color")[..., :3])), 0.01)

    def test_static_default_and_cpu_camera_modes(self) -> None:
        device = sapien.Device("cuda")
        render_system = sapien.render.RenderSystem(device)
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), render_system])
        scene.set_ambient_light([0.5, 0.5, 0.5])
        builder = scene.create_actor_builder()
        builder.add_box_visual(
            half_size=[0.3, 0.3, 0.3], material=[0.8, 0.05, 0.05]
        )
        builder.build_kinematic()

        static_camera = scene.add_camera(
            "static_camera", 64, 64, np.deg2rad(45), 0.05, 10
        )
        static_camera.set_local_pose(sapien.Pose([-3.0, 0.0, 0.0]))
        cpu_camera = scene.add_camera(
            "cpu_camera", 64, 64, np.deg2rad(45), 0.05, 10
        )
        cpu_camera.set_local_pose(sapien.Pose([-3.0, 0.0, 0.0]))

        group = sapien.render.RenderSystemGroup([render_system])
        camera_group = group.create_camera_group(
            [static_camera, cpu_camera], ["Color"]
        )
        camera_group.set_pose_mode(cpu_camera, "cpu")
        group.gpu_init()

        # Free cameras default to 'static': no CUDA row is allocated anywhere.
        self.assertEqual(static_camera.pose_mode, "static")
        with self.assertRaisesRegex(RuntimeError, "no cameras with pose mode 'cuda'"):
            camera_group.cuda_poses
        with self.assertRaisesRegex(RuntimeError, "not 'cuda'"):
            camera_group.get_cuda_pose_index(static_camera)
        with self.assertRaisesRegex(RuntimeError, "pose mode is 'static'"):
            static_camera.set_local_pose(sapien.Pose([-3.0, 0.5, 0.0]))

        def red_mask(image: np.ndarray) -> np.ndarray:
            # channel dominance excludes the gray ambient background
            return (image[..., 0] > 0.2) & (image[..., 0] > image[..., 1] + 0.1)

        def red_columns(camera) -> float:
            camera.take_picture()
            mask = red_mask(camera.get_picture("Color"))
            self.assertGreater(int(np.count_nonzero(mask)), 20)
            return float(np.argwhere(mask)[:, 1].mean())

        group.update_render()
        static_before = red_columns(static_camera)
        cpu_before = red_columns(cpu_camera)

        # A cpu-mode camera moves in real time through its CPU pose; the static
        # camera keeps its snapshot.
        cpu_camera.set_local_pose(sapien.Pose([-3.0, 0.4, 0.0]))
        group.update_render()
        static_after = red_columns(static_camera)
        cpu_after = red_columns(cpu_camera)
        self.assertLess(abs(static_after - static_before), 1.0)
        self.assertGreater(abs(cpu_after - cpu_before), 4.0)

        # Steady state: no CPU state changes, no version movement (zero uploads).
        static_version = static_camera.camera_state_version
        cpu_version = cpu_camera.camera_state_version
        light_version = render_system.scene_light_state_version
        group.update_render()
        group.update_render()
        self.assertEqual(static_camera.camera_state_version, static_version)
        self.assertEqual(cpu_camera.camera_state_version, cpu_version)
        self.assertEqual(render_system.scene_light_state_version, light_version)

        # Static-camera projection stays CPU real-time: a narrower FOV zooms in and
        # the box covers more pixels.
        static_camera.take_picture()
        red_count_before = int(
            np.count_nonzero(red_mask(static_camera.get_picture("Color")))
        )
        static_camera.set_fovx(np.deg2rad(20))
        group.update_render()
        static_camera.take_picture()
        red_count_after = int(
            np.count_nonzero(red_mask(static_camera.get_picture("Color")))
        )
        self.assertGreater(red_count_after, red_count_before * 2)

        # Static tampering is surfaced by the group update itself.
        static_camera.entity.set_pose(sapien.Pose([0.0, 0.1, 0.0]))
        with self.assertRaisesRegex(RuntimeError, "pose mode is 'static'"):
            group.update_render()
        static_camera.entity.set_pose(sapien.Pose())
        group.update_render()

    def test_repeated_dirty_updates_without_capture(self) -> None:
        """A cpu-mode camera that moves every step but is captured only every few
        steps drives several dirty CPU uploads between renders. Each upload must
        complete before the next one rewrites the shared staging buffer, and the
        kept frame must show the newest pose."""
        device = sapien.Device("cuda")
        render_system = sapien.render.RenderSystem(device)
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), render_system])
        scene.set_ambient_light([0.5, 0.5, 0.5])
        builder = scene.create_actor_builder()
        builder.add_box_visual(
            half_size=[0.3, 0.3, 0.3], material=[0.8, 0.05, 0.05]
        )
        builder.build_kinematic()

        camera = scene.add_camera(
            "camera", 64, 64, np.deg2rad(45), 0.05, 10
        )
        camera.set_local_pose(sapien.Pose([-3.0, 0.0, 0.0]))
        group = sapien.render.RenderSystemGroup([render_system])
        camera_group = group.create_camera_group([camera], ["Color"])
        camera_group.set_pose_mode(camera, "cpu")
        group.gpu_init()

        def red_column() -> float:
            camera.take_picture()
            image = camera.get_picture("Color")
            mask = (image[..., 0] > 0.2) & (image[..., 0] > image[..., 1] + 0.1)
            self.assertGreater(int(np.count_nonzero(mask)), 20)
            return float(np.argwhere(mask)[:, 1].mean())

        group.update_render()
        baseline = red_column()

        # Several dirty uploads back to back, with no capture in between.
        for y in (0.1, 0.2, 0.3, 0.4):
            camera.set_local_pose(sapien.Pose([-3.0, y, 0.0]))
            group.update_render()

        # Projection is dirty in the same uncaptured window as the pose.
        camera.set_fovx(np.deg2rad(50))
        group.update_render()

        moved = red_column()
        self.assertGreater(abs(moved - baseline), 4.0)

        # The group is still usable and settles once nothing is dirty.
        version = camera.camera_state_version
        group.update_render()
        group.update_render()
        self.assertEqual(camera.camera_state_version, version)
        self.assertLess(abs(red_column() - moved), 1.0)

    def test_group_seals_point_cloud_and_unseals_light_properties(self) -> None:
        device = sapien.Device("cuda")
        render_system = sapien.render.RenderSystem(device)
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), render_system])

        point_entity = sapien.Entity()
        point_cloud = sapien.render.RenderPointCloudComponent(1)
        point_cloud.set_vertices(np.array([[0.0, 0.0, 0.0]], np.float32))
        point_entity.add_component(point_cloud)
        scene.add_entity(point_entity)

        static_light_entity = sapien.Entity()
        static_light = sapien.render.RenderPointLightComponent()
        static_light_entity.add_component(static_light)
        scene.add_entity(static_light_entity)

        cpu_light_entity = sapien.Entity()
        cpu_light = sapien.render.RenderSpotLightComponent()
        cpu_light.set_pose_mode("cpu")
        cpu_light_entity.add_component(cpu_light)
        scene.add_entity(cpu_light_entity)

        camera = scene.add_camera(
            "camera", 64, 64, np.deg2rad(45), 0.05, 10
        )
        camera.set_local_pose(sapien.Pose([-3.0, 0.0, 0.0]))
        group = sapien.render.RenderSystemGroup([render_system])
        camera_group = group.create_camera_group([camera], ["Color"])
        group.gpu_init()

        with self.assertRaisesRegex(RuntimeError, "point cloud.*static snapshot"):
            point_cloud.set_vertices(np.array([[0.1, 0.0, 0.0]], np.float32))

        # Light properties stay CPU real-time after gpu_init(); only the pose of a
        # static-mode light and setup-only fields are sealed.
        light_version = render_system.scene_light_state_version
        static_light.set_color([0.5, 0.5, 0.5])
        static_light.set_shadow_near(0.05)
        static_light.set_shadow_far(5.0)
        cpu_light.set_inner_fov(0.4)
        cpu_light.set_outer_fov(0.8)
        scene.set_ambient_light([0.3, 0.3, 0.3])
        self.assertGreater(render_system.scene_light_state_version, light_version)

        with self.assertRaisesRegex(RuntimeError, "pose mode is 'static'"):
            static_light.set_local_pose(sapien.Pose([0.0, 0.1, 0.0]))
        with self.assertRaisesRegex(RuntimeError, "setup-only"):
            static_light.set_shadow_map_size(1024)
        with self.assertRaisesRegex(RuntimeError, "setup-only"):
            static_light.disable_shadow()
        with self.assertRaisesRegex(RuntimeError, "sealed"):
            static_light.set_pose_mode("cpu")

        # cpu-mode light poses stay CPU real-time.
        cpu_light_entity.set_pose(sapien.Pose([0.0, 0.2, 0.0]))
        cpu_light.set_local_pose(sapien.Pose([0.0, 0.0, 0.1]))
        group.update_render()

        point_entity.set_pose(sapien.Pose([0.0, 0.1, 0.0]))
        with self.assertRaisesRegex(RuntimeError, "point cloud.*static snapshot"):
            scene.update_render()
        point_entity.set_pose(sapien.Pose())
        scene.update_render()

        static_light_entity.set_pose(sapien.Pose([0.0, 0.1, 0.0]))
        with self.assertRaisesRegex(RuntimeError, "pose mode is 'static'"):
            scene.update_render()
        static_light_entity.set_pose(sapien.Pose())
        scene.update_render()

        del camera_group
        del group
        gc.collect()
        point_entity.set_pose(sapien.Pose([0.0, 0.1, 0.0]))
        static_light.set_local_pose(sapien.Pose([0.0, 0.1, 0.0]))
        static_light.set_shadow_map_size(1024)
        static_light.set_pose_mode("cpu")
        scene.update_render()

    def test_cuda_camera_pose_uses_configured_cuda_stream(self) -> None:
        cudart = _load_cudart()
        stream = ctypes.c_void_p()
        self.assertEqual(
            cudart.cudaStreamCreateWithFlags(
                ctypes.byref(stream), CUDA_STREAM_NON_BLOCKING
            ),
            0,
        )

        group = None
        camera_group = None
        try:
            device = sapien.Device("cuda")
            render_system = sapien.render.RenderSystem(device)
            scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), render_system])
            camera = scene.add_camera(
                "camera", 64, 64, np.deg2rad(45), 0.05, 10
            )
            camera.set_local_pose(sapien.Pose([-3.0, 0.0, 0.0]))

            group = sapien.render.RenderSystemGroup([render_system])
            group.set_cuda_stream(stream.value)
            camera_group = group.create_camera_group([camera], ["Color"])
            camera_group.set_pose_mode(camera, "cuda")
            group.gpu_init()

            final_pose = sapien.Pose([-3.0, 0.4, 0.2])
            for y in (0.1, 0.2, 0.3, 0.4):
                camera_group.set_cuda_pose(
                    camera, sapien.Pose([-3.0, y, 0.2])
                )
                group.update_render()
            self.assertEqual(cudart.cudaStreamSynchronize(stream), 0)

            host_pose = np.empty(7, dtype=np.float32)
            cuda_pose = camera_group.cuda_poses
            self.assertEqual(
                cudart.cudaMemcpy(
                    ctypes.c_void_p(host_pose.ctypes.data),
                    ctypes.c_void_p(cuda_pose.ptr),
                    host_pose.nbytes,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                ),
                0,
            )
            np.testing.assert_allclose(
                host_pose,
                [*final_pose.p, *final_pose.q],
                rtol=0,
                atol=1e-6,
            )
        finally:
            camera_group = None
            group = None
            gc.collect()
            self.assertEqual(cudart.cudaStreamDestroy(stream), 0)

    def _run_light_realtime_updates(self, shader: str) -> None:
        """Static lights keep their pose but their color and shadow parameters stay
        CPU real-time; cpu-mode lights move in real time; ambient stays CPU
        real-time. Raster and RT are covered separately."""
        sapien.render.set_camera_shader_dir(shader)
        if shader == "rt":
            sapien.render.set_ray_tracing_samples_per_pixel(4)
            sapien.render.set_ray_tracing_path_depth(2)
            sapien.render.set_ray_tracing_denoiser("none")

        try:
            device = sapien.Device("cuda")
            render_system = sapien.render.RenderSystem(device)
            scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), render_system])
            builder = scene.create_actor_builder()
            builder.add_box_visual(
                half_size=[0.5, 0.5, 0.5], material=[0.8, 0.8, 0.8]
            )
            builder.build_kinematic()

            light_entity = sapien.Entity()
            static_light = sapien.render.RenderPointLightComponent()
            static_light.set_color([0.0, 0.0, 0.0])
            light_entity.add_component(static_light)
            light_entity.set_pose(sapien.Pose([-2.0, 0.0, 0.0]))
            scene.add_entity(light_entity)

            cpu_light_entity = sapien.Entity()
            cpu_light = sapien.render.RenderPointLightComponent()
            cpu_light.set_pose_mode("cpu")
            cpu_light.set_color([0.0, 0.0, 0.0])
            cpu_light_entity.add_component(cpu_light)
            cpu_light_entity.set_pose(sapien.Pose([-2.0, 0.0, 30.0]))
            scene.add_entity(cpu_light_entity)

            camera = scene.add_camera(
                "camera", 64, 64, np.deg2rad(45), 0.05, 10
            )
            camera.set_local_pose(sapien.Pose([-3.0, 0.0, 0.0]))
            if shader == "rt":
                camera.set_property("toneMapper", 1)
            group = sapien.render.RenderSystemGroup([render_system])
            group.create_camera_group([camera], ["Color"])
            group.gpu_init()

            def brightness() -> float:
                group.update_render()
                camera.take_picture()
                image = camera.get_picture("Color")
                return float(np.mean(image[..., :3]))

            dark = brightness()

            # Static light color is CPU real-time after gpu_init().
            static_light.set_color([20.0, 20.0, 20.0])
            lit = brightness()
            self.assertGreater(lit, dark + 0.05)
            static_light.set_color([0.0, 0.0, 0.0])
            self.assertLess(brightness(), dark + 0.02)

            # A cpu-mode light pose is CPU real-time: bring the far light close.
            cpu_light.set_color([20.0, 20.0, 20.0])
            far_lit = brightness()
            cpu_light_entity.set_pose(sapien.Pose([-2.0, 0.0, 0.0]))
            near_lit = brightness()
            self.assertGreater(near_lit, far_lit + 0.05)

            # Ambient light stays CPU real-time.
            cpu_light.set_color([0.0, 0.0, 0.0])
            base = brightness()
            scene.set_ambient_light([0.6, 0.6, 0.6])
            self.assertGreater(brightness(), base + 0.05)
        finally:
            sapien.render.set_camera_shader_dir("default")

    def test_raster_light_realtime_updates(self) -> None:
        self._run_light_realtime_updates("default")

    def test_rt_light_realtime_updates(self) -> None:
        self._run_light_realtime_updates("rt")

    def test_batched_light_setters(self) -> None:
        lights = [
            sapien.render.RenderPointLightComponent(),
            sapien.render.RenderDirectionalLightComponent(),
            sapien.render.RenderSpotLightComponent(),
        ]

        colors = np.array(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], np.float32
        )
        sapien.render.set_light_colors(lights, colors)
        for light, color in zip(lights, colors):
            np.testing.assert_allclose(light.color, color)

        poses = np.array(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            ],
            np.float32,
        )
        sapien.render.set_light_poses(lights, poses)
        np.testing.assert_allclose(lights[1].local_pose.q, [0.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(lights[2].local_pose.p, [0.0, 0.0, 3.0])

        # Directions apply to directional/spot lights only and keep the position.
        directions = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], np.float32)
        sapien.render.set_light_directions(lights[1:], directions)
        np.testing.assert_allclose(lights[1].local_pose.p, [0.0, 2.0, 0.0])
        rotated = lights[1].local_pose
        np.testing.assert_allclose(
            rotated.to_transformation_matrix()[:3, 0], [0.0, 0.0, -1.0], atol=1e-6
        )
        with self.assertRaisesRegex(RuntimeError, "no direction"):
            sapien.render.set_light_directions([lights[0]], directions[:1])

        # Validation failures leave the batch untouched (validate-then-apply).
        before = [light.color.copy() for light in lights]
        bad_colors = np.array(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 0.0, 0.0]], np.float32
        )
        with self.assertRaisesRegex(RuntimeError, "non-negative"):
            sapien.render.set_light_colors(lights, bad_colors)
        for light, color in zip(lights, before):
            np.testing.assert_allclose(light.color, color)
        with self.assertRaisesRegex(RuntimeError, "one row per light"):
            sapien.render.set_light_poses(lights, poses[:2])
        bad_poses = poses.copy()
        bad_poses[1, 3:] = 0.0
        with self.assertRaisesRegex(RuntimeError, "quaternion"):
            sapien.render.set_light_poses(lights, bad_poses)


class TestSceneGPU(unittest.TestCase):
    def test_implicit_shared_dynamic_requires_and_tracks_cuda_poses(self) -> None:
        sapien.physx.enable_gpu()
        cudart = _load_cudart()
        device = sapien.Device("cuda")

        main_render = sapien.render.RenderSystem(device)
        main_scene = sapien.Scene(
            [sapien.physx.PhysxCpuSystem(), main_render]
        )
        main_scene.set_ambient_light([0.5, 0.5, 0.5])
        camera = main_scene.add_camera(
            "camera", 64, 64, np.deg2rad(45), 0.05, 10
        )
        camera.set_local_pose(sapien.Pose([-3.0, 0.0, 0.0]))

        physx = sapien.physx.PhysxGpuSystem(device)
        shared_render = sapien.render.RenderSystem(device)
        shared_render.batched_render_shared = True
        shared_scene = sapien.Scene([physx, shared_render])
        shared_scene.set_ambient_light([0.5, 0.5, 0.5])
        builder = shared_scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.3, 0.3, 0.3])
        builder.add_box_visual(
            half_size=[0.3, 0.3, 0.3], material=[0.8, 0.05, 0.05]
        )
        builder.initial_pose = sapien.Pose([0.0, 0.0, -50.0])
        actor = builder.build()
        body = actor.find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent
        )

        # The shared render system is intentionally omitted. Camera scene
        # resolution must still discover it and validate its PhysX GPU system.
        group = sapien.render.RenderSystemGroup([main_render])
        camera_group = group.create_camera_group([camera], ["Color"])
        with self.assertRaisesRegex(RuntimeError, "PhysxGpuSystem.*gpu_init"):
            group.gpu_init()

        physx.gpu_init()
        with self.assertRaisesRegex(RuntimeError, "require set_cuda_poses"):
            group.gpu_init()

        pose_buffer = physx.cuda_rigid_body_data
        pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], np.float32)
        self.assertEqual(
            cudart.cudaMemcpy(
                ctypes.c_void_p(
                    pose_buffer.ptr + body.gpu_pose_index * pose_buffer.strides[0]
                ),
                ctypes.c_void_p(pose.ctypes.data),
                pose.nbytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            0,
        )
        group.set_cuda_poses(pose_buffer)
        group.gpu_init()
        group.update_render()
        camera_group.take_picture()
        image = camera_group.get_picture_cuda("Color")
        host = np.empty((64, 64, 4), np.float32)
        self.assertEqual(
            cudart.cudaMemcpy(
                ctypes.c_void_p(host.ctypes.data),
                ctypes.c_void_p(image.ptr),
                host.nbytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            ),
            0,
        )
        red = host[..., 0]
        self.assertGreater(int(np.count_nonzero(red > 0.2)), 20)

        extra_entity = sapien.Entity()
        extra_body = sapien.render.RenderBodyComponent()
        extra_body.attach(
            sapien.render.RenderShapeBox(
                [0.1, 0.1, 0.1], sapien.render.RenderMaterial()
            )
        )
        extra_entity.add_component(extra_body)
        shared_scene.add_entity(extra_entity)
        with self.assertRaisesRegex(RuntimeError, "Modifying a scene"):
            group.update_render()

    def test_mounted_camera_binding_waits_for_physx_gpu_init(self) -> None:
        sapien.physx.enable_gpu()
        device = sapien.Device("cuda")
        physx = sapien.physx.PhysxGpuSystem(device)
        render_system = sapien.render.RenderSystem(device)
        scene = sapien.Scene([physx, render_system])

        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.2, 0.2, 0.2])
        builder.add_box_visual(half_size=[0.2, 0.2, 0.2])
        actor = builder.build()
        camera = scene.add_mounted_camera(
            "camera",
            actor,
            sapien.Pose([-3.0, 0.0, 0.0]),
            64,
            64,
            np.deg2rad(45),
            0.05,
            10,
        )

        group = sapien.render.RenderSystemGroup([render_system])
        camera_group = group.create_camera_group([camera], ["Color"])
        with self.assertRaisesRegex(RuntimeError, "PhysxGpuSystem.*gpu_init"):
            group.gpu_init()

        physx.gpu_init()
        group.set_cuda_poses(physx.cuda_rigid_body_data)
        group.gpu_init()
        with self.assertRaisesRegex(RuntimeError, "mounted GPU parent"):
            camera_group.get_cuda_pose_index(camera)

        # A mounted camera is CUDA-attached to its PhysX parent row even though no
        # pose mode was configured, so it must report the effective mode, not 'static'.
        self.assertEqual(camera.pose_mode, "cuda")

        # CPU debug synchronization may update the mounted entity pose, but the
        # sealed camera still derives its world transform from the GPU parent row.
        body = actor.find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent
        )
        pose_buffer = physx.cuda_rigid_body_data
        pose = np.array([0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0], np.float32)
        cudart = _load_cudart()
        status = cudart.cudaMemcpy(
            ctypes.c_void_p(
                pose_buffer.ptr + body.gpu_pose_index * pose_buffer.strides[0]
            ),
            ctypes.c_void_p(pose.ctypes.data),
            pose.nbytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
        self.assertEqual(status, 0)
        physx.gpu_apply_rigid_dynamic_data()
        physx.sync_poses_gpu_to_cpu()
        scene.update_render()

    def test_cpu_light_on_gpu_body_entity_rejected(self) -> None:
        sapien.physx.enable_gpu()
        device = sapien.Device("cuda")
        physx = sapien.physx.PhysxGpuSystem(device)
        render_system = sapien.render.RenderSystem(device)
        scene = sapien.Scene([physx, render_system])

        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.2, 0.2, 0.2])
        builder.add_box_visual(half_size=[0.2, 0.2, 0.2])
        actor = builder.build()

        # A cpu-mode light on the same entity as a PhysX GPU body has no
        # authoritative CPU pose: gpu_init() must reject the configuration.
        light = sapien.render.RenderPointLightComponent()
        light.set_pose_mode("cpu")
        actor.add_component(light)

        camera = scene.add_camera(
            "camera", 64, 64, np.deg2rad(45), 0.05, 10
        )
        camera.set_local_pose(sapien.Pose([-3.0, 0.0, 0.0]))

        physx.gpu_init()
        group = sapien.render.RenderSystemGroup([render_system])
        group.set_cuda_poses(physx.cuda_rigid_body_data)
        group.create_camera_group([camera], ["Color"])
        with self.assertRaisesRegex(RuntimeError, "cpu-mode light shares its entity"):
            group.gpu_init()

    def test_viewer_controller_camera_stays_cpu_beside_sealed_group(self) -> None:
        """The Viewer keeps its architecture next to a sealed RenderSystemGroup:
        the controller camera stays CPU-managed, joins no camera group, and
        allocates no CUDA pose row, while direct transport and grouped capture
        keep working side by side. The Viewer opens before gpu_init() because the
        group freezes scene topology, including the controller camera node."""
        from sapien.utils import Viewer

        sapien.physx.enable_gpu()
        cudart = _load_cudart()
        device = sapien.Device("cuda")
        physx = sapien.physx.PhysxGpuSystem(device)
        render_system = sapien.render.RenderSystem(device)
        scene = sapien.Scene([physx, render_system])
        scene.set_ambient_light([0.5, 0.5, 0.5])

        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.3, 0.3, 0.3])
        builder.add_box_visual(
            half_size=[0.3, 0.3, 0.3], material=[0.8, 0.05, 0.05]
        )
        builder.initial_pose = sapien.Pose([0.0, 0.0, -50.0])
        actor = builder.build()
        body = actor.find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent
        )

        camera = scene.add_camera(
            "camera", 64, 64, np.deg2rad(45), 0.05, 10
        )
        camera.set_local_pose(sapien.Pose([-3.0, 0.0, 0.0]))

        physx.gpu_init()

        # Apply the pose so the Viewer's direct transport, which refreshes the pose
        # buffer from PhysX simulation state, sees the actor at the origin.
        pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], np.float32)
        pose_buffer = physx.cuda_rigid_body_data
        self.assertEqual(
            cudart.cudaMemcpy(
                ctypes.c_void_p(
                    pose_buffer.ptr + body.gpu_pose_index * pose_buffer.strides[0]
                ),
                ctypes.c_void_p(pose.ctypes.data),
                pose.nbytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            0,
        )
        physx.gpu_apply_rigid_dynamic_data()

        viewer = Viewer(resolutions=(320, 240))
        try:
            viewer.configure_physx_gpu_rendering(physx, "direct")
            viewer.set_scene(scene)
            # Camera-lineset helper visuals would add nodes to the shared render
            # scene after the group froze its topology; keep the overlay off.
            viewer.control_window.show_camera_linesets = False

            group = sapien.render.RenderSystemGroup([render_system])
            group.set_cuda_poses(physx.cuda_rigid_body_data)
            camera_group = group.create_camera_group([camera], ["Color"])
            camera_group.set_pose_mode(camera, "cuda")
            group.gpu_init()

            # The controller camera stays CPU-managed after the group sealed its
            # member cameras: free navigation never raises and needs no pose row.
            viewer.set_camera_xyz(-4.0, 0.0, 0.0)
            viewer.update_render()
            viewer.render()
            viewer.set_camera_xyz(-4.0, 0.5, 0.2)
            viewer.update_render()
            viewer.render()
            segmentation = viewer.window.get_picture("Segmentation")
            self.assertGreater(
                int(np.count_nonzero(segmentation[..., 0] == actor.per_scene_id)),
                20,
            )

            # The sealed group keeps capturing beside the open Viewer, and the
            # controller camera occupies no group CUDA pose row.
            group.update_render()
            camera_group.take_picture()
            image = camera_group.get_picture_cuda("Color")
            host = np.empty((64, 64, 4), np.float32)
            self.assertEqual(
                cudart.cudaMemcpy(
                    ctypes.c_void_p(host.ctypes.data),
                    ctypes.c_void_p(image.ptr),
                    host.nbytes,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                ),
                0,
            )
            self.assertGreater(int(np.count_nonzero(host[..., 0] > 0.2)), 20)
            self.assertEqual(camera_group.cuda_poses.shape, [1, 7])
            self.assertEqual(physx._sync_poses_gpu_to_cpu_count, 0)
        finally:
            viewer.close()

    def test_live_group_protects_shared_scene_from_released_camera(self) -> None:
        sapien.physx.enable_gpu()
        device = sapien.Device("cuda")
        physx = sapien.physx.PhysxGpuSystem(device)
        render_system = sapien.render.RenderSystem(device)
        scene = sapien.Scene([physx, render_system])
        scene.set_ambient_light([0.5, 0.5, 0.5])

        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.3, 0.3, 0.3])
        builder.add_box_visual(
            half_size=[0.3, 0.3, 0.3], material=[0.8, 0.05, 0.05]
        )
        builder.initial_pose = sapien.Pose([0.0, 0.0, -50.0])
        actor = builder.build()
        body = actor.find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent
        )

        camera_a = scene.add_camera(
            "camera_a", 64, 64, np.deg2rad(45), 0.05, 10
        )
        camera_b = scene.add_camera(
            "camera_b", 64, 64, np.deg2rad(45), 0.05, 10
        )
        camera_a.set_local_pose(sapien.Pose([-3.0, 0.0, 0.0]))
        camera_b.set_local_pose(sapien.Pose([-3.0, 0.0, 0.0]))

        physx.gpu_init()
        group_a = sapien.render.RenderSystemGroup([render_system])
        group_b = sapien.render.RenderSystemGroup([render_system])
        group_a.set_cuda_poses(physx.cuda_rigid_body_data)
        group_b.set_cuda_poses(physx.cuda_rigid_body_data)
        camera_group_a = group_a.create_camera_group([camera_a], ["Color"])
        group_b.create_camera_group([camera_b], ["Color"])
        group_a.gpu_init()
        group_b.gpu_init()

        pose_buffer = physx.cuda_rigid_body_data
        pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], np.float32)
        cudart = _load_cudart()
        status = cudart.cudaMemcpy(
            ctypes.c_void_p(
                pose_buffer.ptr + body.gpu_pose_index * pose_buffer.strides[0]
            ),
            ctypes.c_void_p(pose.ctypes.data),
            pose.nbytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
        self.assertEqual(status, 0)

        group_b.update_render()
        camera_b.take_picture()
        before = camera_b.get_picture("Color").copy()

        # Releasing camera A restores its CPU-managed camera state, but its
        # direct render must not upload stale CPU object transforms while group B
        # still owns the shared scene transform buffer.
        del camera_group_a
        del group_a
        gc.collect()
        scene.update_render()
        camera_a.take_picture()
        camera_b.take_picture()
        after = camera_b.get_picture("Color").copy()

        for image in (before, after):
            red = image[..., 0]
            self.assertGreater(int(np.count_nonzero(red > 0.2)), 20)

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
            # Row bounds become final when gpu_init() discovers automatic PhysX
            # body/camera bindings. A deterministic validation failure occurs before
            # camera ownership is sealed, so the group can be corrected and retried.
            group.set_cuda_poses(
                sapien.CudaArray(CudaPoseView([0, 13], [52, 4], "<f4"))
            )
            with self.assertRaisesRegex(RuntimeError, "outside the pose buffer"):
                group.gpu_init()

            group.set_cuda_poses(pose_buffer)
            group.gpu_init()
            with self.assertRaisesRegex(RuntimeError, "already initialized"):
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

    def _run_free_camera_articulation_updates(self, shader: str) -> None:
        """One group update + one capture per frame must show every articulation
        link at its CUDA pose and honor per-frame cuda-camera pose-row writes.

        Contract: after gpu_init() every grouped camera pose is owned per its
        configured mode. Cuda-mode cameras move through the group's CUDA pose rows
        (set_cuda_pose / cuda_poses); CPU set_local_pose() is rejected, and no
        scene.update_render() participates in capture.
        """
        # This is a required GPU test: CUDA or PhysX GPU initialization
        # failures must fail the test instead of being converted to skips.
        sapien.physx.enable_gpu()
        cudart = _load_cudart()

        width = height = 128
        sapien.render.set_camera_shader_dir(shader)
        if shader == "rt":
            sapien.render.set_ray_tracing_samples_per_pixel(4)
            sapien.render.set_ray_tracing_path_depth(2)
            sapien.render.set_ray_tracing_denoiser("none")

        try:
            device = sapien.Device("cuda")
            physx = sapien.physx.PhysxGpuSystem(device)
            physx.set_timestep(1 / 120)
            render_system = sapien.render.RenderSystem(device)
            scene = sapien.Scene([physx, render_system])
            scene.set_ambient_light([0.5, 0.5, 0.5])

            # Three visually distinct links chained along +y. The articulation is
            # built far below the camera frustum: any render path that falls back
            # to stale CPU poses makes the colored links vanish from the image.
            builder = scene.create_articulation_builder()
            builder.set_initial_pose(sapien.Pose([0.0, -0.5, -50.0]))
            colors = ([0.8, 0.05, 0.05], [0.05, 0.8, 0.05], [0.05, 0.05, 0.8])
            link_builder = builder.create_link_builder()
            link_builder.set_name("root")
            link_builder.add_box_collision(half_size=[0.15, 0.15, 0.15])
            link_builder.add_box_visual(half_size=[0.15, 0.15, 0.15], material=colors[0])
            for i, color in enumerate(colors[1:], start=1):
                child = builder.create_link_builder(link_builder)
                child.set_name(f"link{i}")
                child.add_box_collision(half_size=[0.15, 0.15, 0.15])
                child.add_box_visual(half_size=[0.15, 0.15, 0.15], material=color)
                child.set_joint_properties(
                    "revolute",
                    limits=[[-0.5, 0.5]],
                    pose_in_parent=sapien.Pose([0.0, 0.5, 0.0]),
                    pose_in_child=sapien.Pose(),
                )
                link_builder = child
            articulation = builder.build()
            root_link = articulation.get_links()[0]

            # A render-only static body (no CUDA pose source): its transform is
            # a one-time snapshot sealed at gpu_init(). Kept outside the frustum.
            static_entity = sapien.Entity()
            static_body = sapien.render.RenderBodyComponent()
            static_body.attach(
                sapien.render.RenderShapeBox(
                    [0.1, 0.1, 0.1],
                    sapien.render.RenderMaterial(base_color=[0.4, 0.4, 0.4, 1.0]),
                )
            )
            static_entity.add_component(static_body)
            static_entity.set_pose(sapien.Pose([0.0, 0.0, -5.0]))
            scene.add_entity(static_entity)

            # Free camera configured 'cuda': no sibling PhysX body, no GPU pose batch
            # index. Its CPU pose set before group creation seeds the CUDA pose row.
            camera = scene.add_camera("camera", width, height, np.deg2rad(60), 0.05, 20)
            camera.set_local_pose(sapien.Pose([-3.0, 0.0, 1.0]))

            physx.gpu_init()

            group = sapien.render.RenderSystemGroup([render_system])
            group.set_cuda_poses(physx.cuda_rigid_body_data)
            camera_group = group.create_camera_group([camera], ["Color"])
            camera_group.set_pose_mode(camera, "cuda")

            # Steady-state APIs require explicit initialization.
            with self.assertRaisesRegex(RuntimeError, "gpu_init"):
                group.update_render()
            with self.assertRaisesRegex(RuntimeError, "gpu_init"):
                camera_group.take_picture()

            group.gpu_init()

            # gpu_init() seals the group: no further camera groups, no CPU pose
            # writes on member cameras, and the cuda camera has row 0.
            with self.assertRaisesRegex(RuntimeError, "already initialized"):
                group.create_camera_group([camera], ["Color"])
            with self.assertRaisesRegex(RuntimeError, "pose mode is 'cuda'"):
                camera.set_local_pose(sapien.Pose([-3.0, 0.0, 1.0]))
            self.assertEqual(camera_group.get_cuda_pose_index(camera), 0)
            cuda_camera_poses = camera_group.cuda_poses
            self.assertEqual(cuda_camera_poses.shape, [1, 7])

            pose_buffer = physx.cuda_rigid_body_data
            root_row_ptr = (
                pose_buffer.ptr + root_link.gpu_pose_index * pose_buffer.strides[0]
            )

            def set_gpu_root_position(position) -> None:
                pose = np.array([*position, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                status = cudart.cudaMemcpy(
                    ctypes.c_void_p(root_row_ptr),
                    ctypes.c_void_p(pose.ctypes.data),
                    pose.nbytes,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                )
                self.assertEqual(status, 0)
                physx.gpu_apply_articulation_root_pose()
                physx.gpu_update_articulation_kinematics()
                physx.gpu_fetch_articulation_link_pose()
                self.assertEqual(cudart.cudaDeviceSynchronize(), 0)

            def capture() -> np.ndarray:
                # The steady-state contract: one group update, one render, one
                # CUDA readback. No CPU-side scene.update_render() involved.
                group.update_render()
                camera.take_picture()
                image = camera.get_picture_cuda("Color")
                host = np.empty((height, width, 4), dtype=np.float32)
                status = cudart.cudaMemcpy(
                    ctypes.c_void_p(host.ctypes.data),
                    ctypes.c_void_p(image.ptr),
                    host.nbytes,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                )
                self.assertEqual(status, 0)
                return host

            def link_centroid_columns(image: np.ndarray, frame: int) -> np.ndarray:
                rgb = image[..., :3]
                columns = []
                for channel in range(3):
                    others = [c for c in range(3) if c != channel]
                    mask = (
                        (rgb[..., channel] > 0.2)
                        & (rgb[..., channel] > rgb[..., others[0]] + 0.1)
                        & (rgb[..., channel] > rgb[..., others[1]] + 0.1)
                    )
                    count = int(np.count_nonzero(mask))
                    self.assertGreater(
                        count,
                        20,
                        f"frame {frame}: link with color channel {channel} is missing",
                    )
                    columns.append(float(np.argwhere(mask)[:, 1].mean()))
                return np.array(columns)

            # Frame 0: links must appear at their CUDA poses, not the CPU build pose.
            set_gpu_root_position([0.0, -0.5, 1.0])
            columns_0 = link_centroid_columns(capture(), 0)

            # Frame 1: move only the GPU body poses; every link must shift together.
            set_gpu_root_position([0.0, -0.2, 1.0])
            columns_1 = link_centroid_columns(capture(), 1)
            body_shift = columns_1 - columns_0
            for shift in body_shift:
                self.assertGreater(abs(float(shift)), 4.0)
                self.assertEqual(np.sign(shift), np.sign(body_shift[0]))

            # Frame 2: move only the free camera through its CUDA pose row; the
            # links must shift the opposite way in the image.
            camera_group.set_cuda_pose(camera, sapien.Pose([-3.0, 0.3, 1.0]))
            columns_2 = link_centroid_columns(capture(), 2)
            camera_shift = columns_2 - columns_1
            for shift in camera_shift:
                self.assertGreater(abs(float(shift)), 4.0)
                self.assertEqual(np.sign(shift), -np.sign(body_shift[0]))

            # Frame 3: another GPU-only update stays stable on consecutive frames.
            set_gpu_root_position([0.0, 0.1, 1.0])
            columns_3 = link_centroid_columns(capture(), 3)
            for shift in columns_3 - columns_2:
                self.assertGreater(abs(float(shift)), 4.0)
                self.assertEqual(np.sign(shift), np.sign(body_shift[0]))

            # The camera group created once at setup stays usable for the same
            # free camera without recreation, and its capture shows the same
            # links at the same CUDA poses as the direct capture.
            group.update_render()
            camera_group.take_picture()
            group_image = camera_group.get_picture_cuda("Color")
            self.assertEqual(group_image.shape, [1, height, width, 4])
            group_host = np.empty((height, width, 4), dtype=np.float32)
            status = cudart.cudaMemcpy(
                ctypes.c_void_p(group_host.ctypes.data),
                ctypes.c_void_p(group_image.ptr),
                group_host.nbytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
            self.assertEqual(status, 0)
            group_columns = link_centroid_columns(group_host, 4)
            for delta in group_columns - columns_3:
                self.assertLess(abs(float(delta)), 2.0)

            # Writing the CUDA pose row directly is equivalent to
            # set_cuda_pose: move the camera back and expect the image
            # to shift opposite to the frame-2 camera move.
            raw_pose = np.array([-3.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            status = cudart.cudaMemcpy(
                ctypes.c_void_p(cuda_camera_poses.ptr),
                ctypes.c_void_p(raw_pose.ctypes.data),
                raw_pose.nbytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
            self.assertEqual(status, 0)
            columns_5 = link_centroid_columns(capture(), 5)
            for shift in columns_5 - group_columns:
                self.assertGreater(abs(float(shift)), 4.0)
                self.assertEqual(np.sign(shift), np.sign(body_shift[0]))

            # Static snapshots are sealed: moving a CPU-owned static body after
            # gpu_init() is an error surfaced by the next CPU scene update.
            static_entity.set_pose(sapien.Pose([0.0, 0.0, -6.0]))
            with self.assertRaisesRegex(RuntimeError, "static snapshot"):
                scene.update_render()
            static_entity.set_pose(sapien.Pose([0.0, 0.0, -5.0]))
            scene.update_render()

            self.assertEqual(physx._sync_poses_gpu_to_cpu_count, 0)
        finally:
            sapien.render.set_camera_shader_dir("default")

    def test_raster_free_camera_tracks_cuda_articulation_poses(self) -> None:
        self._run_free_camera_articulation_updates("default")

    def test_rt_free_camera_tracks_cuda_articulation_poses(self) -> None:
        self._run_free_camera_articulation_updates("rt")

    def _assert_viewer_staged_pose_updates(self, shader: str) -> None:
        from sapien.utils import Viewer

        sapien.physx.enable_gpu()
        sapien.render.set_viewer_shader_dir(shader)
        if shader == "rt":
            sapien.render.set_ray_tracing_samples_per_pixel(2)
            sapien.render.set_ray_tracing_path_depth(2)
            sapien.render.set_ray_tracing_denoiser("none")

        viewer = None
        try:
            device = sapien.Device("cuda")
            physx = sapien.physx.PhysxGpuSystem(device)
            render_system = sapien.render.RenderSystem(device)
            scene = sapien.Scene([physx, render_system])
            scene.set_ambient_light([0.5, 0.5, 0.5])
            scene.add_ground(0.0, render_half_size=[4.0, 4.0])

            builder = scene.create_actor_builder()
            builder.add_sphere_collision(radius=0.15)
            builder.add_sphere_visual(radius=0.15, material=[0.8, 0.05, 0.05])
            builder.initial_pose = sapien.Pose([0.0, 0.0, 2.0])
            actor = builder.build()
            initial_cpu_pose = actor.pose
            physx.gpu_init()

            viewer = Viewer(resolutions=(320, 240))
            viewer.configure_physx_gpu_rendering(physx, "staged")
            viewer.set_scene(scene)
            viewer.set_camera_xyz(-4.0, 0.0, 2.0)
            viewer.update_render()
            viewer.render()
            initial_transfer_bytes = viewer.pose_transfer_bytes
            initial_sync_count = physx._sync_poses_gpu_to_cpu_count

            for _ in range(180):
                physx.step()
            physx.gpu_fetch_rigid_dynamic_data()
            physx.gpu_fetch_articulation_link_pose()
            viewer.update_render()
            viewer.render()
            staged = viewer.window.get_picture("Segmentation").copy()

            self.assertEqual(viewer.pose_transport, "staged")
            self.assertEqual(viewer.pose_transfer_bytes - initial_transfer_bytes, 7 * 4)
            self.assertEqual(physx._sync_poses_gpu_to_cpu_count, initial_sync_count)
            self.assertTrue(np.allclose(actor.pose.p, initial_cpu_pose.p))
            submitted_pose = viewer.get_entity_viewer_pose(actor)
            self.assertLess(float(submitted_pose.p[2]), float(initial_cpu_pose.p[2]) - 0.5)
            self.assertGreater(np.count_nonzero(staged[..., 0] == actor.per_scene_id), 20)

            if shader == "default":
                viewer.configure_physx_gpu_rendering(physx, "direct")
                viewer.update_render()
                before_selected_read = viewer.pose_transfer_bytes
                direct_pose = viewer.get_entity_viewer_pose(actor)
                self.assertTrue(np.allclose(direct_pose.p, submitted_pose.p))
                self.assertEqual(
                    viewer.pose_transfer_bytes - before_selected_read, 7 * 4
                )
                _ = viewer.get_entity_viewer_pose(actor)
                self.assertEqual(
                    viewer.pose_transfer_bytes - before_selected_read, 7 * 4
                )

            viewer.configure_physx_gpu_rendering(physx, "cpu-debug")
            viewer.update_render()
            viewer.render()
            cpu_debug = viewer.window.get_picture("Segmentation").copy()
            self.assertTrue(np.array_equal(staged, cpu_debug))
        finally:
            if viewer is not None:
                viewer.close()
            sapien.render.set_viewer_shader_dir("default")

    def test_viewer_staged_raster_pose_updates(self) -> None:
        self._assert_viewer_staged_pose_updates("default")

    def test_viewer_staged_rt_pose_updates(self) -> None:
        self._assert_viewer_staged_pose_updates("rt")

    def test_viewer_gpu_spring_and_queued_teleport(self) -> None:
        from sapien.utils import Viewer

        sapien.physx.enable_gpu()
        device = sapien.Device("cuda")
        physx = sapien.physx.PhysxGpuSystem(device)
        scene = sapien.Scene([physx, sapien.render.RenderSystem(device)])
        builder = scene.create_actor_builder()
        builder.add_sphere_collision(radius=0.15)
        builder.add_sphere_visual(radius=0.15)
        actor = builder.build()
        body = actor.find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent
        )
        body.disable_gravity = True
        physx.gpu_init()

        viewer = Viewer(resolutions=(320, 240))
        try:
            viewer.configure_physx_gpu_rendering(physx, "direct")
            viewer.set_scene(scene)
            viewer.update_render()
            initial_pose = viewer.get_entity_viewer_pose(actor)
            initial_sync_count = physx._sync_poses_gpu_to_cpu_count

            self.assertTrue(viewer.begin_gpu_interaction(actor, initial_pose.p))
            viewer.update_gpu_interaction_target(initial_pose.p + [1.0, 0.0, 0.0])
            viewer.apply_interactions()
            physx.step()
            viewer.update_render()
            moved_pose = viewer.get_entity_viewer_pose(actor)
            self.assertGreater(float(moved_pose.p[0]), float(initial_pose.p[0]))

            viewer.end_gpu_interaction()
            viewer.queue_gpu_rigid_dynamic_pose(
                body, sapien.Pose([2.0, 0.0, 1.0]), zero_velocity=True
            )
            viewer.apply_interactions()
            physx.step()
            viewer.update_render()
            teleported_pose = viewer.get_entity_viewer_pose(actor)
            self.assertAlmostEqual(float(teleported_pose.p[0]), 2.0, places=3)
            self.assertAlmostEqual(float(teleported_pose.p[2]), 1.0, places=3)
            self.assertEqual(physx._sync_poses_gpu_to_cpu_count, initial_sync_count)
        finally:
            viewer.close()

    def test_viewer_gpu_articulation_property_commands(self) -> None:
        from sapien.utils import Viewer

        sapien.physx.enable_gpu()
        device = sapien.Device("cuda")
        physx = sapien.physx.PhysxGpuSystem(device)
        scene = sapien.Scene([physx, sapien.render.RenderSystem(device)])

        root = sapien.physx.PhysxArticulationLinkComponent()
        child = sapien.physx.PhysxArticulationLinkComponent(root)
        root.disable_gravity = True
        child.disable_gravity = True
        child.joint.set_type("revolute")
        child.joint.set_limits([[-1.0, 1.0]])
        child.joint.set_pose_in_parent(sapien.Pose([0.3, 0.0, 0.0]))
        root_entity = sapien.Entity().add_component(root)
        child_entity = sapien.Entity().add_component(child)
        scene.add_entity(root_entity)
        scene.add_entity(child_entity)
        articulation = root.articulation
        physx.gpu_init()

        viewer = Viewer(resolutions=(320, 240))
        try:
            viewer.configure_physx_gpu_rendering(physx, "direct")
            viewer.set_scene(scene)
            viewer.update_render()
            viewer.select_entity(root_entity)
            viewer.render()
            qpos, target_qpos, target_qvel = viewer._get_gpu_articulation_state(
                articulation
            )
            self.assertEqual(qpos.shape, (1,))
            self.assertEqual(target_qpos.shape, (1,))
            self.assertEqual(target_qvel.shape, (1,))

            viewer._queue_gpu_articulation_qpos(
                articulation, np.asarray([0.2], dtype=np.float32)
            )
            viewer._queue_gpu_articulation_target_qpos(
                articulation, np.asarray([0.35], dtype=np.float32)
            )
            viewer._queue_gpu_articulation_target_qvel(
                articulation, np.asarray([-0.4], dtype=np.float32)
            )
            viewer.apply_interactions()

            self.assertAlmostEqual(
                physx._gpu_download_articulation_qpos(articulation.gpu_index)[0],
                0.2,
                places=4,
            )
            self.assertAlmostEqual(
                physx._gpu_download_articulation_target_qpos(
                    articulation.gpu_index
                )[0],
                0.35,
                places=4,
            )
            self.assertAlmostEqual(
                physx._gpu_download_articulation_target_qvel(
                    articulation.gpu_index
                )[0],
                -0.4,
                places=4,
            )
            self.assertEqual(physx._sync_poses_gpu_to_cpu_count, 0)
        finally:
            viewer.close()

    def test_viewer_auto_detects_shared_gpu_system_after_reopen(self) -> None:
        from sapien.utils import Viewer

        sapien.physx.enable_gpu()
        device = sapien.Device("cuda")
        base_scene = sapien.Scene(
            [sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(device)]
        )
        cpu_builder = base_scene.create_actor_builder()
        cpu_builder.add_box_collision(half_size=[0.1, 0.1, 0.1])
        cpu_builder.add_box_visual(half_size=[0.1, 0.1, 0.1])
        cpu_actor = cpu_builder.build()

        physx = sapien.physx.PhysxGpuSystem(device)
        shared_render = sapien.render.RenderSystem(device)
        shared_render.batched_render_shared = True
        shared_scene = sapien.Scene([physx, shared_render])
        builder = shared_scene.create_actor_builder()
        builder.add_sphere_collision(radius=0.15)
        builder.add_sphere_visual(radius=0.15)
        builder.build()
        physx.gpu_init()

        for _ in range(2):
            viewer = Viewer(resolutions=(320, 240))
            try:
                viewer.set_scene(base_scene)
                viewer.update_render()
                viewer.render()
                self.assertEqual(viewer.pose_transport, "direct")
                self.assertIs(viewer.window.physx_gpu_system, physx)
                self.assertIs(viewer._physx_gpu_system, physx)
                self.assertFalse(
                    viewer.begin_gpu_interaction(cpu_actor, np.zeros(3))
                )
                self.assertEqual(physx._sync_poses_gpu_to_cpu_count, 0)
            finally:
                viewer.close()
