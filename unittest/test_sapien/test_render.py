import ctypes
import ctypes.util
import gc
import unittest

import numpy as np
import sapien


CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


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
            group.gpu_init()
            # Free cameras in a group expose their CUDA-owned camera buffer and
            # a group pose row seeded from the CPU pose at gpu_init().
            self.assertGreater(camera._cuda_buffer.shape[0], 0)
            self.assertEqual(camera_group.get_free_camera_cuda_pose_index(camera), 0)
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

        # A free-camera-only group needs no unrelated primary object pose source.
        group.gpu_init()
        camera_group.set_free_camera_pose(camera, sapien.Pose([-3.0, 0.2, 0.0]))
        group.update_render()
        camera.take_picture()
        self.assertGreater(float(np.max(camera.get_picture("Color")[..., :3])), 0.01)

        # Every CPU-side pose/configuration source is sealed while the group owns it.
        with self.assertRaisesRegex(RuntimeError, "sealed"):
            camera.set_gpu_pose_batch_index(0)
        with self.assertRaisesRegex(RuntimeError, "sealed"):
            camera.set_fovx(np.deg2rad(50))
        camera.entity.set_pose(sapien.Pose([0.0, 0.1, 0.0]))
        with self.assertRaisesRegex(RuntimeError, "CPU pose changed"):
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


class TestSceneGPU(unittest.TestCase):
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
            camera_group.get_free_camera_cuda_pose_index(camera)

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
        link at its CUDA pose and honor per-frame free-camera CUDA pose writes.

        Contract: after create_camera_group() every camera transform is
        GPU-owned. Free cameras move through the group's CUDA pose rows
        (set_free_camera_pose / cuda_free_camera_poses); CPU set_local_pose()
        is rejected, and no scene.update_render() participates in capture.
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

            # Free camera: no sibling PhysX body, no GPU pose batch index. Its
            # CPU pose set before group creation seeds the CUDA pose row.
            camera = scene.add_camera("camera", width, height, np.deg2rad(60), 0.05, 20)
            camera.set_local_pose(sapien.Pose([-3.0, 0.0, 1.0]))

            physx.gpu_init()

            group = sapien.render.RenderSystemGroup([render_system])
            group.set_cuda_poses(physx.cuda_rigid_body_data)
            camera_group = group.create_camera_group([camera], ["Color"])

            # Steady-state APIs require explicit initialization.
            with self.assertRaisesRegex(RuntimeError, "gpu_init"):
                group.update_render()
            with self.assertRaisesRegex(RuntimeError, "gpu_init"):
                camera_group.take_picture()

            group.gpu_init()

            # gpu_init() seals the group: no further camera groups, no CPU pose
            # writes on member cameras, and the free camera has row 0.
            with self.assertRaisesRegex(RuntimeError, "already initialized"):
                group.create_camera_group([camera], ["Color"])
            with self.assertRaisesRegex(RuntimeError, "owned by a RenderCameraGroup"):
                camera.set_local_pose(sapien.Pose([-3.0, 0.0, 1.0]))
            self.assertEqual(camera_group.get_free_camera_cuda_pose_index(camera), 0)
            free_camera_poses = camera_group.cuda_free_camera_poses
            self.assertEqual(free_camera_poses.shape, [1, 7])

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
            camera_group.set_free_camera_pose(camera, sapien.Pose([-3.0, 0.3, 1.0]))
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
            # set_free_camera_pose: move the camera back and expect the image
            # to shift opposite to the frame-2 camera move.
            raw_pose = np.array([-3.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            status = cudart.cudaMemcpy(
                ctypes.c_void_p(free_camera_poses.ptr),
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
