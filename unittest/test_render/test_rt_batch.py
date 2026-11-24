"""Batched ray tracing: the `rt_batch` pack renders a whole RenderCameraGroup.

Every test below drives a camera group whose cameras use `RenderConfig("rt_batch")`, and
checks the properties the batched path must keep from the per-camera path: agreement with
rasterization on geometry, exact environment isolation, GPU-sourced poses, live light edits,
hard shadows, picture formats, misuse errors and point clouds.
"""

import unittest
from typing import List, Sequence, Tuple

import numpy as np
import sapien

try:
    import torch
except ImportError:  # pragma: no cover - the batched pictures are CUDA buffers
    torch = None

SIZE = 64


def batch_config(**properties) -> "sapien.render.RenderConfig":
    """A `rt_batch` config with the visibility tier: one primary ray, no lighting."""
    config = sapien.render.RenderConfig("rt_batch", preset="visibility")
    config["spp"] = 1
    for name, value in properties.items():
        config[name] = value
    return config


def make_scene(ambient: float = 0.0) -> "sapien.Scene":
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem()])
    scene.set_ambient_light([ambient] * 3)
    return scene


def add_box(scene: "sapien.Scene", pose: "sapien.Pose", color: Sequence[float],
            half_size: float = 0.4) -> None:
    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=[half_size] * 3,
                           material=sapien.render.RenderMaterial(
                               base_color=[*color, 1.0]))
    builder.set_initial_pose(pose)
    builder.build_kinematic()


def render_group(cameras: List["sapien.render.RenderCameraComponent"],
                 pictures: Sequence[str]) -> Tuple["sapien.render.RenderSystemGroup",
                                                   "sapien.render.RenderCameraGroup"]:
    systems = []
    for camera in cameras:
        system = camera.entity.scene.render_system
        if system not in systems:
            systems.append(system)
    group = sapien.render.RenderSystemGroup(systems)
    camera_group = group.create_camera_group(cameras, list(pictures))
    group.gpu_init()
    return group, camera_group


def picture(camera_group: "sapien.render.RenderCameraGroup", name: str) -> np.ndarray:
    """Host copy of a group picture. The DLPack export orders the copy after the frame on the
    current stream, and the temporary view dies here so the group can close."""
    tensor = torch.from_dlpack(camera_group.get_picture_cuda(name))
    result = tensor.cpu().numpy()
    del tensor
    return result


def setUpModule() -> None:
    if torch is None or not torch.cuda.is_available():
        raise unittest.SkipTest("batched pictures are CUDA buffers: requires CUDA-enabled Torch")


class TestBatchAgreesWithRaster(unittest.TestCase):
    def test_depth_and_segmentation_match_the_raster_pack(self) -> None:
        """Visibility-tier Depth and Segmentation must match the raster default pack's
        Position depth and Segmentation, except on silhouette edges."""
        scene = make_scene(ambient=0.5)
        scene.add_ground(0.0)
        add_box(scene, sapien.Pose([0.0, 0.0, 0.6]), [0.8, 0.2, 0.2])
        add_box(scene, sapien.Pose([0.0, 1.2, 0.2]), [0.2, 0.8, 0.2], half_size=0.2)
        pose = sapien.Pose([-3.0, 0.0, 1.2], [0.9, 0.0, 0.42, 0.0])

        raster_config = sapien.render.RenderConfig("default")
        raster_camera = scene.add_camera("raster", SIZE, SIZE, 1.0, 0.05, 30.0,
                                         render_config=raster_config)
        raster_camera.entity.pose = pose
        batch_camera = scene.add_camera("batch", SIZE, SIZE, 1.0, 0.05, 30.0,
                                        render_config=batch_config())
        batch_camera.entity.pose = pose

        scene.update_render()
        raster_camera.take_picture()
        raster_position = raster_camera.get_picture("Position").copy()
        raster_segmentation = raster_camera.get_picture("Segmentation").copy()

        group, camera_group = render_group([batch_camera], ["Depth", "Segmentation"])
        try:
            group.update_render()
            camera_group.take_picture()
            depth = picture(camera_group, "Depth")[0]
            segmentation = picture(camera_group, "Segmentation")[0].copy()
        finally:
            group.close()
            scene.close()

        # Depth is linear depth in metres, positive in front, 0 where the ray missed
        batch_depth = depth
        raster_depth = raster_position[..., 2]
        valid = raster_depth < -1e-4
        self.assertGreater(np.count_nonzero(valid), SIZE * SIZE // 4)

        scale = 30.0
        error = np.abs(batch_depth - np.abs(raster_depth)) / max(scale, 1e-6)
        differing = np.count_nonzero((error > 0.02) & valid)
        self.assertLessEqual(differing / SIZE / SIZE, 0.01,
                             f"{differing} of {SIZE * SIZE} pixels differ by over 2% of far")

        self.assertEqual(np.count_nonzero(segmentation[..., 0]),
                         np.count_nonzero(raster_segmentation[..., 0]))
        self.assertLessEqual(
            np.count_nonzero(segmentation[..., 0] != raster_segmentation[..., 0]) / SIZE / SIZE,
            0.01)


class TestBatchEnvironments(unittest.TestCase):
    def test_markers_stay_in_their_environment(self) -> None:
        """Three scenes, each with its own camera; a red marker exists only in scene 1."""
        colors = [[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]
        scenes = []
        cameras = []
        for index in range(3):
            scene = make_scene(ambient=1.0)
            add_box(scene, sapien.Pose([0.0, 0.0, 0.0]), [0.5, 0.5, 0.5])
            if index == 1:
                add_box(scene, sapien.Pose([0.9, -0.9, 0.0]), colors[0], half_size=0.2)
            camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.05, 20.0,
                                      render_config=batch_config())
            camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
            scenes.append(scene)
            cameras.append(camera)

        group, camera_group = render_group(cameras, ["Albedo"])
        try:
            group.update_render()
            camera_group.take_picture()
            albedo = picture(camera_group, "Albedo")
        finally:
            group.close()
            for scene in scenes:
                scene.close()

        # the marker's red is present only in environment 1
        marker = albedo[..., 0] > albedo[..., 1] + 0.25
        self.assertGreater(np.count_nonzero(marker[1]), 20)
        self.assertEqual(np.count_nonzero(marker[0]), 0)
        self.assertEqual(np.count_nonzero(marker[2]), 0)

    def test_two_cameras_of_one_environment_agree(self) -> None:
        """Two cameras on the same scene must see the same colours from the same pose."""
        scene = make_scene(ambient=1.0)
        add_box(scene, sapien.Pose([0.0, 0.0, 0.0]), [0.2, 0.7, 0.3])
        pose = sapien.Pose([-3.0, 0.0, 0.5])
        first = scene.add_camera("first", SIZE, SIZE, 1.0, 0.05, 20.0,
                                 render_config=batch_config())
        first.entity.pose = pose
        second = scene.add_camera("second", SIZE, SIZE, 1.0, 0.05, 20.0,
                                  render_config=batch_config())
        second.entity.pose = pose

        group, camera_group = render_group([first, second], ["Albedo"])
        try:
            group.update_render()
            camera_group.take_picture()
            albedo = picture(camera_group, "Albedo")
        finally:
            group.close()
            scene.close()

        np.testing.assert_allclose(albedo[0], albedo[1], atol=1e-5)


class TestBatchPoses(unittest.TestCase):
    def test_cuda_camera_pose_changes_the_picture(self) -> None:
        scene = make_scene(ambient=1.0)
        add_box(scene, sapien.Pose([0.0, 0.0, 0.0]), [0.9, 0.2, 0.2])
        add_box(scene, sapien.Pose([1.4, -1.4, 0.0]), [0.2, 0.2, 0.9], half_size=0.3)
        camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.05, 20.0,
                                  render_config=batch_config())
        camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])

        group = sapien.render.RenderSystemGroup([scene.render_system])
        camera_group = group.create_camera_group([camera], ["Albedo"])
        camera_group.set_pose_mode(camera, "cuda")
        group.gpu_init()
        try:
            group.update_render()
            camera_group.take_picture()
            before = picture(camera_group, "Albedo")[0].copy()

            camera_group.set_cuda_pose(camera, sapien.Pose([-3.0, 1.6, 0.9]))
            group.update_render()
            camera_group.take_picture()
            after = picture(camera_group, "Albedo")[0]
            self.assertGreater(np.abs(after - before).mean(), 1e-3)
        finally:
            group.close()
            scene.close()


    def test_cpu_camera_pose_and_odd_size(self) -> None:
        """A cpu-mode camera uploads its moved pose at update_render(); sizes need not be
        multiples of the 8x8 workgroup."""
        width, height = 37, 29
        scene = make_scene(ambient=1.0)
        add_box(scene, sapien.Pose([0.0, 0.0, 0.0]), [0.9, 0.2, 0.2])
        camera = scene.add_camera("camera", width, height, 1.0, 0.05, 20.0,
                                  render_config=batch_config())
        camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])

        group = sapien.render.RenderSystemGroup([scene.render_system])
        camera_group = group.create_camera_group([camera], ["Segmentation"])
        camera_group.set_pose_mode(camera, "cpu")
        group.gpu_init()
        try:
            group.update_render()
            camera_group.take_picture()
            before = picture(camera_group, "Segmentation")[0, ..., 1] > 0
            self.assertEqual(before.shape, (height, width))

            # the camera's left is +y: moving the camera to -y puts the box on the image's left
            camera.entity.pose = sapien.Pose([-3.0, -1.0, 0.0])
            group.update_render()
            camera_group.take_picture()
            after = picture(camera_group, "Segmentation")[0, ..., 1] > 0
        finally:
            group.close()
            scene.close()

        columns = np.arange(width)[None, :]
        self.assertGreater(before.sum(), 10)
        self.assertGreater(after.sum(), 10)
        self.assertLess(float((columns * after).sum() / after.sum()),
                        float((columns * before).sum() / before.sum()) - 3.0)


class TestBatchGpuBodies(unittest.TestCase):
    def test_body_moved_through_the_cuda_pose_buffer_moves_in_the_picture(self) -> None:
        """A PhysX GPU body rendered by a batch group follows its row of the CUDA pose buffer."""
        sapien.physx.enable_gpu()
        device = sapien.Device("cuda")
        physx = sapien.physx.PhysxGpuSystem(device)
        render_system = sapien.render.RenderSystem(device)
        scene = sapien.Scene([physx, render_system])
        scene.set_ambient_light([1.0, 1.0, 1.0])
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.3] * 3)
        builder.add_box_visual(half_size=[0.3] * 3,
                               material=sapien.render.RenderMaterial(
                                   base_color=[0.9, 0.1, 0.1, 1.0]))
        builder.initial_pose = sapien.Pose([0.0, 0.0, 0.0])
        actor = builder.build()
        body = actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.05, 20.0,
                                  render_config=batch_config())
        camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])

        physx.gpu_init()
        group = sapien.render.RenderSystemGroup([render_system])
        camera_group = group.create_camera_group([camera], ["Segmentation"])
        group.set_cuda_poses(physx.cuda_rigid_body_data)
        group.gpu_init()
        # the pose rows are the buffer's content, not the bodies' initial poses: write them
        poses = torch.from_dlpack(physx.cuda_rigid_body_data)

        def place(y: float) -> None:
            poses[body.gpu_pose_index, :7] = torch.tensor(
                [0.0, y, 0.0, 1.0, 0.0, 0.0, 0.0], device=poses.device)
            torch.cuda.synchronize(poses.device)

        try:
            place(0.0)
            group.update_render()
            camera_group.take_picture()
            before = picture(camera_group, "Segmentation")[0, ..., 1] > 0

            # +y is the camera's left: the box must move towards column 0
            place(1.0)
            group.update_render()
            camera_group.take_picture()
            after = picture(camera_group, "Segmentation")[0, ..., 1] > 0
        finally:
            del poses
            group.close()
            scene.close()

        self.assertGreater(np.count_nonzero(before), 20)
        self.assertGreater(np.count_nonzero(after), 20)
        columns = np.arange(SIZE)[None, :]
        before_column = float((columns * before).sum() / before.sum())
        after_column = float((columns * after).sum() / after.sum())
        self.assertLess(after_column, before_column - 5.0)


class TestBatchLighting(unittest.TestCase):
    def test_light_colour_change_after_gpu_init_changes_color(self) -> None:
        scene = make_scene(ambient=0.0)
        add_box(scene, sapien.Pose([0.0, 0.0, 0.0]), [1.0, 1.0, 1.0])
        light = scene.add_directional_light([0.2, 0.4, -1.0], [0.5, 0.5, 0.5], shadow=False)
        camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.05, 20.0,
                                  render_config=sapien.render.RenderConfig("rt_batch"))
        camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])

        group, camera_group = render_group([camera], ["Color"])
        try:
            group.update_render()
            camera_group.take_picture()
            dim = picture(camera_group, "Color")[0, ..., :3].astype(np.float32).copy()

            light.set_color([3.0, 3.0, 3.0])
            group.update_render()
            camera_group.take_picture()
            bright = picture(camera_group, "Color")[0, ..., :3].astype(np.float32)
            # 6x the radiance is about 2.26x after the default gamma-2.2 tone mapping
            self.assertGreater(dim.mean(), 1.0)
            self.assertGreater(bright.mean(), 1.8 * dim.mean())
        finally:
            group.close()
            scene.close()

    def test_shadows_darken_the_ground_under_an_occluder(self) -> None:
        scene = make_scene(ambient=0.1)
        scene.add_ground(0.0)
        # a box above the ground and a light straight down: its shadow is the square below it
        add_box(scene, sapien.Pose([0.0, 0.0, 0.6]), [0.9, 0.9, 0.9], half_size=0.2)
        scene.add_directional_light([0.0, 0.0, -1.0], [3.0, 3.0, 3.0], shadow=False)
        # looking at the origin from 31 degrees above the horizon
        pose = sapien.Pose([-2.5, 0.0, 1.5], [0.9637, 0.0, 0.2669, 0.0])

        def color(config: "sapien.render.RenderConfig", name: str) -> np.ndarray:
            camera = scene.add_camera(name, SIZE, SIZE, 1.0, 0.05, 20.0, render_config=config)
            camera.entity.pose = pose
            group, camera_group = render_group([camera], ["Color"])
            try:
                group.update_render()
                camera_group.take_picture()
                return picture(camera_group, "Color")[0, ..., :3].astype(np.float32)
            finally:
                group.close()

        try:
            unshadowed = color(sapien.render.RenderConfig("rt_batch"), "unshadowed")
            shadowed_config = sapien.render.RenderConfig("rt_batch")
            shadowed_config["shadows"] = True
            shadowed = color(shadowed_config, "shadowed")
        finally:
            scene.close()

        darker = (unshadowed - shadowed).max(axis=-1) > 30.0
        self.assertGreater(np.count_nonzero(darker), 20, "the shadow must be visible")
        self.assertLessEqual(float((shadowed - unshadowed).max()), 1.0,
                             "a shadow ray only removes light")


class TestBatchFormats(unittest.TestCase):
    def test_default_color_is_uint8_and_float_is_available(self) -> None:
        scene = make_scene(ambient=1.0)
        add_box(scene, sapien.Pose([0.0, 0.0, 0.0]), [0.8, 0.3, 0.1])

        default_camera = scene.add_camera("default", SIZE, SIZE, 1.0, 0.05, 20.0,
                                         render_config=sapien.render.RenderConfig("rt_batch"))
        default_camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])

        float_config = sapien.render.RenderConfig("rt_batch")
        float_config.picture_formats = {"Color": "r32g32b32a32Sfloat"}
        float_camera = scene.add_camera("float", SIZE, SIZE, 1.0, 0.05, 20.0,
                                        render_config=float_config)
        float_camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])

        group = sapien.render.RenderSystemGroup([scene.render_system])
        default_group = group.create_camera_group([default_camera], ["Color"])
        float_group = group.create_camera_group([float_camera], ["Color"])
        group.gpu_init()
        try:
            group.update_render()
            default_group.take_picture()
            float_group.take_picture()
            default_picture = picture(default_group, "Color")
            float_picture = picture(float_group, "Color")
        finally:
            group.close()
            scene.close()

        self.assertGreater(int(default_picture[..., :3].max()), 50, "the box must be lit")
        self.assertEqual(default_picture.dtype, np.uint8)
        self.assertEqual(float_picture.dtype, np.float32)
        self.assertEqual(default_picture.shape, (1, SIZE, SIZE, 4))

        # the same image after quantization
        expected = np.round(np.clip(float_picture, 0.0, 1.0) * 255.0)
        self.assertLessEqual(
            float(np.abs(default_picture.astype(np.float32) - expected).max()), 1.0)


class TestBatchMisuse(unittest.TestCase):
    def test_mismatched_configs_and_sizes_raise(self) -> None:
        scene = make_scene(ambient=1.0)
        first = scene.add_camera("first", SIZE, SIZE, 1.0, 0.05, 20.0,
                                 render_config=batch_config())
        second = scene.add_camera("second", SIZE + 8, SIZE, 1.0, 0.05, 20.0,
                                  render_config=batch_config())
        third = scene.add_camera("third", SIZE, SIZE, 1.0, 0.05, 20.0,
                                 render_config=sapien.render.RenderConfig("rt_batch"))
        with sapien.render.RenderSystemGroup([scene.render_system]) as group:
            with self.assertRaisesRegex(RuntimeError, "same width and height"):
                group.create_camera_group([first, second], ["Color"])
            with self.assertRaisesRegex(RuntimeError, "equal render config"):
                group.create_camera_group([first, third], ["Color"])
        scene.close()

    def test_mixing_kinds_raises(self) -> None:
        scene = make_scene(ambient=1.0)
        batch_camera = scene.add_camera("batch", SIZE, SIZE, 1.0, 0.05, 20.0,
                                        render_config=batch_config())
        raster_camera = scene.add_camera("raster", SIZE, SIZE, 1.0, 0.05, 20.0)
        with sapien.render.RenderSystemGroup([scene.render_system]) as group:
            with self.assertRaisesRegex(RuntimeError, "equal render config"):
                group.create_camera_group([batch_camera, raster_camera], ["Color"])
        scene.close()

    def test_per_camera_paths_and_unknown_property_raise(self) -> None:
        scene = make_scene(ambient=1.0)
        add_box(scene, sapien.Pose([0.0, 0.0, 0.0]), [0.8, 0.2, 0.2])
        camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.05, 20.0,
                                  render_config=batch_config())
        camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])

        # a batched pack has no per-camera renderer and no per-camera properties
        with self.assertRaisesRegex(RuntimeError, "batched ray tracing"):
            camera.take_picture()
        with self.assertRaisesRegex(RuntimeError, "batched ray tracing"):
            camera.get_picture("Color")
        with self.assertRaisesRegex(RuntimeError, "batched ray tracing"):
            camera.set_property("exposure", 2.0)

        # an undeclared parameter is rejected when it is assigned, before any group exists
        with self.assertRaisesRegex(RuntimeError, 'rt_batch.*no parameter "not_declared"'):
            batch_config(not_declared=1)

        group, camera_group = render_group([camera], ["Color"])
        try:
            group.update_render()
            camera_group.take_picture()
            self.assertEqual(picture(camera_group, "Color").shape, (1, SIZE, SIZE, 4))
        finally:
            group.close()
            scene.close()


class TestBatchPointCloud(unittest.TestCase):
    def test_point_cloud_renders(self) -> None:
        scene = make_scene(ambient=1.0)
        cloud = sapien.render.RenderPointCloudComponent(1)
        cloud.set_vertices(np.array([[0.0, 0.0, 0.0]], np.float32))
        cloud.set_attribute("color", np.array([[1.0, 0.0, 0.0, 1.0]], np.float32))
        cloud.set_attribute("scale", np.array([0.3], np.float32))
        entity = sapien.Entity()
        entity.add_component(cloud)
        scene.add_entity(entity)

        camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.05, 20.0,
                                  render_config=batch_config())
        camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
        group, camera_group = render_group([camera], ["Albedo", "Segmentation"])
        try:
            group.update_render()
            camera_group.take_picture()
            albedo = picture(camera_group, "Albedo")[0]
        finally:
            group.close()
            scene.close()

        center = albedo[SIZE // 2, SIZE // 2]
        self.assertGreater(float(center[0]), float(center[1]) + 0.3)


if __name__ == "__main__":
    unittest.main()
