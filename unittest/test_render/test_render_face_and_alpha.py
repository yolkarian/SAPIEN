"""Regression tests for mirrored instances, two-sided shadow casting, alpha modes, the catcher's
geometric AOVs, the CPU ray tracer's motion history, and the batched renderer's picture sizing."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import sapien

try:
    import torch
except ImportError:  # pragma: no cover - grouped rendering needs CUDA-enabled Torch
    torch = None

SIZE = 96

# a closed cube of half size 0.4, wound counter-clockwise when seen from outside
CUBE = """v -0.4 -0.4 -0.4
v 0.4 -0.4 -0.4
v 0.4 0.4 -0.4
v -0.4 0.4 -0.4
v -0.4 -0.4 0.4
v 0.4 -0.4 0.4
v 0.4 0.4 0.4
v -0.4 0.4 0.4
f 1 4 3
f 1 3 2
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
"""

# one open quad facing +z
QUAD = """v -0.5 -0.5 0
v 0.5 -0.5 0
v 0.5 0.5 0
v -0.5 0.5 0
f 1 2 3
f 1 3 4
"""


def write_mesh(directory: str, name: str, content: str) -> Path:
    path = Path(directory) / name
    path.write_text(content)
    return path


def rt_config(pack: str = "rt", spp: int = 16) -> "sapien.render.RenderConfig":
    config = sapien.render.RenderConfig(pack)
    config["spp"] = spp
    if "maxDepth" in config:
        config["maxDepth"] = 4
    config.denoiser = "none"
    return config


class TestMirroredInstances(unittest.TestCase):
    """A cube is symmetric under x -> -x: a negative x scale must render exactly like the plain one.

    Vulkan decides front/back facing in object space, so a mirroring instance transform does not
    flip the winding; an inside test that XORs a negative determinant renders the cube's interior.
    """

    def _mean_colour(self, pack: str, scale: list[float], path: Path) -> np.ndarray:
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.3, 0.3, 0.3])
            scene.add_directional_light([1.0, 0.5, -1.0], [2.0, 2.0, 2.0])
            if pack == "shadow_catcher":
                scene.add_ground(-0.6)  # the first render shape is the catcher
            builder = scene.create_actor_builder()
            builder.add_visual_from_file(
                str(path),
                scale=scale,
                material=sapien.render.RenderMaterial(
                    base_color=[0.8, 0.3, 0.2, 1.0], roughness=0.6
                ),
            )
            box = builder.build_kinematic()
            box.set_pose(sapien.Pose([0.0, 0.0, 0.0], [0.92388, 0.0, 0.0, 0.38268]))
            camera = scene.add_camera(
                "camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=rt_config(pack)
            )
            camera.entity.pose = sapien.Pose([-3.0, 0.0, 1.0], [0.976, 0.0, 0.2164, 0.0])
            scene.update_render()
            camera.take_picture()
            picture = camera.get_picture("HdrColor" if pack != "shadow_catcher" else "HdrColor")
            segmentation = camera.get_picture("Segmentation")[..., 1]
            box_id = box.per_scene_id
        mask = segmentation == box_id
        self.assertGreater(int(mask.sum()), 50, "the cube must be visible")
        return picture[mask, :3].mean(axis=0)

    def test_mirrored_cube_renders_like_the_plain_one_in_ray_tracing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_mesh(directory, "cube.obj", CUBE)
            for pack in ("rt", "shadow_catcher"):
                with self.subTest(pack=pack):
                    plain = self._mean_colour(pack, [1.0, 1.0, 1.0], path)
                    mirrored = self._mean_colour(pack, [-1.0, 1.0, 1.0], path)
                    self.assertGreater(float(plain.max()), 0.05)
                    np.testing.assert_allclose(mirrored, plain, rtol=0.1, atol=0.02)


class TestBatchedMirroredInstances(unittest.TestCase):
    def setUp(self) -> None:
        if torch is None or not torch.cuda.is_available():
            self.skipTest("batched pictures are CUDA buffers: requires CUDA-enabled Torch")

    def test_mirrored_cube_renders_like_the_plain_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_mesh(directory, "cube.obj", CUBE)

            def shade(scale: list[float]) -> np.ndarray:
                scene = sapien.Scene(
                    [sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem()]
                )
                scene.set_ambient_light([0.3, 0.3, 0.3])
                scene.add_directional_light([1.0, 0.5, -1.0], [2.0, 2.0, 2.0])
                builder = scene.create_actor_builder()
                builder.add_visual_from_file(
                    str(path),
                    scale=scale,
                    material=sapien.render.RenderMaterial(
                        base_color=[0.8, 0.3, 0.2, 1.0], roughness=0.6
                    ),
                )
                builder.build_kinematic().set_pose(
                    sapien.Pose([0.0, 0.0, 0.0], [0.92388, 0.0, 0.0, 0.38268])
                )
                camera = scene.add_camera(
                    "camera", SIZE, SIZE, 1.0, 0.1, 20.0,
                    render_config=sapien.render.RenderConfig("rt_batch", preset="pbr"),
                )
                camera.entity.pose = sapien.Pose([-3.0, 0.0, 1.0], [0.976, 0.0, 0.2164, 0.0])
                group = sapien.render.RenderSystemGroup([scene.render_system])
                camera_group = group.create_camera_group([camera], ["Color", "Segmentation"])
                group.gpu_init()
                group.update_render()
                camera_group.take_picture()
                color = torch.from_dlpack(camera_group.get_picture_cuda("Color"))
                segmentation = torch.from_dlpack(camera_group.get_picture_cuda("Segmentation"))
                color = color.cpu().numpy()[0][..., :3].astype(np.float32) / 255.0
                mask = segmentation.cpu().numpy()[0][..., 1] > 0
                del segmentation
                group.close()
                scene.close()
                self.assertGreater(int(mask.sum()), 50, "the cube must be visible")
                return color[mask].mean(axis=0)

            plain = shade([1.0, 1.0, 1.0])
            mirrored = shade([-1.0, 1.0, 1.0])
        self.assertGreater(float(plain.max()), 0.05)
        np.testing.assert_allclose(mirrored, plain, rtol=0.1, atol=0.03)


class TestOpenSheetShadows(unittest.TestCase):
    """An opaque open sheet still occludes what lies behind it, in `rt` and in raster."""

    def _floor_brightness(self, pack: str, path: Path, with_sheet: bool) -> float:
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.0, 0.0, 0.0])
            scene.add_directional_light([0.0, 0.0, -1.0], [3.0, 3.0, 3.0], shadow=True)
            floor_builder = scene.create_actor_builder()
            floor_builder.add_box_visual(
                half_size=[3.0, 3.0, 0.05],
                material=sapien.render.RenderMaterial(
                    base_color=[1, 1, 1, 1], roughness=1.0, specular_weight=0.0
                ),
            )
            floor_builder.build_kinematic().set_pose(sapien.Pose([0.0, 0.0, -0.05]))
            if with_sheet:
                sheet_builder = scene.create_actor_builder()
                sheet_builder.add_visual_from_file(
                    str(path),
                    material=sapien.render.RenderMaterial(base_color=[0.5, 0.5, 0.5, 1]),
                )
                sheet_builder.build_kinematic().set_pose(sapien.Pose([0.0, 0.0, 0.5]))
            config = rt_config(pack) if pack != "default" else None
            camera = scene.add_camera(
                "camera", SIZE, SIZE, 1.2, 0.05, 20.0, render_config=config
            )
            camera.entity.pose = sapien.Pose([-1.5, 0.0, 0.25], [0.9961947, 0.0, 0.0871557, 0.0])
            scene.update_render()
            camera.take_picture()
            picture = camera.get_picture("Color" if pack == "default" else "HdrColor")
        patch = picture[SIZE // 2 - 4 : SIZE // 2 + 4, SIZE // 2 - 6 : SIZE // 2 + 6, :3]
        return float(patch.mean())

    def test_open_sheet_shadows_the_floor_in_ray_tracing(self) -> None:
        """`rt` blocks a shadow ray on an opaque back face, so an open sheet casts a shadow. The
        rasterizer's shadow pass culls front faces and therefore never records an open sheet as a
        caster; it is not the reference here."""
        with tempfile.TemporaryDirectory() as directory:
            path = write_mesh(directory, "quad.obj", QUAD)
            unshadowed = self._floor_brightness("rt", path, False)
            shadowed = self._floor_brightness("rt", path, True)
            self.assertGreater(unshadowed, 0.3)
            self.assertLess(shadowed, 0.6 * unshadowed)


class TestAlphaModes(unittest.TestCase):
    """base_color.a follows the material's alpha mode, and a texture's alpha cannot re-open an
    OPAQUE surface."""

    def _coverage(self, alpha_mode: str, base_alpha: float, texture_alpha: float) -> float:
        with sapien.Scene() as scene:
            scene.set_ambient_light([1.0, 1.0, 1.0])
            material = sapien.render.RenderMaterial(base_color=[0.8, 0.3, 0.2, base_alpha])
            material.alpha_mode = alpha_mode
            if texture_alpha < 1.0:
                pixels = np.zeros((4, 4, 4), np.uint8)
                pixels[..., :3] = 200
                pixels[..., 3] = int(round(texture_alpha * 255))
                material.base_color_texture = sapien.render.RenderTexture2D(
                    pixels, "R8G8B8A8Unorm"
                )
            builder = scene.create_actor_builder()
            builder.add_box_visual(half_size=[0.4] * 3, material=material)
            builder.build_kinematic()
            camera = scene.add_camera(
                "camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=rt_config("rt", spp=4)
            )
            camera.entity.pose = sapien.Pose([-1.2, 0.0, 0.0])
            scene.update_render()
            camera.take_picture()
            segmentation = camera.get_picture("Segmentation")[..., 1]
        visible = int((segmentation > 0).sum())
        return visible / float(SIZE * SIZE)

    def test_opaque_ignores_the_base_colour_alpha(self) -> None:
        opaque = self._coverage("opaque", 1.0, 1.0)
        self.assertGreater(opaque, 0.1)
        self.assertAlmostEqual(self._coverage("opaque", 0.2, 1.0), opaque, delta=0.01)

    def test_opaque_ignores_the_base_colour_texture_alpha(self) -> None:
        opaque = self._coverage("opaque", 1.0, 1.0)
        self.assertAlmostEqual(self._coverage("opaque", 1.0, 0.0), opaque, delta=0.01)

    def test_mask_cuts_out_below_the_cutoff(self) -> None:
        self.assertGreater(self._coverage("mask", 1.0, 1.0), 0.1)
        self.assertLess(self._coverage("mask", 0.2, 1.0), 0.01)
        self.assertLess(self._coverage("mask", 1.0, 0.0), 0.01)

    def test_blend_keeps_the_alpha_as_coverage(self) -> None:
        self.assertGreater(self._coverage("blend", 1.0, 1.0), 0.1)
        self.assertLess(self._coverage("blend", 0.0, 1.0), 0.01)


class TestShadowCatcherGeometricAovs(unittest.TestCase):
    def test_unshadowed_catcher_pixels_report_the_geometry(self) -> None:
        """The catcher writes zero alpha where nothing shadows it; the geometric AOVs must still
        describe its surface instead of staying zero."""
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.3, 0.3, 0.3])
            scene.add_directional_light([0.3, 0.2, -1.0], [2.0, 2.0, 2.0])
            ground = scene.add_ground(0.0)
            builder = scene.create_actor_builder()
            builder.add_box_visual(half_size=[0.2] * 3)
            box = builder.build_kinematic()
            box.set_pose(sapien.Pose([0.0, 0.0, 0.2]))
            camera = scene.add_camera(
                "camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=rt_config("shadow_catcher")
            )
            camera.entity.pose = sapien.Pose([0.0, 0.0, 3.0], [0.7071068, 0.0, 0.7071068, 0.0])
            scene.update_render()
            camera.take_picture()
            segmentation = camera.get_picture("Segmentation")[..., 1]
            position = camera.get_picture("Position")
            ground_id = ground.per_scene_id
        ground_pixels = segmentation == ground_id
        # the camera sees only the ground and the box: most pixels are ground, and a catcher pixel
        # outside a shadow must report the ground too (it used to report the empty id 0)
        self.assertGreater(int(ground_pixels.sum()), 0.7 * SIZE * SIZE)
        self.assertGreater(float(np.abs(position[ground_pixels, 2]).min()), 0.5)


class TestCpuRayTracingMotionHistory(unittest.TestCase):
    def test_an_unrendered_move_does_not_report_motion_after_an_idle_frame(self) -> None:
        """MotionVector is measured per scene render version: after a move that was not rendered
        and an idle update_render(), the box reports no motion."""
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.3, 0.3, 0.3])
            builder = scene.create_actor_builder()
            builder.add_box_visual(half_size=[0.4] * 3)
            box = builder.build_kinematic()
            box.set_pose(sapien.Pose([0.0, 0.0, 0.0]))
            camera = scene.add_camera(
                "camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=rt_config("rt", spp=1)
            )
            camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
            scene.update_render()
            camera.take_picture()

            box.set_pose(sapien.Pose([0.0, -0.3, -0.2]))
            scene.update_render()  # the move, no picture
            scene.update_render()  # idle
            camera.take_picture()
            motion = camera.get_picture("MotionVector")
            segmentation = camera.get_picture("Segmentation")[..., 1]
        mask = segmentation > 0
        self.assertGreater(int(mask.sum()), 50)
        np.testing.assert_allclose(motion[mask], 0.0, atol=1e-3)


class TestBatchedPictureSizing(unittest.TestCase):
    def setUp(self) -> None:
        if torch is None or not torch.cuda.is_available():
            self.skipTest("batched pictures are CUDA buffers: requires CUDA-enabled Torch")

    def test_an_eight_bit_picture_does_not_shrink_the_float_pictures(self) -> None:
        """Every picture of a group used to get the byte size of the last one, so
        ["Albedo", "Color"] allocated a quarter of the float32 Albedo."""
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem()])
        scene.set_ambient_light([0.5, 0.5, 0.5])
        builder = scene.create_actor_builder()
        builder.add_box_visual(
            half_size=[3.0, 3.0, 3.0],
            material=sapien.render.RenderMaterial(base_color=[0.7, 0.3, 0.2, 1.0]),
        )
        builder.build_kinematic().set_pose(sapien.Pose([5.0, 0.0, 0.0]))
        camera = scene.add_camera(
            "camera", SIZE, SIZE, 1.0, 0.1, 20.0,
            render_config=sapien.render.RenderConfig("rt_batch"),
        )
        group = sapien.render.RenderSystemGroup([scene.render_system])
        camera_group = group.create_camera_group([camera], ["Albedo", "Color"])
        try:
            group.gpu_init()
            group.update_render()
            camera_group.take_picture()
            albedo = torch.from_dlpack(camera_group.get_picture_cuda("Albedo"))
            values = albedo.cpu().numpy()[0][..., :3]
            del albedo
        finally:
            group.close()
            scene.close()
        # the base colour of the 3 m box, at full precision: the copy itself fails with a CUDA
        # "Copy is larger than memobj size" when the buffer is under-allocated
        self.assertAlmostEqual(float(values[..., 0].mean()), 0.7, delta=0.02)
        self.assertAlmostEqual(float(values[..., 1].mean()), 0.3, delta=0.02)
        self.assertAlmostEqual(float(values[..., 2].mean()), 0.2, delta=0.02)


class TestCastShadowAssignment(unittest.TestCase):
    def setUp(self) -> None:
        if torch is None or not torch.cuda.is_available():
            self.skipTest("grouped rendering requires a CUDA device and CUDA-enabled Torch")

    def test_assigning_the_current_value_keeps_the_group_usable(self) -> None:
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem()])
        scene.set_ambient_light([0.3, 0.3, 0.3])
        builder = scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.4] * 3)
        body_actor = builder.build_kinematic()
        body = body_actor.find_component_by_type(sapien.render.RenderBodyComponent)
        camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0,
                                  render_config=sapien.render.RenderConfig("rt_batch"))
        camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
        group = sapien.render.RenderSystemGroup([scene.render_system])
        camera_group = group.create_camera_group([camera], ["Color"])
        try:
            group.gpu_init()
            body.cast_shadow = body.cast_shadow
            group.update_render()
            camera_group.take_picture()
        finally:
            group.close()
            scene.close()


if __name__ == "__main__":
    unittest.main()
