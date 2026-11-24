"""Per-camera RenderConfig: independent configs in one process, validation, formats, defines."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import sapien

SIZE = 48


def build_box_scene(scene: sapien.Scene) -> None:
    """A red box under a white directional light, seen from a camera at -x looking at origin."""
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([1.0, 0.0, 0.0], [2.0, 2.0, 2.0])
    builder = scene.create_actor_builder()
    builder.add_box_visual(
        half_size=[0.5, 0.5, 0.5],
        material=sapien.render.RenderMaterial(base_color=[1.0, 0.0, 0.0, 1.0]),
    )
    builder.build_kinematic(name="box")


class TestPerCameraConfig(unittest.TestCase):
    def test_raster_and_ray_tracing_cameras_in_one_process(self) -> None:
        """One rasterization and one ray-tracing camera in the same scene render with their own
        pack; the process-wide defaults are never touched."""
        with sapien.Scene() as scene:
            build_box_scene(scene)
            raster_config = sapien.render.RenderConfig("default")
            raster_config["toneMapper"] = 1  # plain sRGB, so a red box reads exactly red
            raster = scene.add_camera(
                "raster", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=raster_config
            )
            rt_config = sapien.render.RenderConfig("rt")
            rt_config["spp"] = 1
            rt_config["toneMapper"] = 1
            rt_config.denoiser = "none"
            rt = scene.add_camera(
                "rt", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=rt_config
            )
            for camera in (raster, rt):
                camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])

            self.assertFalse(raster.render_config.is_ray_tracing)
            self.assertTrue(rt.render_config.is_ray_tracing)
            # the default camera config still renders rasterization
            self.assertFalse(
                sapien.render.get_default_render_config("camera").is_ray_tracing
            )

            scene.update_render()
            raster.take_picture()
            rt.take_picture()
            for name, camera in (("raster", raster), ("rt", rt)):
                with self.subTest(camera=name):
                    center = camera.get_picture("Color")[SIZE // 2, SIZE // 2, :3]
                    self.assertGreater(float(center[0]), float(center[1]) + 0.5)
                    self.assertGreater(float(center[0]), 0.5)

    def test_unknown_property_on_config_raises_when_the_camera_is_created(self) -> None:
        config = sapien.render.RenderConfig("rt")
        config["notAProperty"] = 1
        with sapien.Scene() as scene:
            with self.assertRaisesRegex(RuntimeError, "notAProperty.*rt"):
                scene.add_camera(
                    "camera", 32, 32, 1.0, 0.1, 10.0, render_config=config
                )

    def test_wrong_property_type_on_config_raises(self) -> None:
        config = sapien.render.RenderConfig("rt")
        config["spp"] = 1.5  # spp is an int push-constant member
        with sapien.Scene() as scene:
            with self.assertRaisesRegex(RuntimeError, "spp.*rt"):
                scene.add_camera(
                    "camera", 32, 32, 1.0, 0.1, 10.0, render_config=config
                )

        config = sapien.render.RenderConfig("default")
        config["exposure"] = 3  # exposure is a float specialization constant
        with sapien.Scene() as scene:
            with self.assertRaisesRegex(RuntimeError, "exposure.*default"):
                scene.add_camera(
                    "camera", 32, 32, 1.0, 0.1, 10.0, render_config=config
                )

    def test_denoiser_on_raster_pack_raises(self) -> None:
        config = sapien.render.RenderConfig("default")
        config.denoiser = "oidn"
        with sapien.Scene() as scene:
            with self.assertRaisesRegex(RuntimeError, "denoiser"):
                scene.add_camera(
                    "camera", 32, 32, 1.0, 0.1, 10.0, render_config=config
                )

    def test_set_property_validates_once_the_camera_has_a_renderer(self) -> None:
        with sapien.Scene() as scene:
            build_box_scene(scene)
            camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0)
            camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
            scene.update_render()
            with self.assertRaisesRegex(RuntimeError, "notAProperty.*default"):
                camera.set_property("notAProperty", 1)
            with self.assertRaisesRegex(RuntimeError, "shadeFlat"):
                camera.set_property("shadeFlat", 0.5)

    def test_set_property_before_the_renderer_validates_at_attach(self) -> None:
        """A property set before the camera is added is validated when its renderer appears."""
        config = sapien.render.RenderConfig("rt")
        config["spp"] = 1
        camera = sapien.render.RenderCameraComponent(SIZE, SIZE, config)
        camera.set_property("spp", 2)  # not validated yet: no renderer
        self.assertEqual(camera.get_property("spp"), 2)
        camera.set_property("notAProperty", 3)
        entity = sapien.Entity()
        entity.add_component(camera)
        with sapien.Scene() as scene:
            with self.assertRaisesRegex(RuntimeError, "notAProperty.*rt"):
                scene.add_entity(entity)
            self.assertNotIn(entity, scene.get_entities())

    def test_picture_formats_apply_to_one_camera_only(self) -> None:
        with sapien.Scene() as scene:
            build_box_scene(scene)
            uint8_config = sapien.render.RenderConfig("default")
            uint8_config["toneMapper"] = 1
            uint8_config.picture_formats = {"Color": "r8g8b8a8unorm"}
            uint8_camera = scene.add_camera(
                "uint8", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=uint8_config
            )
            float_config = sapien.render.RenderConfig("default")
            float_config["toneMapper"] = 1
            float_camera = scene.add_camera(
                "float", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=float_config
            )
            for camera in (uint8_camera, float_camera):
                camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])

            scene.update_render()
            uint8_camera.take_picture()
            float_camera.take_picture()
            uint8_color = uint8_camera.get_picture("Color")
            float_color = float_camera.get_picture("Color")
            self.assertEqual(uint8_color.dtype, np.uint8)
            self.assertEqual(float_color.dtype, np.float32)
            # both cameras shade the same way; only the target precision differs
            self.assertGreater(float(float_color[..., 0].max()), 0.5)
            self.assertAlmostEqual(
                int(uint8_color[..., 0].max()),
                round(float(float_color[..., 0].max()) * 255),
                delta=2,
            )

    def test_msaa(self) -> None:
        for msaa in (1, 2, 4, 8):
            config = sapien.render.RenderConfig("default")
            config.msaa = msaa
            self.assertEqual(config.msaa, msaa)
        with self.assertRaises(RuntimeError):
            config.msaa = 3


class TestShaderDefines(unittest.TestCase):
    """Uses a private copy of the self-contained `trivial` pack (it has no includes)."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        source = Path(sapien.__file__).parent / "vulkan_shader" / "trivial"
        self.pack = Path(self.directory.name) / "pack"
        shutil.copytree(source, self.pack)
        self.addCleanup(sapien.render.clear_cache, False, False, True)

    def _center_color(self, config: sapien.render.RenderConfig) -> np.ndarray:
        with sapien.Scene() as scene:
            camera = scene.add_camera(
                "camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=config
            )
            camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
            builder = scene.create_actor_builder()
            builder.add_box_visual(
                half_size=[0.5, 0.5, 0.5],
                material=sapien.render.RenderMaterial(base_color=[1.0, 0.0, 0.0, 1.0]),
            )
            builder.build_kinematic(name="box")
            scene.update_render()
            camera.take_picture()
            return camera.get_picture("Color")[SIZE // 2, SIZE // 2, :3].copy()

    def test_defines_reach_the_shader(self) -> None:
        shader = self.pack / "gbuffer.frag"
        text = shader.read_text()
        marker = "outColor = materialBuffer.baseColor;"
        self.assertIn(marker, text)
        shader.write_text(
            text.replace(
                marker,
                marker + "\n#ifdef BLUE_VARIANT\n  outColor = vec4(0, 0, 1, 1);\n#endif",
            )
        )

        plain = sapien.render.RenderConfig(str(self.pack))
        np.testing.assert_allclose(self._center_color(plain), [1.0, 0.0, 0.0], atol=1e-3)

        variant = sapien.render.RenderConfig(str(self.pack))
        variant.defines = {"BLUE_VARIANT": ""}
        np.testing.assert_allclose(self._center_color(variant), [0.0, 0.0, 1.0], atol=1e-3)

        # a define with a value is passed as `#define NAME value`
        value_variant = sapien.render.RenderConfig(str(self.pack))
        value_variant.defines = {"BLUE_VARIANT": "", "UNUSED": "1"}
        np.testing.assert_allclose(
            self._center_color(value_variant), [0.0, 0.0, 1.0], atol=1e-3
        )


class TestRayTracingDefaults(unittest.TestCase):
    def _render_brightness(
        self, config: sapien.render.RenderConfig, frames: int = 1
    ) -> list[float]:
        """Ambient light reaches the ground only after at least one bounce."""
        values = []
        with sapien.Scene() as scene:
            scene.add_ground(0.0)
            camera = scene.add_camera(
                "camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=config
            )
            camera.entity.pose = sapien.Pose([0.0, 0.0, 2.0], [0.7071068, 0.0, 0.7071068, 0.0])
            scene.set_ambient_light([0.5, 0.5, 0.5])
            for _ in range(frames):
                scene.update_render()
                camera.take_picture()
                values.append(float(camera.get_picture("HdrColor")[..., :3].mean()))
        return values

    def test_defaults_apply_when_keys_are_absent(self) -> None:
        config = sapien.render.RenderConfig("rt")
        config.denoiser = "none"
        values = self._render_brightness(config, frames=3)
        # with the default depth 8 the ambient light arrives at the first frame
        self.assertGreater(values[0], 1e-3)

    def test_explicit_depth_zero_blacks_out_the_indirect_light(self) -> None:
        config = sapien.render.RenderConfig("rt")
        config["maxDepth"] = 0
        config.denoiser = "none"
        values = self._render_brightness(config, frames=4)
        self.assertLess(values[-1], 1e-3)
