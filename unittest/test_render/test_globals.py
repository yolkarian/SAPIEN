"""RenderConfig, its process-wide defaults, and the per-consumer copy semantics."""

import unittest
from pathlib import Path

import numpy as np
import sapien


class TestGlobals(unittest.TestCase):
    def test_device_summary(self) -> None:
        summary = sapien.render.get_device_summary()
        self.assertIsInstance(summary, str)

    def test_imgui_ini_filename(self) -> None:
        self.assertEqual(
            sapien.render.get_imgui_ini_filename(),
            str(Path.home() / ".sapien" / "imgui.ini"),
        )

    def test_render_config_repr_and_eq(self) -> None:
        config = sapien.render.RenderConfig("default")
        self.assertTrue(config.shader_pack.endswith("default"))
        self.assertFalse(config.is_ray_tracing)
        self.assertEqual(config, sapien.render.RenderConfig("default"))
        self.assertNotEqual(config, sapien.render.RenderConfig("rt"))
        self.assertIn("RenderConfig", repr(config))

        rt = sapien.render.RenderConfig("rt")
        self.assertTrue(rt.is_ray_tracing)

    def test_render_config_properties(self) -> None:
        config = sapien.render.RenderConfig("rt")
        config["spp"] = 7
        config["aperture"] = 0.02
        self.assertEqual(config["spp"], 7)
        self.assertAlmostEqual(config["aperture"], 0.02)
        self.assertIn("spp", config)
        self.assertNotIn("maxDepth", config)
        # an unassigned parameter reads its pack.json default
        self.assertEqual(config["maxDepth"], 8)

        properties = config.properties
        # values live in the renderer as float32, so a float reads back with float32 precision
        self.assertEqual(properties["spp"], 7)
        self.assertAlmostEqual(properties["aperture"], 0.02, places=6)
        properties["spp"] = 99  # a copy
        self.assertEqual(config["spp"], 7)

        del config["spp"]
        self.assertNotIn("spp", config)
        self.assertEqual(config["spp"], 32)

    def test_render_config_parameters_and_presets(self) -> None:
        config = sapien.render.RenderConfig("rt_batch")
        self.assertEqual(
            list(config.parameters),
            ["exposure", "toneMapper", "spp", "shading", "shadows", "maxShadowLights"],
        )
        shading = config.parameters["shading"]
        self.assertEqual(shading["kind"], "variant")
        self.assertEqual(shading["type"], "enum")
        self.assertEqual(shading["default"], "lambert")
        self.assertEqual(shading["values"], ["visibility", "lambert", "pbr"])
        self.assertEqual(config.presets, ["visibility", "lambert", "pbr"])

    def test_render_config_rejects_non_numeric_property(self) -> None:
        config = sapien.render.RenderConfig("rt")
        with self.assertRaisesRegex(RuntimeError, "spp"):
            config["spp"] = "many"

    def test_render_config_copy_is_independent(self) -> None:
        config = sapien.render.RenderConfig("rt")
        config["spp"] = 3
        copy = config.copy()
        self.assertEqual(copy, config)
        copy["spp"] = 11
        self.assertEqual(config["spp"], 3)
        self.assertNotEqual(copy, config)

    def test_invalid_shader_pack_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no_such_pack"):
            sapien.render.RenderConfig("no_such_pack")
        with self.assertRaises(RuntimeError):
            sapien.render.RenderConfig(str(Path(sapien.__file__).parent / "example"))

    def test_invalid_pack_assignment_keeps_the_previous_pack(self) -> None:
        config = sapien.render.RenderConfig("default")
        with self.assertRaisesRegex(RuntimeError, "no_such_pack"):
            config.shader_pack = "no_such_pack"
        self.assertTrue(config.shader_pack.endswith("default"))

    def test_default_render_config_round_trip(self) -> None:
        original_camera = sapien.render.get_default_render_config("camera")
        original_viewer = sapien.render.get_default_render_config("viewer")
        self.addCleanup(
            sapien.render.set_default_render_config, "camera", original_camera
        )
        self.addCleanup(
            sapien.render.set_default_render_config, "viewer", original_viewer
        )
        try:
            self.assertTrue(original_camera.shader_pack.endswith("default"))
            self.assertTrue(original_viewer.shader_pack.endswith("default"))

            config = sapien.render.RenderConfig("rt")
            config["spp"] = 5
            sapien.render.set_default_render_config("camera", config)
            sapien.render.set_default_render_config("viewer", config)

            # get returns a copy the caller may edit without touching the stored default
            camera_default = sapien.render.get_default_render_config("camera")
            self.assertEqual(camera_default["spp"], 5)
            self.assertTrue(camera_default.is_ray_tracing)
            camera_default["spp"] = 100
            self.assertEqual(
                sapien.render.get_default_render_config("camera")["spp"], 5
            )
            self.assertEqual(
                sapien.render.get_default_render_config("viewer")["spp"], 5
            )

            with self.assertRaisesRegex(RuntimeError, "target"):
                sapien.render.get_default_render_config("invalid")
            with self.assertRaisesRegex(RuntimeError, "target"):
                sapien.render.set_default_render_config("invalid", config)
        finally:
            sapien.render.set_default_render_config("camera", original_camera)
            sapien.render.set_default_render_config("viewer", original_viewer)

    def test_camera_copies_the_default_at_construction(self) -> None:
        original = sapien.render.get_default_render_config("camera")
        self.addCleanup(sapien.render.set_default_render_config, "camera", original)
        try:
            config = sapien.render.RenderConfig("rt")
            config["spp"] = 9
            sapien.render.set_default_render_config("camera", config)
            with sapien.Scene() as scene:
                camera = scene.add_camera("camera", 32, 32, 1.0, 0.1, 10.0)
                # later edits to the config and the stored default do not reach the camera
                config["spp"] = 3
                sapien.render.set_default_render_config(
                    "camera", sapien.render.RenderConfig("default")
                )
                copied = camera.render_config
                self.assertTrue(copied.is_ray_tracing)
                self.assertEqual(copied["spp"], 9)

                # render_config is a copy: editing it does not change the camera
                copied["spp"] = 40
                self.assertEqual(camera.render_config["spp"], 9)
        finally:
            sapien.render.set_default_render_config("camera", original)

    def test_explicit_config_wins_over_the_default(self) -> None:
        with sapien.Scene() as scene:
            config = sapien.render.RenderConfig("rt")
            config["spp"] = 2
            camera = scene.add_camera(
                "camera", 32, 32, 1.0, 0.1, 10.0, render_config=config
            )
            self.assertTrue(camera.render_config.is_ray_tracing)
            self.assertEqual(camera.render_config["spp"], 2)
