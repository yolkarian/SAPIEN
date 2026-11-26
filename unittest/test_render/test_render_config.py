"""Per-camera RenderConfig: independent configs in one process, pack.json validation, formats,
presets and the shader variants of pack.json."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import sapien

SIZE = 48

# the parameters every built-in pack declares, written out so a shader and its pack.json cannot
# drift apart unnoticed: name -> (kind, type, default, minimum, maximum)
BUILTIN_PACKS = {
    "default": {
        "exposure": ("uniform", "float", 1.0, 0.0, None),
        "toneMapper": ("uniform", "int", 2, 0.0, 5.0),
        "ambientOcclusionStrength": ("uniform", "float", 0.65, 0.0, None),
        "ambientOcclusionRadius": ("uniform", "float", 0.3, 0.0, None),
    },
    "vr_default": {
        "exposure": ("uniform", "float", 1.0, 0.0, None),
        "toneMapper": ("uniform", "int", 2, 0.0, 5.0),
        "ambientOcclusionStrength": ("uniform", "float", 0.65, 0.0, None),
        "ambientOcclusionRadius": ("uniform", "float", 0.3, 0.0, None),
    },
    "point": {"exposure": ("uniform", "float", 1.0, 0.0, None)},
    "minimal": {},
    "trivial": {},
    "vertex_color": {},
    "rt": {
        "spp": ("uniform", "int", 32, 1.0, None),
        "maxDepth": ("uniform", "int", 8, 0.0, None),
        "aperture": ("uniform", "float", 0.0, 0.0, None),
        "focusPlane": ("uniform", "float", 1.0, 0.0, None),
        "russianRoulette": ("uniform", "bool", True, None, None),
        "russianRouletteMinBounces": ("uniform", "int", 2, 0.0, None),
        "clampIndirect": ("uniform", "float", 0.0, 0.0, None),
        "lightSampling": ("variant", "enum", "mis", None, None),
        "exposure": ("uniform", "float", 1.0, 0.0, None),
        "toneMapper": ("uniform", "int", 2, 0.0, 5.0),
        "bloomIntensity": ("uniform", "float", 0.0, 0.0, None),
        "bloomThreshold": ("uniform", "float", 1.0, 0.0, None),
        "bloomRadius": ("uniform", "float", 4.0, 0.5, 32.0),
        "bloomKnee": ("uniform", "float", 0.5, 0.0, 1.0),
        "exposureEV": ("uniform", "float", 0.0, None, None),
        "whiteBalanceTemp": ("uniform", "float", 6500.0, 1500.0, 25000.0),
        "whiteBalanceTint": ("uniform", "float", 0.0, -1.0, 1.0),
        "contrast": ("uniform", "float", 1.0, 0.0, None),
        "saturation": ("uniform", "float", 1.0, 0.0, None),
        "cdlSlope": ("uniform", "float", 1.0, 0.0, None),
        "cdlOffset": ("uniform", "float", 0.0, None, None),
        "cdlPower": ("uniform", "float", 1.0, 0.01, None),
        "lensDistortionK1": ("uniform", "float", 0.0, None, None),
        "lensDistortionK2": ("uniform", "float", 0.0, None, None),
        "chromaticAberration": ("uniform", "float", 0.0, None, None),
        "vignetteStrength": ("uniform", "float", 0.0, 0.0, None),
        "vignetteSmoothness": ("uniform", "float", 0.5, 0.01, 2.0),
        "filmGrain": ("uniform", "float", 0.0, 0.0, None),
        "sensorReadNoise": ("uniform", "float", 0.0, 0.0, None),
        "sensorShotNoise": ("uniform", "float", 0.0, 0.0, None),
        "noiseSeed": ("uniform", "int", 0, None, None),
    },
    "shadow_catcher": {
        "spp": ("uniform", "int", 32, 1.0, None),
        "maxDepth": ("uniform", "int", 8, 0.0, None),
        "aperture": ("uniform", "float", 0.0, 0.0, None),
        "focusPlane": ("uniform", "float", 1.0, 0.0, None),
        "russianRoulette": ("uniform", "bool", True, None, None),
        "russianRouletteMinBounces": ("uniform", "int", 2, 0.0, None),
        "exposure": ("uniform", "float", 1.0, 0.0, None),
        "toneMapper": ("uniform", "int", 2, 0.0, 5.0),
    },
    "rt_batch": {
        "exposure": ("uniform", "float", 1.0, 0.0, None),
        "toneMapper": ("uniform", "int", 0, 0.0, 5.0),
        "spp": ("uniform", "int", 1, 1.0, 32.0),
        "shading": ("variant", "enum", "lambert", None, None),
        "shadows": ("variant", "bool", False, None, None),
        "maxShadowLights": ("variant", "int", 4, 1.0, 16.0),
    },
}

# a pack whose entry point and manifest make it usable as a rasterization camera
CAMERA_PACKS = ["default", "vr_default", "point", "minimal", "trivial", "vertex_color", "rt"]
# packs whose Color output is the shaded box; `trivial` and `minimal` have their own minimal
# display passes (base colour and a bare lighting sum), `point` displays only the point-cloud
# pass (black without points) and `vertex_color` only vertex colours
SHADING_PACKS = ["default", "vr_default", "rt"]
# packs no per-camera renderer exists for
NON_CAMERA_PACKS = ["rt_batch"]


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


def copy_trivial_pack(directory: str, manifest: dict | None = None) -> Path:
    """A writable copy of the `trivial` pack with `manifest` written as its pack.json;
    None keeps the original `{"parameters": {}}` manifest. `trivial` includes
    `../common/scene.glsl`, so that directory is staged alongside the pack."""
    shader_root = Path(sapien.__file__).parent / "vulkan_shader"
    pack = Path(directory) / "pack"
    shutil.copytree(shader_root / "trivial", pack)
    shutil.copytree(shader_root / "common", Path(directory) / "common")
    if manifest is not None:
        (pack / "pack.json").write_text(json.dumps(manifest))
    return pack


class TestPackParameters(unittest.TestCase):
    """`cfg.parameters` reports the pack.json of every built-in pack."""

    def test_builtin_packs_declare_the_expected_parameters(self) -> None:
        for pack, expected in BUILTIN_PACKS.items():
            with self.subTest(pack=pack):
                config = sapien.render.RenderConfig(pack)
                parameters = config.parameters
                self.assertEqual(list(parameters), list(expected))
                for name, (kind, type_, default, minimum, maximum) in expected.items():
                    declared = parameters[name]
                    self.assertEqual(declared["kind"], kind)
                    self.assertEqual(declared["type"], type_)
                    if type_ == "float":
                        # values live in the renderer as float32
                        self.assertAlmostEqual(declared["default"], default, places=6)
                    else:
                        self.assertEqual(declared["default"], default)
                    self.assertEqual(declared.get("minimum"), minimum)
                    self.assertEqual(declared.get("maximum"), maximum)
                    self.assertTrue(
                        declared["description"], f"{pack}.{name} needs a description"
                    )
                    # the effective default is what cfg[name] reports before any assignment
                    if type_ == "float":
                        self.assertAlmostEqual(config[name], default, places=6)
                    else:
                        self.assertEqual(config[name], default)
                    self.assertNotIn(name, config)

    def test_rt_batch_declares_its_parameters_and_presets(self) -> None:
        config = sapien.render.RenderConfig("rt_batch")
        self.assertEqual(config.parameters["shading"]["values"], ["visibility", "lambert", "pbr"])
        self.assertEqual(config.parameters["shading"]["kind"], "variant")
        self.assertEqual(config.parameters["shadows"]["type"], "bool")
        self.assertEqual(config.parameters["maxShadowLights"]["minimum"], 1.0)
        self.assertEqual(config.parameters["maxShadowLights"]["maximum"], 16.0)
        self.assertEqual(config.presets, ["visibility", "lambert", "pbr"])
        self.assertEqual(config.kind, "batch_ray_tracing")

    def test_raster_packs_declare_no_presets(self) -> None:
        self.assertEqual(sapien.render.RenderConfig("default").presets, [])

    def test_every_camera_pack_renders_with_its_defaults(self) -> None:
        """pack.json and the shaders agree: a camera of every built-in camera pack renders its
        defaults instead of raising about an undeclared uniform."""
        for pack in CAMERA_PACKS:
            with self.subTest(pack=pack):
                config = sapien.render.RenderConfig(pack)
                if config.is_ray_tracing:
                    config.denoiser = "none"
                with sapien.Scene() as scene:
                    build_box_scene(scene)
                    camera = scene.add_camera(
                        "camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=config
                    )
                    camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
                    scene.update_render()
                    camera.take_picture()
                    picture = camera.get_picture("Color")
                    self.assertEqual(picture.shape, (SIZE, SIZE, 4))
                    self.assertTrue(np.all(np.isfinite(picture)))
                    if pack in SHADING_PACKS:
                        center = picture[SIZE // 2, SIZE // 2, :3]
                        self.assertGreater(float(center[0]), 0.1)

    def test_a_batch_pack_has_no_per_camera_renderer(self) -> None:
        for pack in NON_CAMERA_PACKS:
            with self.subTest(pack=pack):
                config = sapien.render.RenderConfig(pack)
                self.assertTrue(config.is_batch_ray_tracing)
                with sapien.Scene() as scene:
                    camera = scene.add_camera(
                        "camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=config
                    )
                    # the camera exists (a RenderCameraGroup owns its rendering) but has no
                    # renderer of its own, so the per-camera path raises
                    with self.assertRaisesRegex(RuntimeError, "RenderCameraGroup"):
                        camera.take_picture()
                    with self.assertRaisesRegex(RuntimeError, "RenderCameraGroup"):
                        camera.get_picture("Color")


class TestAssignmentValidation(unittest.TestCase):
    def test_unknown_parameter_raises_naming_key_and_pack(self) -> None:
        config = sapien.render.RenderConfig("rt")
        with self.assertRaisesRegex(RuntimeError, "notAParameter.*rt"):
            config["notAParameter"] = 1

    def test_wrong_type_raises_naming_key_and_pack(self) -> None:
        config = sapien.render.RenderConfig("rt")
        with self.assertRaisesRegex(RuntimeError, "spp.*int"):
            config["spp"] = 1.5
        with self.assertRaisesRegex(RuntimeError, "spp"):
            config["spp"] = "many"
        with self.assertRaisesRegex(RuntimeError, "russianRoulette"):
            config["russianRoulette"] = 2

    def test_out_of_range_raises_naming_key_and_pack(self) -> None:
        config = sapien.render.RenderConfig("rt")
        with self.assertRaisesRegex(RuntimeError, "spp.*minimum"):
            config["spp"] = 0
        with self.assertRaisesRegex(RuntimeError, "toneMapper.*maximum"):
            config["toneMapper"] = 6
        self.assertNotIn("spp", config)

    def test_int_is_accepted_for_a_float_parameter_and_stored_as_float(self) -> None:
        config = sapien.render.RenderConfig("default")
        config["exposure"] = 3
        self.assertIsInstance(config["exposure"], float)
        self.assertIsInstance(config.properties["exposure"], float)
        self.assertEqual(config["exposure"], 3.0)
        self.assertIsInstance(config["exposure"], float)

    def test_effective_defaults_and_del(self) -> None:
        config = sapien.render.RenderConfig("rt")
        self.assertEqual(config["spp"], 32)
        self.assertNotIn("spp", config)
        config["spp"] = 7
        self.assertEqual(config["spp"], 7)
        self.assertIn("spp", config)
        self.assertEqual(config.properties, {"spp": 7})

        # the dict from cfg.properties is a copy
        config.properties["spp"] = 99
        self.assertEqual(config["spp"], 7)

        del config["spp"]
        self.assertEqual(config["spp"], 32)
        self.assertNotIn("spp", config)

    def test_unknown_parameter_read_and_del_raise(self) -> None:
        config = sapien.render.RenderConfig("default")
        with self.assertRaisesRegex(RuntimeError, "notAParameter"):
            config["notAParameter"]
        with self.assertRaisesRegex(RuntimeError, "notAParameter"):
            del config["notAParameter"]

    def test_preset_expands_into_assignments(self) -> None:
        config = sapien.render.RenderConfig("rt_batch", preset="pbr")
        self.assertEqual(config["shading"], "pbr")
        self.assertIn("shading", config)
        self.assertEqual(config.properties, {"shading": "pbr"})

        # a preset only sets what it names: everything else stays at its default
        self.assertEqual(config["spp"], 1)
        self.assertFalse(config["shadows"])

    def test_unknown_preset_raises_naming_the_pack_and_its_presets(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "toon.*rt_batch.*visibility"):
            sapien.render.RenderConfig("rt_batch", preset="toon")
        with self.assertRaisesRegex(RuntimeError, "toon.*default.*no presets"):
            sapien.render.RenderConfig("default", preset="toon")

    def test_shader_pack_switch_revalidates_assignments(self) -> None:
        config = sapien.render.RenderConfig("rt")
        config["spp"] = 4
        config["maxDepth"] = 4  # only rt declares maxDepth
        pack = config.shader_pack
        with self.assertRaisesRegex(RuntimeError, 'rt_batch.*no parameter "maxDepth"'):
            config.shader_pack = "rt_batch"
        # a rejected switch changes nothing
        self.assertEqual(config.shader_pack, pack)
        self.assertEqual(config["maxDepth"], 4)

        # a parameter both packs declare switches fine
        del config["maxDepth"]
        config.shader_pack = "rt_batch"
        self.assertTrue(config.shader_pack.endswith("rt_batch"))
        self.assertEqual(config["spp"], 4)

    def test_shader_pack_switch_keeps_out_of_range_values_out(self) -> None:
        config = sapien.render.RenderConfig("rt")
        config["spp"] = 40  # inside rt's unbounded range, above rt_batch's maximum of 32
        with self.assertRaisesRegex(RuntimeError, "spp.*maximum"):
            config.shader_pack = "rt_batch"
        self.assertTrue(config.shader_pack.endswith("rt"))


class TestPackManifestFiles(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(sapien.render.clear_cache, False, False, True)

    def test_missing_pack_json_raises_with_the_directory(self) -> None:
        pack = copy_trivial_pack(self.directory.name)
        (pack / "pack.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "pack.json.*customize_shaders"):
            sapien.render.RenderConfig(str(pack))
        # the directory is named in the message
        with self.assertRaisesRegex(RuntimeError, str(pack.name)):
            sapien.render.RenderConfig(str(pack))

    def test_malformed_pack_json_raises_with_the_parser_message(self) -> None:
        pack = copy_trivial_pack(self.directory.name)
        (pack / "pack.json").write_text('{"parameters": {"a": {"kind": "uniform"}}}')
        with self.assertRaisesRegex(RuntimeError, r'pack\.json: parameter "a" needs "type"'):
            sapien.render.RenderConfig(str(pack))
        (pack / "pack.json").write_text(
            '{"parameters": {"a": {"kind": "uniform", "type": "int"}}}'
        )
        with self.assertRaisesRegex(RuntimeError, r'parameter "a" needs "default"'):
            sapien.render.RenderConfig(str(pack))

        (pack / "pack.json").write_text("{")
        with self.assertRaisesRegex(RuntimeError, "pack.json is not valid JSON"):
            sapien.render.RenderConfig(str(pack))

    def test_edited_pack_json_is_read_again_after_clearing_the_cache(self) -> None:
        manifest = {
            "parameters": {
                "gain": {"kind": "uniform", "type": "int", "default": 1, "minimum": 1}
            }
        }
        pack = copy_trivial_pack(self.directory.name, manifest)
        self.assertEqual(sapien.render.RenderConfig(str(pack))["gain"], 1)

        manifest["parameters"]["gain"]["default"] = 5
        (pack / "pack.json").write_text(json.dumps(manifest))
        # cached: the old manifest is still in use
        self.assertEqual(sapien.render.RenderConfig(str(pack))["gain"], 1)

        sapien.render.clear_cache(models=False, images=False, shaders=True)
        self.assertEqual(sapien.render.RenderConfig(str(pack))["gain"], 5)

    def test_a_manifest_uniform_missing_from_the_shader_raises(self) -> None:
        manifest = {
            "parameters": {
                "gain": {"kind": "uniform", "type": "float", "default": 1.0}
            }
        }
        pack = copy_trivial_pack(self.directory.name, manifest)
        config = sapien.render.RenderConfig(str(pack))
        with sapien.Scene() as scene:
            with self.assertRaisesRegex(
                RuntimeError, r'pack\.json declares uniform "gain" as float but the shader'
            ):
                scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=config)

    def test_a_manifest_uniform_of_the_wrong_type_raises(self) -> None:
        """The shader declares `shadeFlat` as an int; a float manifest entry must not reach it."""
        manifest = {
            "parameters": {
                "shadeFlat": {"kind": "uniform", "type": "float", "default": 1.0}
            }
        }
        pack = copy_trivial_pack(self.directory.name, manifest)
        config = sapien.render.RenderConfig(str(pack))
        with sapien.Scene() as scene:
            with self.assertRaisesRegex(
                RuntimeError, r'declares uniform "shadeFlat" as float but the shader declares'
            ):
                scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=config)


class TestVariantParameters(unittest.TestCase):
    """A variant parameter selects shader source through a definition, exactly as `defines` did."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
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

    def _pack_with_blue_variant(self, manifest: dict) -> Path:
        pack = copy_trivial_pack(self.directory.name, manifest)
        shader = pack / "gbuffer.frag"
        text = shader.read_text()
        marker = "outColor = material.baseColor;"
        self.assertIn(marker, text)
        shader.write_text(
            text.replace(
                marker,
                marker + "\n#ifdef BLUE_VARIANT\n  outColor = vec4(0, 0, 1, 1);\n#endif",
            )
        )
        return pack

    def test_variant_reaches_the_shader(self) -> None:
        manifest = {
            "parameters": {
                "blue": {
                    "kind": "variant",
                    "type": "bool",
                    "default": False,
                    "define": "BLUE_VARIANT",
                    "description": "paint the box blue",
                }
            }
        }
        pack = self._pack_with_blue_variant(manifest)

        plain = sapien.render.RenderConfig(str(pack))
        np.testing.assert_allclose(self._center_color(plain), [1.0, 0.0, 0.0], atol=1e-3)

        variant = sapien.render.RenderConfig(str(pack))
        variant["blue"] = True
        np.testing.assert_allclose(self._center_color(variant), [0.0, 0.0, 1.0], atol=1e-3)

    def test_numeric_and_enum_variants_define_their_value(self) -> None:
        """A numeric variant defines `NAME value` and an enum variant its mapped name."""
        manifest = {
            "parameters": {
                "scale": {
                    "kind": "variant",
                    "type": "float",
                    "default": 2.0,
                    "define": "SCALE",
                    "description": "scales the stored colour",
                },
                "tint": {
                    "kind": "variant",
                    "type": "enum",
                    "values": ["red", "blue"],
                    "default": "red",
                    "define": {"red": "TINT_RED", "blue": "TINT_BLUE"},
                    "description": "which channel the colour goes to",
                },
            }
        }
        pack = copy_trivial_pack(self.directory.name, manifest)
        shader = pack / "gbuffer.frag"
        text = shader.read_text()
        marker = "outColor = material.baseColor;"
        shader.write_text(
            text.replace(
                marker,
                marker
                + "\n  outColor = vec4(SCALE, 0, 0, 1)\n"
                + "#ifdef TINT_BLUE\n"
                + "      .bgra;\n"
                + "#else\n"
                + "      .rgba;\n"
                + "#endif",
            )
        )

        np.testing.assert_allclose(
            self._center_color(sapien.render.RenderConfig(str(pack))), [2.0, 0.0, 0.0], atol=1e-3
        )

        blue = sapien.render.RenderConfig(str(pack))
        blue["tint"] = "blue"
        blue["scale"] = 4  # an int for a float variant is accepted and stored as a float
        np.testing.assert_allclose(self._center_color(blue), [0.0, 0.0, 4.0], atol=1e-3)

        with self.assertRaisesRegex(RuntimeError, "tint.*no value"):
            blue["tint"] = "green"

    def test_set_property_rejects_a_variant_parameter(self) -> None:
        manifest = {
            "parameters": {
                "blue": {
                    "kind": "variant",
                    "type": "bool",
                    "default": False,
                    "define": "BLUE_VARIANT",
                    "description": "paint the box blue",
                }
            }
        }
        pack = copy_trivial_pack(self.directory.name, manifest)
        camera = sapien.render.RenderCameraComponent(
            SIZE, SIZE, sapien.render.RenderConfig(str(pack))
        )
        with self.assertRaisesRegex(RuntimeError, "variant.*RenderConfig"):
            camera.set_property("blue", True)
        # a batch pack rejects every property, variant or not
        batch_camera = sapien.render.RenderCameraComponent(
            SIZE, SIZE, sapien.render.RenderConfig("rt_batch")
        )
        with self.assertRaisesRegex(RuntimeError, "batched ray tracing"):
            batch_camera.set_property("shadows", True)

    def test_get_property_returns_the_config_default(self) -> None:
        config = sapien.render.RenderConfig("rt")
        camera = sapien.render.RenderCameraComponent(SIZE, SIZE, config)
        # no runtime override yet: the config's effective value, i.e. the pack.json default
        self.assertEqual(camera.get_property("spp"), 32)
        self.assertEqual(camera.get_property("maxDepth"), 8)
        camera.set_property("spp", 5)
        self.assertEqual(camera.get_property("spp"), 5)

    def test_bool_parameters_round_trip_as_0_or_1(self) -> None:
        config = sapien.render.RenderConfig("rt")
        config["russianRoulette"] = True
        self.assertIs(config["russianRoulette"], True)
        camera = sapien.render.RenderCameraComponent(SIZE, SIZE, config)
        # get_property reports a bool parameter as its int form
        self.assertEqual(camera.get_property("russianRoulette"), 1)
        camera.set_property("russianRoulette", False)
        self.assertEqual(camera.get_property("russianRoulette"), 0)
        camera.set_property("russianRoulette", True)
        self.assertEqual(camera.get_property("russianRoulette"), 1)


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

    def test_denoiser_on_raster_pack_raises(self) -> None:
        config = sapien.render.RenderConfig("default")
        config.denoiser = "oidn"
        with sapien.Scene() as scene:
            with self.assertRaisesRegex(RuntimeError, "denoiser"):
                scene.add_camera(
                    "camera", 32, 32, 1.0, 0.1, 10.0, render_config=config
                )

    def test_set_property_validates_against_the_manifest(self) -> None:
        with sapien.Scene() as scene:
            build_box_scene(scene)
            camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0)
            camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
            scene.update_render()
            with self.assertRaisesRegex(RuntimeError, 'no parameter "notAProperty"'):
                camera.set_property("notAProperty", 1)
            with self.assertRaisesRegex(RuntimeError, 'no parameter "shadeFlat"'):
                camera.set_property("shadeFlat", 0.5)
            with self.assertRaisesRegex(RuntimeError, "toneMapper.*maximum"):
                camera.set_property("toneMapper", 7)
            self.assertEqual(camera.get_property("toneMapper"), 2)

    def test_set_property_before_the_renderer_validates_at_attach(self) -> None:
        """A property set before the camera is added is validated when its renderer appears."""
        config = sapien.render.RenderConfig("rt")
        config["spp"] = 1
        camera = sapien.render.RenderCameraComponent(SIZE, SIZE, config)
        camera.set_property("spp", 2)  # validated against pack.json right away
        self.assertEqual(camera.get_property("spp"), 2)
        # an unknown name is rejected at assignment time, before any renderer exists
        with self.assertRaisesRegex(RuntimeError, "notAProperty.*rt"):
            camera.set_property("notAProperty", 3)
        # as is an out-of-range value of a declared parameter
        with self.assertRaisesRegex(RuntimeError, "spp.*minimum"):
            camera.set_property("spp", 0)
        self.assertEqual(camera.get_property("spp"), 2)

        entity = sapien.Entity()
        entity.add_component(camera)
        with sapien.Scene() as scene:
            scene.add_entity(entity)
            self.assertEqual(camera.get_property("spp"), 2)

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

    def test_an_unknown_parameter_name_is_not_reported_as_a_shader_mismatch(self) -> None:
        """A typo never reaches renderer creation: pack.json answers first."""
        config = sapien.render.RenderConfig("default")
        with self.assertRaisesRegex(RuntimeError, 'no parameter "exposur"'):
            config["exposur"] = 1.0
        self.assertEqual(list(config.properties), [])


class TestViewerRenderPanel(unittest.TestCase):
    """The render panel builds one widget per uniform parameter of `cfg.parameters`.

    The widgets can be constructed without a display; only showing the window needs one, so the
    panel's parameter plumbing is exercised with a stand-in window.
    """

    class _Window:
        """A stand-in for `RenderWindow`: records the properties the panel sets."""

        def __init__(self, config: "sapien.render.RenderConfig"):
            self.render_config = config
            self.denoiser = "none"

        def get_camera_property_int(self, name: str) -> int:
            return int(self.render_config[name])

        def get_camera_property_float(self, name: str) -> float:
            return float(self.render_config[name])

        def set_camera_property(self, name: str, value) -> None:
            # the real renderer rejects a value its pack does not accept
            self.render_config[name] = value

    class _Viewer:
        def __init__(self, config: "sapien.render.RenderConfig"):
            self.window = TestViewerRenderPanel._Window(config)
            self.config = config
            self.render_scene = object()

        @property
        def render_config(self):
            return self.config

        @render_config.setter
        def render_config(self, config):
            self.config = config

    def test_panel_widgets_follow_the_pack_parameters(self) -> None:
        from sapien.utils.viewer.render_window import RenderOptionsWindow

        try:
            import sapien.internal_renderer  # noqa: F401
        except ImportError:  # pragma: no cover - the wheel always ships it
            self.skipTest("the internal renderer widgets are not available")

        config = sapien.render.RenderConfig("default")
        panel = RenderOptionsWindow()
        panel.viewer = self._Viewer(config)
        panel.shader_list = [Path(config.shader_pack)]
        panel.shader_types = ["rast"]
        panel.shader_type = "rast"
        panel.shader_dir = config.shader_pack
        panel.shader_index = 0

        panel.build()
        self.assertIsNotNone(panel.ui_window)

        # every uniform parameter is readable and settable through the panel
        uniforms = [name for name, p in config.parameters.items() if p["kind"] == "uniform"]
        self.assertEqual(uniforms, ["exposure", "toneMapper",
                                    "ambientOcclusionStrength", "ambientOcclusionRadius"])
        self.assertAlmostEqual(panel.get_parameter("exposure"), 1.0, places=6)
        panel.set_parameter("exposure", 3)
        self.assertEqual(panel.viewer.window.get_camera_property_float("exposure"), 3)
        panel.set_parameter("toneMapper", 1)
        self.assertEqual(panel.viewer.window.get_camera_property_int("toneMapper"), 1)

        # a value the pack's range rejects never reaches the renderer
        with self.assertRaisesRegex(RuntimeError, "toneMapper.*maximum"):
            panel.set_parameter("toneMapper", 9)
        self.assertEqual(panel.viewer.window.get_camera_property_int("toneMapper"), 1)

    def test_panel_does_not_list_a_pack_it_cannot_display(self) -> None:
        """The pack list skips a batched pack, which has no window renderer."""
        from sapien.utils.viewer.render_window import RenderOptionsWindow

        config = sapien.render.RenderConfig("default")
        panel = RenderOptionsWindow()
        panel.viewer = self._Viewer(config)
        panel.viewer.render_config = config
        panel.init(panel.viewer)
        names = panel.shader_name_list
        self.assertNotIn("rt_batch", names)
        # common and internal hold includes and one internal shader, not packs
        self.assertNotIn("common", names)
        self.assertNotIn("internal", names)
        self.assertIn("default", names)
        self.assertIn("rt", names)


class TestViewerPackSwitch(unittest.TestCase):
    """The Viewer render panel switches packs without a window through this helper.

    The panel itself needs a display, but the pack switch is a pure RenderConfig operation, so
    the carry-over rule can be exercised here.
    """

    def test_build_config_for_pack_carries_over_compatible_assignments(self) -> None:
        from sapien.utils.viewer.render_window import build_config_for_pack

        # spp and maxDepth exist only in rt: switching to default drops them
        rt = sapien.render.RenderConfig("rt")
        rt["spp"] = 4
        rt["maxDepth"] = 2
        config = build_config_for_pack("default", rt)
        self.assertTrue(config.shader_pack.endswith("default"))
        self.assertEqual(config.properties, {})
        self.assertEqual(config["toneMapper"], 2)  # the new pack's default
        self.assertAlmostEqual(config["exposure"], 1.0, places=6)

        # a name both packs share survives; rt_batch's spp range (1..32) still applies
        batch = sapien.render.RenderConfig("rt_batch")
        batch["spp"] = 8
        batch["shadows"] = True
        carried = build_config_for_pack("rt", batch)
        self.assertEqual(carried.properties, {"spp": 8})

        # a value inside both ranges survives the switch unchanged
        batch["spp"] = 32
        self.assertEqual(build_config_for_pack("rt_batch", batch).properties["spp"], 32)

    def test_build_config_for_pack_keeps_the_new_pack_defaults_for_variants(self) -> None:
        from sapien.utils.viewer.render_window import build_config_for_pack

        # a variant assignment of the source pack survives when the target declares it
        batch = sapien.render.RenderConfig("rt_batch", preset="pbr")
        copy = build_config_for_pack("rt_batch", batch)
        self.assertEqual(copy["shading"], "pbr")
        # a numeric assignment the target declares with another type is dropped
        batch["spp"] = 8
        rt = build_config_for_pack("default", batch)
        self.assertNotIn("spp", rt.properties)


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

    def test_the_manifest_default_reaches_the_renderer(self) -> None:
        """`maxDepth` comes from pack.json: dropping the assignment renders its default (8)."""
        config = sapien.render.RenderConfig("rt")
        config["maxDepth"] = 0
        config.denoiser = "none"
        del config["maxDepth"]
        self.assertEqual(config["maxDepth"], 8)
        values = self._render_brightness(config, frames=3)
        self.assertGreater(values[0], 1e-3)


if __name__ == "__main__":
    unittest.main()
