"""The post-processing chain of pack.json ("post"): pass order, pass-local parameters, the images
between passes, renderer images read by name, and where pass files are looked up."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import sapien

SIZE = 32
SHADER_ROOT = Path(sapien.__file__).parent / "vulkan_shader"

# multiplies the colour by the pass parameter `gain`
GAIN_PASS = """#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0, rgba32f) uniform readonly image2D inColor;
layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D outColor;
layout(push_constant) uniform Parameters { float gain; };
void main() {
  ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
  if (any(greaterThanEqual(pixel, imageSize(outColor)))) return;
  vec4 color = imageLoad(inColor, pixel);
  imageStore(outColor, pixel, vec4(color.rgb * gain, color.a));
}
"""

# replaces the colour by the renderer's Albedo image
ALBEDO_PASS = """#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0, rgba32f) uniform readonly image2D inColor;
layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D outColor;
layout(set = 0, binding = 2, rgba32f) uniform readonly image2D Albedo;
void main() {
  ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
  if (any(greaterThanEqual(pixel, imageSize(outColor)))) return;
  imageStore(outColor, pixel, imageLoad(Albedo, pixel));
}
"""

# writes the pixel's own uv in r/g and a constant 1 in b, so a pass after it maps a known image
# and a multiplicative pass reveals its factor in b
UV_PASS = """#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0, rgba32f) uniform readonly image2D inColor;
layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D outColor;
void main() {
  ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
  ivec2 size = imageSize(outColor);
  if (any(greaterThanEqual(pixel, size))) return;
  vec2 uv = (vec2(pixel) + 0.5) / vec2(size);
  imageStore(outColor, pixel, vec4(uv, 1.0, 1.0));
}
"""

# a single lit pixel at the centre of the image, the rest black
DELTA_PASS = """#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0, rgba32f) uniform readonly image2D inColor;
layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D outColor;
void main() {
  ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
  ivec2 size = imageSize(outColor);
  if (any(greaterThanEqual(pixel, size))) return;
  bool lit = pixel == size / 2;
  imageStore(outColor, pixel, vec4(lit ? 4.0 : 0.0, 0.0, 0.0, 1.0));
}
"""

# a diagonal step edge whose two sides differ by 0.036 once sRGB encoded: above fxaa's 0.0312
# floor but below lMax * 0.125, so the default threshold leaves it alone and 0 smooths it
EDGE_PASS = """#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0, rgba32f) uniform readonly image2D inColor;
layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D outColor;
void main() {
  ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
  ivec2 size = imageSize(outColor);
  if (any(greaterThanEqual(pixel, size))) return;
  bool above = pixel.y + pixel.x > size.x / 2 + size.y / 2;
  float value = above ? 0.45 : 0.40;
  imageStore(outColor, pixel, vec4(value, value, value, 1.0));
}
"""


def srgb(linear: np.ndarray) -> np.ndarray:
    """The sRGB encoding of tone mapper 1."""
    linear = np.maximum(linear, 0.0)
    encoded = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    return np.clip(encoded, 0.0, 1.0)


def linear_from_srgb(encoded: np.ndarray) -> np.ndarray:
    """The inverse of `srgb`; used to read a `Color` pixel back as the radiance it encodes."""
    encoded = np.asarray(encoded, dtype=np.float64)
    return np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)


# Bradford XYZ -> LMS and linear Rec.709 (D65) -> XYZ, the pair color_grade.comp composes.
_BRADFORD = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ]
)
_REC709_TO_XYZ = np.array(
    [
        [0.4123908, 0.3575843, 0.1804808],
        [0.2126390, 0.7151687, 0.0721923],
        [0.0193308, 0.1191948, 0.9505322],
    ]
)


def kelvin_chromaticity(temperature: float) -> np.ndarray:
    """The CIE (x, y) of the Planckian locus, as color_grade.comp computes it."""
    t = float(np.clip(temperature, 1500.0, 25000.0))
    inv_t = 1e3 / t
    if t <= 4000.0:
        x = ((-0.2661239 * inv_t - 0.2343589) * inv_t + 0.8776956) * inv_t + 0.179910
    else:
        x = ((-3.0258469 * inv_t + 2.1070379) * inv_t + 0.2226347) * inv_t + 0.240390
    if t <= 2222.0:
        y = ((-1.1063814 * x - 1.34811020) * x + 2.18555832) * x - 0.20219683
    elif t <= 4000.0:
        y = ((-0.9549476 * x - 1.37418593) * x + 2.09137015) * x - 0.16748867
    else:
        y = ((3.0817580 * x - 5.87338670) * x + 3.75112997) * x - 0.37001483
    return np.array([x, max(y, 1e-4)])


def bradford_adapt(linear_rgb: np.ndarray, temperature: float) -> np.ndarray:
    """The white balance color_grade.comp applies: a Bradford adaptation from the 6500 K
    reference white to `temperature`, in linear Rec.709."""
    to_lms = _BRADFORD @ _REC709_TO_XYZ

    def xyz(t: float) -> np.ndarray:
        x, y = kelvin_chromaticity(t)
        return np.array([x / y, 1.0, (1.0 - x - y) / y])

    scale = (_BRADFORD @ xyz(6500.0)) / np.maximum(_BRADFORD @ xyz(temperature), 1e-4)
    return np.linalg.inv(to_lms) @ ((to_lms @ np.asarray(linear_rgb, dtype=np.float64)) * scale)


class PostChain(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(sapien.render.clear_cache, shaders=True)

    def pack(self, post: list[str], base: str = "rt") -> Path:
        """A copy of a built-in pack that runs the passes `post`, shipping the passes `gain` and
        `albedo` itself; `tonemap` is the built-in pass. The copy of `default` samples the
        result of the passes instead of tone mapping in its composite pass."""
        root = Path(self.directory.name) / f"case{len(list(Path(self.directory.name).iterdir()))}"
        pack = root / "pack"
        shutil.copytree(SHADER_ROOT / base, pack)
        shutil.copytree(SHADER_ROOT / "common", root / "common")
        if base == "default":
            composite = (pack / "composite0.frag").read_text()
            for old, new in (
                ("layout (constant_id = 0) const float exposure = 1.0;\n", ""),
                ("layout (constant_id = 1) const int toneMapper = 2;\n", ""),
                ("uniform sampler2D samplerLighting;", "uniform sampler2D samplerPostColor;"),
                ("texture(samplerLighting, inUV)", "texture(samplerPostColor, inUV)"),
                ("  outColor.rgb = sapienToneMap(outColor.rgb, exposure, toneMapper);\n", ""),
            ):
                self.assertIn(old, composite)
                composite = composite.replace(old, new)
            (pack / "composite0.frag").write_text(composite)
        (pack / "post").mkdir()
        (pack / "post" / "gain.comp").write_text(GAIN_PASS)
        (pack / "post" / "albedo.comp").write_text(ALBEDO_PASS)
        (pack / "post" / "uv.comp").write_text(UV_PASS)
        (pack / "post" / "delta.comp").write_text(DELTA_PASS)
        (pack / "post" / "edge.comp").write_text(EDGE_PASS)
        manifest = json.loads((pack / "pack.json").read_text())
        manifest["post"] = post
        # Keep only post-pass parameters whose pass is in `post`
        pass_params = {
            "bloom": ("bloomIntensity", "bloomThreshold", "bloomRadius", "bloomKnee"),
            "color_grade": (
                "exposureEV",
                "whiteBalanceTemp",
                "whiteBalanceTint",
                "contrast",
                "saturation",
                "cdlSlope",
                "cdlOffset",
                "cdlPower",
            ),
            "tonemap": ("exposure", "toneMapper"),
            "lens": (
                "lensDistortionK1",
                "lensDistortionK2",
                "chromaticAberration",
                "vignetteStrength",
                "vignetteSmoothness",
                "filmGrain",
                "sensorReadNoise",
                "sensorShotNoise",
                "noiseSeed",
            ),
        }
        for pass_name, params in pass_params.items():
            if pass_name not in post:
                for name in params:
                    manifest["parameters"].pop(name, None)
        if "gain" in post:
            manifest["parameters"]["gain"] = {"kind": "uniform", "type": "float", "default": 1.0}
        if "fxaa" in post:
            manifest["parameters"]["fxaaEdgeThreshold"] = {
                "kind": "uniform",
                "type": "float",
                "default": 0.125,
            }
            manifest["parameters"]["fxaaSubpix"] = {
                "kind": "uniform",
                "type": "float",
                "default": 0.75,
            }
        (pack / "pack.json").write_text(json.dumps(manifest))
        return pack

    def render(self, pack: Path | str, **parameters) -> dict[str, np.ndarray]:
        """The pictures of a lit box: the chain's source as "HdrColor" (`Lighting` for a
        rasterization pack), Albedo and, when the pack has one, Color."""
        config = sapien.render.RenderConfig(str(pack))
        if "spp" in config.parameters:
            config["spp"] = 1
        config["toneMapper"] = 1
        for name, value in parameters.items():
            config[name] = value
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.2, 0.2, 0.2])
            scene.add_point_light([-2.0, 1.0, 2.0], [8.0, 8.0, 8.0])
            builder = scene.create_actor_builder()
            builder.add_box_visual(
                half_size=[0.5, 0.5, 0.5],
                material=sapien.render.RenderMaterial(base_color=[0.8, 0.5, 0.2, 1.0]),
            )
            builder.build_kinematic()
            camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=config)
            camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
            scene.update_render()
            camera.take_picture()
            available = camera.get_picture_names()
            pictures = {
                name: camera.get_picture(name)[..., :3].copy()
                for name in ("HdrColor", "Lighting", "Albedo", "Color")
                if name in available
            }
            if "Lighting" in pictures:
                pictures["HdrColor"] = pictures.pop("Lighting")
            return pictures

    def test_builtin_rt_color_is_the_tone_mapped_radiance(self) -> None:
        pictures = self.render("rt", exposure=0.5)
        self.assertGreater(float(pictures["HdrColor"].max()), 0.2)
        np.testing.assert_allclose(pictures["Color"], srgb(0.5 * pictures["HdrColor"]), atol=1e-4)

    def test_passes_run_in_the_listed_order(self) -> None:
        """The curve is not linear, so a gain before it and a gain after it differ."""
        before = self.render(self.pack(["gain", "tonemap"]), gain=0.25)
        np.testing.assert_allclose(before["Color"], srgb(0.25 * before["HdrColor"]), atol=1e-4)
        after = self.render(self.pack(["tonemap", "gain"]), gain=0.25)
        np.testing.assert_allclose(after["Color"], 0.25 * srgb(after["HdrColor"]), atol=1e-4)
        lit = before["HdrColor"].max(axis=-1) > 0.1
        self.assertGreater(float(np.abs(before["Color"] - after["Color"])[lit].mean()), 0.05)

    def test_longer_chains_alternate_the_images_between_passes(self) -> None:
        pictures = self.render(self.pack(["gain", "gain", "gain", "tonemap"]), gain=0.5)
        np.testing.assert_allclose(pictures["Color"], srgb(0.125 * pictures["HdrColor"]), atol=1e-4)

    def test_a_pass_reads_renderer_images_by_name(self) -> None:
        pictures = self.render(self.pack(["tonemap", "albedo"]))
        self.assertGreater(float(pictures["Albedo"].max()), 0.5)
        np.testing.assert_array_equal(pictures["Color"], pictures["Albedo"])

    def test_a_pass_parameter_is_a_camera_property(self) -> None:
        config = sapien.render.RenderConfig(str(self.pack(["gain", "tonemap"])))
        self.assertEqual(config["gain"], 1.0)
        config["gain"] = 0.5
        with sapien.Scene() as scene:
            camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=config)
            scene.update_render()
            camera.take_picture()
            self.assertEqual(camera.get_property("gain"), 0.5)
            camera.set_property("gain", 2.0)
            self.assertEqual(camera.get_property("gain"), 2.0)

    def test_without_passes_there_is_no_color(self) -> None:
        pack = self.pack([])
        config = sapien.render.RenderConfig(str(pack))
        config["spp"] = 1
        with sapien.Scene() as scene:
            camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=config)
            scene.update_render()
            camera.take_picture()
            self.assertIn("HdrColor", camera.get_picture_names())
            self.assertNotIn("Color", camera.get_picture_names())

    def test_an_unknown_pass_raises_naming_both_locations(self) -> None:
        pack = self.pack(["tonemap", "sharpen"])
        with self.assertRaisesRegex(RuntimeError, r'"sharpen".*pack/post/sharpen\.comp.*nor at'):
            sapien.render.RenderConfig(str(pack))

    def test_post_on_a_batch_pack_raises(self) -> None:
        pack = self.pack(["tonemap"], base="rt_batch")
        with self.assertRaisesRegex(RuntimeError, "not supported by batched ray tracing"):
            sapien.render.RenderConfig(str(pack))

    def test_raster_passes_run_in_the_listed_order(self) -> None:
        before = self.render(self.pack(["gain", "tonemap"], base="default"), gain=0.25)
        self.assertGreater(float(before["HdrColor"].max()), 0.2)
        np.testing.assert_allclose(before["Color"], srgb(0.25 * before["HdrColor"]), atol=1e-4)
        after = self.render(self.pack(["tonemap", "gain"], base="default"), gain=0.25)
        np.testing.assert_allclose(after["Color"], 0.25 * srgb(after["HdrColor"]), atol=1e-4)

    def test_a_raster_pass_parameter_changes_between_frames(self) -> None:
        """Pass parameters are recorded with the frame: a change must reach the next one."""
        config = sapien.render.RenderConfig(str(self.pack(["tonemap"], base="default")))
        config["toneMapper"] = 1
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.5, 0.5, 0.5])
            builder = scene.create_actor_builder()
            builder.add_box_visual(half_size=[0.5, 0.5, 0.5])
            builder.build_kinematic()
            camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=config)
            camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
            for exposure in (1.0, 0.25, 0.5):
                camera.set_property("exposure", exposure)
                scene.update_render()
                camera.take_picture()
                lighting = camera.get_picture("Lighting")[..., :3]
                self.assertGreater(float(lighting.max()), 0.2)
                np.testing.assert_allclose(
                    camera.get_picture("Color")[..., :3], srgb(exposure * lighting), atol=1e-4
                )

    def test_a_raster_pass_cannot_read_other_images(self) -> None:
        pack = self.pack(["tonemap", "albedo"], base="default")
        with self.assertRaisesRegex(RuntimeError, 'reads the image "Albedo"'):
            self.render(pack)

    def test_a_raster_pack_without_lighting_raises(self) -> None:
        pack = self.pack(["tonemap"], base="trivial")
        manifest = json.loads((pack / "pack.json").read_text())
        manifest["parameters"] = {
            "exposure": {"kind": "uniform", "type": "float", "default": 1.0},
            "toneMapper": {"kind": "uniform", "type": "int", "default": 2},
        }
        (pack / "pack.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RuntimeError, "needs the render target Lighting"):
            self.render(pack)

    def test_a_pass_without_the_chain_images_raises(self) -> None:
        pack = self.pack(["gain", "tonemap"])
        (pack / "post" / "gain.comp").write_text(GAIN_PASS.replace("outColor", "result"))
        with self.assertRaisesRegex(RuntimeError, "gain.comp.*inColor and outColor"):
            self.render(pack)

    def test_new_tone_mappers_and_presets(self) -> None:
        """PBR Neutral (3), AgX (4) and Filmic (5) produce distinct bounded outputs in [0, 1],
        and rt exposes preview and production presets."""
        cfg = sapien.render.RenderConfig("rt", preset="preview")
        self.assertEqual(cfg["spp"], 4)
        self.assertEqual(cfg["maxDepth"], 4)
        prod = sapien.render.RenderConfig("rt", preset="production")
        self.assertEqual(prod["spp"], 64)
        self.assertEqual(prod["maxDepth"], 12)
        self.assertEqual(prod["clampIndirect"], 10.0)

        # Keep one scene/camera and switch toneMapper at runtime (avoids extra Vulkan instances)
        rt_cfg = sapien.render.RenderConfig("rt")
        rt_cfg["spp"] = 1
        rt_cfg["exposure"] = 2.5
        rt_cfg.denoiser = "none"
        outputs = {}
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.2, 0.2, 0.2])
            scene.add_point_light([-2.0, 1.0, 2.0], [8.0, 8.0, 8.0])
            b = scene.create_actor_builder()
            b.add_box_visual(
                half_size=[0.5, 0.5, 0.5],
                material=sapien.render.RenderMaterial(base_color=[0.8, 0.5, 0.2, 1.0]),
            )
            b.build_kinematic()
            cam = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=rt_cfg)
            cam.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
            scene.update_render()
            for tm in (1, 2, 3, 4, 5):
                cam.set_property("toneMapper", tm)
                cam.take_picture()
                color = cam.get_picture("Color")[..., :3].copy()
                self.assertTrue(np.all(np.isfinite(color)))
                self.assertGreaterEqual(float(color.min()), 0.0)
                self.assertLessEqual(float(color.max()), 1.0)
                outputs[tm] = color
        for tm in (3, 4, 5):
            self.assertGreater(float(np.abs(outputs[tm] - outputs[1]).max()), 0.02)
            self.assertGreater(float(np.abs(outputs[tm] - outputs[2]).max()), 0.01)

    def test_bloom_adds_halo_around_bright_emitter(self) -> None:
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.0, 0.0, 0.0])
            b = scene.create_actor_builder()
            b.add_box_visual(
                half_size=[0.25, 0.25, 0.25],
                material=sapien.render.RenderMaterial(
                    base_color=[0, 0, 0, 1], emission=[8.0, 8.0, 8.0, 1.0], specular_weight=0.0
                ),
            )
            b.build_kinematic()
            cfg = sapien.render.RenderConfig("rt")
            cfg["spp"] = 1
            cfg["toneMapper"] = 1
            cfg.denoiser = "none"
            cam = scene.add_camera("cam", 64, 64, 1.0, 0.1, 20.0, render_config=cfg)
            cam.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
            scene.update_render()
            cam.take_picture()
            no_bloom = cam.get_picture("Color")[..., :3].copy()
            seg = cam.get_picture("Segmentation")[..., 1]
            sky = seg == 0
            self.assertEqual(float(no_bloom[sky].max()), 0.0)

            cam.set_property("bloomIntensity", 1.0)
            cam.set_property("bloomThreshold", 1.0)
            cam.set_property("bloomRadius", 5.0)
            cam.take_picture()
            with_bloom = cam.get_picture("Color")[..., :3].copy()
            self.assertGreater(float(with_bloom[sky].max()), 0.1)

    def test_color_grade_and_lens_effects(self) -> None:
        """Exercise EV exposure, white balance, saturation, ASC CDL, vignette, chromatic
        aberration, lens distortion and sensor noise at runtime on one RT camera."""
        cfg = sapien.render.RenderConfig("rt")
        cfg["spp"] = 1
        cfg["toneMapper"] = 1
        cfg.denoiser = "none"
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.2, 0.2, 0.2])
            scene.add_point_light([-2.0, 1.0, 2.0], [8.0, 8.0, 8.0])
            b = scene.create_actor_builder()
            b.add_box_visual(
                half_size=[0.5, 0.5, 0.5],
                material=sapien.render.RenderMaterial(base_color=[0.8, 0.5, 0.2, 1.0]),
            )
            b.build_kinematic()
            cam = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=cfg)
            cam.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
            scene.update_render()

            def reset_and_capture(**overrides) -> tuple[np.ndarray, np.ndarray]:
                defaults = {
                    "exposure": 1.0,
                    "exposureEV": 0.0,
                    "whiteBalanceTemp": 6500.0,
                    "whiteBalanceTint": 0.0,
                    "contrast": 1.0,
                    "saturation": 1.0,
                    "cdlSlope": 1.0,
                    "cdlOffset": 0.0,
                    "cdlPower": 1.0,
                    "lensDistortionK1": 0.0,
                    "lensDistortionK2": 0.0,
                    "chromaticAberration": 0.0,
                    "vignetteStrength": 0.0,
                    "vignetteSmoothness": 0.5,
                    "filmGrain": 0.0,
                    "sensorReadNoise": 0.0,
                    "sensorShotNoise": 0.0,
                    "noiseSeed": 0,
                }
                defaults.update(overrides)
                for k, v in defaults.items():
                    cam.set_property(k, v)
                cam.take_picture()
                return cam.get_picture("HdrColor")[..., :3].copy(), cam.get_picture("Color")[..., :3].copy()

            hdr_base, base = reset_and_capture()
            hdr_ev, ev_up = reset_and_capture(exposureEV=1.0)
            np.testing.assert_allclose(ev_up, srgb(2.0 * hdr_ev), atol=1e-4)

            hdr_w, warm = reset_and_capture(whiteBalanceTemp=10000.0)
            hdr_c, cool = reset_and_capture(whiteBalanceTemp=3500.0)
            lit = hdr_w.max(axis=-1) > 0.1
            warm_rb = float((warm[lit, 0] / np.maximum(warm[lit, 2], 1e-4)).mean())
            cool_rb = float((cool[lit, 0] / np.maximum(cool[lit, 2], 1e-4)).mean())
            self.assertGreater(warm_rb, cool_rb * 1.3)

            _, grey = reset_and_capture(saturation=0.0)
            np.testing.assert_allclose(grey[..., 0], grey[..., 1], atol=1e-4)
            np.testing.assert_allclose(grey[..., 0], grey[..., 2], atol=1e-4)

            hdr_cdl, cdl = reset_and_capture(cdlSlope=0.5, cdlOffset=0.1, cdlPower=1.5)
            np.testing.assert_allclose(cdl, srgb(np.maximum(hdr_cdl * 0.5 + 0.1, 0.0) ** 1.5), atol=1e-4)

            hdr_vig, vig = reset_and_capture(vignetteStrength=1.2, vignetteSmoothness=1.0)
            ref_vig = srgb(hdr_vig)
            center_ratio = float(vig[SIZE // 2, SIZE // 2].mean() / max(ref_vig[SIZE // 2, SIZE // 2].mean(), 1e-5))
            corner_ratio = float(vig[0, 0].mean() / max(ref_vig[0, 0].mean(), 1e-5))
            self.assertGreater(center_ratio, 0.95)
            self.assertLess(corner_ratio, 0.4)

            hdr_ca, ca = reset_and_capture(chromaticAberration=0.08)
            ref_ca = srgb(hdr_ca)
            self.assertGreater(float(np.abs(ca[..., 0] - ref_ca[..., 0]).max()), 0.1)
            self.assertGreater(float(np.abs(ca[..., 2] - ref_ca[..., 2]).max()), 0.1)

            hdr_d, dist = reset_and_capture(lensDistortionK1=0.6)
            self.assertGreater(float(np.abs(dist - srgb(hdr_d)).max()), 0.1)

            hdr_n0, n0 = reset_and_capture(sensorReadNoise=0.05, noiseSeed=1)
            hdr_n1, n1 = reset_and_capture(sensorReadNoise=0.05, noiseSeed=2)
            self.assertGreater(float(np.std(n0 - srgb(hdr_n0))), 0.02)
            self.assertGreater(float(np.abs((n0 - srgb(hdr_n0)) - (n1 - srgb(hdr_n1))).mean()), 0.02)

    def test_fxaa_smooths_diagonal_edge_and_preserves_flat_interior(self) -> None:
        pack_no_aa = self.pack(["tonemap"], base="default")
        pack_fxaa = self.pack(["tonemap", "fxaa"], base="default")
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.8, 0.8, 0.8])
            b = scene.create_actor_builder()
            b.add_box_visual(
                half_size=[0.5, 0.5, 0.5],
                material=sapien.render.RenderMaterial(base_color=[0.9, 0.9, 0.9, 1.0]),
            )
            box = b.build_kinematic()
            # Rotate by 25 degrees around X so silhouette edges are diagonal
            box.set_pose(sapien.Pose([0.0, 0.0, 0.0], [np.cos(0.22), np.sin(0.22), 0.0, 0.0]))
            c0 = scene.add_camera("c0", 64, 64, 1.0, 0.1, 20.0, render_config=sapien.render.RenderConfig(str(pack_no_aa)))
            c1 = scene.add_camera("c1", 64, 64, 1.0, 0.1, 20.0, render_config=sapien.render.RenderConfig(str(pack_fxaa)))
            for c in (c0, c1):
                c.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
            scene.update_render()
            c0.take_picture()
            c1.take_picture()
            img0 = c0.get_picture("Color")[..., :3]
            img1 = c1.get_picture("Color")[..., :3]
            diff = np.abs(img1 - img0)
            # Edge pixels change while flat center and flat background stay unchanged
            self.assertGreater(float(diff.max()), 0.05)
            np.testing.assert_allclose(img1[30:34, 30:34], img0[30:34, 30:34], atol=1e-5)
            np.testing.assert_allclose(img1[0:4, 0:4], img0[0:4, 0:4], atol=1e-5)

    def test_white_balance_is_a_bradford_adaptation_in_one_lms_space(self) -> None:
        """White balance adapts in one LMS space. At 3500 K a neutral surface comes out in the
        proportions of a Bradford adaptation: red at about 0.47x green. The old code derived the
        scale in Bradford LMS but applied it in Hunt-Pointer-Estevez LMS, which put red at
        about 0.06x green and gave everything a strong cyan cast."""
        grey = np.array([0.25, 0.25, 0.25])
        pictures = {}
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.0, 0.0, 0.0])
            builder = scene.create_actor_builder()
            builder.add_box_visual(
                half_size=[0.5, 0.5, 0.5],
                material=sapien.render.RenderMaterial(
                    base_color=[0.0, 0.0, 0.0, 1.0],
                    emission=[*grey, 1.0],
                    specular_weight=0.0,
                ),
            )
            builder.build_kinematic()
            config = sapien.render.RenderConfig("rt")
            config["spp"] = 1
            config["maxDepth"] = 0
            config["toneMapper"] = 1
            config["exposure"] = 1.0
            camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=config)
            camera.entity.pose = sapien.Pose([-2.0, 0.0, 0.0])
            scene.update_render()
            for temperature in (6500.0, 3500.0, 10000.0):
                camera.set_property("whiteBalanceTemp", temperature)
                camera.set_property("whiteBalanceTint", 0.0)
                camera.take_picture()
                # the box fills the centre: its radiance is the emission, free of path noise
                pictures[temperature] = linear_from_srgb(
                    camera.get_picture("Color")[SIZE // 2, SIZE // 2, :3]
                )

        # 6500 K is the reference white, so the pass is the identity there.
        np.testing.assert_allclose(pictures[6500.0], grey, rtol=1e-4, atol=1e-5)

        for temperature in (3500.0, 10000.0):
            expected = bradford_adapt(grey, temperature)
            np.testing.assert_allclose(pictures[temperature], expected, rtol=2e-3, atol=2e-4)
            # the failing case of the review: the red of grey stays well above 0.3x green
            self.assertGreater(float(pictures[temperature][0]), 0.3 * pictures[temperature][1])

        # above 6500 K the image warms and below it cools, as on a camera: the red-to-blue
        # ratio grows with the setting
        self.assertGreater(
            float(pictures[10000.0][0] / pictures[10000.0][2]),
            float(pictures[3500.0][0] / pictures[3500.0][2]) * 1.3,
        )

    def test_white_balance_tint_sign_convention(self) -> None:
        """Positive whiteBalanceTint shifts neutral grey toward magenta (R, B > G) and negative
        shifts it toward green (G > R, B), matching standard camera tint sliders."""
        grey = np.array([0.25, 0.25, 0.25])
        results = {}
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.0, 0.0, 0.0])
            builder = scene.create_actor_builder()
            builder.add_box_visual(
                half_size=[0.5, 0.5, 0.5],
                material=sapien.render.RenderMaterial(
                    base_color=[0.0, 0.0, 0.0, 1.0],
                    emission=[*grey, 1.0],
                    specular_weight=0.0,
                ),
            )
            builder.build_kinematic()
            config = sapien.render.RenderConfig("rt")
            config["spp"] = 1
            config["maxDepth"] = 0
            config["toneMapper"] = 1
            config["exposure"] = 1.0
            camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=config)
            camera.entity.pose = sapien.Pose([-2.0, 0.0, 0.0])
            scene.update_render()
            for tint in (-0.5, 0.0, 0.5):
                camera.set_property("whiteBalanceTemp", 6500.0)
                camera.set_property("whiteBalanceTint", tint)
                camera.take_picture()
                results[tint] = linear_from_srgb(
                    camera.get_picture("Color")[SIZE // 2, SIZE // 2, :3]
                )

        np.testing.assert_allclose(results[0.0], grey, rtol=1e-4, atol=1e-5)
        # +tint -> magenta (green drops relative to red and blue)
        self.assertGreater(float(results[0.5][0]), float(results[0.5][1]) * 1.05, str(results[0.5]))
        self.assertGreater(float(results[0.5][2]), float(results[0.5][1]) * 1.05, str(results[0.5]))
        # -tint -> green (green rises relative to red and blue)
        self.assertGreater(
            float(results[-0.5][1]), float(results[-0.5][0]) * 1.05, str(results[-0.5])
        )
        self.assertGreater(
            float(results[-0.5][1]), float(results[-0.5][2]) * 1.05, str(results[-0.5])
        )

    def test_an_undeclared_pass_parameter_keeps_its_neutral_default(self) -> None:
        """A pass whose parameters pack.json does not declare renders with the neutral values
        the chain seeds, not with the zero-filled push constants: color_grade with cdlSlope and
        cdlPower 0 rendered black, and fxaa with threshold 0 smoothed a low-contrast edge the
        documented defaults leave alone."""
        cases = {
            "color_grade": (
                ("exposureEV", "whiteBalanceTemp", "whiteBalanceTint", "contrast", "saturation",
                 "cdlSlope", "cdlOffset", "cdlPower"),
                ["color_grade", "tonemap"],
            ),
            "fxaa": (("fxaaEdgeThreshold", "fxaaSubpix"), ["edge", "tonemap", "fxaa"]),
        }
        for pass_name, (names, passes) in cases.items():
            with self.subTest(pass_=pass_name):
                declared = self.render(self.pack(passes))
                pack = self.pack(passes)
                manifest = json.loads((pack / "pack.json").read_text())
                for name in names:
                    del manifest["parameters"][name]
                (pack / "pack.json").write_text(json.dumps(manifest))
                config = sapien.render.RenderConfig(str(pack))
                for name in names:
                    self.assertNotIn(name, config.parameters)
                np.testing.assert_allclose(
                    self.render(pack)["Color"], declared["Color"], atol=1e-6
                )

        # color_grade at its neutral values is the identity: the same picture as without it
        np.testing.assert_allclose(
            self.render(self.pack(["color_grade", "tonemap"]))["Color"],
            self.render(self.pack(["tonemap"]))["Color"],
            atol=1e-6,
        )

        # fxaa at its neutral values leaves the low-contrast edge of `edge` alone; the
        # zero-filled push constants smooth it
        flat = self.render(self.pack(["edge", "tonemap"]))["Color"]
        smoothed = self.render(self.pack(["edge", "tonemap", "fxaa"]))["Color"]
        np.testing.assert_allclose(smoothed, flat, atol=1e-6)
        zeroed = self.pack(["edge", "tonemap", "fxaa"])
        manifest = json.loads((zeroed / "pack.json").read_text())
        manifest["parameters"]["fxaaEdgeThreshold"]["default"] = 0.0
        manifest["parameters"]["fxaaSubpix"]["default"] = 0.0
        (zeroed / "pack.json").write_text(json.dumps(manifest))
        self.assertGreater(float(np.abs(self.render(zeroed)["Color"] - flat).max()), 1e-3)

    def test_lens_effects_are_radial_on_a_non_square_image(self) -> None:
        """Distortion and vignette use the pixel radius, not the uv radius: on a 96x48 image
        the pixel 10 to the right of the centre and the pixel 10 above it are displaced and
        darkened alike. The old code squared a uv x offset twice as large, so those two
        pixels were treated as different radii."""
        width, height, offset, k1, strength = 96, 48, 10, 0.5, 1.5
        pack = self.pack(["uv", "lens"])
        config = sapien.render.RenderConfig(str(pack))
        config["spp"] = 1
        config["lensDistortionK1"] = k1
        config["lensDistortionK2"] = 0.0
        config["chromaticAberration"] = 0.0
        config["vignetteStrength"] = strength
        config["vignetteSmoothness"] = 1.0
        config["filmGrain"] = 0.0
        config["sensorReadNoise"] = 0.0
        config["sensorShotNoise"] = 0.0
        with sapien.Scene() as scene:
            camera = scene.add_camera(
                "camera", width, height, 1.0, 0.1, 20.0, render_config=config
            )
            scene.update_render()
            camera.take_picture()
            picture = camera.get_picture("Color")[..., :3].copy()

        # `uv` wrote (uv, 1), so at each probe the lens pass's r/g is the uv it sampled (the uv
        # field is linear, so the bilinear fetch reproduces it exactly) and b is the vignette
        # factor it multiplied in. Both probes are the same number of pixels from the centre.
        cx, cy = width // 2, height // 2
        measurements = {}
        for horizontal in (True, False):
            pixel = (cx + offset if horizontal else cx, cy if horizontal else cy + offset)
            value = picture[pixel[1], pixel[0], :3].astype(np.float64)
            # the lens pass multiplies all three channels by the vignette factor it wrote in b
            sampled = (value[:2] / value[2] * 2.0 - 1.0) * np.array([width / height, 1.0])
            measurements[horizontal] = (sampled, float(value[2]))
            # the shader's own formula: the distortion moves the sample outwards by
            # 1 + k1 r^2 in the pixel radius r
            uv = (np.array(pixel, dtype=np.float64) + 0.5) / np.array([width, height])
            p = (uv * 2.0 - 1.0) * np.array([width / height, 1.0])
            np.testing.assert_allclose(sampled, p * (1.0 + k1 * float(p @ p)), atol=1e-5)
            expected_vignette = max(1.0 - strength * 0.5 * float(p @ p), 0.0)
            np.testing.assert_allclose(float(value[2]), expected_vignette, atol=1e-5)

        horizontal_sample, horizontal_vignette = measurements[True]
        vertical_sample, vertical_vignette = measurements[False]
        # both probes are the same pixel radius from the centre, so they are displaced by the
        # same amount and darkened alike; only the direction differs
        np.testing.assert_allclose(
            float(np.hypot(*horizontal_sample)), float(np.hypot(*vertical_sample)), rtol=1e-6
        )
        np.testing.assert_allclose(horizontal_vignette, vertical_vignette, rtol=1e-6)
        # in pixels the sample moved outwards by 1 + k1 r^2: r is about 10.5 px here, so the
        # displacement is visibly larger than the probe distance. Without the aspect ratio the
        # horizontal probe would have had r^2 = 0.048 against the vertical 0.192.
        self.assertGreater(float(np.hypot(*horizontal_sample)) * 0.5 * height, offset * 1.04)
        self.assertLess(horizontal_vignette, 0.95)

        # chromatic aberration keeps the documented magnitude: red and blue are sampled at
        # (1 +/- 2 * chromaticAberration) times the green pixel radius. `uv` writes (uv, 1), so
        # the red channel reads back the x of the red sample and green the y of the green one.
        aberration = 0.05
        config["lensDistortionK1"] = 0.0
        config["chromaticAberration"] = aberration
        config["vignetteStrength"] = 0.0
        with sapien.Scene() as scene:
            camera = scene.add_camera(
                "camera", width, height, 1.0, 0.1, 20.0, render_config=config
            )
            scene.update_render()
            camera.take_picture()
            picture = camera.get_picture("Color")[..., :3].copy()
        pixel = (cx + offset, cy)
        value = picture[pixel[1], pixel[0], :3].astype(np.float64)
        # the channels read back uvs: scaling the centred pixel radius by a factor scales the
        # centred uv by the same factor
        centred_uv = np.array([pixel[0] + 0.5, pixel[1] + 0.5]) / np.array([width, height]) * 2 - 1
        np.testing.assert_allclose(
            float(value[0]), 0.5 + 0.5 * float(centred_uv[0]) * (1.0 + 2.0 * aberration),
            atol=1e-5,
        )
        np.testing.assert_allclose(float(value[1]), 0.5 + 0.5 * float(centred_uv[1]), atol=1e-5)
        # the blue sample sits inside the red one: (1 - 2 CA) R = 0.5 at the probe, so the red
        # channel alone shows both the direction and the magnitude
        self.assertGreater(float(value[0]), 0.5 + 0.5 * float(centred_uv[0]))

    def test_bloom_of_a_single_pixel_decreases_monotonically_at_large_radius(self) -> None:
        """A one-pixel emitter at the largest supported radius spreads into a smooth glow:
        every annulus around it is dimmer than the previous one and none is empty. The two-ring
        25-tap gather left gaps between its taps from about radius 5 on."""
        radius = 8
        size = 3 * radius + 4
        pack = self.pack(["delta", "bloom"])
        config = sapien.render.RenderConfig(str(pack))
        config["spp"] = 1
        config["bloomThreshold"] = 1.0
        config["bloomKnee"] = 0.0
        config["bloomIntensity"] = 1.0
        config["bloomRadius"] = float(radius)
        with sapien.Scene() as scene:
            camera = scene.add_camera("camera", size, size, 1.0, 0.1, 20.0, render_config=config)
            scene.update_render()
            camera.take_picture()
            picture = camera.get_picture("Color")[..., 0].copy()

        centre = size // 2
        self.assertGreater(float(picture[centre, centre]), 4.0)
        yy, xx = np.mgrid[0:size, 0:size]
        distance = np.hypot(xx - centre, yy - centre)
        profile = []
        for ring in range(1, radius):
            annulus = (distance >= ring - 0.5) & (distance < ring + 0.5)
            self.assertGreater(int(annulus.sum()), 0)
            self.assertGreater(float(picture[annulus].min()), 0.0)
            profile.append(float(picture[annulus].mean()))
        for inner, outer in zip(profile, profile[1:]):
            self.assertGreater(inner, outer * 1.05)


if __name__ == "__main__":
    unittest.main()
