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


def srgb(linear: np.ndarray) -> np.ndarray:
    """The sRGB encoding of tone mapper 1."""
    linear = np.maximum(linear, 0.0)
    encoded = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    return np.clip(encoded, 0.0, 1.0)


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


if __name__ == "__main__":
    unittest.main()
