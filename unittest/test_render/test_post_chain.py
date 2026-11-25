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
        if "gain" in post:
            manifest["parameters"]["gain"] = {"kind": "uniform", "type": "float", "default": 1.0}
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
        manifest = json.loads((pack / "pack.json").read_text())
        for name in ("exposure", "toneMapper"):  # no pass declares them any more
            del manifest["parameters"][name]
        (pack / "pack.json").write_text(json.dumps(manifest))
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


if __name__ == "__main__":
    unittest.main()
