"""Regression tests for the scene-wide raster descriptor set (common/scene.glsl)."""

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

import numpy as np
import sapien

try:
    import torch
except ImportError:  # pragma: no cover - depends on the test environment
    torch = None

SIZE = 48
# Columns of bodies at y = +0.4 / -0.4 seen from the camera below (1.0 rad fovy, 3 m away).
LEFT, RIGHT = SIZE // 2 - 6, SIZE // 2 + 6


def _scene_with_camera(
    render_config: sapien.render.RenderConfig | None = None,
) -> tuple[sapien.Scene, sapien.render.RenderCameraComponent]:
    """A camera at x = -3 looking along +x at the origin."""
    scene = sapien.Scene()
    scene.set_ambient_light([0.5, 0.5, 0.5])
    camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.05, 10, render_config=render_config)
    camera.entity.pose = sapien.Pose([-3, 0, 0])
    return scene, camera


def _box(
    scene: sapien.Scene,
    half_size: float,
    material: sapien.render.RenderMaterial | Sequence[float],
    position: Sequence[float] = (0.0, 0.0, 0.0),
) -> sapien.render.RenderBodyComponent:
    """A kinematic box visual; returns its render body."""
    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=[half_size] * 3, material=material)
    actor = builder.build_kinematic()
    actor.set_pose(sapien.Pose(position))
    return actor.find_component_by_type(sapien.render.RenderBodyComponent)


def _solid_texture(rgba: Sequence[int]) -> sapien.render.RenderTexture:
    """A 2x2 single-colour 2D texture."""
    pixels = np.empty((2, 2, 4), np.uint8)
    pixels[:] = rgba
    return sapien.render.RenderTexture(pixels, 2, "R8G8B8A8Unorm")


def _render(scene: sapien.Scene, camera: sapien.render.RenderCameraComponent) -> np.ndarray:
    scene.update_render()
    camera.take_picture()
    return camera.get_picture("Color")[..., :3]


class TestGpuScene(unittest.TestCase):
    def _custom_pack(self, struct: str, shading: str) -> sapien.render.RenderConfig:
        """A private copy of the `trivial` pack that declares CustomObjectData as `struct` and
        computes its colour with `shading` (GLSL statements assigning outColor)."""
        root = Path(sapien.__file__).parent / "vulkan_shader"
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        pack = Path(directory.name) / "pack"
        shutil.copytree(root / "trivial", pack)
        shutil.copytree(root / "common", Path(directory.name) / "common")
        for name in ("gbuffer.vert", "gbuffer.frag"):
            lines = (pack / name).read_text().splitlines(keepends=True)
            lines.insert(1, f"#define SAPIEN_CUSTOM_OBJECT_DATA {struct}\n")
            (pack / name).write_text("".join(lines))
        frag = pack / "gbuffer.frag"
        marker = "outColor = material.baseColor;"
        self.assertIn(marker, frag.read_text())
        frag.write_text(frag.read_text().replace(marker, shading))
        return sapien.render.RenderConfig(str(pack))

    def test_material_edit_after_first_frame(self) -> None:
        scene, camera = _scene_with_camera()
        self.addCleanup(scene.close)
        material = sapien.render.RenderMaterial(base_color=[1, 0, 0, 1])
        _box(scene, 0.8, material)
        self.assertGreater(float(_render(scene, camera)[SIZE // 2, SIZE // 2, 0]), 0.3)
        material.base_color = [0, 0, 1, 1]
        color = _render(scene, camera)[SIZE // 2, SIZE // 2]
        self.assertGreater(float(color[2]), float(color[0]) + 0.2)

    def test_more_materials_than_max_num_materials(self) -> None:
        # materials[] starts with max_num_materials (128) entries and grows with the scene.
        scene, camera = _scene_with_camera()
        self.addCleanup(scene.close)
        for i in range(150):
            _box(scene, 0.01, [i / 150, 0, 0, 1], (0, 0, 20))  # out of view
        _box(scene, 0.8, [0, 0, 1, 1])
        color = _render(scene, camera)[SIZE // 2, SIZE // 2]
        self.assertGreater(float(color[2]), float(color[0]) + 0.2)

    def test_many_distinct_textures(self) -> None:
        scene, camera = _scene_with_camera()
        self.addCleanup(scene.close)
        for i in range(24):
            pixels = np.zeros((2, 2, 4), np.uint8)
            pixels[:] = [(i * 10) % 256, (i * 37) % 256, (i * 71) % 256, 255]
            material = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1])
            material.base_color_texture = sapien.render.RenderTexture2D(pixels, "R8G8B8A8Unorm")
            _box(scene, 0.2, material, (0, -1.0 + 0.3 * (i % 8), -0.5 + 0.3 * (i // 8)))
        self.assertGreater(float(np.std(_render(scene, camera))), 0.02)

    def test_custom_object_property_reaches_shader(self) -> None:
        config = self._custom_pack(
            "struct CustomObjectData { float gain; };",
            "outColor = vec4(customObjectData[instance.customDataIndex].gain, 0, 0, 1);",
        )
        scene, camera = _scene_with_camera(config)
        self.addCleanup(scene.close)
        _box(scene, 0.8, [1, 1, 1, 1]).set_property("gain", 0.75)
        self.assertAlmostEqual(float(_render(scene, camera)[SIZE // 2, SIZE // 2, 0]), 0.75,
                               delta=0.02)

    def test_custom_object_texture_reaches_shader(self) -> None:
        config = self._custom_pack(
            "struct CustomObjectData { int decal; };",
            "outColor = texture(textures[nonuniformEXT("
            "customObjectData[instance.customDataIndex].decal)], inUV);",
        )
        scene, camera = _scene_with_camera(config)
        self.addCleanup(scene.close)
        body = _box(scene, 0.8, [1, 1, 1, 1])
        # an unset texture field is slot 0, the white default texture
        np.testing.assert_allclose(_render(scene, camera)[SIZE // 2, SIZE // 2], [1, 1, 1],
                                   atol=0.01)
        body.set_texture("decal", _solid_texture([0, 255, 0, 255]))  # after the first frame
        np.testing.assert_allclose(_render(scene, camera)[SIZE // 2, SIZE // 2], [0, 1, 0],
                                   atol=0.01)

    def test_custom_texture_array_and_padded_stride(self) -> None:
        # 28 bytes are declared, but customObjectData[] elements are 32 bytes apart: the second
        # body reads its own fields only when the upload uses the padded stride.
        config = self._custom_pack(
            "struct CustomObjectData { int decals[2]; vec3 tint; };",
            "CustomObjectData data = customObjectData[instance.customDataIndex];\n"
            "  outColor = vec4(data.tint, 1) * texture(textures[nonuniformEXT(data.decals[1])], inUV);",
        )
        scene, camera = _scene_with_camera(config)
        self.addCleanup(scene.close)
        white, green = _solid_texture([255] * 4), _solid_texture([0, 255, 0, 255])
        left = _box(scene, 0.3, [1, 1, 1, 1], (0, 0.4, 0))
        left.set_property("tint", [1, 1, 0])
        left.set_texture_array("decals", [white, green])
        right = _box(scene, 0.3, [1, 1, 1, 1], (0, -0.4, 0))
        right.set_property("tint", [0, 0, 1])
        right.set_texture_array("decals", [green, white])
        color = _render(scene, camera)
        np.testing.assert_allclose(color[SIZE // 2, LEFT], [0, 1, 0], atol=0.01)
        np.testing.assert_allclose(color[SIZE // 2, RIGHT], [0, 0, 1], atol=0.01)

    def test_vertex_color_pack_renders_vertex_colors(self) -> None:
        # A quad at x = 0 spanning y, z in [-1, 1]: red at (-1, -1), blue at (1, 1). 10 pixels
        # from the centre is (y, z) = -+0.68, where the nearer corner weighs 0.68.
        scene, camera = _scene_with_camera(sapien.render.RenderConfig("vertex_color"))
        self.addCleanup(scene.close)
        builder = scene.create_actor_builder()
        builder.add_visual_from_file("../assets/models/vertex_color.obj")
        builder.build_kinematic()
        color = _render(scene, camera)
        np.testing.assert_allclose(color[SIZE // 2 + 10, SIZE // 2 + 10], [0.84, 0.32, 0], atol=0.05)
        np.testing.assert_allclose(color[SIZE // 2 - 10, SIZE // 2 - 10], [0.16, 0.32, 0.68], atol=0.05)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(),
                         "requires a CUDA device and CUDA-enabled Torch")
    def test_material_edit_after_gpu_init(self) -> None:
        # Grouped cameras get no per-frame CPU upload; material edits must still reach them.
        scene, camera = _scene_with_camera()
        self.addCleanup(scene.close)
        material = sapien.render.RenderMaterial(base_color=[1, 0, 0, 1])
        _box(scene, 0.8, material)
        group = sapien.render.RenderSystemGroup([scene.render_system])
        self.addCleanup(group.close)
        cameras = group.create_camera_group([camera], ["Color"])
        group.gpu_init()

        def center() -> "torch.Tensor":
            group.update_render()
            cameras.take_picture()
            picture = torch.from_dlpack(cameras.get_picture_cuda("Color"))
            torch.cuda.synchronize(picture.device)
            return picture[0, SIZE // 2, SIZE // 2, :3].float().cpu()

        before = center()
        self.assertGreater(float(before[0]), float(before[2]))
        material.base_color = [0, 0, 1, 1]
        after = center()
        self.assertGreater(float(after[2]), float(after[0]))


if __name__ == "__main__":
    unittest.main()
