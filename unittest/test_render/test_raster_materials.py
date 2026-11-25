"""The OpenPBR material in the rasterization packs (common/raster_material.glsl): the coat and the
fuzz layered over the packs' base BRDF, the specular colour and the occlusion texture, in a
deferred pack (`default`, which carries them in the `Material` G-buffer target), a forward pack
(`minimal`) and `point` (deferred, with its own image-based light). The deferred packs are read
before tone mapping (`Lighting`); `minimal` only has its gamma-encoded `Color`."""

import unittest
from typing import Sequence

import numpy as np
import sapien

SIZE = 32
PICTURE = {"default": "Lighting", "minimal": "Color", "point": "Lighting"}
PACKS = tuple(PICTURE)


def centre(pack: str, material: sapien.render.RenderMaterial,
           light_direction: Sequence[float] | None = (1.0, 0.0, 0.0),
           ambient: float = 0.0) -> np.ndarray:
    """Colour of the picture centre: a box face seen straight on from x = -3, lit by a
    directional light (by default from behind the camera) and the ambient light."""
    with sapien.Scene() as scene:
        scene.set_ambient_light([ambient] * 3)
        builder = scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.5] * 3, material=material)
        builder.build_kinematic()
        if light_direction is not None:
            scene.add_directional_light(list(light_direction), [6.0, 6.0, 6.0], shadow=False)
        camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.05, 10.0,
                                  render_config=sapien.render.RenderConfig(pack))
        camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
        scene.update_render()
        camera.take_picture()
        return camera.get_picture(PICTURE[pack])[SIZE // 2, SIZE // 2, :3].astype(np.float64)


def black(**fields) -> sapien.render.RenderMaterial:
    """A black base without base specular, so only the layers can reflect."""
    material = sapien.render.RenderMaterial(base_color=[0.0, 0.0, 0.0, 1.0], specular_weight=0.0)
    for name, value in fields.items():
        setattr(material, name, value)
    return material


class TestRasterLayers(unittest.TestCase):
    def test_the_coat_reflects_over_a_black_base(self) -> None:
        for pack in PACKS:
            with self.subTest(pack=pack):
                bare = centre(pack, black())
                coated = centre(pack, black(coat_weight=1.0, coat_roughness=0.3))
                self.assertGreater(float((coated - bare).min()), 0.2, str((bare, coated)))

    def test_the_fuzz_reflects_over_a_black_base(self) -> None:
        # the sheen lobe vanishes where the half vector is the normal: light at 45 degrees
        grazing = (0.7071068, 0.0, -0.7071068)
        for pack in PACKS:
            with self.subTest(pack=pack):
                bare = centre(pack, black(), grazing)
                fuzzy = centre(pack, black(fuzz_weight=1.0, fuzz_color=[1.0, 1.0, 1.0],
                                           fuzz_roughness=0.5), grazing)
                self.assertGreater(float((fuzzy - bare).min()), 0.05, str((bare, fuzzy)))

    def test_specular_colour_tints_the_dielectric_highlight(self) -> None:
        for pack in PACKS:
            with self.subTest(pack=pack):
                bare = centre(pack, black(specular_weight=1.0, roughness=0.3,
                                          specular_color=[0.0, 0.0, 0.0]))
                tinted = centre(pack, black(specular_weight=1.0, roughness=0.3,
                                            specular_color=[1.0, 0.0, 0.0])) - bare
                self.assertGreater(float(tinted[0]), 0.1, str(tinted))
                self.assertLess(float(abs(tinted[1:]).max()), 0.05 * float(tinted[0]), str(tinted))

    def test_the_occlusion_texture_darkens_ambient_light_only(self) -> None:
        pixels = np.full((2, 2, 4), 64, np.uint8)
        pixels[..., 3] = 255
        occlusion = sapien.render.RenderTexture2D(pixels, "R8G8B8A8Unorm")
        for pack in PACKS:
            with self.subTest(pack=pack):
                plain = sapien.render.RenderMaterial(base_color=[1.0, 1.0, 1.0, 1.0])
                occluded = sapien.render.RenderMaterial(base_color=[1.0, 1.0, 1.0, 1.0])
                occluded.occlusion_texture = occlusion
                ambient = centre(pack, plain, None, ambient=0.5)
                self.assertGreater(float(ambient.min()), 0.1)
                self.assertLess(float(centre(pack, occluded, None, ambient=0.5).max()),
                                0.8 * float(ambient.min()))
                # direct light is not occluded
                np.testing.assert_allclose(centre(pack, occluded), centre(pack, plain), atol=1e-3)


if __name__ == "__main__":
    unittest.main()
