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

    def test_direct_fuzz_is_not_dimmed_by_a_second_cosine(self) -> None:
        """The fuzz lobe already returns f * cos(wi), so a directional light must see it as is. For
        fuzz roughness 0.5 with the camera on the face normal, the lobe at 85 degrees is 0.37 of
        the one at 60 degrees; a second cosine makes that 0.065, and 0.29 instead of 0.64 in the
        `minimal` pack's gamma-encoded `Color`."""
        oblique = (0.5, 0.0, -0.8660254)  # light 60 degrees off the face normal
        grazing = (0.0871557, 0.0, -0.9961947)  # and 85 degrees
        fuzz = dict(fuzz_weight=1.0, fuzz_color=[1.0, 1.0, 1.0], fuzz_roughness=0.5)
        for pack in PACKS:
            with self.subTest(pack=pack):
                near = float(centre(pack, black(**fuzz), oblique).mean())
                far = float(centre(pack, black(**fuzz), grazing).mean())
                self.assertGreater(near, 0.02, str(near))
                ratio = far / near
                if PICTURE[pack] == "Color":  # only `minimal` is gamma encoded
                    ratio = ratio ** 2.2
                self.assertAlmostEqual(ratio, 0.37, delta=0.12, msg=str((near, far)))

    def test_metal_specular_weight_above_one_is_clamped(self) -> None:
        """A metal's F0 is `base_color * specular_weight`, and the spec's range for that weight is
        [0, 1]: the documented `specular_weight = 2 * specular` migration gave 2, which forward
        raster used to scale a metal's reflectance by 1.8 where the deferred pack clamped it at
        pack time. A rough metal with the light behind the camera puts the lobe's peak on the
        centre pixel (0.43 of the light's radiance at 8-bit 173; unclamped 0.86 and 238)."""
        for pack in PACKS:
            with self.subTest(pack=pack):
                def metal(weight: float) -> np.ndarray:
                    return centre(pack, sapien.render.RenderMaterial(
                        base_color=[0.9, 0.9, 0.9, 1], metallic=1.0, specular_weight=weight,
                        roughness=1.0))
                one = metal(1.0)
                self.assertGreater(float(one.min()), 0.05, str(one))
                np.testing.assert_allclose(metal(2.0), one, atol=1e-3, err_msg=str(one))

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

    def test_zero_coat_ior_renders_finite(self) -> None:
        """coat_ior = 0 must not divide by zero in G-buffer/forward material evaluation."""
        for pack in PACKS:
            with self.subTest(pack=pack):
                pixel = centre(pack, black(coat_weight=1.0, coat_ior=0.0, coat_roughness=0.2))
                self.assertTrue(np.all(np.isfinite(pixel)), f"{pack}: {pixel}")

    def test_inplace_alpha_transition_updates_raster_visibility(self) -> None:
        """Mutating a material's base_color alpha across opaque/translucent/invisible thresholds
        must re-record raster draw lists without rebuilding scene topology."""
        for pack in ("default", "minimal"):
            with self.subTest(pack=pack), sapien.Scene() as scene:
                scene.set_ambient_light([0.8, 0.8, 0.8])
                wall_builder = scene.create_actor_builder()
                wall_builder.add_box_visual(
                    half_size=[0.05, 2.0, 2.0],
                    material=sapien.render.RenderMaterial(
                        base_color=[1.0, 1.0, 1.0, 1.0], specular_weight=0.0, roughness=1.0
                    ),
                )
                wall_builder.build_kinematic().set_pose(sapien.Pose([2.0, 0.0, 0.0]))

                box_mat = sapien.render.RenderMaterial(
                    base_color=[1.0, 0.0, 0.0, 1.0], specular_weight=0.0, roughness=1.0
                )
                box_builder = scene.create_actor_builder()
                box_builder.add_box_visual(half_size=[0.4, 0.4, 0.4], material=box_mat)
                box_builder.build_kinematic().set_pose(sapien.Pose([0.0, 0.0, 0.0]))

                camera = scene.add_camera(
                    "camera",
                    SIZE,
                    SIZE,
                    1.0,
                    0.05,
                    10.0,
                    render_config=sapien.render.RenderConfig(pack),
                )
                camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])

                def sample() -> np.ndarray:
                    scene.update_render()
                    camera.take_picture()
                    return camera.get_picture(PICTURE[pack])[SIZE // 2, SIZE // 2, :3].astype(
                        np.float64
                    )

                opaque = sample()
                self.assertGreater(float(opaque[0]), 0.1)
                self.assertLess(float(opaque[1]), 0.05 * float(opaque[0]))

                box_mat.base_color = [1.0, 0.0, 0.0, 0.0]
                invisible = sample()
                self.assertGreater(float(invisible[1]), 0.1, str(invisible))
                self.assertAlmostEqual(
                    float(invisible[0]), float(invisible[1]), delta=0.05 * float(invisible[1])
                )

                box_mat.base_color = [1.0, 0.0, 0.0, 0.5]
                translucent = sample()
                self.assertGreater(float(translucent[1]), 0.05, str(translucent))
                self.assertGreater(float(translucent[0]), float(translucent[1]) * 1.2)

                box_mat.base_color = [1.0, 0.0, 0.0, 1.0]
                restored = sample()
                np.testing.assert_allclose(restored, opaque, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
