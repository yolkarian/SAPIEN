import unittest
from pathlib import Path
import numpy as np

import sapien


class TestMaterial(unittest.TestCase):
    def test_defaults(self):
        material = sapien.render.RenderMaterial()

        self.assertTrue(np.allclose(material.emission, [0.0, 0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(material.base_color, [1.0, 1.0, 1.0, 1.0]))
        self.assertAlmostEqual(material.base_weight, 1.0)
        self.assertAlmostEqual(material.specular_weight, 1.0)
        self.assertTrue(np.allclose(material.specular_color, [1.0, 1.0, 1.0]))
        self.assertAlmostEqual(material.specular_roughness_anisotropy, 0.0)
        self.assertAlmostEqual(material.roughness, 0.45)
        self.assertAlmostEqual(material.metallic, 0.0)
        self.assertAlmostEqual(material.ior, 1.5)
        self.assertAlmostEqual(material.transmission, 0.0)
        self.assertTrue(np.allclose(material.transmission_color, [1.0, 1.0, 1.0]))
        self.assertAlmostEqual(material.transmission_depth, 0.0)
        self.assertAlmostEqual(material.coat_weight, 0.0)
        self.assertTrue(np.allclose(material.coat_color, [1.0, 1.0, 1.0]))
        self.assertAlmostEqual(material.coat_roughness, 0.0)
        self.assertAlmostEqual(material.coat_ior, 1.6)
        self.assertAlmostEqual(material.fuzz_weight, 0.0)
        self.assertTrue(np.allclose(material.fuzz_color, [1.0, 1.0, 1.0]))
        self.assertAlmostEqual(material.fuzz_roughness, 0.5)
        self.assertFalse(material.thin_walled)

        self.assertIsNone(material.emission_texture)
        self.assertIsNone(material.base_color_texture)
        self.assertIsNone(material.metallic_texture)
        self.assertIsNone(material.roughness_texture)
        self.assertIsNone(material.normal_texture)
        self.assertIsNone(material.transmission_texture)
        self.assertIsNone(material.occlusion_texture)

    def test_creation(self):
        emission = np.random.rand(4) + 1.0
        base_color = np.random.rand(4) + [1, 1, 1, 0]
        specular_weight = 0.8
        roughness = 0.9
        metallic = 0.5
        transmission = 0.3
        ior = 1.3
        mat = sapien.render.RenderMaterial(
            emission,
            base_color,
            specular_weight,
            roughness,
            metallic,
            transmission,
            ior,
        )

        self.assertTrue(np.allclose(emission, mat.emission))
        self.assertTrue(np.allclose(base_color, mat.base_color))
        self.assertTrue(np.allclose(specular_weight, mat.specular_weight))
        self.assertTrue(np.allclose(roughness, mat.roughness))
        self.assertTrue(np.allclose(metallic, mat.metallic))
        self.assertTrue(np.allclose(transmission, mat.transmission))
        self.assertTrue(np.allclose(ior, mat.ior))

    def test_setters_round_trip(self):
        mat = sapien.render.RenderMaterial()

        emission = np.random.rand(4) + 1.0
        base_color = np.random.rand(4) + [1, 1, 1, 0]
        base_weight = 0.9
        specular_weight = 0.7
        specular_color = np.random.rand(3)
        specular_roughness_anisotropy = 0.4
        roughness = 0.8
        metallic = 0.3
        transmission = 0.2
        transmission_color = np.random.rand(3)
        transmission_depth = 0.05
        ior = 1.5
        coat_weight = 0.6
        coat_color = np.random.rand(3)
        coat_roughness = 0.25
        coat_ior = 1.7
        fuzz_weight = 0.35
        fuzz_color = np.random.rand(3)
        fuzz_roughness = 0.65
        thin_walled = True

        mat.set_emission(emission)
        mat.set_base_color(base_color)
        mat.set_base_weight(base_weight)
        mat.set_specular_weight(specular_weight)
        mat.set_specular_color(specular_color)
        mat.set_specular_roughness_anisotropy(specular_roughness_anisotropy)
        mat.set_roughness(roughness)
        mat.set_metallic(metallic)
        mat.set_transmission(transmission)
        mat.set_transmission_color(transmission_color)
        mat.set_transmission_depth(transmission_depth)
        mat.set_ior(ior)
        mat.set_coat_weight(coat_weight)
        mat.set_coat_color(coat_color)
        mat.set_coat_roughness(coat_roughness)
        mat.set_coat_ior(coat_ior)
        mat.set_fuzz_weight(fuzz_weight)
        mat.set_fuzz_color(fuzz_color)
        mat.set_fuzz_roughness(fuzz_roughness)
        mat.set_thin_walled(thin_walled)

        self.assertTrue(np.allclose(emission, mat.get_emission()))
        self.assertTrue(np.allclose(base_color, mat.get_base_color()))
        self.assertTrue(np.allclose(base_weight, mat.get_base_weight()))
        self.assertTrue(np.allclose(specular_weight, mat.get_specular_weight()))
        self.assertTrue(np.allclose(specular_color, mat.get_specular_color()))
        self.assertTrue(
            np.allclose(
                specular_roughness_anisotropy,
                mat.get_specular_roughness_anisotropy(),
            )
        )
        self.assertTrue(np.allclose(roughness, mat.get_roughness()))
        self.assertTrue(np.allclose(metallic, mat.get_metallic()))
        self.assertTrue(np.allclose(transmission, mat.get_transmission()))
        self.assertTrue(np.allclose(transmission_color, mat.get_transmission_color()))
        self.assertTrue(np.allclose(transmission_depth, mat.get_transmission_depth()))
        self.assertTrue(np.allclose(ior, mat.get_ior()))
        self.assertTrue(np.allclose(coat_weight, mat.get_coat_weight()))
        self.assertTrue(np.allclose(coat_color, mat.get_coat_color()))
        self.assertTrue(np.allclose(coat_roughness, mat.get_coat_roughness()))
        self.assertTrue(np.allclose(coat_ior, mat.get_coat_ior()))
        self.assertTrue(np.allclose(fuzz_weight, mat.get_fuzz_weight()))
        self.assertTrue(np.allclose(fuzz_color, mat.get_fuzz_color()))
        self.assertTrue(np.allclose(fuzz_roughness, mat.get_fuzz_roughness()))
        self.assertTrue(thin_walled == mat.get_thin_walled())

        self.assertTrue(np.allclose(emission, mat.emission))
        self.assertTrue(np.allclose(base_color, mat.base_color))
        self.assertTrue(np.allclose(base_weight, mat.base_weight))
        self.assertTrue(np.allclose(specular_weight, mat.specular_weight))
        self.assertTrue(np.allclose(specular_color, mat.specular_color))
        self.assertTrue(
            np.allclose(
                specular_roughness_anisotropy, mat.specular_roughness_anisotropy
            )
        )
        self.assertTrue(np.allclose(roughness, mat.roughness))
        self.assertTrue(np.allclose(metallic, mat.metallic))
        self.assertTrue(np.allclose(transmission, mat.transmission))
        self.assertTrue(np.allclose(transmission_color, mat.transmission_color))
        self.assertTrue(np.allclose(transmission_depth, mat.transmission_depth))
        self.assertTrue(np.allclose(ior, mat.ior))
        self.assertTrue(np.allclose(coat_weight, mat.coat_weight))
        self.assertTrue(np.allclose(coat_color, mat.coat_color))
        self.assertTrue(np.allclose(coat_roughness, mat.coat_roughness))
        self.assertTrue(np.allclose(coat_ior, mat.coat_ior))
        self.assertTrue(np.allclose(fuzz_weight, mat.fuzz_weight))
        self.assertTrue(np.allclose(fuzz_color, mat.fuzz_color))
        self.assertTrue(np.allclose(fuzz_roughness, mat.fuzz_roughness))
        self.assertTrue(thin_walled == mat.thin_walled)

    def test_texture(self):
        mat = sapien.render.RenderMaterial()

        color_tex = sapien.render.RenderTexture2D(
            str(
                Path(".") / "assets" / "texture" / "Metal" / "Metal038_1K-JPG_Color.jpg"
            )
        )
        normal_tex = sapien.render.RenderTexture2D(
            str(
                Path(".")
                / "assets"
                / "texture"
                / "Metal"
                / "Metal038_1K-JPG_NormalGL.jpg"
            ),
            srgb=False,
        )
        roughness_tex = sapien.render.RenderTexture2D(
            str(
                Path(".")
                / "assets"
                / "texture"
                / "Metal"
                / "Metal038_1K-JPG_Roughness.jpg"
            ),
            srgb=False,
        )
        metallic_tex = sapien.render.RenderTexture2D(
            str(
                Path(".")
                / "assets"
                / "texture"
                / "Metal"
                / "Metal038_1K-JPG_Metalness.jpg"
            ),
            srgb=False,
        )

        emission_tex = sapien.render.RenderTexture2D(
            np.array([[1]], dtype=np.float32), "R32Sfloat"
        )
        transmission_tex = sapien.render.RenderTexture2D(
            np.array([[0]], dtype=np.float32), "R32Sfloat"
        )
        occlusion_tex = sapien.render.RenderTexture2D(
            np.array([[1]], dtype=np.float32), "R32Sfloat"
        )

        # property
        mat.base_color_texture = color_tex
        mat.normal_texture = normal_tex
        mat.metallic_texture = metallic_tex
        mat.roughness_texture = roughness_tex
        mat.emission_texture = emission_tex
        mat.transmission_texture = transmission_tex
        mat.occlusion_texture = occlusion_tex
        self.assertEqual(mat.base_color_texture, color_tex)
        self.assertEqual(mat.normal_texture, normal_tex)
        self.assertEqual(mat.metallic_texture, metallic_tex)
        self.assertEqual(mat.roughness_texture, roughness_tex)
        self.assertEqual(mat.emission_texture, emission_tex)
        self.assertEqual(mat.transmission_texture, transmission_tex)
        self.assertEqual(mat.occlusion_texture, occlusion_tex)

        # get set
        mat.set_base_color_texture(None)
        self.assertIsNone(mat.get_base_color_texture())
        mat.set_normal_texture(None)
        self.assertIsNone(mat.get_normal_texture())
        mat.set_metallic_texture(None)
        self.assertIsNone(mat.get_metallic_texture())
        mat.set_roughness_texture(None)
        self.assertIsNone(mat.get_roughness_texture())
        mat.set_emission_texture(None)
        self.assertIsNone(mat.get_emission_texture())
        mat.set_transmission_texture(None)
        self.assertIsNone(mat.get_transmission_texture())
        mat.set_occlusion_texture(None)
        self.assertIsNone(mat.get_occlusion_texture())
