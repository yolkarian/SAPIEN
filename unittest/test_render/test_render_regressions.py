"""Regression tests for renderer defects that produced wrong images without raising."""

import base64
import json
import shutil
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
import sapien

SIZE = 48


def write_constant_exr(path: Path, width: int, height: int, value: float) -> None:
    """Write an uncompressed 32-bit float RGB scanline OpenEXR image of one constant value."""

    def attribute(name: str, type_name: str, payload: bytes) -> bytes:
        return (
            name.encode() + b"\0" + type_name.encode() + b"\0"
            + struct.pack("<i", len(payload)) + payload
        )

    # channel: name, pixel type (2 = float), pLinear + 3 reserved bytes, x/y sampling
    channels = b"".join(
        name + b"\0" + struct.pack("<iBxxxii", 2, 0, 1, 1) for name in (b"B", b"G", b"R")
    ) + b"\0"
    window = struct.pack("<iiii", 0, 0, width - 1, height - 1)
    header = (
        attribute("channels", "chlist", channels)
        + attribute("compression", "compression", b"\0")
        + attribute("dataWindow", "box2i", window)
        + attribute("displayWindow", "box2i", window)
        + attribute("lineOrder", "lineOrder", b"\0")
        + attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0))
        + attribute("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
        + attribute("screenWindowWidth", "float", struct.pack("<f", 1.0))
        + b"\0"
    )
    row = np.full(3 * width, value, np.float32).tobytes()
    line_size = 8 + len(row)
    data_start = 8 + len(header) + 8 * height
    with open(path, "wb") as file:
        file.write(struct.pack("<ii", 20000630, 2))
        file.write(header)
        for y in range(height):
            file.write(struct.pack("<Q", data_start + y * line_size))
        for y in range(height):
            file.write(struct.pack("<ii", y, len(row)) + row)


def png_1x1(rgba: list[int]) -> bytes:
    """A 1x1 RGBA8 PNG image of colour `rgba`."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    pixels = zlib.compress(b"\0" + bytes(rgba))  # one scanline, filter type 0
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels)
            + chunk(b"IEND", b""))


# a Collada triangle whose Phong effect has a grey diffuse colour and an ambient texture
COLLADA_AMBIENT_TEXTURE = """<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><up_axis>Z_UP</up_axis></asset>
  <library_images><image id="ambient-image"><init_from>ambient.png</init_from></image></library_images>
  <library_effects>
    <effect id="effect">
      <profile_COMMON>
        <newparam sid="ambient-surface">
          <surface type="2D"><init_from>ambient-image</init_from></surface>
        </newparam>
        <newparam sid="ambient-sampler">
          <sampler2D><source>ambient-surface</source></sampler2D>
        </newparam>
        <technique sid="common">
          <phong>
            <ambient><texture texture="ambient-sampler" texcoord="UVMap"/></ambient>
            <diffuse><color>0.5 0.5 0.5 1</color></diffuse>
          </phong>
        </technique>
      </profile_COMMON>
    </effect>
  </library_effects>
  <library_materials>
    <material id="material"><instance_effect url="#effect"/></material>
  </library_materials>
  <library_geometries>
    <geometry id="mesh">
      <mesh>
        <source id="positions">
          <float_array id="positions-array" count="9">0 0 0 1 0 0 0 1 0</float_array>
          <technique_common>
            <accessor source="#positions-array" count="3" stride="3">
              <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="vertices"><input semantic="POSITION" source="#positions"/></vertices>
        <triangles material="material" count="1">
          <input semantic="VERTEX" source="#vertices" offset="0"/>
          <p>0 1 2</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="scene">
      <node id="node">
        <instance_geometry url="#mesh">
          <bind_material>
            <technique_common>
              <instance_material symbol="material" target="#material"/>
            </technique_common>
          </bind_material>
        </instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>
"""


class RayTracingCase(unittest.TestCase):
    """Cameras of the ray-tracing shader pack."""

    def setUp(self) -> None:
        self.render_config = sapien.render.RenderConfig("rt")
        self.render_config["spp"] = 1
        self.render_config.denoiser = "none"


class TestRasterLightCapacity(unittest.TestCase):
    def test_black_lights_add_no_phantom_light(self) -> None:
        """When the default pack held 10 point lights, an 11th made the lighting loop read the
        textured-light slot as a point light whose colour is that light's direction. Black
        lights, textured or not, must add nothing."""
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.0, 0.0, 0.0])
            scene.add_ground(0.0)
            camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0)
            # look straight down at the ground below the textured light
            camera.entity.pose = sapien.Pose([0.0, 0.0, 2.0], [0.7071068, 0.0, 0.7071068, 0.0])

            for _ in range(11):
                scene.add_point_light([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])

            # a black textured light pointing along +x one metre above the ground
            light = sapien.render.RenderTexturedLightComponent()
            light.texture = sapien.render.RenderTexture2D(
                np.ones((4, 4), np.float32), "R32Sfloat"
            )
            light.color = [0.0, 0.0, 0.0]
            entity = sapien.Entity()
            entity.add_component(light)
            entity.pose = sapien.Pose([0.0, 0.0, 1.0])
            scene.add_entity(entity)

            scene.update_render()
            camera.take_picture()
            color = camera.get_picture("Color")[..., :3]

            self.assertLess(float(color.max()), 0.02)


class TestEnvironmentPrefilter(unittest.TestCase):
    def test_hdr_environment_keeps_radiance_above_one(self) -> None:
        """Rough reflections read prefiltered mip levels, which used to be clamped to [0, 1]."""
        radiance = 4.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "constant.exr"
            write_constant_exr(path, 64, 32, radiance)
            with sapien.Scene() as scene:
                scene.set_ambient_light([0.0, 0.0, 0.0])
                scene.set_environment_map(str(path))
                camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0)
                camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])

                material = sapien.render.RenderMaterial(
                    base_color=[1.0, 1.0, 1.0, 1.0], metallic=1.0, roughness=1.0
                )
                builder = scene.create_actor_builder()
                builder.add_sphere_visual(radius=1.0, material=material)
                builder.build_kinematic(name="sphere")

                scene.update_render()
                camera.take_picture()
                lighting = camera.get_picture("Lighting")[SIZE // 2, SIZE // 2, :3]

        # The reflection is the prefiltered radiance times the BRDF's directional albedo, which
        # is at most 1 (about 0.36 for single-scattering GGX at roughness 1). A prefilter
        # clamped to [0, 1] therefore cannot exceed 1; the unclamped one gives about 1.44.
        self.assertGreater(float(lighting.min()), 1.0)
        self.assertLess(float(lighting.max()), radiance)


class TestRayTracingPointCloud(RayTracingCase):
    def test_point_clouds_render_beside_meshes(self) -> None:
        """Point sets were looked up by TLAS instance index, which counts the meshes first."""
        with sapien.Scene() as scene:
            scene.set_ambient_light([1.0, 1.0, 1.0])
            camera = scene.add_camera(
                "camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=self.render_config
            )
            camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])

            builder = scene.create_actor_builder()
            builder.add_box_visual(half_size=[0.1, 0.1, 0.1])
            builder.build_kinematic(name="box").set_pose(sapien.Pose([0.0, 5.0, 5.0]))

            def add_cloud(position: list[float], color: list[float]) -> None:
                cloud = sapien.render.RenderPointCloudComponent(1)
                cloud.set_vertices(np.array([position], np.float32))
                cloud.set_attribute("color", np.array([color], np.float32))
                cloud.set_attribute("scale", np.array([0.3], np.float32))
                entity = sapien.Entity()
                entity.add_component(cloud)
                scene.add_entity(entity)

            add_cloud([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0])  # view center
            add_cloud([0.0, 5.0, -5.0], [0.0, 1.0, 0.0, 1.0])  # out of view

            scene.update_render()
            camera.take_picture()
            albedo = camera.get_picture("Albedo")[SIZE // 2, SIZE // 2, :3]

        self.assertGreater(float(albedo[0]), 0.8)
        self.assertLess(float(albedo[1]), 0.2)


class TestRayTracingSampling(RayTracingCase):
    def _lit_scene(self, scene: sapien.Scene) -> sapien.render.RenderCameraComponent:
        scene.set_ambient_light([1.0, 1.0, 1.0])
        scene.add_ground(0.0)
        camera = scene.add_camera(
            "camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=self.render_config
        )
        camera.entity.pose = sapien.Pose([0.0, 0.0, 2.0], [0.7071068, 0.0, 0.7071068, 0.0])
        return camera

    def test_noise_changes_when_accumulation_restarts(self) -> None:
        """Every restart reused one random sequence, so moving scenes showed a fixed pattern."""
        with sapien.Scene() as scene:
            camera = self._lit_scene(scene)

            def capture() -> np.ndarray:
                # writing the pose restarts accumulation, as a moving scene does every frame
                camera.entity.pose = camera.entity.pose
                scene.update_render()
                camera.take_picture()
                return camera.get_picture("HdrColor")[..., :3].copy()

            first, second = capture(), capture()

        self.assertGreater(float(first.std()), 1e-3)  # the estimate is noisy at 1 spp
        self.assertGreater(float(np.abs(first - second).mean()), 1e-3)

    def test_property_change_restarts_accumulation(self) -> None:
        """Samples taken before a property edit kept their weight in the accumulated image."""
        with sapien.Scene() as scene:
            camera = self._lit_scene(scene)

            def capture() -> float:
                scene.update_render()
                camera.take_picture()
                return float(camera.get_picture("HdrColor")[..., :3].mean())

            # with no bounce the ambient light never reaches the ground
            camera.set_property("maxDepth", 0)
            dark = [capture() for _ in range(8)][-1]
            camera.set_property("maxDepth", 4)
            lit = capture()

        self.assertLess(dark, 1e-3)
        # one fresh frame (about 0.28 here); blended with the eight dark frames it would be 0.03
        self.assertGreater(lit, 0.15)


class TestShaderReload(unittest.TestCase):
    """Uses a private copy of the `trivial` pack and the shared `common` shader directory."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        shader_root = Path(sapien.__file__).parent / "vulkan_shader"
        self.pack = Path(self.directory.name) / "pack"
        shutil.copytree(shader_root / "trivial", self.pack)
        # trivial now includes common/scene.glsl via "../common/..."; stage that directory too.
        shutil.copytree(shader_root / "common", Path(self.directory.name) / "common")
        self.addCleanup(sapien.render.clear_cache, False, False, True)

    def _picture(self, name: str) -> np.ndarray:
        """Render a red box with the private pack and return its picture `name`."""
        with sapien.Scene() as scene:
            camera = scene.add_camera(
                "camera",
                SIZE,
                SIZE,
                1.0,
                0.1,
                20.0,
                render_config=sapien.render.RenderConfig(str(self.pack)),
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
            return camera.get_picture(name).copy()

    def _center_color(self) -> np.ndarray:
        return self._picture("Color")[SIZE // 2, SIZE // 2, :3]

    def test_edited_shader_is_used_after_clearing_the_shader_cache(self) -> None:
        np.testing.assert_allclose(self._center_color(), [1.0, 0.0, 0.0], atol=1e-3)

        shader = self.pack / "gbuffer.frag"
        text = shader.read_text()
        marker = "outColor = material.baseColor;"
        self.assertIn(marker, text)
        shader.write_text(text.replace(marker, "outColor = material.baseColor.gbra;"))

        sapien.render.clear_cache(models=False, images=False, shaders=True)
        np.testing.assert_allclose(self._center_color(), [0.0, 0.0, 1.0], atol=1e-3)

    def test_compile_error_names_file_and_line(self) -> None:
        shader = self.pack / "gbuffer.frag"
        lines = shader.read_text().splitlines()
        line = next(i for i, text in enumerate(lines) if "outPosition = inPosition;" in text)
        lines[line] = "  outPosition = notDeclared;"
        shader.write_text("\n".join(lines) + "\n")

        with self.assertRaisesRegex(RuntimeError, rf"gbuffer\.frag:{line + 1}: 'notDeclared'"):
            self._center_color()

    def test_scalar_integer_output_renders(self) -> None:
        """A scalar uint output got the R32Uint format, which render targets did not accept."""
        shader = self.pack / "gbuffer.frag"
        text = shader.read_text()
        declaration = "layout(location = 2) out uvec4 outSegmentation;"
        assignment = "outSegmentation = inSeg;"
        self.assertIn(declaration, text)
        self.assertIn(assignment, text)
        text = text.replace(declaration, declaration + "\nlayout(location = 3) out uint outLabel;")
        shader.write_text(text.replace(assignment, assignment + "\n  outLabel = 7u;"))

        label = self._picture("Label")
        self.assertEqual(label.dtype, np.uint32)
        self.assertEqual(label.shape, (SIZE, SIZE))
        self.assertEqual(int(label[SIZE // 2, SIZE // 2]), 7)
        self.assertEqual(int(label[0, 0]), 0)


class TestShadowCatcher(unittest.TestCase):
    def test_bodies_without_shadows_stay_visible_through_glass(self) -> None:
        """The ray that decides whether a shadow_catcher path goes on traced only shadow
        casters, so a body with cast_shadow=False vanished behind glass and from reflections."""
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.5, 0.5, 0.5])
            scene.add_ground(0.0)  # the first render shape, id 1, is the catcher
            builder = scene.create_actor_builder()
            builder.add_box_visual(
                half_size=[0.3, 0.3, 0.3],
                material=sapien.render.RenderMaterial(
                    base_color=[1.0, 0.0, 0.0, 1.0], specular_weight=0.0, roughness=1.0
                ),
            )
            red = builder.build_kinematic()
            red.set_pose(sapien.Pose([1.5, 0.0, 0.4]))
            red.find_component_by_type(sapien.render.RenderBodyComponent).cast_shadow = False
            builder = scene.create_actor_builder()
            builder.add_box_visual(
                half_size=[0.02, 1.0, 1.0],
                material=sapien.render.RenderMaterial(
                    base_color=[1.0, 1.0, 1.0, 1.0], roughness=0.0, transmission=1.0
                ),
            )
            builder.build_kinematic().set_pose(sapien.Pose([0.6, 0.0, 0.4]))

            config = sapien.render.RenderConfig("shadow_catcher")
            config["spp"] = 16
            config.denoiser = "none"
            camera = scene.add_camera("camera", SIZE, SIZE, 0.6, 0.1, 20.0, render_config=config)
            camera.entity.pose = sapien.Pose([-0.5, 0.0, 0.4])
            scene.update_render()
            camera.take_picture()
            centre = camera.get_picture("HdrColor")[
                SIZE // 2 - 3 : SIZE // 2 + 3, SIZE // 2 - 3 : SIZE // 2 + 3, :3
            ].mean(axis=(0, 1))
        # the red box seen through the glass, not the grey ambient light behind it
        self.assertGreater(float(centre[0]), 0.2)
        self.assertGreater(float(centre[0]), 3.0 * float(centre[1]))


class TestGltfMaterialImport(unittest.TestCase):
    def _triangle_gltf(
        self, directory: Path, material: dict, image: bytes | None = None
    ) -> Path:
        """Write a one-triangle glTF file whose only material is `material`; `image` (PNG
        bytes) becomes texture 0."""
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32).tobytes()
        uri = "data:application/octet-stream;base64," + base64.b64encode(positions).decode()
        document = {
            "asset": {"version": "2.0"},
            "extensionsUsed": sorted(material.get("extensions", {})),
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
            "materials": [material],
            "buffers": [{"uri": uri, "byteLength": len(positions)}],
            "bufferViews": [{"buffer": 0, "byteLength": len(positions)}],
            "accessors": [{
                "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                "min": [0, 0, 0], "max": [1, 1, 0],
            }],
        }
        if image is not None:
            document["images"] = [
                {"uri": "data:image/png;base64," + base64.b64encode(image).decode()}
            ]
            document["textures"] = [{"source": 0}]
        path = directory / "triangle.gltf"
        path.write_text(json.dumps(document))
        return path

    def _import(self, material: dict, image: bytes | None = None) -> sapien.render.RenderMaterial:
        """The material of a one-triangle glTF file whose only material is `material`."""
        with tempfile.TemporaryDirectory() as directory:
            path = self._triangle_gltf(Path(directory), material, image)
            parts = sapien.render.RenderShapeTriangleMesh(str(path)).parts
            self.assertEqual(len(parts), 1)
            return parts[0].material

    def test_metallic_roughness_and_specular_extension(self) -> None:
        """The specular colour used to be read into the factor and the level computed from an
        uninitialized colour."""
        material = {
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.2, 0.4, 0.6, 1.0],
                "metallicFactor": 0.3,
                "roughnessFactor": 0.7,
            },
            "doubleSided": True,
            "extensions": {
                "KHR_materials_specular": {
                    "specularFactor": 0.5,
                    "specularColorFactor": [0.2, 0.5, 0.9],
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._triangle_gltf(Path(directory), material)
            shape = sapien.render.RenderShapeTriangleMesh(str(path))
            parts = shape.parts
            self.assertEqual(len(parts), 1)
            imported = parts[0].material

            np.testing.assert_allclose(imported.base_color, [0.2, 0.4, 0.6, 1.0], atol=1e-6)
            self.assertAlmostEqual(imported.metallic, 0.3, places=6)
            self.assertAlmostEqual(imported.roughness, 0.7, places=6)
            # KHR_materials_specular maps to OpenPBR specular_weight and specular_color directly
            self.assertAlmostEqual(imported.specular_weight, 0.5, places=6)
            np.testing.assert_allclose(imported.specular_color, [0.2, 0.5, 0.9], atol=1e-6)

            # load_scene imports materials through the same code
            node = sapien.render.load_scene(str(path))
            meshes = []
            pending = [node]
            while pending:
                current = pending.pop()
                pending.extend(current.children)
                if current.mesh is not None:
                    meshes.append(current.mesh)
            self.assertEqual(len(meshes), 1)
            from_scene = meshes[0].parts[0].material
            np.testing.assert_allclose(from_scene.base_color, [0.2, 0.4, 0.6, 1.0], atol=1e-6)
            self.assertAlmostEqual(from_scene.specular_weight, 0.5, places=6)

    def test_specular_extension_spares_metals_and_folds_bright_colours(self) -> None:
        """KHR_materials_specular leaves the metal BRDF alone, which OpenPBR's specular_weight
        scales; a colour above 1 moves its excess into the weight."""
        metal = self._import({
            "pbrMetallicRoughness": {"metallicFactor": 1.0},
            "extensions": {
                "KHR_materials_specular": {
                    "specularFactor": 0.2,
                    "specularColorFactor": [0.5, 0.5, 0.5],
                }
            },
        })
        self.assertAlmostEqual(metal.specular_weight, 1.0, places=6)
        np.testing.assert_allclose(metal.specular_color, [1.0, 1.0, 1.0], atol=1e-6)

        bright = self._import({
            "pbrMetallicRoughness": {"metallicFactor": 0.0},
            "extensions": {
                "KHR_materials_specular": {
                    "specularFactor": 0.5,
                    "specularColorFactor": [2.0, 1.0, 0.5],
                }
            },
        })
        self.assertAlmostEqual(bright.specular_weight, 1.0, places=6)
        np.testing.assert_allclose(bright.specular_color, [1.0, 0.5, 0.25], atol=1e-6)

    def test_ior_extension(self) -> None:
        """0 is the specular-glossiness mode's infinite index and 1 is a legal index; both used
        to become 1.5."""
        def ior(value: float) -> float:
            return self._import({"extensions": {"KHR_materials_ior": {"ior": value}}}).ior

        self.assertGreater(ior(0.0), 1000.0)
        self.assertAlmostEqual(ior(1.0), 1.0, places=6)
        self.assertAlmostEqual(ior(1.33), 1.33, places=5)

    def test_transmission_takes_the_base_colour_tint(self) -> None:
        """KHR_materials_transmission tints by the base colour, which OpenPBR does through
        transmission_color; a volume's attenuation colour takes its place."""
        thin = self._import({
            "pbrMetallicRoughness": {"baseColorFactor": [1.0, 0.8, 0.1, 1.0]},
            "extensions": {"KHR_materials_transmission": {"transmissionFactor": 1.0}},
        })
        self.assertTrue(thin.thin_walled)
        np.testing.assert_allclose(thin.transmission_color, [1.0, 0.8, 0.1], atol=1e-6)
        self.assertEqual(thin.transmission_depth, 0.0)

        solid = self._import({
            "pbrMetallicRoughness": {"baseColorFactor": [1.0, 0.8, 0.1, 1.0]},
            "extensions": {
                "KHR_materials_transmission": {"transmissionFactor": 1.0},
                "KHR_materials_volume": {
                    "thicknessFactor": 0.1,
                    "attenuationDistance": 0.5,
                    "attenuationColor": [0.2, 0.4, 0.9],
                },
            },
        })
        self.assertFalse(solid.thin_walled)
        np.testing.assert_allclose(solid.transmission_color, [0.2, 0.4, 0.9], atol=1e-6)
        self.assertAlmostEqual(solid.transmission_depth, 0.5, places=6)

    def test_occlusion_texture_strength(self) -> None:
        imported = self._import(
            {"occlusionTexture": {"index": 0, "strength": 0.3}}, png_1x1([0, 0, 0, 255])
        )
        self.assertIsNotNone(imported.occlusion_texture)
        self.assertAlmostEqual(imported.occlusion_strength, 0.3, places=6)

    def test_collada_ambient_texture_is_not_occlusion(self) -> None:
        """Collada's ambient texture arrives as a light map, which only glTF uses for
        occlusion."""
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "ambient.png").write_bytes(png_1x1([255, 0, 0, 255]))
            path = Path(directory) / "triangle.dae"
            path.write_text(COLLADA_AMBIENT_TEXTURE)
            parts = sapien.render.RenderShapeTriangleMesh(str(path)).parts
            self.assertEqual(len(parts), 1)
            material = parts[0].material
        np.testing.assert_allclose(material.base_color, [0.5, 0.5, 0.5, 1.0], atol=1e-6)
        self.assertIsNone(material.occlusion_texture)


class TestOcclusionStrength(unittest.TestCase):
    def test_strength_scales_the_occlusion_of_ambient_light(self) -> None:
        black = sapien.render.RenderTexture2D(
            np.tile(np.array([0, 0, 0, 255], np.uint8), (2, 2, 1)), "R8G8B8A8Unorm"
        )

        def centre(strength: float | None) -> float:
            material = sapien.render.RenderMaterial(
                base_color=[1.0, 1.0, 1.0, 1.0], specular_weight=0.0, roughness=1.0
            )
            if strength is not None:
                material.occlusion_texture = black
                material.occlusion_strength = strength
            with sapien.Scene() as scene:
                scene.set_ambient_light([0.5, 0.5, 0.5])
                builder = scene.create_actor_builder()
                builder.add_box_visual(half_size=[0.5] * 3, material=material)
                builder.build_kinematic()
                config = sapien.render.RenderConfig("default")
                config["ambientOcclusionStrength"] = 0.0
                camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.05, 10.0,
                                          render_config=config)
                camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
                scene.update_render()
                camera.take_picture()
                return float(camera.get_picture("Lighting")[SIZE // 2, SIZE // 2, :3].mean())

        plain = centre(None)
        self.assertGreater(plain, 0.1)
        self.assertAlmostEqual(centre(1.0) / plain, 0.0, delta=0.02)
        self.assertAlmostEqual(centre(0.5) / plain, 0.5, delta=0.03)
        self.assertAlmostEqual(centre(0.0) / plain, 1.0, delta=0.02)


class TestIdleSceneUpdates(unittest.TestCase):
    def test_changes_after_idle_frames_are_rendered(self) -> None:
        """World matrices are only recomputed when something changed; a move, a light
        direction change and a new body after idle frames must still show up."""
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.0, 0.0, 0.0])
            light = scene.add_directional_light([1.0, 0.0, 0.0], [1.0, 1.0, 1.0])
            camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0)
            camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])

            def box(position: list[float]) -> sapien.Entity:
                builder = scene.create_actor_builder()
                builder.add_box_visual(half_size=[0.3, 0.3, 0.3])
                actor = builder.build_kinematic()
                actor.set_pose(sapien.Pose(position))
                return actor

            def center() -> float:
                scene.update_render()
                camera.take_picture()
                return float(camera.get_picture("Lighting")[SIZE // 2, SIZE // 2, :3].mean())

            actor = box([0.0, 5.0, 0.0])  # out of view
            self.assertLess(max(center() for _ in range(3)), 1e-3)

            actor.set_pose(sapien.Pose([0.0, 0.0, 0.0]))
            lit = center()
            self.assertGreater(lit, 0.1)
            self.assertAlmostEqual(center(), lit, places=5)  # idle again, unchanged

            # turn the light around: it now lights the far side of the box
            light.pose = sapien.Pose(q=sapien.math.shortest_rotation([1, 0, 0], [-1, 0, 0]))
            self.assertLess(center(), 0.5 * lit)
            light.pose = sapien.Pose()

            # the light's own local pose, not its entity's
            light.local_pose = sapien.Pose(q=sapien.math.shortest_rotation([1, 0, 0], [-1, 0, 0]))
            self.assertLess(center(), 0.5 * lit)
            light.local_pose = sapien.Pose()
            self.assertAlmostEqual(center(), lit, places=4)

            actor.set_pose(sapien.Pose([0.0, 5.0, 0.0]))
            self.assertLess(center(), 1e-3)
            box([0.0, 0.0, 0.0])  # added after idle frames
            self.assertAlmostEqual(center(), lit, places=4)

    def test_physics_moves_after_idle_frames_are_rendered(self) -> None:
        """Poses written by a physics step (not Entity.set_pose) must reach the render too."""
        with sapien.Scene() as scene:
            scene.set_timestep(1 / 60)
            scene.set_ambient_light([0.0, 0.0, 0.0])
            scene.add_directional_light([1.0, 0.0, 0.0], [1.0, 1.0, 1.0])
            scene.add_ground(-0.5)
            camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0)
            camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])

            builder = scene.create_actor_builder()
            builder.add_box_collision(half_size=[0.3, 0.3, 0.3])
            builder.add_box_visual(half_size=[0.3, 0.3, 0.3])
            builder.initial_pose = sapien.Pose([0.0, 0.0, 3.0])  # above the view
            builder.build()

            def center() -> float:
                scene.update_render()
                camera.take_picture()
                return float(camera.get_picture("Lighting")[SIZE // 2, SIZE // 2, :3].mean())

            self.assertLess(max(center() for _ in range(3)), 1e-3)
            for _ in range(120):  # falls onto the ground, into the view
                scene.step()
            self.assertGreater(center(), 0.1)


class TestObjectTransformUploads(unittest.TestCase):
    def test_a_move_waits_for_another_cameras_queued_upload(self) -> None:
        """Camera C draws 4 million triangles; camera B's frame and its upload of the box's
        transform queue behind it (the copy waits for earlier draws). Meanwhile the box moves
        again and camera A uploads. B must still show the box where it was for its frame: the
        scene's one staging buffer and command buffer were rewritten without waiting, so B's
        queued copy carried A's transform."""
        before, after, elsewhere = [0.0, -0.4, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.4]
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.6, 0.6, 0.6])
            builder = scene.create_actor_builder()
            builder.add_box_visual(
                half_size=[0.2, 0.2, 0.2],
                material=sapien.render.RenderMaterial(base_color=[1.0, 0.0, 0.0, 1.0]),
            )
            box = builder.build_kinematic()
            # a 1000 x 1000 grid of tiny triangles, both windings, 10 m behind the box cameras
            xs = np.linspace(-1.0, 1.0, 1001, dtype=np.float32)
            gy, gz = np.meshgrid(xs, xs, indexing="ij")
            vertices = np.stack([np.full(gy.size, 10.0, np.float32), gy.ravel(), gz.ravel()], -1)
            index = np.arange(gy.size).reshape(gy.shape)
            a, b, c, d = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
            triangles = np.concatenate(
                [np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)]
            ).astype(np.uint32)
            triangles = np.ascontiguousarray(np.concatenate([triangles, triangles[:, ::-1]]))
            body = sapien.render.RenderBodyComponent()
            body.attach(
                sapien.render.RenderShapeTriangleMesh(
                    vertices,
                    triangles,
                    np.zeros((0, 3), np.float32),
                    np.zeros((0, 2), np.float32),
                    sapien.render.RenderMaterial(base_color=[1.0, 1.0, 1.0, 1.0]),
                )
            )
            grid = sapien.Entity()
            grid.add_component(body)
            scene.add_entity(grid)
            heavy = scene.add_camera("C", 1024, 1024, 0.4, 0.1, 20.0)
            heavy.entity.pose = sapien.Pose([5.0, 0.0, 0.0])
            cameras = []
            for name in ("B", "A"):
                camera = scene.add_camera(name, SIZE, SIZE, 1.0, 0.1, 5.0)
                camera.entity.pose = sapien.Pose([-2.0, 0.0, 0.0])
                cameras.append(camera)
            camera_b, camera_a = cameras

            def picture(camera: sapien.render.RenderCameraComponent, position) -> np.ndarray:
                box.set_pose(sapien.Pose(position))
                scene.update_render()
                camera.take_picture()
                return camera.get_picture("Color")[..., :3].copy()

            for camera in (heavy, camera_b, camera_a):
                picture(camera, before)
            reference = picture(camera_b, after)
            # the scene is meaningful only if the box shows where it is
            self.assertGreater(float(np.abs(picture(camera_b, elsewhere) - reference).max()), 0.3)

            box.set_pose(sapien.Pose(before))
            scene.update_render()
            heavy.take_picture()
            box.set_pose(sapien.Pose(after))
            scene.update_render()
            camera_b.take_picture()  # its transform copy queues behind C's draws
            box.set_pose(sapien.Pose(elsewhere))
            scene.update_render()
            camera_a.take_picture()
            np.testing.assert_allclose(
                camera_b.get_picture("Color")[..., :3], reference, atol=0.02
            )


if __name__ == "__main__":
    unittest.main()
