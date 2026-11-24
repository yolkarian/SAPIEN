"""Regression tests for renderer defects that produced wrong images without raising."""

import base64
import json
import shutil
import struct
import tempfile
import unittest
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


class RayTracingCase(unittest.TestCase):
    """Cameras of the ray-tracing shader pack."""

    def setUp(self) -> None:
        self.render_config = sapien.render.RenderConfig("rt")
        self.render_config["spp"] = 1
        self.render_config.denoiser = "none"


class TestRasterLightCapacity(unittest.TestCase):
    def test_lights_beyond_shader_capacity_add_no_phantom_light(self) -> None:
        """The default pack holds 10 point lights. An 11th made the lighting loop read the
        textured-light slot as a point light whose colour is that light's direction."""
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

    def _center_color(self) -> np.ndarray:
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
            return camera.get_picture("Color")[SIZE // 2, SIZE // 2, :3].copy()

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


class TestGltfMaterialImport(unittest.TestCase):
    def _triangle_gltf(self, directory: Path, material: dict) -> Path:
        """Write a one-triangle glTF file whose only material is `material`."""
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
        path = directory / "triangle.gltf"
        path.write_text(json.dumps(document))
        return path

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
                    "specularColorFactor": [1.0, 1.0, 1.0],
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
            # F0 = 0.04 * factor * colour in glTF and 0.08 * specular in SAPIEN
            self.assertAlmostEqual(imported.specular, 0.25, places=5)

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
            self.assertAlmostEqual(from_scene.specular, 0.25, places=5)


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


if __name__ == "__main__":
    unittest.main()
