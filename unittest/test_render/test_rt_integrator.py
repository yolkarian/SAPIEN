"""The `rt` path tracer's integrator: light sampling (next-event estimation) of the environment
and of emissive meshes combined with BSDF sampling by multiple importance sampling, shadow rays
through transparent occluders, the indirect clamp, the defaults of its parameters, and material
edits after the first frame.

Light sampling, BSDF sampling and their MIS combination estimate the same image, so their means
must agree; light sampling must also be clearly less noisy where light comes from a small
source. Images are accumulated over frames of one static scene.
"""

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import sapien

SIZE = 48


def write_exr(path: Path, image: np.ndarray) -> None:
    """Write an uncompressed 32-bit float RGB scanline OpenEXR file of image [height, width, 3]."""
    height, width, _ = image.shape

    def attribute(name: str, type_name: str, payload: bytes) -> bytes:
        return name.encode() + b"\0" + type_name.encode() + b"\0" + struct.pack("<i", len(payload)) + payload

    channels = b"".join(n + b"\0" + struct.pack("<iBxxxii", 2, 0, 1, 1) for n in (b"B", b"G", b"R")) + b"\0"
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
    image = np.asarray(image, np.float32)
    # a scanline holds the whole row of each channel in alphabetical order: B, G, R
    rows = [np.concatenate([image[y, :, 2], image[y, :, 1], image[y, :, 0]]).tobytes() for y in range(height)]
    data_start = 8 + len(header) + 8 * height
    with open(path, "wb") as file:
        file.write(struct.pack("<ii", 20000630, 2))
        file.write(header)
        offset = data_start
        for row in rows:
            file.write(struct.pack("<Q", offset))
            offset += 8 + len(row)
        for y, row in enumerate(rows):
            file.write(struct.pack("<ii", y, len(row)) + row)


def config(mode: str = "mis", spp: int = 16, depth: int = 4, clamp: float = 0.0) -> sapien.render.RenderConfig:
    cfg = sapien.render.RenderConfig("rt")
    cfg["spp"] = spp
    cfg["maxDepth"] = depth
    cfg["lightSampling"] = mode
    cfg["clampIndirect"] = clamp
    cfg["russianRoulette"] = False
    cfg.denoiser = "none"
    return cfg


def render(scene: sapien.Scene, camera: sapien.render.RenderCameraComponent, frames: int) -> np.ndarray:
    """HDR radiance [SIZE, SIZE, 3] accumulated over `frames` frames of the static scene."""
    scene.update_render()
    for _ in range(frames):
        camera.take_picture()
    return camera.get_picture("HdrColor")[..., :3]


def noise(image: np.ndarray, mask: np.ndarray) -> float:
    """Pixel noise of a smooth image region: the spread of horizontal neighbour differences."""
    both = mask[:, 1:] & mask[:, :-1]
    return float(np.std((image[:, 1:] - image[:, :-1])[both]) / np.sqrt(2.0))


def disk(radius_fraction: float) -> np.ndarray:
    """Pixels within radius_fraction * SIZE / 2 of the image centre."""
    y, x = np.mgrid[:SIZE, :SIZE] + 0.5 - SIZE / 2
    return x * x + y * y < (radius_fraction * SIZE / 2) ** 2


class EnvironmentLighting(unittest.TestCase):
    """A sphere at the origin, seen from x = -3, lit only by an environment map."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def environment(self, image: np.ndarray) -> str:
        path = str(Path(self.directory.name) / "environment.exr")
        write_exr(Path(path), image)
        return path

    def sphere(self, environment: str, material: sapien.render.RenderMaterial, cfg, frames: int) -> np.ndarray:
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.0, 0.0, 0.0])
            scene.set_environment_map(environment)
            camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=cfg)
            camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
            builder = scene.create_actor_builder()
            builder.add_sphere_visual(radius=1.0, material=material)
            builder.build_kinematic()
            return render(scene, camera, frames)

    def test_white_furnace(self) -> None:
        """A white Lambertian sphere in a uniform environment reflects exactly the environment's
        radiance, whichever strategy estimates it."""
        environment = self.environment(np.full((32, 64, 3), 2.0, np.float32))
        white = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1], specular_weight=0.0, roughness=1.0)
        for mode in ("mis", "nee", "bsdf"):
            with self.subTest(mode=mode):
                image = self.sphere(environment, white, config(mode, spp=16), frames=8)
                self.assertAlmostEqual(float(image[disk(0.5)].mean()), 2.0, delta=0.03)

    def test_layers_redistribute_energy(self) -> None:
        """A white base under a clear coat and a white fuzz: albedo scaling keeps the layered
        reflectance at most the base's, losing only what the rough lobes' single scattering
        loses, so the sphere reflects nearly, and never more than, the environment's radiance."""
        environment = self.environment(np.full((32, 64, 3), 2.0, np.float32))
        layered = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1], roughness=0.5)
        layered.coat_weight = 1.0
        layered.coat_roughness = 0.1
        layered.fuzz_weight = 1.0
        layered.fuzz_roughness = 0.5
        mean = float(self.sphere(environment, layered, config("mis", spp=16), frames=8)[disk(0.5)].mean())
        self.assertLess(mean, 2.0 * 1.01)
        self.assertGreater(mean, 2.0 * 0.9)

    def test_small_bright_environment_region(self) -> None:
        """A bright band on a dim sky: the strategies agree on a glossy sphere (the MIS weights
        use the pdf of the whole lobe mixture), and light sampling cuts the noise."""
        image = np.full((32, 64, 3), 0.05, np.float32)
        image[10:12] = 30.0  # a thin ring around the sky lights every side of the sphere
        environment = self.environment(image)
        glossy = sapien.render.RenderMaterial(base_color=[0.8, 0.8, 0.8, 1], specular_weight=1.0, roughness=0.4)
        means, noises = {}, {}
        for mode in ("mis", "nee", "bsdf"):
            picture = self.sphere(environment, glossy, config(mode, spp=32), frames=24).mean(axis=-1)
            means[mode] = float(picture[disk(0.6)].mean())
            noises[mode] = noise(picture, disk(0.6))
        for mode in ("mis", "nee"):
            self.assertAlmostEqual(means[mode], means["bsdf"], delta=0.05 * means["bsdf"], msg=str(means))
            # a table misaligned with the map would make light sampling noisier than BSDF
            self.assertLess(noises[mode], 0.75 * noises["bsdf"], str(noises))


class EmissiveMeshLighting(unittest.TestCase):
    """A white floor lit only by an emissive sphere above it."""

    def floor(self, cfg, emitter_position: list[float], move_to: list[float] | None = None,
              emission_to: list[float] | None = None, frames: int = 24,
              floor_material: sapien.render.RenderMaterial | None = None) -> np.ndarray:
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.0, 0.0, 0.0])
            scene.add_ground(0.0, render_material=floor_material or sapien.render.RenderMaterial(
                base_color=[1, 1, 1, 1], specular_weight=0.0, roughness=1.0))
            builder = scene.create_actor_builder()
            material = sapien.render.RenderMaterial(
                emission=[20.0, 20.0, 20.0, 1.0], base_color=[0, 0, 0, 1], specular_weight=0.0)
            builder.add_sphere_visual(radius=0.3, material=material)
            emitter = builder.build_kinematic()
            emitter.set_pose(sapien.Pose(emitter_position))
            # looking straight down at the floor; the emitter is outside the view
            camera = scene.add_camera("camera", SIZE, SIZE, 0.6, 0.1, 20.0, render_config=cfg)
            camera.entity.pose = sapien.Pose([0.0, 0.0, 3.0], [0.7071068, 0.0, 0.7071068, 0.0])
            if move_to is not None or emission_to is not None:
                render(scene, camera, 1)
            if move_to is not None:
                emitter.set_pose(sapien.Pose(move_to))  # only the TLAS transform changes
            if emission_to is not None:
                material.emission = emission_to
            return render(scene, camera, frames)

    def test_strategies_agree_and_light_sampling_cuts_noise(self) -> None:
        means, noises = {}, {}
        everywhere = np.ones((SIZE, SIZE), bool)
        for mode in ("mis", "nee", "bsdf"):
            picture = self.floor(config(mode, spp=16, depth=1), [1.2, 0.0, 0.6]).mean(axis=-1)
            means[mode] = float(picture.mean())
            noises[mode] = noise(picture, everywhere)
        self.assertGreater(means["bsdf"], 0.05)
        for mode in ("mis", "nee"):
            self.assertAlmostEqual(means[mode], means["bsdf"], delta=0.05 * means["bsdf"], msg=str(means))
            # area sampling of a sphere wastes the samples on its far side
            self.assertLess(noises[mode], 0.75 * noises["bsdf"], str(noises))

    def test_a_moved_emitter_is_sampled_where_it_is(self) -> None:
        """Light sampling reads the emitter's current transform, not the one of the last
        rebuild: moving it gives the image of a scene built with it there."""
        cfg = config("nee", spp=16, depth=1)
        built_there = self.floor(cfg, [-1.0, 0.5, 0.8]).mean(axis=-1)
        moved_there = self.floor(cfg, [1.2, 0.0, 0.6], move_to=[-1.0, 0.5, 0.8]).mean(axis=-1)
        for half in (np.s_[:, : SIZE // 2], np.s_[:, SIZE // 2:]):
            self.assertAlmostEqual(float(moved_there[half].mean()), float(built_there[half].mean()),
                                   delta=0.05 * float(built_there[half].mean()))

    def test_an_emission_edit_relights_the_floor(self) -> None:
        """Light sampling reads the emitter's current radiance: recoloured after the first frame,
        it lights the floor in the new colour only."""
        floor = self.floor(config("nee", spp=16, depth=1), [1.2, 0.0, 0.6],
                           emission_to=[0.0, 0.0, 20.0, 1.0])
        self.assertGreater(float(floor[..., 2].mean()), 0.05)
        self.assertLess(float(floor[..., 0].mean()), 1e-3 * float(floor[..., 2].mean()))

    def test_strategies_agree_on_every_lobe(self) -> None:
        """Light sampling evaluates the BSDF, BSDF sampling divides by its pdf: on a floor that
        uses every OpenPBR lobe of the first batch the two must still agree, which they do only
        when evaluation, pdf and sampling describe the same distribution."""
        floor = sapien.render.RenderMaterial(base_color=[0.6, 0.5, 0.4, 1], metallic=0.3, roughness=0.35)
        floor.specular_color = [1.0, 0.8, 0.6]
        floor.specular_roughness_anisotropy = 0.6
        floor.coat_weight = 0.7
        floor.coat_roughness = 0.2
        floor.coat_color = [0.9, 0.9, 1.0]
        floor.fuzz_weight = 0.5
        floor.fuzz_color = [0.8, 0.6, 1.0]
        floor.fuzz_roughness = 0.4
        means = {mode: float(self.floor(config(mode, spp=16, depth=1), [1.2, 0.0, 0.6],
                                        floor_material=floor).mean())
                 for mode in ("mis", "nee", "bsdf")}
        self.assertGreater(means["bsdf"], 0.05)
        for mode in ("mis", "nee"):
            self.assertAlmostEqual(means[mode], means["bsdf"], delta=0.05 * means["bsdf"], msg=str(means))


class Shadows(unittest.TestCase):
    """Direct light (maxDepth 0) on a white floor beside an occluder: the camera looks straight
    down at the occluder's shadow, the light comes in at 45 degrees from -x."""

    def shadow(self, material: sapien.render.RenderMaterial | None, cloud: bool = False) -> np.ndarray:
        """Mean RGB radiance of the floor at the shadow centre."""
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.0, 0.0, 0.0])
            scene.add_ground(0.0, render_material=sapien.render.RenderMaterial(
                base_color=[1, 1, 1, 1], specular_weight=0.0, roughness=1.0))
            scene.add_directional_light([0.7071068, 0.0, -0.7071068], [2.0, 2.0, 2.0])
            if material is not None:  # its shadow is centred on the origin
                builder = scene.create_actor_builder()
                builder.add_box_visual(half_size=[0.4, 0.4, 0.02], material=material)
                builder.build_kinematic().set_pose(sapien.Pose([-0.5, 0.0, 0.5]))
            if cloud:
                points = sapien.render.RenderPointCloudComponent(1)
                points.set_vertices(np.array([[-0.5, 0.0, 0.5]], np.float32))
                points.set_attribute("color", np.array([[1.0, 1.0, 1.0, 1.0]], np.float32))
                points.set_attribute("scale", np.array([0.3], np.float32))
                cloud_entity = sapien.Entity()
                cloud_entity.add_component(points)
                scene.add_entity(cloud_entity)
            camera = scene.add_camera("camera", SIZE, SIZE, 0.3, 0.1, 20.0, render_config=config(spp=4, depth=0))
            # straight down at the shadow centre; the occluder is outside these pixels
            camera.entity.pose = sapien.Pose([0.0, 0.0, 2.0], [0.7071068, 0.0, 0.7071068, 0.0])
            centre = np.s_[SIZE // 2 - 4:SIZE // 2 + 4, SIZE // 2 - 4:SIZE // 2 + 4]
            return render(scene, camera, 2)[centre].mean(axis=(0, 1))

    def test_occluders_pass_their_transmittance(self) -> None:
        lit = self.shadow(None)
        self.assertGreater(float(lit.mean()), 0.1)
        opaque = self.shadow(sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]))
        np.testing.assert_allclose(opaque, 0.0, atol=1e-3 * float(lit.mean()))
        # each of the box's two faces covers 0.25: 0.75 * 0.75 of the light gets through
        sheer = self.shadow(sapien.render.RenderMaterial(base_color=[1, 1, 1, 0.25]))
        np.testing.assert_allclose(sheer / lit, 0.75 * 0.75, atol=0.01)
        # a clear red filter passes red only: transmission_color tints light entering the body
        filter_material = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1], transmission=1.0)
        filter_material.transmission_color = [1.0, 0.0, 0.0]
        red = self.shadow(filter_material)
        np.testing.assert_allclose(red / lit, [1.0, 0.0, 0.0], atol=0.01)

    def test_point_clouds_cast_shadows(self) -> None:
        """Point sets are opaque geometry, which never runs any-hit: the shadow ray must still
        count as blocked."""
        lit = self.shadow(None)
        blocked = self.shadow(None, cloud=True)
        np.testing.assert_allclose(blocked, 0.0, atol=1e-3 * float(lit.mean()))

    def test_cast_shadow_false_disables_shadows_in_rt_and_raster(self) -> None:
        """A render body with cast_shadow=False stays visible to camera rays while casting no
        shadow in both `rt` and `default` rasterization, and toggling it back restores the shadow."""
        for pack in ("rt", "default"):
            with self.subTest(pack=pack), sapien.Scene() as scene:
                scene.set_ambient_light([0.0, 0.0, 0.0])
                scene.add_ground(
                    0.0,
                    render_material=sapien.render.RenderMaterial(
                        base_color=[1, 1, 1, 1], specular_weight=0.0, roughness=1.0
                    ),
                )
                light = scene.add_directional_light(
                    [0.7071068, 0.0, -0.7071068], [2.0, 2.0, 2.0], shadow=True
                )
                light.shadow_half_size = 2.0
                builder = scene.create_actor_builder()
                builder.add_box_visual(
                    half_size=[0.4, 0.4, 0.05],
                    material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 1]),
                )
                occluder = builder.build_kinematic()
                occluder.set_pose(sapien.Pose([-0.5, 0.0, 0.5]))
                body = occluder.find_component_by_type(sapien.render.RenderBodyComponent)
                self.assertTrue(body.cast_shadow)

                cfg = config(spp=4, depth=0) if pack == "rt" else sapien.render.RenderConfig("default")
                cfg["toneMapper"] = 1
                # Camera looking at shadow centre on the floor
                cam = scene.add_camera("cam", SIZE, SIZE, 0.3, 0.1, 20.0, render_config=cfg)
                cam.entity.pose = sapien.Pose([0.0, 0.0, 2.0], [0.7071068, 0.0, 0.7071068, 0.0])
                # Second camera looking directly at the occluder to verify it remains visible
                vis_cam = scene.add_camera("vis", SIZE, SIZE, 0.8, 0.1, 20.0, render_config=cfg)
                vis_cam.entity.pose = sapien.Pose([-2.0, 0.0, 0.5])

                pic = "HdrColor" if pack == "rt" else "Lighting"
                centre = np.s_[SIZE // 2 - 3 : SIZE // 2 + 3, SIZE // 2 - 3 : SIZE // 2 + 3]

                scene.update_render()
                cam.take_picture()
                shadowed = float(cam.get_picture(pic)[centre][..., :3].mean())

                body.cast_shadow = False
                self.assertFalse(body.cast_shadow)
                scene.update_render()
                cam.take_picture()
                vis_cam.take_picture()
                unshadowed = float(cam.get_picture(pic)[centre][..., :3].mean())
                occluder_seg = vis_cam.get_picture("Segmentation")[..., 1]

                self.assertGreater(unshadowed, 0.2)
                self.assertLess(shadowed, 0.05 * unshadowed)
                # Occluder is still visible to camera rays
                self.assertTrue(np.any(occluder_seg == occluder.per_scene_id))

                # Re-enabling restores the shadow
                body.cast_shadow = True
                scene.update_render()
                cam.take_picture()
                restored = float(cam.get_picture(pic)[centre][..., :3].mean())
                self.assertLess(restored, 0.05 * unshadowed)


class Transmission(unittest.TestCase):
    """A clear box of half size 0.5 at the origin, seen face-on from x = -3 in front of a white
    emissive wall at x = 2 and a black sky: the centre pixel sees only light that crossed both
    faces of the box straight on."""

    def centre(self, glass: sapien.render.RenderMaterial) -> np.ndarray:
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.0, 0.0, 0.0])
            builder = scene.create_actor_builder()
            builder.add_box_visual(half_size=[0.5, 0.5, 0.5], material=glass)
            builder.add_box_visual(sapien.Pose([2.0, 0.0, 0.0]), half_size=[0.05, 2.0, 2.0],
                                   material=sapien.render.RenderMaterial(
                                       emission=[1, 1, 1, 1], base_color=[0, 0, 0, 1],
                                       specular_weight=0.0))
            builder.build_kinematic()
            camera = scene.add_camera("camera", SIZE, SIZE, 0.2, 0.1, 20.0,
                                      render_config=config(spp=16, depth=6))
            camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
            centre = np.s_[SIZE // 2 - 2:SIZE // 2 + 2, SIZE // 2 - 2:SIZE // 2 + 2]
            return render(scene, camera, 4)[centre].mean(axis=(0, 1))

    def test_transmission_colour_where_it_is_reached(self) -> None:
        def glass(**fields) -> sapien.render.RenderMaterial:
            material = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1], roughness=0.0,
                                                    transmission=1.0)
            material.transmission_color = [0.5, 1.0, 1.0]
            for name, value in fields.items():
                setattr(material, name, value)
            return material

        # transmission_depth 0: the colour tints the light where it enters the body
        surface = self.centre(glass())
        # thin walls: each of the two sheets tints
        thin = self.centre(glass(thin_walled=True))
        # the colour is reached after 0.25 of medium; the ray crosses 1.0 of it
        deep = self.centre(glass(transmission_depth=0.25))
        self.assertGreater(float(surface[1]), 0.5)
        for picture, ratio in ((surface, 0.5), (thin, 0.25), (deep, 0.5 ** 4)):
            self.assertAlmostEqual(float(picture[0] / picture[1]), ratio, delta=0.02, msg=str(picture))


class IndirectClamp(unittest.TestCase):
    def test_clamp_limits_only_light_after_the_first_bounce(self) -> None:
        """A clamp close to 0 removes indirect light and keeps the direct light: the image equals
        the direct-only one. Without it, the box walls add indirect light."""

        def picture(cfg) -> float:
            with sapien.Scene() as scene:
                scene.set_ambient_light([0.0, 0.0, 0.0])
                scene.add_ground(0.0, render_material=sapien.render.RenderMaterial(
                    base_color=[1, 1, 1, 1], specular_weight=0.0, roughness=1.0))
                builder = scene.create_actor_builder()
                for x in (-1.0, 1.0):  # two white walls beside the lit floor
                    builder.add_box_visual(sapien.Pose([x, 0.0, 0.5]), half_size=[0.05, 1.0, 0.5],
                                           material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1], specular_weight=0.0))
                builder.build_kinematic()
                scene.add_point_light([0.0, 0.0, 0.9], [4.0, 4.0, 4.0])
                camera = scene.add_camera("camera", SIZE, SIZE, 0.6, 0.1, 20.0, render_config=cfg)
                camera.entity.pose = sapien.Pose([0.0, 0.0, 2.5], [0.7071068, 0.0, 0.7071068, 0.0])
                return float(render(scene, camera, 8).mean())

        direct = picture(config(spp=8, depth=0))
        clamped = picture(config(spp=8, depth=3, clamp=1e-6))
        full = picture(config(spp=8, depth=3))
        self.assertAlmostEqual(clamped, direct, delta=0.01 * direct)
        self.assertGreater(full, 1.03 * direct)


class MaterialEdits(unittest.TestCase):
    """A box seen from x = -3, lit by nothing: the picture centre is the box's emission."""

    def box(self, material: sapien.render.RenderMaterial, cameras: int = 1):
        scene = sapien.Scene()
        self.addCleanup(scene.close)
        scene.set_ambient_light([0.0, 0.0, 0.0])
        builder = scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.8] * 3, material=material)
        builder.build_kinematic()
        result = []
        for index in range(cameras):
            camera = scene.add_camera(f"camera{index}", SIZE, SIZE, 1.0, 0.1, 10.0,
                                      render_config=config(spp=1, depth=1))
            camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
            result.append(camera)
        scene.update_render()
        return result

    @staticmethod
    def centre(camera: sapien.render.RenderCameraComponent) -> np.ndarray:
        camera.take_picture()
        return camera.get_picture("HdrColor")[SIZE // 2, SIZE // 2, :3]

    def test_an_edit_replaces_the_accumulated_samples(self) -> None:
        material = sapien.render.RenderMaterial(emission=[1, 0, 0, 1], base_color=[0, 0, 0, 1])
        (camera,) = self.box(material)
        for _ in range(4):
            self.centre(camera)
        material.emission = [0, 0, 1, 1]
        # no update_render: kept samples would leave 4/5 of the red
        np.testing.assert_allclose(self.centre(camera), [0, 0, 1], atol=0.02)

    def test_a_new_texture_reaches_every_camera_of_the_scene(self) -> None:
        material = sapien.render.RenderMaterial(emission=[1, 1, 1, 1], base_color=[0, 0, 0, 1])
        cameras = self.box(material, cameras=2)
        for camera in cameras:
            np.testing.assert_allclose(self.centre(camera), [1, 1, 1], atol=0.02)
        pixels = np.zeros((2, 2, 4), np.uint8)
        pixels[:] = [0, 255, 0, 255]
        material.emission_texture = sapien.render.RenderTexture2D(pixels, "R8G8B8A8Unorm")
        # textures[] grows: the first camera rebuilds the scene's tables, the second rebinds them
        for camera in cameras:
            np.testing.assert_allclose(self.centre(camera), [0, 1, 0], atol=0.02)


class Defaults(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = sapien.render.RenderConfig("rt")
        self.assertTrue(bool(cfg["russianRoulette"]))
        self.assertEqual(cfg["lightSampling"], "mis")
        self.assertEqual(cfg["clampIndirect"], 0.0)


if __name__ == "__main__":
    unittest.main()
