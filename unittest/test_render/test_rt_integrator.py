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


def annulus(inner: float, outer: float) -> np.ndarray:
    """Pixels between radius_fraction inner and outer * SIZE / 2 of the image centre: the rim of
    a sphere of radius 1 seen from x = -3 at SIZE, fovy 1 spans 0.58 to 0.65."""
    return disk(outer) & ~disk(inner)


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

    def test_the_fuzz_conserves_the_white_furnace(self) -> None:
        """A white base under white fuzz of any weight reflects exactly the environment's
        radiance in a uniform environment: the sheen's directional albedo, which is tabulated,
        plus the base dimmed by exactly that albedo add up to one. Multiplying the sheen lobe by
        an extra cosine loses 10 to 40% of the tabulated albedo away from normal incidence, so
        the sphere would come out 7% dark (and the base is not compensated)."""
        environment = self.environment(np.full((32, 64, 3), 2.0, np.float32))
        plain = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1], specular_weight=0.0,
                                            roughness=0.5)
        fuzzy = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1], specular_weight=0.0,
                                            roughness=0.5)
        fuzzy.fuzz_weight = 1.0
        fuzzy.fuzz_roughness = 0.5  # the tabulated albedo spreads most around this roughness
        cfg = config("mis", spp=32)
        bare = float(self.sphere(environment, plain, cfg, frames=24)[disk(0.5)].mean())
        sheen = float(self.sphere(environment, fuzzy, cfg, frames=24)[disk(0.5)].mean())
        self.assertAlmostEqual(sheen / bare, 1.0, delta=0.025, msg=str((bare, sheen)))

    def test_a_denser_coat_does_not_brighten_the_grazing_rim(self) -> None:
        """A near-black sphere in a uniform environment: at the rim the coat's Fresnel alone
        reflects (0.372 of the environment's radiance over the annulus at cos 0.28-0.42), and the
        base adds little. Entering the base with the coat's own ratio (1.5 / 1.6 < 1) made its
        Fresnel total-internal-reflect, so the rim went to 1.051 while the base behind it should
        stay dark -- the spec's bright ring."""
        environment = self.environment(np.full((32, 64, 3), 2.0, np.float32))
        dark = sapien.render.RenderMaterial(base_color=[0.05, 0.05, 0.05, 1], roughness=0.3)
        coated = sapien.render.RenderMaterial(base_color=[0.05, 0.05, 0.05, 1], roughness=0.3)
        coated.coat_weight = 1.0
        coated.coat_roughness = 0.3  # the default coat_ior 1.6 sits above ior 1.5
        bare = self.sphere(environment, dark, config("mis", spp=16), frames=16)
        with_coat = self.sphere(environment, coated, config("mis", spp=16), frames=16)
        # the annulus just inside the silhouette: view cosines 0.28 to 0.42 for SIZE = 48 at
        # x = -3 with fovy 1; disk(0.5) only reaches cosine 0.62 and never shows the ring
        rim = annulus(0.58, 0.62)
        base = float(bare[rim].mean())
        sheen = float(with_coat[rim].mean())
        # the coat's own reflection is all the annulus may gain: per pixel 1.22 to 1.37 times the
        # bare sphere (0.32-0.45 against 0.25-0.37 of the environment's radiance). The
        # total-internal-reflection ring made it 2.0 to 3.4 times (0.63 to 1.54).
        self.assertGreater(base, 0.15, str(base))
        self.assertGreater(sheen, 1.1 * base, str((base, sheen)))
        self.assertLess(sheen, 1.7 * base, str((base, sheen)))

    def test_metal_specular_weight_above_one_is_clamped(self) -> None:
        """The spec's metal range is [0, 1]; the documented migration `specular_weight = 2 *
        specular` produced 2. A metal sphere of albedo 0.9 must reflect the same radiance at
        weight 2 as at weight 1 (0.878 of the environment's, over the disk at cos 0.62-1);
        unclamped, the F82 Fresnel doubles it to 1.76."""
        environment = self.environment(np.full((32, 64, 3), 2.0, np.float32))
        def sphere(weight: float) -> float:
            metal = sapien.render.RenderMaterial(base_color=[0.9, 0.9, 0.9, 1], metallic=1.0,
                                                 specular_weight=weight, roughness=0.35)
            image = self.sphere(environment, metal, config("bsdf", spp=16), frames=16)
            return float(image[disk(0.5)].mean())
        one = sphere(1.0)
        self.assertGreater(one, 1.0, str(one))  # 2 * 0.878
        self.assertAlmostEqual(sphere(2.0), one, delta=0.05 * one, msg=str(one))

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

    def test_coated_and_modulated_glass_conserves_white_furnace(self) -> None:
        """Transmissive glass under a denser coat or modulated specular_weight must stay finite
        and never exceed the white furnace: Snell refraction must keep the physical IOR ratio so
        entry and exit 1 / eta^2 factors cancel, and reflection + transmission Fresnel must sum
        to one."""
        environment = self.environment(np.full((32, 64, 3), 2.0, np.float32))
        cases = {
            "denser_coat": dict(ior=1.15, coat_weight=1.0, coat_ior=1.6, coat_roughness=0.1),
            "low_specular_weight": dict(ior=1.5, specular_weight=0.1),
            "high_specular_weight": dict(ior=1.5, specular_weight=25.0),
        }
        for name, fields in cases.items():
            with self.subTest(case=name):
                glass = sapien.render.RenderMaterial(
                    base_color=[1, 1, 1, 1], roughness=0.2, transmission=1.0
                )
                for field, value in fields.items():
                    setattr(glass, field, value)
                image = self.sphere(environment, glass, config("mis", spp=16, depth=8), frames=8)
                self.assertTrue(np.all(np.isfinite(image)), name)
                mean = float(image[disk(0.5)].mean())
                self.assertGreater(mean, 2.0 * 0.7, f"{name}: {mean}")
                self.assertLess(mean, 2.0 * 1.03, f"{name}: {mean}")


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

    def test_a_material_edit_moves_the_emitter_off_the_light_table(self) -> None:
        """Giving the emitter a coat (or a partly transparent base) dims its visible emission, so
        the emissive table must drop it at the next frame even though emission itself did not
        change and no update_render ran. A stale table makes light sampling use the raw radiance
        while the BSDF hit applies the emission factor: nee and bsdf then disagree."""
        def pictures(edit) -> dict[str, np.ndarray]:
            """Two frames per mode: the bare emitter, then the same emitter after `edit`."""
            with sapien.Scene() as scene:
                scene.set_ambient_light([0.0, 0.0, 0.0])
                scene.add_ground(0.0, render_material=sapien.render.RenderMaterial(
                    base_color=[1, 1, 1, 1], specular_weight=0.0, roughness=1.0))
                builder = scene.create_actor_builder()
                material = sapien.render.RenderMaterial(
                    emission=[20.0, 20.0, 20.0, 1.0], base_color=[0, 0, 0, 1],
                    specular_weight=0.0)
                builder.add_sphere_visual(radius=0.3, material=material)
                builder.build_kinematic().set_pose(sapien.Pose([1.2, 0.0, 0.6]))
                cameras = {}
                for mode in ("nee", "bsdf"):
                    camera = scene.add_camera(mode, SIZE, SIZE, 0.6, 0.1, 20.0,
                                              render_config=config(mode, spp=16, depth=1))
                    camera.entity.pose = sapien.Pose([0.0, 0.0, 3.0],
                                                     [0.7071068, 0.0, 0.7071068, 0.0])
                    cameras[mode] = camera
                result = {mode: [render(scene, camera, 8).mean(axis=-1)]
                          for mode, camera in cameras.items()}
                edit(material)
                for mode, camera in cameras.items():
                    result[mode].append(render(scene, camera, 8).mean(axis=-1))
                return result

        def coat(material: sapien.render.RenderMaterial) -> None:
            material.coat_weight = 1.0
            material.coat_color = [0.6, 0.6, 0.6]
            material.coat_roughness = 0.3

        def coverage(material: sapien.render.RenderMaterial) -> None:
            material.base_color = [0, 0, 0, 0.5]

        for name, edit in (("coat", coat), ("coverage", coverage)):
            with self.subTest(edit=name):
                result = pictures(edit)
                for index in (0, 1):
                    nee = float(result["nee"][index].mean())
                    bsdf = float(result["bsdf"][index].mean())
                    self.assertAlmostEqual(nee, bsdf, delta=0.05 * bsdf,
                                           msg=f"{name} frame {index}: {nee} vs {bsdf}")
                # the edit dims the emitter: the floor must darken
                self.assertLess(float(result["bsdf"][1].mean()),
                                0.95 * float(result["bsdf"][0].mean()), name)

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

    def test_a_modulated_emitter_is_not_light_sampled(self) -> None:
        """A coat or a partly transparent base dims an emitter's visible emission below its raw
        radiance, so such a mesh must stay out of the emissive light-sampling table, exactly like
        an emission-textured one: its BSDF hits then carry MIS weight 1 and NEE does not sample
        it, which keeps mis, nee and bsdf on the same image."""
        def lit(modulated: str | None) -> dict[str, float]:
            def floor(cfg) -> np.ndarray:
                with sapien.Scene() as scene:
                    scene.set_ambient_light([0.0, 0.0, 0.0])
                    scene.add_ground(0.0, render_material=sapien.render.RenderMaterial(
                        base_color=[1, 1, 1, 1], specular_weight=0.0, roughness=1.0))
                    builder = scene.create_actor_builder()
                    material = sapien.render.RenderMaterial(
                        emission=[20.0, 20.0, 20.0, 1.0], base_color=[0, 0, 0, 1],
                        specular_weight=0.0)
                    if modulated == "coat":
                        material.coat_weight = 1.0
                        material.coat_color = [0.6, 0.6, 0.6]
                        material.coat_roughness = 0.3
                    elif modulated == "coverage":
                        material.base_color = [0, 0, 0, 0.5]
                    builder.add_sphere_visual(radius=0.3, material=material)
                    builder.build_kinematic().set_pose(sapien.Pose([1.2, 0.0, 0.6]))
                    camera = scene.add_camera("camera", SIZE, SIZE, 0.6, 0.1, 20.0,
                                              render_config=cfg)
                    camera.entity.pose = sapien.Pose([0.0, 0.0, 3.0],
                                                     [0.7071068, 0.0, 0.7071068, 0.0])
                    return render(scene, camera, 16).mean(axis=-1)
            return {mode: float(floor(config(mode, spp=16, depth=1)).mean())
                    for mode in ("mis", "nee", "bsdf")}

        bare = lit(None)
        self.assertGreater(bare["bsdf"], 0.02, str(bare))
        for case in ("coat", "coverage"):
            with self.subTest(case=case):
                means = lit(case)
                # the three techniques must estimate one image: NEE sampling the raw radiance of
                # an emitter whose visible emission is modulated would double it
                for mode in ("mis", "nee", "bsdf"):
                    self.assertAlmostEqual(means[mode], means["bsdf"],
                                           delta=0.05 * means["bsdf"], msg=str(means))
                # and the modulation is visible: the emitter is dimmer than the bare one
                self.assertLess(means["bsdf"], 0.95 * bare["bsdf"], str((bare, means)))

    def test_single_sided_emissive_mesh_emits_only_from_front_face(self) -> None:
        """A single-sided opaque emissive quad facing away from the floor must not light the
        floor in mis, nee, or bsdf; flipping its winding to face the floor must light it
        identically across all three strategies."""
        def floor_mean(facing_floor: bool) -> dict[str, float]:
            verts = np.array(
                [[-0.4, -0.4, 0.6], [0.4, -0.4, 0.6], [0.4, 0.4, 0.6], [-0.4, 0.4, 0.6]],
                dtype=np.float32,
            )
            tris = (
                np.array([[0, 2, 1], [0, 3, 2]], dtype=np.uint32)
                if facing_floor
                else np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
            )
            means = {}
            for mode in ("mis", "nee", "bsdf"):
                with sapien.Scene() as scene:
                    scene.set_ambient_light([0.0, 0.0, 0.0])
                    scene.add_ground(
                        0.0,
                        render_material=sapien.render.RenderMaterial(
                            base_color=[1, 1, 1, 1], specular_weight=0.0, roughness=1.0
                        ),
                    )
                    mat = sapien.render.RenderMaterial(
                        emission=[20.0, 20.0, 20.0, 1.0],
                        base_color=[0, 0, 0, 1],
                        specular_weight=0.0,
                    )
                    body = sapien.render.RenderBodyComponent()
                    body.attach(
                        sapien.render.RenderShapeTriangleMesh(
                            verts,
                            tris,
                            np.zeros((0, 3), np.float32),
                            np.zeros((0, 2), np.float32),
                            mat,
                        )
                    )
                    emitter = sapien.Entity()
                    emitter.add_component(body)
                    scene.add_entity(emitter)
                    camera = scene.add_camera(
                        "camera",
                        SIZE,
                        SIZE,
                        0.6,
                        0.1,
                        20.0,
                        render_config=config(mode, spp=16, depth=1),
                    )
                    # Look obliquely at the floor beside the quad so the quad itself is out of view
                    camera.entity.pose = sapien.Pose(
                        [1.4, 0.0, 2.0], [0.7071068, 0.0, 0.7071068, 0.0]
                    )
                    means[mode] = float(render(scene, camera, 8).mean())
            return means

        back = floor_mean(facing_floor=False)
        for mode, val in back.items():
            self.assertLess(val, 1e-4, f"{mode} back-face emitted light: {back}")

        front = floor_mean(facing_floor=True)
        self.assertGreater(front["bsdf"], 0.05, str(front))
        for mode in ("mis", "nee"):
            self.assertAlmostEqual(
                front[mode], front["bsdf"], delta=0.06 * front["bsdf"], msg=str(front)
            )


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

    def test_a_transparent_bead_passes_its_background(self) -> None:
        """A bead's alpha is its transmission. Camera rays only ever meet the near surface of a
        bead (its intersection shader reports the near root), so the 1 / eta^2 factor of the
        refraction has no exit to cancel it: a clear bead over a white background must still
        pass that background, not the 1 / 1.2^2 = 0.69 of it."""
        def centre(alpha: float) -> float:
            with sapien.Scene() as scene:
                scene.set_ambient_light([0.0, 0.0, 0.0])
                builder = scene.create_actor_builder()
                builder.add_box_visual(sapien.Pose([2.0, 0.0, 0.0]), half_size=[0.05, 2.0, 2.0],
                                       material=sapien.render.RenderMaterial(
                                           emission=[1, 1, 1, 1], base_color=[0, 0, 0, 1],
                                           specular_weight=0.0))
                builder.build_kinematic()
                points = sapien.render.RenderPointCloudComponent(1)
                points.set_vertices(np.array([[0.0, 0.0, 0.0]], np.float32))
                points.set_attribute("scale", np.array([0.5], np.float32))
                points.set_attribute("color",
                                     np.array([[1.0, 1.0, 1.0, alpha]], np.float32))
                entity = sapien.Entity()
                entity.add_component(points)
                scene.add_entity(entity)
                camera = scene.add_camera("camera", SIZE, SIZE, 0.2, 0.1, 20.0,
                                          render_config=config("bsdf", spp=16, depth=2))
                camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
                centre = np.s_[SIZE // 2 - 2:SIZE // 2 + 2, SIZE // 2 - 2:SIZE // 2 + 2]
                return float(render(scene, camera, 4)[centre].mean())

        clear = centre(0.0)
        self.assertGreater(clear, 0.8, str(clear))
        # alpha 0.5 leaves half the light transmitted; the bead reflects the black sky with the
        # rest, so the centre keeps about half the background and, before the fix, a third
        half = centre(0.5)
        self.assertGreater(half, 0.45 * clear, str((clear, half)))
        self.assertLess(half, 0.6 * clear, str((clear, half)))

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

    def test_a_coat_never_brightens_the_transmitted_light(self) -> None:
        """A coat on a glass slab only adds its own reflection and absorbs: the radiance
        transmitted through the slab must not grow with the coat. At coat_ior 1.6 over ior 1.5
        the coat's ratio is below 1, and entering the body with it made a round trip through the
        slab gain a factor (ior / eta)^2 -- 1.55 at coat 0.5, 2.30 at coat 1."""
        def glass(**fields) -> sapien.render.RenderMaterial:
            material = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1], roughness=0.0,
                                                    transmission=1.0)
            for name, value in fields.items():
                setattr(material, name, value)
            return material

        bare = float(self.centre(glass()).mean())
        self.assertGreater(bare, 0.5)
        for coat in (0.5, 1.0):
            with self.subTest(coat=coat):
                coated = float(self.centre(glass(coat_weight=coat, coat_roughness=0.05)).mean())
                self.assertLess(coated, 1.02 * bare, str((bare, coated)))

    def test_clear_glass_without_specular_transmits(self) -> None:
        """specular_weight 0 at ior 1, and a coat whose IOR equals the base's, both put the
        base's IOR ratio at exactly 1. There the refraction half vector wo + eta * wi vanishes,
        which made the transmission weight zero: the slab must still pass its background."""
        def glass(**fields) -> sapien.render.RenderMaterial:
            material = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1], roughness=0.0,
                                                    transmission=1.0)
            for name, value in fields.items():
                setattr(material, name, value)
            return material

        reference = float(self.centre(glass()).mean())
        self.assertGreater(reference, 0.5)
        clear = [glass(specular_weight=0.0), glass(ior=1.0)]
        for index, material in enumerate(clear):
            with self.subTest(case=index):
                picture = float(self.centre(material).mean())
                # no reflectance at the interface: more light than through the plain glass slab
                self.assertGreater(picture, 1.02 * reference, str((reference, picture)))
                self.assertLess(picture, 1.25 * reference, str((reference, picture)))
        # the coat still reflects on both sides, so this slab passes less, but not nothing
        coated = float(self.centre(glass(coat_weight=1.0, coat_ior=1.5)).mean())
        self.assertGreater(coated, 0.7 * reference, str((reference, coated)))

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

    def test_clamp_keeps_bsdf_direct_emission_unclamped(self) -> None:
        """A tight clampIndirect must not clamp BSDF direct lighting from an emissive mesh:
        mis, nee, and bsdf must still agree on the floor illuminated by an emissive sphere."""
        means = {}
        for mode in ("mis", "nee", "bsdf"):
            with sapien.Scene() as scene:
                scene.set_ambient_light([0.0, 0.0, 0.0])
                scene.add_ground(
                    0.0,
                    render_material=sapien.render.RenderMaterial(
                        base_color=[1, 1, 1, 1], specular_weight=0.0, roughness=1.0
                    ),
                )
                builder = scene.create_actor_builder()
                builder.add_sphere_visual(
                    radius=0.35,
                    material=sapien.render.RenderMaterial(
                        emission=[20.0, 20.0, 20.0, 1.0],
                        base_color=[0, 0, 0, 1],
                        specular_weight=0.0,
                    ),
                )
                builder.build_kinematic().set_pose(sapien.Pose([1.0, 0.0, 0.7]))
                camera = scene.add_camera(
                    "camera",
                    SIZE,
                    SIZE,
                    0.6,
                    0.1,
                    20.0,
                    render_config=config(mode, spp=16, depth=2, clamp=1e-4),
                )
                camera.entity.pose = sapien.Pose(
                    [0.0, 0.0, 3.0], [0.7071068, 0.0, 0.7071068, 0.0]
                )
                means[mode] = float(render(scene, camera, 12).mean())

        self.assertGreater(means["bsdf"], 0.05, str(means))
        for mode in ("mis", "nee"):
            self.assertAlmostEqual(
                means[mode], means["bsdf"], delta=0.06 * means["bsdf"], msg=str(means)
            )


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

    def test_an_edit_waits_for_another_cameras_frame_in_flight(self) -> None:
        """Camera B's long frame is still tracing when the emission turns blue and camera A of
        the same scene renders, rewriting the scene's material table. B's frame must stay red:
        frames of the timeline render path signalled no access fence, so the table changed
        mid-trace and the rays traced after the write saw blue."""
        material = sapien.render.RenderMaterial(emission=[1, 0, 0, 1], base_color=[0, 0, 0, 1])
        (quick,) = self.box(material)
        scene = quick.entity.scene
        heavy = scene.add_camera("heavy", 256, 256, 1.0, 0.1, 10.0,
                                 render_config=config(spp=1024, depth=1))
        heavy.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
        scene.update_render()
        for camera in (heavy, quick):
            camera.take_picture()
            camera.get_picture("HdrColor")
        heavy.take_picture()  # about 0.1 s of tracing on an RTX 3090
        material.emission = [0, 0, 1, 1]
        quick.take_picture()
        self.assertLess(float(heavy.get_picture("HdrColor")[..., 2].max()), 0.01)
        np.testing.assert_allclose(
            quick.get_picture("HdrColor")[SIZE // 2, SIZE // 2, :3], [0, 0, 1], atol=0.02)


class Defaults(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = sapien.render.RenderConfig("rt")
        self.assertTrue(bool(cfg["russianRoulette"]))
        self.assertEqual(cfg["lightSampling"], "mis")
        self.assertEqual(cfg["clampIndirect"], 0.0)


if __name__ == "__main__":
    unittest.main()
