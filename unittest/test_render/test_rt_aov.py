"""Ray tracing (`rt`) AOV batch 1: Depth, MotionVector, Emission, and the direct/indirect x
diffuse/specular radiance split (DiffuseDirect, SpecularDirect, DiffuseIndirect,
SpecularIndirect).
"""

import unittest

import numpy as np
import sapien


SIZE = 64


def rt_config(mode: str = "mis", spp: int = 8, depth: int = 4) -> sapien.render.RenderConfig:
    cfg = sapien.render.RenderConfig("rt")
    cfg["spp"] = spp
    cfg["maxDepth"] = depth
    cfg["lightSampling"] = mode
    cfg["russianRoulette"] = False
    cfg.denoiser = "none"
    return cfg


def render_pictures(
    scene: sapien.Scene,
    camera: sapien.render.RenderCameraComponent,
    names: tuple[str, ...],
    frames: int = 1,
) -> dict[str, np.ndarray]:
    scene.update_render()
    for _ in range(frames):
        camera.take_picture()
    return {name: camera.get_picture(name).copy() for name in names}


class RtDepthAov(unittest.TestCase):
    def test_depth_matches_camera_distance_and_sky_is_zero(self) -> None:
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.2, 0.2, 0.2])
            builder = scene.create_actor_builder()
            # Front face at x = 0.0, half-width 0.5 along y and z
            builder.add_box_visual(
                half_size=[0.5, 0.5, 0.5],
                material=sapien.render.RenderMaterial(base_color=[0.8, 0.8, 0.8, 1.0]),
            )
            box = builder.build_kinematic()
            box.set_pose(sapien.Pose([0.5, 0.0, 0.0]))

            camera = scene.add_camera(
                "camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=rt_config(spp=1, depth=1)
            )
            camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])

            pics = render_pictures(scene, camera, ("Depth", "Position", "Segmentation"))
            depth = pics["Depth"]
            position = pics["Position"]
            seg = pics["Segmentation"][..., 1]

            self.assertEqual(depth.shape, (SIZE, SIZE))
            self.assertEqual(depth.dtype, np.float32)

            hit = seg > 0
            sky = ~hit
            self.assertTrue(hit.any())
            self.assertTrue(sky.any())

            # Camera at x = -3 looking along +X, box front face at x = 0 -> depth = 3.0 m
            centre = depth[SIZE // 2 - 2 : SIZE // 2 + 2, SIZE // 2 - 2 : SIZE // 2 + 2]
            np.testing.assert_allclose(centre, 3.0, atol=1e-3)
            # Matches -Position.z on every hit pixel, and is 0 on sky
            np.testing.assert_allclose(depth[hit], -position[hit, 2], atol=1e-4)
            np.testing.assert_array_equal(depth[sky], 0.0)


class RtMotionVectorAov(unittest.TestCase):
    def test_object_camera_combined_and_sky_motion(self) -> None:
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.2, 0.2, 0.2])
            builder = scene.create_actor_builder()
            builder.add_box_visual(
                half_size=[0.4, 0.4, 0.4],
                material=sapien.render.RenderMaterial(base_color=[0.7, 0.3, 0.3, 1.0]),
            )
            box = builder.build_kinematic()
            box.set_pose(sapien.Pose([0.0, 0.0, 0.0]))

            camera = scene.add_camera(
                "camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=rt_config(spp=1, depth=1)
            )
            camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])

            # Frame 0: initial frame must have finite zero motion everywhere (no zero-matrix NaN)
            f0 = render_pictures(scene, camera, ("MotionVector", "Segmentation"))
            mv0 = f0["MotionVector"]
            self.assertEqual(mv0.shape, (SIZE, SIZE, 2))
            self.assertEqual(mv0.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(mv0)))
            np.testing.assert_allclose(mv0, 0.0, atol=1e-5)

            seg0 = f0["Segmentation"][..., 1] > 0
            y0, x0 = np.nonzero(seg0)
            cx0, cy0 = float(x0.mean()), float(y0.mean())

            # Frame 1: move box toward -Y (right in image) and -Z (down in image)
            # Camera is at (-3, 0, 0) looking +X; in ROS camera coords +Y is left, +Z is up.
            dy_world = -0.3
            dz_world = -0.2
            box.set_pose(sapien.Pose([0.0, dy_world, dz_world]))
            f1 = render_pictures(scene, camera, ("MotionVector", "Segmentation"))
            mv1 = f1["MotionVector"]
            seg1 = f1["Segmentation"][..., 1] > 0
            y1, x1 = np.nonzero(seg1)
            cx1, cy1 = float(x1.mean()), float(y1.mean())

            # Measure motion on interior pixels of the moved box
            box_mv = mv1[seg1].mean(axis=0)
            # Matches the observed screen centroid displacement (curr - prev)
            self.assertGreater(box_mv[0], 2.0)
            self.assertGreater(box_mv[1], 1.0)
            np.testing.assert_allclose(box_mv[0], cx1 - cx0, atol=0.6)
            np.testing.assert_allclose(box_mv[1], cy1 - cy0, atol=0.6)
            # Analytical pinhole displacement on the front face (z = 2.6 m)
            fx = camera.get_intrinsic_matrix()[0, 0]
            fy = camera.get_intrinsic_matrix()[1, 1]
            front_z = 3.0 - 0.4
            centre_y = int(round(cy1))
            centre_x = int(round(cx1))
            np.testing.assert_allclose(
                mv1[centre_y, centre_x],
                [-dy_world * fx / front_z, -dz_world * fy / front_z],
                atol=0.15,
            )
            # Static sky has zero motion when only the object moved
            np.testing.assert_allclose(mv1[~seg1], 0.0, atol=1e-5)

            # Frame 2: another update_render with no movement settles motion back to zero
            f2 = render_pictures(scene, camera, ("MotionVector",))
            np.testing.assert_allclose(f2["MotionVector"], 0.0, atol=1e-5)

            # Frame 3: camera moves in -Y (right), so the static box shifts left (-x in image)
            camera.entity.pose = sapien.Pose([-3.0, -0.25, 0.0])
            f3 = render_pictures(scene, camera, ("MotionVector", "Segmentation"))
            mv3 = f3["MotionVector"]
            seg3 = f3["Segmentation"][..., 1] > 0
            self.assertLess(float(mv3[seg3, 0].mean()), -2.0)
            np.testing.assert_allclose(
                mv3[seg3, 0].mean(), -0.25 * fx / front_z, atol=0.5
            )
            # Pure camera translation leaves sky at infinity stationary
            np.testing.assert_allclose(mv3[~seg3], 0.0, atol=1e-5)

            # Frame 4: settle, then move camera and box together by the same translation -> 0 on box
            scene.update_render()
            box.set_pose(sapien.Pose([0.0, dy_world + 0.2, dz_world]))
            camera.entity.pose = sapien.Pose([-3.0, -0.25 + 0.2, 0.0])
            f4 = render_pictures(scene, camera, ("MotionVector", "Segmentation"))
            mv4 = f4["MotionVector"]
            seg4 = f4["Segmentation"][..., 1] > 0
            np.testing.assert_allclose(mv4[seg4], 0.0, atol=1e-4)

            # Frame 5: settle, then yaw camera left (+Z rotation) -> sky shifts right (+x)
            scene.update_render()
            angle = 0.05
            camera.entity.pose = sapien.Pose(
                [-3.0, -0.05, 0.0], [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]
            )
            f5 = render_pictures(scene, camera, ("MotionVector", "Segmentation"))
            mv5 = f5["MotionVector"]
            sky5 = f5["Segmentation"][..., 1] == 0
            self.assertTrue(sky5.any())
            self.assertGreater(float(mv5[sky5, 0].mean()), 1.0)


class RtRadianceSplitAov(unittest.TestCase):
    SPLIT_NAMES = (
        "HdrColor",
        "Emission",
        "DiffuseDirect",
        "SpecularDirect",
        "DiffuseIndirect",
        "SpecularIndirect",
        "Segmentation",
    )

    def test_conservation_and_lobe_separation(self) -> None:
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.1, 0.1, 0.1])
            # Pure Lambertian red floor
            floor_mat = sapien.render.RenderMaterial(
                base_color=[0.8, 0.2, 0.2, 1.0],
                metallic=0.0,
                specular_weight=0.0,
                roughness=1.0,
            )
            floor = scene.add_ground(0.0, render_material=floor_mat)

            # Left: pure metallic sphere (no diffuse lobe)
            metal_mat = sapien.render.RenderMaterial(
                base_color=[0.9, 0.8, 0.6, 1.0], metallic=1.0, roughness=0.25
            )
            b_metal = scene.create_actor_builder()
            b_metal.add_sphere_visual(radius=0.55, material=metal_mat)
            metal = b_metal.build_kinematic()
            metal.set_pose(sapien.Pose([0.0, 0.8, 0.55]))

            # Right: glossy dielectric sphere (both diffuse and specular lobes, sweeps highlights)
            glossy_mat = sapien.render.RenderMaterial(
                base_color=[0.2, 0.7, 0.2, 1.0],
                metallic=0.0,
                specular_weight=1.0,
                roughness=0.2,
            )
            glossy_mat.coat_weight = 0.5
            glossy_mat.coat_roughness = 0.1
            b_glossy = scene.create_actor_builder()
            b_glossy.add_sphere_visual(radius=0.55, material=glossy_mat)
            glossy = b_glossy.build_kinematic()
            glossy.set_pose(sapien.Pose([0.0, -0.8, 0.55]))

            # Back emissive panel + bounce wall
            emit_mat = sapien.render.RenderMaterial(
                base_color=[0.0, 0.0, 0.0, 1.0],
                specular_weight=0.0,
                emission=[3.0, 3.0, 2.5, 1.0],
            )
            b_emit = scene.create_actor_builder()
            b_emit.add_box_visual(half_size=[0.05, 0.6, 0.4], material=emit_mat)
            emitter = b_emit.build_kinematic()
            emitter.set_pose(sapien.Pose([1.4, 0.0, 1.4]))

            # White bounce wall beside the scene to ensure clear indirect bounce
            wall_mat = sapien.render.RenderMaterial(
                base_color=[0.9, 0.9, 0.9, 1.0], specular_weight=0.0, roughness=1.0
            )
            b_wall = scene.create_actor_builder()
            b_wall.add_box_visual(half_size=[2.0, 0.05, 1.5], material=wall_mat)
            b_wall.build_kinematic().set_pose(sapien.Pose([0.0, 1.8, 1.5]))

            scene.add_directional_light([0.5, -0.3, -0.8], [2.0, 2.0, 2.0])
            scene.add_point_light([-1.0, 0.0, 2.0], [6.0, 6.0, 6.0])

            camera = scene.add_camera(
                "camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=rt_config("mis", spp=8, depth=4)
            )
            camera.entity.pose = sapien.Pose([-3.2, 0.0, 1.3], [0.985, 0.0, 0.174, 0.0])

            # Accumulate 4 frames to also verify multi-frame AOV accumulation
            pics = render_pictures(scene, camera, self.SPLIT_NAMES, frames=4)
            hdr = pics["HdrColor"][..., :3]
            em = pics["Emission"][..., :3]
            dd = pics["DiffuseDirect"][..., :3]
            sd = pics["SpecularDirect"][..., :3]
            di = pics["DiffuseIndirect"][..., :3]
            si = pics["SpecularIndirect"][..., :3]
            seg = pics["Segmentation"][..., 1]

            # Exact conservation across the whole image (objects, emitter, floor, sky)
            total = em + dd + sd + di + si
            np.testing.assert_allclose(total, hdr, rtol=1e-4, atol=1e-4)

            # Identify each actor's interior mask via entity per-scene ID (eroded by 1 pixel so
            # subpixel Halton jitter does not straddle neighbouring diffuse/specular surfaces)
            def interior(mask: np.ndarray) -> np.ndarray:
                out = mask.copy()
                out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = False
                out[1:-1, 1:-1] &= (
                    mask[:-2, 1:-1]
                    & mask[2:, 1:-1]
                    & mask[1:-1, :-2]
                    & mask[1:-1, 2:]
                    & mask[:-2, :-2]
                    & mask[:-2, 2:]
                    & mask[2:, :-2]
                    & mask[2:, 2:]
                )
                return out

            floor_mask = interior(seg == floor.per_scene_id)
            metal_mask = interior(seg == metal.per_scene_id)
            glossy_mask = interior(seg == glossy.per_scene_id)
            emit_mask = interior(seg == emitter.per_scene_id)
            sky_mask = interior(seg == 0)
            self.assertTrue(floor_mask.any())
            self.assertTrue(metal_mask.any())
            self.assertTrue(glossy_mask.any())
            self.assertTrue(emit_mask.any())
            self.assertTrue(sky_mask.any())

            # Pure Lambertian floor: zero specular, non-zero diffuse direct & indirect
            np.testing.assert_allclose(sd[floor_mask], 0.0, atol=1e-6)
            np.testing.assert_allclose(si[floor_mask], 0.0, atol=1e-6)
            self.assertGreater(float(dd[floor_mask].mean()), 0.02)
            self.assertGreater(float(di[floor_mask].mean()), 0.005)

            # Pure metal: zero diffuse, non-zero specular direct & indirect
            np.testing.assert_allclose(dd[metal_mask], 0.0, atol=1e-6)
            np.testing.assert_allclose(di[metal_mask], 0.0, atol=1e-6)
            self.assertGreater(float(sd[metal_mask].mean()), 0.02)
            self.assertGreater(float(si[metal_mask].mean()), 0.02)

            # Glossy dielectric: both diffuse and specular direct & indirect are active
            self.assertGreater(float(dd[glossy_mask].mean()), 0.05)
            self.assertGreater(float(sd[glossy_mask].mean()), 0.01)
            self.assertGreater(float(sd[glossy_mask].max()), 0.1)
            self.assertGreater(float(di[glossy_mask].mean()), 0.005)
            self.assertGreater(float(si[glossy_mask].mean()), 0.003)

            # Emissive panel: Emission carries its radiance; diffuse/specular are 0
            self.assertGreater(float(em[emit_mask].mean()), 2.0)
            np.testing.assert_allclose(dd[emit_mask], 0.0, atol=1e-6)
            np.testing.assert_allclose(sd[emit_mask], 0.0, atol=1e-6)

            # Sky: Emission carries ambient/env background; all scattering splits are 0
            np.testing.assert_allclose(em[sky_mask], 0.1, atol=1e-4)
            np.testing.assert_allclose(dd[sky_mask], 0.0, atol=1e-6)
            np.testing.assert_allclose(sd[sky_mask], 0.0, atol=1e-6)
            np.testing.assert_allclose(di[sky_mask], 0.0, atol=1e-6)
            np.testing.assert_allclose(si[sky_mask], 0.0, atol=1e-6)

    def test_direct_split_is_invariant_across_sampling_strategies(self) -> None:
        """An emissive sphere lighting a Lambertian floor: BSDF hits on the emitter at bounce 1
        belong to DiffuseDirect (C D L), so DiffuseDirect agrees across mis, nee and bsdf."""
        means = {}
        indirect_means = {}
        for mode in ("mis", "nee", "bsdf"):
            with sapien.Scene() as scene:
                scene.set_ambient_light([0.0, 0.0, 0.0])
                scene.add_ground(
                    0.0,
                    render_material=sapien.render.RenderMaterial(
                        base_color=[1.0, 1.0, 1.0, 1.0], specular_weight=0.0, roughness=1.0
                    ),
                )
                b = scene.create_actor_builder()
                b.add_sphere_visual(
                    radius=0.35,
                    material=sapien.render.RenderMaterial(
                        emission=[20.0, 20.0, 20.0, 1.0],
                        base_color=[0.0, 0.0, 0.0, 1.0],
                        specular_weight=0.0,
                    ),
                )
                b.build_kinematic().set_pose(sapien.Pose([1.0, 0.0, 0.7]))
                camera = scene.add_camera(
                    "camera",
                    48,
                    48,
                    0.6,
                    0.1,
                    20.0,
                    render_config=rt_config(mode, spp=16, depth=2),
                )
                camera.entity.pose = sapien.Pose([0.0, 0.0, 3.0], [0.7071068, 0.0, 0.7071068, 0.0])
                pics = render_pictures(
                    scene,
                    camera,
                    ("HdrColor", "Emission", "DiffuseDirect", "SpecularDirect", "DiffuseIndirect", "SpecularIndirect"),
                    frames=16,
                )
                total = (
                    pics["Emission"][..., :3]
                    + pics["DiffuseDirect"][..., :3]
                    + pics["SpecularDirect"][..., :3]
                    + pics["DiffuseIndirect"][..., :3]
                    + pics["SpecularIndirect"][..., :3]
                )
                np.testing.assert_allclose(total, pics["HdrColor"][..., :3], rtol=1e-4, atol=1e-4)
                np.testing.assert_allclose(pics["SpecularDirect"][..., :3], 0.0, atol=1e-6)
                np.testing.assert_allclose(pics["SpecularIndirect"][..., :3], 0.0, atol=1e-6)
                means[mode] = float(pics["DiffuseDirect"][..., :3].mean())
                indirect_means[mode] = float(pics["DiffuseIndirect"][..., :3].mean())

        self.assertGreater(means["bsdf"], 0.05)
        for mode in ("mis", "nee"):
            self.assertAlmostEqual(
                means[mode], means["bsdf"], delta=0.06 * means["bsdf"], msg=str(means)
            )
            # Only the floor and emitter exist (emitter has base_color 0), so 2-bounce indirect is 0
            self.assertLess(indirect_means[mode], 1e-4)
        self.assertLess(indirect_means["bsdf"], 1e-4)


if __name__ == "__main__":
    unittest.main()
