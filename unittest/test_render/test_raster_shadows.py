"""Rasterization shadow upgrades: cascaded directional shadows, receiver slope bias,
PCSS soft shadows (directional, spot, point), point-light cube PCF, and scene-owned
shadow maps shared across cameras."""

import unittest

import numpy as np
import sapien

try:
    import torch
except ImportError:
    torch = None


def add_box(
    scene: sapien.Scene,
    half_size,
    pose: sapien.Pose,
    color=(0.8, 0.8, 0.8, 1.0),
) -> sapien.Entity:
    builder = scene.create_actor_builder()
    builder.add_box_visual(
        half_size=half_size,
        material=sapien.render.RenderMaterial(
            base_color=list(color), specular_weight=0.0, roughness=1.0
        ),
    )
    actor = builder.build_kinematic()
    actor.set_pose(pose)
    return actor


def make_default_config(pack: str = "default") -> sapien.render.RenderConfig:
    cfg = sapien.render.RenderConfig(pack)
    if "ambientOcclusionStrength" in cfg.parameters:
        cfg["ambientOcclusionStrength"] = 0.0
    if "toneMapper" in cfg.parameters:
        cfg["toneMapper"] = 1
    return cfg


class TestRasterShadows(unittest.TestCase):
    def test_directional_cascades_near_detail_and_full_coverage(self) -> None:
        """With a large shadow_half_size (10 m) and coarse 256 px map, cascade 0 (0.15x)
        still resolves a 4 cm thin bar near the origin, while cascade 2 (1.0x) still
        shadows an occluder 6 m away."""
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.0, 0.0, 0.0])
            scene.add_ground(
                0.0,
                render_material=sapien.render.RenderMaterial(
                    base_color=[1, 1, 1, 1], specular_weight=0.0, roughness=1.0
                ),
            )
            light = scene.add_directional_light(
                [0.7071068, 0.0, -0.7071068],
                [2.0, 2.0, 2.0],
                shadow=True,
                shadow_scale=10.0,
                shadow_near=-10.0,
                shadow_far=10.0,
                shadow_map_size=256,
            )
            self.assertEqual(light.softness, 0.0)

            # Thin 4 cm bar (2 cm tall) inside cascade 0 ([-1.5, 1.5]): at 256 px, single-map
            # texel would be 7.8 cm (2x wider than the bar); cascade 0 texel is 1.17 cm.
            # Light shines along (+X, 0, -Z) so the ~6 cm shadow falls at (0.3, 0, 0), running
            # horizontally along Y across the overhead camera's image.
            add_box(scene, [0.02, 0.4, 0.01], sapien.Pose([0.0, 0.0, 0.3]))
            # Wide occluder inside cascade 2 (outside cascades 0 and 1, which end at 4.0 m);
            # box spans x in [5.6, 6.4] at z=1.0, so its 45-deg shadow falls at x in [6.6, 7.4]
            add_box(scene, [0.4, 0.4, 0.05], sapien.Pose([6.0, 0.0, 1.0]))

            cfg = make_default_config("default")
            near_cam = scene.add_camera("near", 128, 128, 0.4, 0.05, 20.0, render_config=cfg)
            near_cam.entity.pose = sapien.Pose(
                [0.3, 0.0, 0.8], [0.7071068, 0.0, 0.7071068, 0.0]
            )

            far_cam = scene.add_camera("far", 64, 64, 0.4, 0.1, 20.0, render_config=cfg)
            # Centered on the shadow at (7.0, 0, 0), clear of the occluder box itself
            far_cam.entity.pose = sapien.Pose(
                [7.0, 0.0, 1.5], [0.7071068, 0.0, 0.7071068, 0.0]
            )

            scene.update_render()
            near_cam.take_picture()
            far_cam.take_picture()

            near_lum = near_cam.get_picture("Lighting")[..., :3].mean(axis=-1)
            # Overhead camera rows run along -X, sweeping perpendicular to the 4 cm shadow bar
            profile = near_lum[:, 60:68].mean(axis=1)
            lit_near = float(np.percentile(profile, 90))
            dark_near = float(profile.min())
            self.assertGreater(lit_near, 0.3)
            self.assertLess(
                dark_near,
                0.25 * lit_near,
                f"cascade 0 should sharply resolve a 4 cm bar (dark={dark_near}, lit={lit_near})",
            )
            # Shadow should be narrow (not smeared across the entire view)
            shadow_fraction = float((profile < 0.5 * lit_near).mean())
            self.assertLess(shadow_fraction, 0.35)

            far_lum = far_cam.get_picture("Lighting")[..., :3].mean(axis=-1)
            far_centre = float(far_lum[28:36, 28:36].mean())
            self.assertLess(
                far_centre,
                0.2 * lit_near,
                f"cascade 2 must still cover occluders at 6 m (centre={far_centre}, lit={lit_near})",
            )

    def test_directional_pcss_penumbra_widens_with_softness_and_height(self) -> None:
        """PCSS produces hard shadows at softness=0 and height-dependent soft penumbras at
        softness>0 (contact hardening: higher occluder casts a wider penumbra)."""
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.0, 0.0, 0.0])
            scene.add_ground(
                0.0,
                render_material=sapien.render.RenderMaterial(
                    base_color=[1, 1, 1, 1], specular_weight=0.0, roughness=1.0
                ),
            )
            light = scene.add_directional_light(
                [0.5, 0.0, -0.8660254],
                [2.0, 2.0, 2.0],
                shadow=True,
                shadow_scale=4.0,
                shadow_near=-10.0,
                shadow_far=10.0,
                shadow_map_size=1024,
            )
            # Low plate at z=0.06 casts shadow around x = 0.0, y = -0.5;
            # High plate at z=1.40 casts shadow around x = 0.0, y = +0.5.
            low_box = add_box(scene, [0.25, 0.25, 0.02], sapien.Pose([-0.035, -0.5, 0.06]))
            high_box = add_box(scene, [0.25, 0.25, 0.02], sapien.Pose([-0.81, 0.5, 1.40]))
            occluder_ids = {low_box.per_scene_id, high_box.per_scene_id}

            cfg = make_default_config("default")
            # Two overhead cameras centered on the low and high shadow patches
            low_cam = scene.add_camera("low", 128, 128, 0.8, 0.1, 20.0, render_config=cfg)
            low_cam.entity.pose = sapien.Pose(
                [0.0, -0.5, 1.6], [0.7071068, 0.0, 0.7071068, 0.0]
            )
            high_cam = scene.add_camera("high", 128, 128, 0.8, 0.1, 20.0, render_config=cfg)
            high_cam.entity.pose = sapien.Pose(
                [0.0, 0.5, 1.6], [0.7071068, 0.0, 0.7071068, 0.0]
            )

            def penumbra_pixels(cam: sapien.render.RenderCameraComponent) -> int:
                lum = cam.get_picture("Lighting")[..., :3].mean(axis=-1)
                seg = cam.get_picture("Segmentation")[..., 1]
                floor_mask = ~np.isin(seg, list(occluder_ids))
                floor_lum = lum[floor_mask]
                lit = float(np.percentile(floor_lum, 95))
                self.assertGreater(lit, 0.3)
                norm = floor_lum / lit
                return int(((norm > 0.15) & (norm < 0.85)).sum())

            # Hard shadow baseline (softness = 0)
            light.softness = 0.0
            scene.update_render()
            low_cam.take_picture()
            high_cam.take_picture()
            hard_high = penumbra_pixels(high_cam)

            # Soft shadow (softness = 0.10): high occluder penumbra widens significantly,
            # while low contact occluder stays much sharper than high occluder
            light.softness = 0.10
            scene.update_render()
            low_cam.take_picture()
            high_cam.take_picture()
            soft_low = penumbra_pixels(low_cam)
            soft_high = penumbra_pixels(high_cam)

            self.assertGreater(
                soft_high,
                hard_high * 2 + 50,
                f"PCSS softness=0.10 should widen high-occluder penumbra "
                f"(hard={hard_high}, soft={soft_high})",
            )
            self.assertGreater(
                soft_high,
                soft_low * 1.8,
                f"PCSS must harden near contact (soft_low={soft_low}, soft_high={soft_high})",
            )

    def test_sloped_receiver_has_no_shadow_acne(self) -> None:
        """A steep 60-degree sloped receiver lit at a grazing angle shows no shadow acne
        under directional (hard and soft), spot, and point lights."""
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.0, 0.0, 0.0])
            # Large tilted plane: normal tilted 60 deg from +Z toward +X
            angle = np.deg2rad(60.0)
            q = [float(np.cos(angle / 2)), 0.0, float(np.sin(angle / 2)), 0.0]
            add_box(scene, [2.0, 2.0, 0.05], sapien.Pose([0.0, 0.0, 0.0], q))

            dir_light = scene.add_directional_light(
                [0.0, 0.0, -1.0],
                [2.0, 2.0, 2.0],
                shadow=True,
                shadow_scale=5.0,
                shadow_near=-10.0,
                shadow_far=10.0,
                shadow_map_size=512,
            )
            pt_light = scene.add_point_light(
                [0.0, 0.0, 2.5],
                [0.0, 0.0, 0.0],
                shadow=True,
                shadow_near=0.1,
                shadow_far=10.0,
                shadow_map_size=512,
            )
            spot_light = scene.add_spot_light(
                [0.0, 0.0, 2.5],
                [0.0, 0.0, -1.0],
                inner_fov=0.9,
                outer_fov=1.2,
                color=[0.0, 0.0, 0.0],
                shadow=True,
                shadow_near=0.1,
                shadow_far=10.0,
                shadow_map_size=512,
            )

            cam = scene.add_camera(
                "cam", 96, 96, 0.5, 0.1, 20.0, render_config=make_default_config("default")
            )
            # Look perpendicular to the tilted plane from its lit side
            cam.entity.pose = sapien.Pose([1.732, 0.0, 1.0], [0.8660254, 0.0, 0.5, 0.0])
            # Rotate camera so its +X looks toward origin (-1.732, 0, -1.0)
            cam.entity.pose = sapien.Pose(
                [1.732, 0.0, 1.0],
                sapien.math.shortest_rotation([1, 0, 0], [-1.732, 0.0, -1.0]),
            )

            def check_no_acne(light_comp, label: str) -> None:
                light_comp.shadow = False
                scene.update_render()
                cam.take_picture()
                ref = cam.get_picture("Lighting")[24:72, 24:72, :3].mean(axis=-1)

                light_comp.shadow = True
                scene.update_render()
                cam.take_picture()
                lum = cam.get_picture("Lighting")[24:72, 24:72, :3].mean(axis=-1)

                self.assertGreater(float(np.median(ref)), 0.05, f"{label}: plane should be lit")
                ratio = float((lum / np.maximum(ref, 1e-6)).min())
                self.assertGreater(
                    ratio,
                    0.98,
                    f"{label}: shadow acne detected on 60 deg slope (min ratio={ratio:.4f})",
                )

            # 1. Directional hard & soft
            check_no_acne(dir_light, "directional hard")
            dir_light.softness = 0.06
            check_no_acne(dir_light, "directional PCSS")

            # 2. Point light hard & soft
            dir_light.color = [0.0, 0.0, 0.0]
            pt_light.color = [15.0, 15.0, 15.0]
            check_no_acne(pt_light, "point PCF")
            pt_light.radius = 0.15
            check_no_acne(pt_light, "point PCSS")

            # 3. Spot light hard & soft
            pt_light.color = [0.0, 0.0, 0.0]
            spot_light.color = [15.0, 15.0, 15.0]
            check_no_acne(spot_light, "spot PCF")
            spot_light.radius = 0.15
            check_no_acne(spot_light, "spot PCSS")

    def test_point_and_spot_light_pcf_and_pcss(self) -> None:
        """Point-light cube PCF+PCSS and spot-light PCSS cast dark core shadows, clean lit
        surroundings, and visibly softer edges when radius > 0."""
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.0, 0.0, 0.0])
            scene.add_ground(
                0.0,
                render_material=sapien.render.RenderMaterial(
                    base_color=[1, 1, 1, 1], specular_weight=0.0, roughness=1.0
                ),
            )
            # Occluder floating at (0, 0, 1.0)
            add_box(scene, [0.3, 0.3, 0.03], sapien.Pose([0.0, 0.0, 1.0]))

            pt = scene.add_point_light(
                [-0.6, 0.0, 2.0],
                [15.0, 15.0, 15.0],
                shadow=True,
                shadow_near=0.1,
                shadow_far=10.0,
                shadow_map_size=512,
            )
            spot = scene.add_spot_light(
                [-0.6, 0.0, 2.0],
                [0.6, 0.0, -2.0],
                inner_fov=1.0,
                outer_fov=1.4,
                color=[0.0, 0.0, 0.0],
                shadow=True,
                shadow_near=0.1,
                shadow_far=10.0,
                shadow_map_size=512,
            )

            # Overhead camera looking at shadow centered at (+0.6, 0, 0)
            cam = scene.add_camera(
                "cam", 128, 128, 1.0, 0.1, 20.0, render_config=make_default_config("default")
            )
            cam.entity.pose = sapien.Pose([0.6, 0.0, 1.8], [0.7071068, 0.0, 0.7071068, 0.0])

            def analyze():
                scene.update_render()
                cam.take_picture()
                lum = cam.get_picture("Lighting")[..., :3].mean(axis=-1)
                lit = float(np.percentile(lum, 95))
                centre = float(lum[60:68, 60:68].mean())
                norm = lum / max(lit, 1e-6)
                penumbra = int(((norm > 0.15) & (norm < 0.85)).sum())
                return lit, centre, penumbra

            # Point light: hard PCF vs soft PCSS
            lit_p0, dark_p0, pen_p0 = analyze()
            self.assertGreater(lit_p0, 0.2)
            self.assertLess(dark_p0, 0.1 * lit_p0)

            pt.radius = 0.25
            lit_p1, dark_p1, pen_p1 = analyze()
            self.assertLess(dark_p1, 0.2 * lit_p1)
            self.assertGreater(
                pen_p1,
                pen_p0 + 80,
                f"point light radius=0.25 should widen penumbra (hard={pen_p0}, soft={pen_p1})",
            )

            # Spot light: hard PCF vs soft PCSS
            pt.color = [0.0, 0.0, 0.0]
            spot.color = [15.0, 15.0, 15.0]
            lit_s0, dark_s0, pen_s0 = analyze()
            self.assertGreater(lit_s0, 0.2)
            self.assertLess(dark_s0, 0.1 * lit_s0)

            spot.radius = 0.25
            lit_s1, dark_s1, pen_s1 = analyze()
            self.assertLess(dark_s1, 0.2 * lit_s1)
            self.assertGreater(
                pen_s1,
                pen_s0 + 80,
                f"spot light radius=0.25 should widen penumbra (hard={pen_s0}, soft={pen_s1})",
            )

    def test_scene_owned_shadow_maps_shared_across_cameras(self) -> None:
        """Multiple cameras viewing the same scene share one set of scene-owned shadow maps:
        all cameras see dynamic occluder updates regardless of render order, and extra cameras
        do not duplicate the shadow map VRAM."""
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.0, 0.0, 0.0])
            scene.add_ground(
                0.0,
                render_material=sapien.render.RenderMaterial(
                    base_color=[1, 1, 1, 1], specular_weight=0.0, roughness=1.0
                ),
            )
            # 2048^2 directional (3 cascades = 48 MiB) + 2048^2 point (6 faces = 96 MiB) = 144 MiB
            scene.add_directional_light(
                [0.7071068, 0.0, -0.7071068],
                [2.0, 2.0, 2.0],
                shadow=True,
                shadow_scale=4.0,
                shadow_map_size=2048,
            )
            scene.add_point_light(
                [0.0, 2.0, 2.0],
                [0.0, 0.0, 0.0],
                shadow=True,
                shadow_map_size=2048,
            )
            occluder = add_box(scene, [0.35, 0.35, 0.05], sapien.Pose([-0.5, 0.0, 0.5]))

            cam_pose = sapien.Pose([0.0, 0.0, 1.5], [0.7071068, 0.0, 0.7071068, 0.0])
            cam1 = scene.add_camera(
                "cam1", 64, 64, 0.8, 0.1, 20.0, render_config=make_default_config("default")
            )
            cam1.entity.pose = cam_pose

            scene.update_render()
            cam1.take_picture()

            free_after_one = None
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
                free_after_one = torch.cuda.mem_get_info()[0]

            # Add 4 more cameras sharing the same scene (mixing default and minimal packs)
            extra_cams = []
            for idx, pack in enumerate(["default", "default", "minimal", "default"]):
                c = scene.add_camera(
                    f"extra_{idx}",
                    64,
                    64,
                    0.8,
                    0.1,
                    20.0,
                    render_config=make_default_config(pack),
                )
                c.entity.pose = cam_pose
                extra_cams.append(c)

            for c in extra_cams:
                c.take_picture()

            if free_after_one is not None:
                torch.cuda.synchronize()
                free_after_five = torch.cuda.mem_get_info()[0]
                delta_mib = (free_after_one - free_after_five) / (1024 * 1024)
                # Per-camera shadow maps would cost 4 * 144 MiB = 576 MiB; shared maps only
                # allocate tiny 64x64 camera render targets (< 10 MiB total).
                self.assertLess(
                    delta_mib,
                    80.0,
                    f"4 extra cameras on the same scene consumed {delta_mib:.1f} MiB "
                    "(expected shared scene shadow maps, not 576 MiB of duplicates)",
                )

            def centre_brightness(cam: sapien.render.RenderCameraComponent, pack: str) -> float:
                pic = "Color" if pack == "minimal" else "Lighting"
                return float(cam.get_picture(pic)[28:36, 28:36, :3].mean())

            # Frame 1: occluder at (-0.5, 0, 0.5) shadows origin for every camera
            self.assertLess(centre_brightness(cam1, "default"), 0.05)
            for idx, (c, pack) in enumerate(
                zip(extra_cams, ["default", "default", "minimal", "default"])
            ):
                self.assertLess(
                    centre_brightness(c, pack),
                    0.05,
                    f"extra camera {idx} ({pack}) should see initial shadow",
                )

            # Frame 2: move occluder away, render an extra camera FIRST, then cam1 and minimal
            occluder.set_pose(sapien.Pose([-3.0, 0.0, 0.5]))
            scene.update_render()
            extra_cams[0].take_picture()
            cam1.take_picture()
            extra_cams[2].take_picture()

            self.assertGreater(centre_brightness(extra_cams[0], "default"), 0.3)
            self.assertGreater(centre_brightness(cam1, "default"), 0.3)
            self.assertGreater(centre_brightness(extra_cams[2], "minimal"), 0.3)

    def test_point_cloud_shadows_do_not_depend_on_render_order(self) -> None:
        """Only the point pack draws point clouds into the scene's shared shadow maps. A default
        camera used to reuse maps a point camera had just rendered, so whether it showed the
        point cloud's shadow depended on which camera rendered first."""

        def centres(point_first: bool) -> tuple[float, float]:
            with sapien.Scene() as scene:
                scene.set_ambient_light([0.0, 0.0, 0.0])
                scene.add_ground(
                    0.0,
                    render_material=sapien.render.RenderMaterial(
                        base_color=[1, 1, 1, 1], specular_weight=0.0, roughness=1.0
                    ),
                )
                scene.add_directional_light(
                    [0.7071068, 0.0, -0.7071068],
                    [2.0, 2.0, 2.0],
                    shadow=True,
                    shadow_scale=4.0,
                    shadow_map_size=1024,
                )
                # a 0.3 m point above and beside the view: its shadow falls on the origin
                points = sapien.render.RenderPointCloudComponent(1)
                points.set_vertices(np.array([[-0.5, 0.0, 0.5]], np.float32))
                points.set_attribute("color", np.array([[1.0, 1.0, 1.0, 1.0]], np.float32))
                points.set_attribute("scale", np.array([0.3], np.float32))
                cloud = sapien.Entity()
                cloud.add_component(points)
                scene.add_entity(cloud)

                cameras = {}
                for pack in ("default", "point"):
                    cameras[pack] = scene.add_camera(
                        pack, 64, 64, 0.8, 0.1, 20.0, render_config=make_default_config(pack)
                    )
                    cameras[pack].entity.pose = sapien.Pose(
                        [0.0, 0.0, 1.5], [0.7071068, 0.0, 0.7071068, 0.0]
                    )
                scene.update_render()
                for pack in ("point", "default") if point_first else ("default", "point"):
                    cameras[pack].take_picture()
                return tuple(
                    float(cameras[pack].get_picture("Lighting")[28:36, 28:36, :3].mean())
                    for pack in ("default", "point")
                )

        default_after_point, point_shadowed = centres(point_first=True)
        default_first, _ = centres(point_first=False)
        self.assertGreater(default_first, 0.2)
        self.assertLess(point_shadowed, 0.2 * default_first)
        self.assertAlmostEqual(default_after_point, default_first, delta=1e-3)

    def test_a_light_change_waits_for_another_cameras_shadow_pass(self) -> None:
        """Camera B's frame draws the scene's shared shadow maps from a 2-million-triangle
        occluder. While that frame runs, the light turns and camera A uploads the new light
        matrices into the scene's shared light buffers. B's picture must equal the same frame
        rendered alone: an upload that does not wait rasterises B's maps with A's light."""
        first, turned = [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]

        def build(direction):
            scene = sapien.Scene()
            scene.set_ambient_light([0.0, 0.0, 0.0])
            scene.add_ground(
                0.0,
                render_material=sapien.render.RenderMaterial(
                    base_color=[1, 1, 1, 1], specular_weight=0.0, roughness=1.0
                ),
            )
            light = scene.add_directional_light(
                direction, [2.0, 2.0, 2.0], shadow=True, shadow_scale=3.0
            )
            # a grid of tiny triangles over a 0.3 m square 0.5 m up, both windings: slow to
            # rasterise, and its shadow falls 0.5 m beside it along the light
            xs = np.linspace(-0.15, 0.15, 708, dtype=np.float32)
            gx, gy = np.meshgrid(xs, xs, indexing="ij")
            vertices = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 0.5, np.float32)], -1)
            index = np.arange(gx.size).reshape(gx.shape)
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
                    sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
                )
            )
            occluder = sapien.Entity()
            occluder.add_component(body)
            scene.add_entity(occluder)
            # far outside every shadow map: moving it starts a new shadow epoch
            marker = add_box(scene, [0.1] * 3, sapien.Pose([50.0, 0.0, 0.0]))
            cameras = []
            for name in ("B", "A"):
                camera = scene.add_camera(
                    name, 128, 128, 1.2, 0.1, 20.0, render_config=make_default_config()
                )
                camera.entity.pose = sapien.Pose(
                    [0.0, 0.0, 2.5], [0.7071068, 0.0, 0.7071068, 0.0]
                )
                cameras.append(camera)
            return scene, light, marker, cameras

        def alone(direction) -> np.ndarray:
            scene, _, _, (camera, _) = build(direction)
            with scene:
                scene.update_render()
                camera.take_picture()
                return camera.get_picture("Lighting")[..., :3].copy()

        reference = alone(first)
        # the scene is meaningful only if turning the light moves the shadow
        self.assertGreater(float(np.abs(alone(turned) - reference).max()), 0.3)

        scene, light, marker, (camera_b, camera_a) = build(first)
        with scene:
            scene.update_render()
            for camera in (camera_b, camera_a):
                camera.take_picture()
                camera.get_picture("Lighting")
            marker.set_pose(sapien.Pose([50.0, 1.0, 0.0]))
            scene.update_render()
            camera_b.take_picture()  # B draws the shared maps in this frame
            light.entity.set_pose(
                sapien.Pose([0, 0, 0], sapien.math.shortest_rotation([1, 0, 0], turned))
            )
            scene.update_render()
            camera_a.take_picture()
            picture = camera_b.get_picture("Lighting")[..., :3]
            np.testing.assert_allclose(picture, reference, atol=0.02)

    def test_off_axis_point_and_spot_receivers_have_no_shadow_acne(self) -> None:
        """An unoccluded flat floor illuminated obliquely (~56 deg off-axis) by a point light or
        wide spotlight must show no self-shadowing acne under PCF or PCSS."""
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.0, 0.0, 0.0])
            scene.add_ground(
                0.0,
                render_material=sapien.render.RenderMaterial(
                    base_color=[1, 1, 1, 1], specular_weight=0.0, roughness=1.0
                ),
            )
            pt_light = scene.add_point_light(
                [-1.5, 0.0, 1.0],
                [15.0, 15.0, 15.0],
                shadow=True,
                shadow_near=0.1,
                shadow_far=10.0,
                shadow_map_size=512,
            )
            spot_light = scene.add_spot_light(
                [-1.5, 0.0, 1.0],
                [1.5, 0.0, -1.0],
                inner_fov=1.0,
                outer_fov=1.4,
                color=[0.0, 0.0, 0.0],
                shadow=True,
                shadow_near=0.1,
                shadow_far=10.0,
                shadow_map_size=512,
            )
            cam = scene.add_camera(
                "cam", 96, 96, 0.8, 0.1, 20.0, render_config=make_default_config("default")
            )
            cam.entity.pose = sapien.Pose([0.0, 0.0, 2.0], [0.7071068, 0.0, 0.7071068, 0.0])

            def check_no_acne(light_comp, label: str) -> None:
                light_comp.shadow = False
                scene.update_render()
                cam.take_picture()
                ref = cam.get_picture("Lighting")[16:80, 16:80, :3].mean(axis=-1)

                light_comp.shadow = True
                scene.update_render()
                cam.take_picture()
                lum = cam.get_picture("Lighting")[16:80, 16:80, :3].mean(axis=-1)

                self.assertGreater(float(np.median(ref)), 0.05, f"{label}: floor should be lit")
                ratio = float((lum / np.maximum(ref, 1e-6)).min())
                self.assertGreater(
                    ratio,
                    0.98,
                    f"{label}: off-axis shadow acne detected (min ratio={ratio:.4f})",
                )

            check_no_acne(pt_light, "off-axis point PCF")
            pt_light.radius = 0.25
            check_no_acne(pt_light, "off-axis point PCSS")

            pt_light.color = [0.0, 0.0, 0.0]
            spot_light.color = [15.0, 15.0, 15.0]
            check_no_acne(spot_light, "off-axis spot PCF")
            spot_light.radius = 0.25
            check_no_acne(spot_light, "off-axis spot PCSS")

    def test_exceeding_shadow_limit_raises_and_recovers(self) -> None:
        """Exceeding the shader pack's directional shadow count must raise cleanly before
        writing descriptors, and disabling the excess shadow light must recover cleanly."""
        with sapien.Scene() as scene:
            scene.set_ambient_light([0.0, 0.0, 0.0])
            scene.add_ground(
                0.0,
                render_material=sapien.render.RenderMaterial(
                    base_color=[1, 1, 1, 1], specular_weight=0.0, roughness=1.0
                ),
            )
            scene.add_directional_light([0.0, 0.0, -1.0], [2.0, 2.0, 2.0], shadow=True)
            extra = scene.add_directional_light([0.5, 0.0, -1.0], [1.0, 1.0, 1.0], shadow=True)
            cam = scene.add_camera(
                "cam", 32, 32, 0.8, 0.1, 20.0, render_config=make_default_config("default")
            )
            cam.entity.pose = sapien.Pose([0.0, 0.0, 2.0], [0.7071068, 0.0, 0.7071068, 0.0])

            scene.update_render()
            with self.assertRaisesRegex(RuntimeError, "too many directional"):
                cam.take_picture()

            extra.shadow = False
            scene.update_render()
            cam.take_picture()
            lit = float(cam.get_picture("Lighting")[..., :3].mean())
            self.assertGreater(lit, 0.2)


if __name__ == "__main__":
    unittest.main()
