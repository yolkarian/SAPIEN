"""Half-resolution GTAO pass + depth-aware bilateral upsample in default and vr_default packs."""

import unittest

import numpy as np
import sapien


def add_white_box(scene: sapien.Scene, half_size, pose: sapien.Pose) -> sapien.Entity:
    builder = scene.create_actor_builder()
    builder.add_box_visual(
        half_size=half_size,
        material=sapien.render.RenderMaterial(
            base_color=[1.0, 1.0, 1.0, 1.0], specular_weight=0.0, roughness=1.0
        ),
    )
    actor = builder.build_kinematic()
    actor.set_pose(pose)
    return actor


def make_corner_scene() -> tuple[sapien.Scene, sapien.Entity]:
    """An ambient-only white diffuse floor with a tall wall at x = 0 forming a 90 deg crease."""
    scene = sapien.Scene()
    scene.set_ambient_light([1.0, 1.0, 1.0])
    scene.add_ground(
        0.0,
        render_material=sapien.render.RenderMaterial(
            base_color=[1.0, 1.0, 1.0, 1.0], specular_weight=0.0, roughness=1.0
        ),
    )
    # Wall along x = 0, spanning y in [-3, 3], z in [0, 2]
    wall = add_white_box(scene, [0.1, 3.0, 1.0], sapien.Pose([0.1, 0.0, 1.0]))
    return scene, wall


class TestRasterGTAO(unittest.TestCase):
    def test_half_res_gtao_corner_darkening_and_controls(self) -> None:
        """GTAO runs at half resolution (ceil(W/2) x ceil(H/2)), keeps open flat planes
        unoccluded, darkens concave 90-degree corners, respects strength=0 disabling, and
        scales spatial reach with ambientOcclusionRadius."""
        scene, wall = make_corner_scene()
        try:
            cfg = sapien.render.RenderConfig("default")
            cfg["ambientOcclusionStrength"] = 1.0
            cfg["ambientOcclusionRadius"] = 0.4
            cfg["toneMapper"] = 1

            # Odd dimensions verify half-res ceil sizing: 97x73 -> (37, 49)
            cam = scene.add_camera("cam", 97, 73, 0.9, 0.1, 20.0, render_config=cfg)
            # Look obliquely down so both the crease at x=0 and open floor past x=-1.0 are visible
            cam.entity.pose = sapien.Pose(
                [-2.0, 0.0, 1.4],
                sapien.math.shortest_rotation([1, 0, 0], [1.6, 0.0, -1.4]),
            )

            # get_picture_names() must also work before the first take_picture()
            self.assertIn("AmbientOcclusion", cam.get_picture_names())
            scene.update_render()
            cam.take_picture()

            ao_img = cam.get_picture("AmbientOcclusion")
            self.assertEqual(ao_img.shape, (37, 49))
            self.assertTrue(np.all(np.isfinite(ao_img)))
            self.assertGreaterEqual(float(ao_img.min()), 0.0)
            self.assertLessEqual(float(ao_img.max()), 1.0001)

            # Use world-space position (via Position) to pick floor pixels near the crease
            # vs open floor far from the wall (distance > 0.85 m >> radius 0.4 m)
            pos = cam.get_picture("Position")
            seg = cam.get_picture("Segmentation")[..., 1]
            floor_mask = (seg != wall.per_scene_id) & (pos[..., 3] < 1.0)
            # Camera model matrix transforms camera-space xyz to world space
            model = cam.get_model_matrix()
            pts_cam = pos[..., :3]
            pts_world = pts_cam @ model[:3, :3].T + model[:3, 3]
            dist_to_wall = -pts_world[..., 0]  # wall face sits at x = 0

            crease_mask = floor_mask & (dist_to_wall > 0.02) & (dist_to_wall < 0.16)
            mid_mask = floor_mask & (dist_to_wall > 0.25) & (dist_to_wall < 0.50)
            open_mask = floor_mask & (dist_to_wall > 0.85) & (dist_to_wall < 1.30)
            self.assertGreater(int(crease_mask.sum()), 30)
            self.assertGreater(int(mid_mask.sum()), 30)
            self.assertGreater(int(open_mask.sum()), 30)

            lum = cam.get_picture("Lighting")[..., :3].mean(axis=-1)
            crease_val = float(lum[crease_mask].mean())
            mid_r04 = float(lum[mid_mask].mean())
            open_val = float(lum[open_mask].mean())

            # Half-res raw AO shows strong darkening in the crease and ~1.0 on open floor
            self.assertLess(float(ao_img.min()), 0.72)

            # Strength = 0.0 completely disables AO (crease matches open floor, raw AO == 1)
            cam.set_property("ambientOcclusionStrength", 0.0)
            cam.take_picture()
            lum_off = cam.get_picture("Lighting")[..., :3].mean(axis=-1)
            ao_off = cam.get_picture("AmbientOcclusion")
            self.assertGreater(float(ao_off.min()), 0.999)
            open_off = float(lum_off[open_mask].mean())
            self.assertAlmostEqual(
                float(lum_off[crease_mask].mean()),
                open_off,
                delta=0.01,
            )

            # Open floor with AO enabled matches AO disabled (zero planar self-occlusion even on
            # odd resolutions), while the concave crease is strongly darkened
            self.assertGreater(
                open_val,
                0.98 * open_off,
                f"Open flat floor should stay unoccluded (open={open_val:.4f}, off={open_off:.4f})",
            )
            self.assertLess(
                crease_val,
                0.80 * open_val,
                f"GTAO should darken 90 deg crease (crease={crease_val:.3f}, open={open_val:.3f})",
            )

            # Larger radius extends darkening farther from the wall into mid_mask
            cam.set_property("ambientOcclusionStrength", 1.0)
            cam.set_property("ambientOcclusionRadius", 0.12)
            cam.take_picture()
            mid_small = float(cam.get_picture("Lighting")[..., :3].mean(axis=-1)[mid_mask].mean())

            cam.set_property("ambientOcclusionRadius", 0.80)
            cam.take_picture()
            mid_large = float(cam.get_picture("Lighting")[..., :3].mean(axis=-1)[mid_mask].mean())

            self.assertGreater(
                mid_small,
                0.98 * open_off,
                f"Radius 0.12 m must not darken floor 0.25..0.50 m away "
                f"(mid={mid_small:.4f}, off={open_off:.4f})",
            )
            self.assertLess(
                mid_large,
                mid_small - 0.05 * open_off,
                f"Larger radius should darken farther from wall (r=0.12: {mid_small:.3f}, "
                f"r=0.4: {mid_r04:.3f}, r=0.8: {mid_large:.3f})",
            )
        finally:
            scene.close()

    def test_depth_edge_preservation_no_foreground_halo(self) -> None:
        """A foreground object far in front of a background wall must not cast a dark AO halo
        onto the wall across the depth discontinuity."""
        with sapien.Scene() as scene:
            scene.set_ambient_light([1.0, 1.0, 1.0])
            # Distant background wall at x = +3.5 (distance 3.5 m from camera at origin)
            bg_wall = add_white_box(scene, [0.05, 4.0, 4.0], sapien.Pose([3.5, 0.0, 0.0]))
            # Small foreground box at x = +1.2 (2.3 m in front of background wall >> radius 0.35 m)
            fg_box = add_white_box(scene, [0.08, 0.18, 0.18], sapien.Pose([1.2, 0.0, 0.0]))

            cfg = sapien.render.RenderConfig("default")
            cfg["ambientOcclusionStrength"] = 1.0
            cfg["ambientOcclusionRadius"] = 0.35
            cam = scene.add_camera("cam", 128, 128, 0.9, 0.1, 20.0, render_config=cfg)
            cam.entity.pose = sapien.Pose([0.0, 0.0, 0.0])

            scene.update_render()
            cam.take_picture()

            seg = cam.get_picture("Segmentation")[..., 1]
            lum = cam.get_picture("Lighting")[..., :3].mean(axis=-1)
            fg_mask = seg == fg_box.per_scene_id
            bg_mask = seg == bg_wall.per_scene_id
            self.assertGreater(int(fg_mask.sum()), 100)

            # Build a 4-pixel ring on the background wall right outside the foreground silhouette
            dilated = fg_mask.copy()
            for _ in range(4):
                d = dilated.copy()
                d[1:, :] |= dilated[:-1, :]
                d[:-1, :] |= dilated[1:, :]
                d[:, 1:] |= dilated[:, :-1]
                d[:, :-1] |= dilated[:, 1:]
                dilated = d
            halo_ring = dilated & bg_mask
            far_bg = (~dilated) & bg_mask
            self.assertGreater(int(halo_ring.sum()), 50)

            ring_min = float(lum[halo_ring].min())
            far_med = float(np.median(lum[far_bg]))
            self.assertGreater(
                ring_min,
                0.94 * far_med,
                f"Depth discontinuity halo detected around foreground box "
                f"(ring_min={ring_min:.3f}, far_bg={far_med:.3f})",
            )

    def test_dark_background_ao_stays_off_foreground_edges(self) -> None:
        """The gaps of a ridged far wall are dark in the AO target, and a flat box in front of
        it is not. The 3x3 upsample must weight its taps by depth: a plain blur darkens the
        box's silhouette with the background's AO. At even sizes the GTAO pass and the
        upsample must also read the same full-resolution depth texel for an AO texel; a float
        mismatch between the two darkened the silhouette at 10 of these 33 sizes on an
        RTX 3090."""
        with sapien.Scene() as scene:
            scene.set_ambient_light([1.0, 1.0, 1.0])
            material = sapien.render.RenderMaterial(
                base_color=[1.0, 1.0, 1.0, 1.0], specular_weight=0.0, roughness=1.0
            )
            # a wall at x = 3.4 behind a grid of 0.2 m cubes with 0.1 m gaps between them
            builder = scene.create_actor_builder()
            builder.add_box_visual(sapien.Pose([3.4, 0.0, 0.0]), [0.05, 2.0, 2.0], material)
            for y in np.arange(-1.2, 1.21, 0.3):
                for z in np.arange(-1.2, 1.21, 0.3):
                    builder.add_box_visual(
                        sapien.Pose([3.25, float(y), float(z)]), [0.1, 0.1, 0.1], material
                    )
            ridged = builder.build_kinematic()
            fg_box = add_white_box(scene, [0.08, 0.18, 0.18], sapien.Pose([1.2, 0.05, 0.05]))

            cfg = sapien.render.RenderConfig("default")
            cfg["ambientOcclusionStrength"] = 1.0
            cfg["ambientOcclusionRadius"] = 0.35
            for size in range(96, 161, 2):
                with self.subTest(size=size):
                    cam = scene.add_camera(
                        f"cam{size}", size, size, 0.9, 0.1, 20.0, render_config=cfg
                    )
                    scene.update_render()
                    cam.take_picture()

                    # the scene is meaningful only if the background's AO is dark somewhere
                    seg = cam.get_picture("Segmentation")[..., 1]
                    self.assertLess(float(cam.get_picture("AmbientOcclusion").min()), 0.7)
                    self.assertGreater(int((seg == ridged.per_scene_id).sum()), size * size // 4)

                    fg_mask = seg == fg_box.per_scene_id
                    eroded = fg_mask.copy()
                    for _ in range(2):
                        e = eroded.copy()
                        e[1:, :] &= eroded[:-1, :]
                        e[:-1, :] &= eroded[1:, :]
                        e[:, 1:] &= eroded[:, :-1]
                        e[:, :-1] &= eroded[:, 1:]
                        eroded = e
                    edge = fg_mask & ~eroded
                    self.assertGreater(int(edge.sum()), 50)
                    self.assertGreater(int(eroded.sum()), 100)

                    lum = cam.get_picture("Lighting")[..., :3].mean(axis=-1)
                    edge_min = float(lum[edge].min())
                    inner = float(np.median(lum[eroded]))
                    self.assertGreater(
                        edge_min,
                        0.97 * inner,
                        f"background AO bled onto the foreground edge (edge_min={edge_min:.3f}, "
                        f"inner={inner:.3f})",
                    )

    def test_vr_default_and_orthographic_camera(self) -> None:
        """Both vr_default and orthographic cameras run the half-res GTAO pass cleanly."""
        scene, wall = make_corner_scene()
        try:
            vr_cfg = sapien.render.RenderConfig("vr_default")
            vr_cfg["ambientOcclusionStrength"] = 1.0
            vr_cfg["ambientOcclusionRadius"] = 0.4
            vr_cam = scene.add_camera("vr", 64, 64, 0.9, 0.1, 20.0, render_config=vr_cfg)
            vr_cam.entity.pose = sapien.Pose(
                [-1.4, 0.0, 1.2],
                sapien.math.shortest_rotation([1, 0, 0], [1.4, 0.0, -1.0]),
            )

            ortho_cfg = sapien.render.RenderConfig("default")
            ortho_cfg["ambientOcclusionStrength"] = 1.0
            ortho_cfg["ambientOcclusionRadius"] = 0.4
            ortho_cam = scene.add_camera("ortho", 64, 64, 0.9, 0.1, 20.0, render_config=ortho_cfg)
            ortho_cam.set_orthographic_parameters(0.1, 20.0, 1.0)
            ortho_cam.entity.pose = vr_cam.entity.pose

            scene.update_render()
            vr_cam.take_picture()
            ortho_cam.take_picture()

            for label, cam in [("vr_default", vr_cam), ("orthographic", ortho_cam)]:
                ao = cam.get_picture("AmbientOcclusion")
                self.assertEqual(ao.shape, (32, 32), f"{label} AO shape mismatch")
                self.assertTrue(np.all(np.isfinite(ao)), f"{label} AO has non-finite values")
                self.assertLess(
                    float(ao.min()),
                    0.75,
                    f"{label} should darken the corner (min AO={ao.min():.3f})",
                )
                self.assertGreater(
                    float(np.percentile(ao, 95)),
                    0.95,
                    f"{label} open plane should stay unoccluded",
                )
        finally:
            scene.close()


if __name__ == "__main__":
    unittest.main()
