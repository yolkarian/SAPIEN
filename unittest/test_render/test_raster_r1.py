"""R1 grouped raster rendering: single-submit command buffer and target/pass trimming."""

import gc
import unittest

import numpy as np
import sapien
import torch


def _build_scene(
    box_color: list[float],
    box_pos: list[float],
    with_transparent: bool = True,
) -> tuple[sapien.Scene, sapien.Entity]:
    scene = sapien.Scene()
    scene.set_ambient_light([0.15, 0.15, 0.15])
    scene.add_directional_light(
        [0.4, -0.3, -1.0],
        [2.2, 2.0, 1.8],
        shadow=True,
        shadow_map_size=512,
        shadow_scale=4.0,
    )
    scene.add_point_light([-1.5, 1.5, 2.0], [4.0, 3.5, 3.0], shadow=False)

    floor = scene.create_actor_builder()
    floor.add_box_visual(
        half_size=[4.0, 4.0, 0.05],
        material=sapien.render.RenderMaterial(base_color=[0.75, 0.75, 0.75, 1.0], roughness=0.8),
    )
    floor.build_static().set_pose(sapien.Pose([0.0, 0.0, -0.05]))

    # Corner wall so GTAO has strong non-trivial occlusion in the crease
    wall = scene.create_actor_builder()
    wall.add_box_visual(
        half_size=[0.05, 4.0, 1.5],
        material=sapien.render.RenderMaterial(base_color=[0.65, 0.65, 0.7, 1.0], roughness=0.7),
    )
    wall.build_static().set_pose(sapien.Pose([1.0, 0.0, 1.5]))

    box_builder = scene.create_actor_builder()
    box_builder.add_box_visual(
        half_size=[0.35, 0.35, 0.35],
        material=sapien.render.RenderMaterial(base_color=box_color, roughness=0.35),
    )
    box = box_builder.build_kinematic()
    box.set_pose(sapien.Pose(box_pos))

    if with_transparent:
        glass_builder = scene.create_actor_builder()
        glass_builder.add_box_visual(
            half_size=[0.2, 0.2, 0.2],
            material=sapien.render.RenderMaterial(
                base_color=[0.2, 0.6, 0.9, 0.45], roughness=0.2
            ),
        )
        glass = glass_builder.build_kinematic()
        glass.set_pose(sapien.Pose([-0.55, 0.45, 0.25]))

    return scene, box


def _look_at_pose(eye: list[float], target: list[float]) -> sapien.Pose:
    forward = np.array(target, dtype=np.float32) - np.array(eye, dtype=np.float32)
    forward /= np.linalg.norm(forward)
    left = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float32), forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, 0] = forward
    mat[:3, 1] = left
    mat[:3, 2] = up
    mat[:3, 3] = eye
    return sapien.Pose(mat)


class RasterR1Test(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for grouped rendering tests")

    def test_grouped_default_matches_standalone_and_shares_shadows(self) -> None:
        cfg = sapien.render.RenderConfig("default")
        targets = ["Color", "Lighting", "Albedo", "AmbientOcclusion", "GbufferDepth"]

        scene0, _ = _build_scene([0.85, 0.25, 0.2, 1.0], [0.1, -0.2, 0.35])
        scene1, _ = _build_scene([0.2, 0.75, 0.3, 1.0], [-0.1, 0.3, 0.45])
        pose_a = _look_at_pose([-2.4, -1.2, 1.5], [0.2, 0.0, 0.35])
        pose_b = _look_at_pose([-2.2, 1.4, 1.3], [0.2, 0.0, 0.35])

        # Reference standalone renders (separate cameras with identical poses)
        ref_cams = [
            scene0.add_camera("ref0a", 96, 72, 1.0, 0.1, 20.0, render_config=cfg),
            scene0.add_camera("ref0b", 96, 72, 1.0, 0.1, 20.0, render_config=cfg),
            scene1.add_camera("ref1a", 96, 72, 1.0, 0.1, 20.0, render_config=cfg),
        ]
        ref_cams[0].entity.pose = pose_a
        ref_cams[1].entity.pose = pose_b
        ref_cams[2].entity.pose = pose_a
        scene0.update_render()
        scene1.update_render()
        refs = []
        for rc in ref_cams:
            rc.take_picture()
            refs.append({t: rc.get_picture(t).copy() for t in targets})

        # Grouped cameras in one RenderCameraGroup (2 cameras share scene0, 1 views scene1)
        g_cams = [
            scene0.add_camera("g0a", 96, 72, 1.0, 0.1, 20.0, render_config=cfg),
            scene0.add_camera("g0b", 96, 72, 1.0, 0.1, 20.0, render_config=cfg),
            scene1.add_camera("g1a", 96, 72, 1.0, 0.1, 20.0, render_config=cfg),
        ]
        g_cams[0].entity.pose = pose_a
        g_cams[1].entity.pose = pose_b
        g_cams[2].entity.pose = pose_a

        rsg = sapien.render.RenderSystemGroup([scene0.render_system, scene1.render_system])
        group = rsg.create_camera_group(g_cams, targets)
        try:
            rsg.gpu_init()
            # Take two frames to exercise command buffer reuse across frames
            for _ in range(2):
                rsg.update_render()
                group.take_picture()
                torch.cuda.synchronize()
                for t in targets:
                    view = torch.from_dlpack(group.get_picture_cuda(t))
                    batch = view.cpu().numpy()
                    del view
                    for idx in range(len(g_cams)):
                        np.testing.assert_allclose(batch[idx], refs[idx][t], atol=1e-4, rtol=1e-4)
        finally:
            gc.collect()
            rsg.close()
            scene0.close()
            scene1.close()

    def test_target_and_pass_trimming_parity_and_vram(self) -> None:
        """Trimming drops unreferenced targets/passes while preserving exact output parity."""
        scene, _ = _build_scene([0.8, 0.3, 0.25, 1.0], [0.0, 0.0, 0.35])
        pose = _look_at_pose([-2.2, -1.0, 1.4], [0.1, 0.0, 0.35])

        # Capture full standalone baselines for both default and minimal packs
        def_cfg = sapien.render.RenderConfig("default")
        min_cfg = sapien.render.RenderConfig("minimal")
        ref_def = scene.add_camera("ref_def", 80, 64, 1.0, 0.1, 20.0, render_config=def_cfg)
        ref_min = scene.add_camera("ref_min", 80, 64, 1.0, 0.1, 20.0, render_config=min_cfg)
        ref_def.entity.pose = pose
        ref_min.entity.pose = pose
        scene.update_render()
        ref_def.take_picture()
        ref_min.take_picture()
        full_def_names = set(ref_def.get_picture_names())
        full_min_names = set(ref_min.get_picture_names())
        self.assertGreaterEqual(len(full_def_names), 16)
        self.assertIn("PositionSegmentation", full_min_names)

        cases = [
            (
                def_cfg,
                ["Color"],
                ref_def.get_picture("Color").copy(),
                {"Custom", "DepthLinear", "SegmentationView0", "SegmentationView1", "Position"},
                None,
            ),
            (
                def_cfg,
                ["Lighting"],
                ref_def.get_picture("Lighting").copy(),
                {
                    "Color",
                    "Custom",
                    "Line",
                    "LineDepth",
                    "Point",
                    "PointDepth",
                    "PositionRaw",
                    "Segmentation",
                    "DepthLinear",
                    "Position",
                },
                None,
            ),
            (
                def_cfg,
                ["Albedo"],
                ref_def.get_picture("Albedo").copy(),
                None,
                {"Albedo", "GbufferDepth"},
            ),
            (
                def_cfg,
                ["AmbientOcclusion"],
                ref_def.get_picture("AmbientOcclusion").copy(),
                None,
                {"AmbientOcclusion", "Normal", "GbufferDepth"},
            ),
            (
                def_cfg,
                ["GbufferDepth"],
                ref_def.get_picture("GbufferDepth").copy(),
                None,
                {"GbufferDepth"},
            ),
            (
                min_cfg,
                ["Color"],
                ref_min.get_picture("Color").copy(),
                {"PositionSegmentation"},
                {"Color", "ColorRaw", "GbufferDepth"},
            ),
        ]

        try:
            for idx, (cfg, req, expected, excluded, exact_set) in enumerate(cases):
                cam = scene.add_camera(f"trim_{idx}", 80, 64, 1.0, 0.1, 20.0, render_config=cfg)
                cam.entity.pose = pose
                rsg = sapien.render.RenderSystemGroup([scene.render_system])
                group = rsg.create_camera_group([cam], req)
                try:
                    rsg.gpu_init()
                    active_names = set(cam.get_picture_names())
                    if excluded is not None:
                        for name in excluded:
                            self.assertNotIn(name, active_names, f"target {name} was not trimmed")
                    if exact_set is not None:
                        self.assertEqual(active_names, exact_set)
                    rsg.update_render()
                    group.take_picture()
                    torch.cuda.synchronize()
                    view = torch.from_dlpack(group.get_picture_cuda(req[0]))
                    got = view.cpu().numpy()[0]
                    del view
                    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)
                finally:
                    gc.collect()
                    rsg.close()
        finally:
            scene.close()

        # Verify real VRAM savings across 64 256x256 cameras: measure trimmed ["Albedo"] first,
        # then full targets (so VMA cannot hide the full group's allocation inside a prior pool).
        n_cams = 64
        vram_scene, _ = _build_scene([0.7, 0.4, 0.2, 1.0], [0.0, 0.0, 0.35], with_transparent=False)
        cams_trim = [
            vram_scene.add_camera(f"vt_{i}", 256, 256, 1.0, 0.1, 20.0, render_config=def_cfg)
            for i in range(n_cams)
        ]
        torch.cuda.synchronize()
        free_before_trim, _ = torch.cuda.mem_get_info()
        rsg_trim = sapien.render.RenderSystemGroup([vram_scene.render_system])
        grp_trim = rsg_trim.create_camera_group(cams_trim, ["Albedo"])
        try:
            rsg_trim.gpu_init()
            rsg_trim.update_render()
            grp_trim.take_picture()
            torch.cuda.synchronize()
            free_after_trim, _ = torch.cuda.mem_get_info()
            used_trim_mib = (free_before_trim - free_after_trim) / (1024 * 1024)
        finally:
            gc.collect()
            rsg_trim.close()

        all_targets = [
            "Color",
            "Lighting",
            "Albedo",
            "Position",
            "PositionRaw",
            "Normal",
            "Specular",
            "Segmentation",
            "Custom",
            "Emission",
            "DepthLinear",
            "SegmentationView0",
            "SegmentationView1",
        ]
        cams_full = [
            vram_scene.add_camera(f"vf_{i}", 256, 256, 1.0, 0.1, 20.0, render_config=def_cfg)
            for i in range(n_cams)
        ]
        torch.cuda.synchronize()
        free_before_full, _ = torch.cuda.mem_get_info()
        rsg_full = sapien.render.RenderSystemGroup([vram_scene.render_system])
        grp_full = rsg_full.create_camera_group(cams_full, all_targets)
        try:
            rsg_full.gpu_init()
            rsg_full.update_render()
            grp_full.take_picture()
            torch.cuda.synchronize()
            free_after_full, _ = torch.cuda.mem_get_info()
            used_full_mib = (free_before_full - free_after_full) / (1024 * 1024)
        finally:
            gc.collect()
            rsg_full.close()
            vram_scene.close()

        # 64 x 256x256 x ~15 extra 16B/px targets is ~1000 MiB (plus exported CUDA buffers).
        # Even accounting for VMA block rounding, the full group must allocate far more VRAM.
        self.assertGreater(
            used_full_mib - used_trim_mib,
            200.0,
            f"expected >=200 MiB VRAM savings, got trim={used_trim_mib:.1f} MiB vs "
            f"full={used_full_mib:.1f} MiB",
        )

    def test_cuda_poses_material_edits_and_property_rebuild(self) -> None:
        cfg = sapien.render.RenderConfig("default")
        scene, box = _build_scene([0.9, 0.15, 0.15, 1.0], [0.0, 0.0, 0.35], with_transparent=False)
        body = box.find_component_by_type(sapien.render.RenderBodyComponent)
        shape = body.render_shapes[0]
        shape.set_gpu_pose_batch_index(0)
        mat = shape.material

        pose0 = _look_at_pose([-2.2, 0.0, 1.1], [0.0, 0.0, 0.35])
        pose1 = _look_at_pose([-1.6, 0.8, 1.1], [0.0, 0.0, 0.35])
        cam = scene.add_camera("cam", 80, 64, 1.0, 0.1, 20.0, render_config=cfg)
        cam.entity.pose = pose0

        poses = torch.tensor([[0.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)

        rsg = sapien.render.RenderSystemGroup([scene.render_system])
        rsg.set_cuda_poses(sapien.CudaArray(poses))
        group = rsg.create_camera_group([cam], ["Color"])
        group.set_pose_mode(cam, "cuda")
        try:
            rsg.gpu_init()
            rsg.update_render()
            group.take_picture()
            torch.cuda.synchronize()
            view = torch.from_dlpack(group.get_picture_cuda("Color"))
            img0 = view.cpu().numpy()[0].copy()
            del view

            # 1. Move body via CUDA pose buffer and camera via set_cuda_pose
            poses[0, 1] = 0.6
            group.set_cuda_pose(cam, pose1)
            rsg.update_render()
            group.take_picture()
            torch.cuda.synchronize()
            view = torch.from_dlpack(group.get_picture_cuda("Color"))
            img1 = view.cpu().numpy()[0].copy()
            del view
            self.assertGreater(float(np.abs(img1 - img0).mean()), 0.02)

            # 2. Edit material base color in place (no re-record required)
            mat.base_color = [0.1, 0.2, 0.95, 1.0]
            group.take_picture()
            torch.cuda.synchronize()
            view = torch.from_dlpack(group.get_picture_cuda("Color"))
            img2 = view.cpu().numpy()[0].copy()
            del view
            # Blue channel rises while red channel drops on the box
            self.assertGreater(float((img2[..., 2] - img1[..., 2]).max()), 0.2)
            self.assertGreater(float((img1[..., 0] - img2[..., 0]).max()), 0.2)

            # 3. Change specialization constant (triggers pipeline rebuild + re-record);
            # scene/light buffers must stay populated rather than turning black!
            cam.set_property("toneMapper", 0)
            cam.set_property("exposure", 1.5)
            group.take_picture()
            torch.cuda.synchronize()
            view = torch.from_dlpack(group.get_picture_cuda("Color"))
            img3 = view.cpu().numpy()[0].copy()
            del view
            self.assertGreater(float(img3[..., :3].mean()), 0.25)
            self.assertGreater(float(np.abs(img3 - img2).mean()), 0.01)
        finally:
            gc.collect()
            rsg.close()
            scene.close()

    def test_unknown_target_raises_and_close_restores_standalone(self) -> None:
        cfg = sapien.render.RenderConfig("default")
        scene, _ = _build_scene([0.8, 0.4, 0.2, 1.0], [0.0, 0.0, 0.35], with_transparent=False)
        pose = _look_at_pose([-2.2, 0.0, 1.2], [0.0, 0.0, 0.35])
        cam = scene.add_camera("cam", 64, 64, 1.0, 0.1, 20.0, render_config=cfg)
        cam.entity.pose = pose

        # Unknown target raises a clear error at gpu_init()
        rsg_bad = sapien.render.RenderSystemGroup([scene.render_system])
        rsg_bad.create_camera_group([cam], ["NotARealTarget"])
        try:
            with self.assertRaisesRegex(RuntimeError, "NotARealTarget"):
                rsg_bad.gpu_init()
        finally:
            gc.collect()
            rsg_bad.close()

        # Trim to Albedo, close the group, and verify standalone rendering restores all targets
        rsg = sapien.render.RenderSystemGroup([scene.render_system])
        group = rsg.create_camera_group([cam], ["Albedo"])
        try:
            rsg.gpu_init()
            self.assertEqual(set(cam.get_picture_names()), {"Albedo", "GbufferDepth"})
            rsg.update_render()
            group.take_picture()
            torch.cuda.synchronize()
            view = torch.from_dlpack(group.get_picture_cuda("Albedo"))
            albedo_grouped = view.cpu().numpy()[0].copy()
            del view
        finally:
            gc.collect()
            rsg.close()

        # Standalone capture after close must restore Color and all other pack targets
        scene.update_render()
        cam.take_picture()
        restored_names = set(cam.get_picture_names())
        self.assertIn("Color", restored_names)
        self.assertIn("Lighting", restored_names)
        self.assertIn("Position", restored_names)
        color = cam.get_picture("Color")
        self.assertGreater(float(color[..., :3].mean()), 0.1)
        np.testing.assert_allclose(cam.get_picture("Albedo"), albedo_grouped, atol=1e-4, rtol=1e-4)
        scene.close()


if __name__ == "__main__":
    unittest.main()
