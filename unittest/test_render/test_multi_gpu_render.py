"""Vulkan rendering on several CUDA devices in one process.

Every test needs at least two render-capable CUDA devices and CUDA-enabled Torch and is
skipped otherwise. Each device renders the same scene built from shared (device-independent)
materials and meshes, so the images of all devices must agree.
"""

import gc
import os
import subprocess
import sys
import textwrap
import threading
import unittest
from typing import List, Tuple

import numpy as np
import sapien

try:
    import torch
except ImportError:  # pragma: no cover - depends on the test environment
    torch = None

DEVICES = ("cuda:0", "cuda:1")
SIZE = 64
FRAMES = 20
TIMESTEP = 0.02
SPEED = 2.0


def _has_two_cuda_devices() -> bool:
    return torch is not None and torch.cuda.is_available() and torch.cuda.device_count() >= 2


def _build_scene(
    device: str, material: sapien.render.RenderMaterial
) -> Tuple[sapien.Scene, sapien.render.RenderCameraComponent]:
    """A unit box with `material` in front of a camera, rendered on `device`."""
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(device)])
    scene.set_ambient_light([1.0, 1.0, 1.0])
    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=[0.5, 0.5, 0.5], material=material)
    builder.build_kinematic()
    camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0)
    camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
    return scene, camera


def _capture(scene: sapien.Scene, camera: sapien.render.RenderCameraComponent) -> np.ndarray:
    scene.update_render()
    camera.take_picture()
    return camera.get_picture("Color")[..., :3].copy()


def _center(image: np.ndarray) -> np.ndarray:
    """Mean color of the image center, which the box covers."""
    quarter = SIZE // 4
    return image[quarter:-quarter, quarter:-quarter].reshape(-1, 3).mean(axis=0)


def _box_column(image: "torch.Tensor") -> float:
    """Mean image column of the box in an H x W x 4 picture (the background has alpha 0)."""
    mask = (image[..., 3] > 0.5).float()
    columns = torch.arange(image.shape[1], device=image.device, dtype=torch.float32)
    return float((mask * columns).sum() / mask.sum())


class _PhysxRenderJob:
    """One device's GPU PhysX scene with a box drifting sideways, rendered by its own group."""

    def __init__(self, device: str, velocity: float) -> None:
        config = sapien.physx.PhysxSceneConfig()
        config.gravity = [0.0, 0.0, 0.0]
        sapien.physx.set_scene_config(config)
        self.system = sapien.physx.PhysxGpuSystem(device)
        self.system.set_timestep(TIMESTEP)
        self.scene = sapien.Scene([self.system, sapien.render.RenderSystem(device)])
        self.scene.set_ambient_light([1.0, 1.0, 1.0])
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.2, 0.2, 0.2])
        builder.add_box_visual(
            half_size=[0.2, 0.2, 0.2],
            material=sapien.render.RenderMaterial(base_color=[1.0, 1.0, 1.0, 1.0]),
        )
        builder.build()
        camera = self.scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0)
        camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
        self.system.gpu_init()
        bodies = self.system.cuda_rigid_dynamic_data.torch()
        self.system.gpu_fetch_rigid_dynamic_data()
        bodies[:, 7:13] = 0.0
        bodies[:, 8] = velocity
        self.system.gpu_apply_rigid_dynamic_data()
        del bodies
        self.group = sapien.render.RenderSystemGroup([self.scene.render_system])
        self.group.set_cuda_poses(self.system.cuda_rigid_body_data)
        self.camera_group = self.group.create_camera_group([camera], ["Color"])
        self.group.gpu_init()

    def frame(self) -> float:
        """Step physics, render, and return the box's image column read through DLPack."""
        self.system.step()
        self.system.gpu_fetch_rigid_dynamic_data()
        self.group.update_render()
        self.camera_group.take_picture()
        picture = torch.from_dlpack(self.camera_group.get_picture_cuda("Color"))
        return _box_column(picture[0])

    def close(self) -> None:
        self.system.wait_idle()
        gc.collect()
        self.group.close()
        self.scene.close()
        self.system.close()


@unittest.skipUnless(_has_two_cuda_devices(), "requires two CUDA devices and Torch")
class TestMultiGpuRender(unittest.TestCase):
    def tearDown(self) -> None:
        gc.collect()

    def test_shared_material_renders_and_updates_on_every_device(self) -> None:
        red, green = [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]
        # reference: a material created green renders the expected green image
        scene, camera = _build_scene(DEVICES[0], sapien.render.RenderMaterial(base_color=green))
        try:
            expected_green = _capture(scene, camera)
        finally:
            scene.close()

        material = sapien.render.RenderMaterial(base_color=red)
        jobs = [_build_scene(device, material) for device in DEVICES]
        try:
            before = [_capture(scene, camera) for scene, camera in jobs]
            np.testing.assert_allclose(before[0], before[1], atol=1e-3)
            red_center = _center(before[0])
            self.assertGreater(red_center[0], red_center[1] + 0.2)

            # one host-side edit must reach the copy uploaded to each device
            material.base_color = green
            for scene, camera in jobs:
                np.testing.assert_allclose(_capture(scene, camera), expected_green, atol=1e-3)
        finally:
            for scene, _ in jobs:
                scene.close()

    def test_camera_group_outputs_live_on_their_render_device(self) -> None:
        material = sapien.render.RenderMaterial(base_color=[0.2, 0.4, 0.8, 1.0])
        jobs = [_build_scene(device, material) for device in DEVICES]
        groups: List[sapien.render.RenderSystemGroup] = []
        try:
            reference = _center(_capture(*jobs[0]))
            images = []
            current = torch.cuda.current_device()
            for index, (scene, camera) in enumerate(jobs):
                group = sapien.render.RenderSystemGroup([scene.render_system])
                groups.append(group)
                camera_group = group.create_camera_group([camera], ["Color"])
                group.gpu_init()
                group.update_render()
                camera_group.take_picture()
                picture = camera_group.get_picture_cuda("Color")
                self.assertEqual(picture.cuda_id, index)
                tensor = picture.torch()
                self.assertEqual(tensor.device, torch.device("cuda", index))
                torch.cuda.synchronize(tensor.device)
                images.append(tensor[0, ..., :3].cpu().numpy())
                del tensor, picture
            # rendering on another device must not switch the caller's current device
            self.assertEqual(torch.cuda.current_device(), current)
            np.testing.assert_allclose(images[0], images[1], atol=1e-3)
            np.testing.assert_allclose(_center(images[0]), reference, atol=0.02)
        finally:
            gc.collect()
            for group in groups:
                group.close()
            for scene, _ in jobs:
                scene.close()

    def test_render_system_group_rejects_mixed_devices(self) -> None:
        material = sapien.render.RenderMaterial()
        jobs = [_build_scene(device, material) for device in DEVICES]
        try:
            with self.assertRaisesRegex(RuntimeError, "device"):
                sapien.render.RenderSystemGroup([scene.render_system for scene, _ in jobs])
        finally:
            for scene, _ in jobs:
                scene.close()

    def test_viewer_rejects_a_scene_of_another_device(self) -> None:
        """The viewer renders through the default render device and raises for a scene that
        renders on another one."""
        from sapien.utils import Viewer

        default_system = sapien.render.RenderSystem()
        default_id = default_system.device.cuda_id
        default_system.close()
        material = sapien.render.RenderMaterial()
        jobs = [_build_scene(device, material) for device in DEVICES]
        ids = [scene.render_system.device.cuda_id for scene, _ in jobs]
        viewer = None
        try:
            if default_id not in ids:
                self.skipTest("the default render device is none of the tested devices")
            own = jobs[ids.index(default_id)][0]
            foreign = jobs[1 - ids.index(default_id)][0]
            viewer = Viewer(resolutions=(SIZE, SIZE))
            viewer.set_scene(own)
            viewer.update_render()
            viewer.render()
            with self.assertRaisesRegex(RuntimeError, "renders on device"):
                viewer.set_scene(foreign)
        finally:
            if viewer is not None:
                viewer.close()
            for scene, _ in jobs:
                scene.close()

    def test_devices_render_concurrently_from_threads(self) -> None:
        material = sapien.render.RenderMaterial(base_color=[0.9, 0.6, 0.1, 1.0])
        jobs = [_build_scene(device, material) for device in DEVICES]
        try:
            # the first capture on each device uploads the shared mesh and material
            # concurrently from both threads
            results: List[List[np.ndarray]] = [[] for _ in jobs]
            errors: List[BaseException] = []

            def render(index: int) -> None:
                try:
                    scene, camera = jobs[index]
                    for _ in range(FRAMES):
                        results[index].append(_capture(scene, camera))
                except BaseException as error:  # reported on the main thread
                    errors.append(error)

            threads = [threading.Thread(target=render, args=(i,)) for i in range(len(jobs))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            reference = results[0][0]
            for frames in results:
                self.assertEqual(len(frames), FRAMES)
                for image in frames:
                    np.testing.assert_allclose(image, reference, atol=1e-3)
        finally:
            for scene, _ in jobs:
                scene.close()

    def test_render_groups_follow_physx_from_threads(self) -> None:
        """One thread per GPU steps its PhysX system and renders its group; the boxes move
        exactly as when one thread drives both devices in turn."""
        sapien.physx.enable_gpu()
        old_config = sapien.physx.get_scene_config()
        velocities = (SPEED, -SPEED)

        def run(threaded: bool) -> List[List[float]]:
            jobs = [_PhysxRenderJob(device, v) for device, v in zip(DEVICES, velocities)]
            columns: List[List[float]] = [[] for _ in jobs]
            try:
                if not threaded:
                    for _ in range(FRAMES):
                        for job, out in zip(jobs, columns):
                            out.append(job.frame())
                    return columns
                errors: List[BaseException] = []

                def drive(index: int) -> None:
                    try:
                        for _ in range(FRAMES):
                            columns[index].append(jobs[index].frame())
                    except BaseException as error:  # reported on the main thread
                        errors.append(error)

                threads = [threading.Thread(target=drive, args=(i,)) for i in range(len(jobs))]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                self.assertEqual(errors, [])
                return columns
            finally:
                for job in jobs:
                    job.close()

        try:
            sequential = run(threaded=False)
            threaded = run(threaded=True)
        finally:
            sapien.physx.set_scene_config(old_config)
        shifts = [frames[-1] - frames[0] for frames in sequential]
        self.assertGreater(min(abs(shift) for shift in shifts), 2.0, "the boxes must move")
        self.assertLess(shifts[0] * shifts[1], 0.0, "the devices' boxes move in opposite ways")
        np.testing.assert_allclose(threaded, sequential, atol=0.5)

    def test_render_group_rejects_a_stream_of_another_device(self) -> None:
        scene, _ = _build_scene(DEVICES[1], sapien.render.RenderMaterial())
        group = sapien.render.RenderSystemGroup([scene.render_system])
        try:
            foreign = torch.cuda.Stream(device=0)
            with self.assertRaisesRegex(RuntimeError, "cuda:1"):
                group.set_cuda_stream(foreign.cuda_stream)
            group.set_cuda_stream(torch.cuda.Stream(device=1).cuda_stream)
        finally:
            group.close()
            scene.close()

    def test_rendering_on_one_device_creates_no_context_on_another(self) -> None:
        """A process that simulates and renders on cuda:1 never claims memory on cuda:0, so
        per-GPU ranks that keep every GPU visible do not each pay for a context on the first."""
        script = textwrap.dedent(
            """
            import ctypes, gc
            import sapien

            sapien.physx.enable_gpu()
            config = sapien.physx.PhysxSceneConfig()
            config.gravity = [0.0, 0.0, 0.0]
            sapien.physx.set_scene_config(config)
            system = sapien.physx.PhysxGpuSystem("cuda:1")
            scene = sapien.Scene([system, sapien.render.RenderSystem("cuda:1")])
            builder = scene.create_actor_builder()
            builder.add_box_collision(half_size=[0.2, 0.2, 0.2])
            builder.add_box_visual(half_size=[0.2, 0.2, 0.2])
            builder.build()
            camera = scene.add_camera("camera", 32, 32, 1.0, 0.1, 20.0)
            camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
            system.gpu_init()
            group = sapien.render.RenderSystemGroup([scene.render_system])
            group.set_cuda_poses(system.cuda_rigid_body_data)
            camera_group = group.create_camera_group([camera], ["Color"])
            group.gpu_init()
            system.step()
            system.gpu_fetch_rigid_dynamic_data()
            group.update_render()
            camera_group.take_picture()
            assert camera_group.get_picture_cuda("Color").cuda_id == 1
            gc.collect()
            group.close()
            scene.close()
            system.close()

            cuda = ctypes.CDLL("libcuda.so.1")
            device = ctypes.c_int()
            assert cuda.cuDeviceGet(ctypes.byref(device), 0) == 0
            flags, active = ctypes.c_uint(), ctypes.c_int()
            assert cuda.cuDevicePrimaryCtxGetState(device, ctypes.byref(flags), ctypes.byref(active)) == 0
            print("CUDA0_PRIMARY_CONTEXT_ACTIVE", active.value)
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CUDA0_PRIMARY_CONTEXT_ACTIVE 0", result.stdout)

    def test_process_seeing_one_gpu_simulates_and_renders_on_it(self) -> None:
        """With CUDA_VISIBLE_DEVICES naming one GPU, default PhysX and render systems both pick
        that GPU although Vulkan still enumerates every GPU."""
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        second = visible.split(",")[1] if visible else "1"
        expected_pci = sapien.Device(DEVICES[1]).pci_string
        script = textwrap.dedent(
            """
            import gc
            import sapien, torch

            sapien.physx.enable_gpu()
            physx_system = sapien.physx.PhysxGpuSystem()
            render_system = sapien.render.RenderSystem()
            print("PHYSX", physx_system.device.cuda_id, physx_system.device.pci_string)
            print("RENDER", render_system.device.cuda_id, render_system.device.pci_string)
            scene = sapien.Scene([physx_system, render_system])
            scene.set_ambient_light([1.0, 1.0, 1.0])
            builder = scene.create_actor_builder()
            builder.add_box_collision(half_size=[0.2, 0.2, 0.2])
            builder.add_box_visual(half_size=[0.2, 0.2, 0.2])
            builder.build()
            camera = scene.add_camera("camera", 32, 32, 1.0, 0.1, 20.0)
            camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
            physx_system.gpu_init()
            group = sapien.render.RenderSystemGroup([render_system])
            group.set_cuda_poses(physx_system.cuda_rigid_body_data)
            camera_group = group.create_camera_group([camera], ["Color"])
            group.gpu_init()
            group.update_render()
            camera_group.take_picture()
            picture = torch.from_dlpack(camera_group.get_picture_cuda("Color"))
            print("BOX_VISIBLE", bool(picture[..., :3].max() > 0.1))
            del picture
            gc.collect()
            group.close()
            scene.close()
            physx_system.close()
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": second},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"PHYSX 0 {expected_pci}", result.stdout)
        self.assertIn(f"RENDER 0 {expected_pci}", result.stdout)
        self.assertIn("BOX_VISIBLE True", result.stdout)

    def test_shutdown_releases_every_device(self) -> None:
        material = sapien.render.RenderMaterial()
        jobs = [_build_scene(device, material) for device in DEVICES]
        for scene, camera in jobs:
            _capture(scene, camera)
        for scene, _ in jobs:
            render_system = scene.render_system
            scene.close()
            render_system.close()
        del jobs, scene, camera, material, render_system
        gc.collect()
        self.assertTrue(sapien.render.can_shutdown(), sapien.render.get_live_resources())
        sapien.render.shutdown()
        self.assertFalse(sapien.render.get_live_resources()["engine_exists"])

        # a fresh engine can render on the non-default device again
        scene, camera = _build_scene(DEVICES[1], sapien.render.RenderMaterial())
        try:
            self.assertGreater(_center(_capture(scene, camera)).sum(), 0.1)
        finally:
            scene.close()


if __name__ == "__main__":
    unittest.main()
