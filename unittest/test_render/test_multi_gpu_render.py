"""Vulkan rendering on several CUDA devices in one process.

Every test needs at least two render-capable CUDA devices and CUDA-enabled Torch and is
skipped otherwise. Each device renders the same scene built from shared (device-independent)
materials and meshes, so the images of all devices must agree.
"""

import gc
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
