"""CUDA interop and threading of RenderSystemGroup / RenderCameraGroup on one GPU.

Torch supplies streams, delay kernels and assertions. Delay kernels make every ordering
deterministic: a missing dependency lets a late write land after (or before) the copy that
take_picture() makes into the exported picture buffers.
"""

import gc
import threading
import time
import unittest
from typing import List, Tuple

import sapien

try:
    import torch
except ImportError:  # pragma: no cover - depends on the test environment
    torch = None

SIZE = 32
SLEEP_CYCLES = 400_000_000


def _has_cuda() -> bool:
    return torch is not None and torch.cuda.is_available()


def _build_scene() -> Tuple[sapien.Scene, sapien.render.RenderCameraComponent]:
    """A lit box in front of a camera, rendered on the default device."""
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem()])
    scene.set_ambient_light([1.0, 1.0, 1.0])
    builder = scene.create_actor_builder()
    builder.add_box_visual(
        half_size=[0.5, 0.5, 0.5],
        material=sapien.render.RenderMaterial(base_color=[0.8, 0.3, 0.1, 1.0]),
    )
    builder.build_kinematic()
    camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0)
    camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])
    return scene, camera


@unittest.skipUnless(_has_cuda(), "requires a CUDA device and CUDA-enabled Torch")
class TestBatchedRenderInterop(unittest.TestCase):
    def setUp(self) -> None:
        self.scene, camera = _build_scene()
        self.group = sapien.render.RenderSystemGroup([self.scene.render_system])
        self.camera_group = self.group.create_camera_group([camera], ["Color"])
        self.group.gpu_init()
        self.device = torch.device("cuda", self.camera_group.get_picture_cuda("Color").cuda_id)
        # Torch work below (delays, fills) must run on the group's stream: the default render
        # device is the first CUDA device in Vulkan order, which need not be Torch's cuda:0.
        self.addCleanup(torch.cuda.set_device, torch.cuda.current_device())
        torch.cuda.set_device(self.device)
        # Load every kernel used below first: a first launch synchronizes the whole context
        # and would hide a missing dependency.
        torch.cuda._sleep(1)
        self.group.update_render()
        self.camera_group.take_picture()
        picture = self._picture()
        self.expected = picture.clone()
        picture.fill_(-1.0)
        self.camera_group.take_picture()
        torch.cuda.synchronize(self.device)
        del picture
        self.assertGreater(float(self.expected[..., :3].max()), 0.1, "the box must be visible")

    def tearDown(self) -> None:
        torch.cuda.synchronize(self.device)
        gc.collect()
        self.group.close()
        self.scene.close()

    def _picture(self) -> "torch.Tensor":
        """Zero-copy view of the exported picture buffer, ordered on the current stream."""
        return torch.from_dlpack(self.camera_group.get_picture_cuda("Color"))

    def _assert_rendered(self, picture: "torch.Tensor") -> None:
        torch.cuda.synchronize(self.device)
        torch.testing.assert_close(picture, self.expected, rtol=0.0, atol=0.0)

    def test_take_picture_waits_for_the_group_stream(self) -> None:
        """A write queued on the group's stream before take_picture() lands before its copy."""
        picture = self._picture()  # Torch's default stream is the group's default stream
        torch.cuda._sleep(SLEEP_CYCLES)
        picture.fill_(-1.0)
        self.camera_group.take_picture()
        self._assert_rendered(picture)

    def test_gpu_wait_for_orders_a_consumer_on_another_stream(self) -> None:
        """A consumer on another stream fenced with gpu_wait_for() finishes before the next copy."""
        side = torch.cuda.Stream(self.device)
        with torch.cuda.stream(side):
            picture = self._picture()
            torch.cuda._sleep(SLEEP_CYCLES)
            picture.fill_(-1.0)
            consumed = torch.ones(1, device=self.device)
            # Torch's DLPack export makes the group's stream wait for the current stream.
            self.camera_group.gpu_wait_for(consumed)
        self.camera_group.take_picture()
        self._assert_rendered(picture)

    def test_take_picture_waits_without_holding_the_gil(self) -> None:
        """take_picture() blocks for the previous copy, which waits for the stream, while other
        Python threads keep running."""
        torch.cuda._sleep(SLEEP_CYCLES)
        self.camera_group.take_picture()  # its copy waits for the delay
        durations: List[float] = []
        errors: List[BaseException] = []
        started, finished = threading.Event(), threading.Event()

        def capture() -> None:
            try:
                started.set()
                begin = time.perf_counter()
                self.camera_group.take_picture()  # waits for the previous copy
                durations.append(time.perf_counter() - begin)
            except BaseException as error:  # surfaced in the main thread below
                errors.append(error)
            finally:
                finished.set()

        worker = threading.Thread(target=capture)
        worker.start()
        started.wait()
        progress = 0
        while not finished.is_set():
            progress += 1
        worker.join()
        self.assertEqual(errors, [])
        self.assertGreater(durations[0], 0.03, "take_picture() did not wait for the stream")
        self.assertGreater(progress, 1000, "the main thread stalled while take_picture() waited")

    def test_concurrent_calls_into_one_group_are_serialized(self) -> None:
        errors: List[BaseException] = []

        def drive() -> None:
            try:
                for _ in range(30):
                    self.group.update_render()
                    self.camera_group.take_picture()
            except BaseException as error:  # surfaced in the main thread below
                errors.append(error)

        threads = [threading.Thread(target=drive) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self._assert_rendered(self._picture())

    def test_close_rejects_views_requested_while_it_runs(self) -> None:
        """A view requested from another thread while close() runs raises instead of outliving
        the buffers it views."""
        busy = torch.cuda.Stream(self.device)
        with torch.cuda.stream(busy):
            torch.cuda._sleep(4 * SLEEP_CYCLES)  # close() waits for the device
        errors: List[BaseException] = []

        def close() -> None:
            try:
                self.group.close()
            except BaseException as error:  # surfaced in the main thread below
                errors.append(error)

        closer = threading.Thread(target=close)
        closer.start()
        time.sleep(0.1)  # close() holds the group while it waits for the device
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.camera_group.get_picture_cuda("Color")
        closer.join()
        self.assertEqual(errors, [])
        self.assertTrue(self.camera_group.is_closed)
        self.assertEqual(self.camera_group.outstanding_cuda_view_count, 0)


if __name__ == "__main__":
    unittest.main()
