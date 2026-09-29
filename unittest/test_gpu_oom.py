"""Opt-in GPU OOM recovery test; run alone on an otherwise idle device.

SAPIEN_TEST_GPU_OOM=1 enables this test. Reserve device memory directly rather
than constructing scenes with giant PhysX heaps: those also allocate pinned host
memory and can leak pages through kernel/driver failure paths on some machines.
One attempted scene uses a bounded 256 MiB heap, with 128 MiB GPU headroom left.
The failed creation and all recovery assertions deliberately share one process.
"""

import ctypes
import gc
import os
import unittest

import sapien


@unittest.skipUnless(
    os.environ.get("SAPIEN_TEST_GPU_OOM") == "1",
    "requires explicit SAPIEN_TEST_GPU_OOM=1 and an isolated idle GPU",
)
class TestGpuSceneCreationFailure(unittest.TestCase):
    cuda: ctypes.CDLL

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.cuda = ctypes.CDLL("libcuda.so.1")
        except OSError as error:
            raise unittest.SkipTest("CUDA driver unavailable") from error
        cls.cuda.cuInit.argtypes = [ctypes.c_uint]
        status = cls.cuda.cuInit(0)
        if status == 100:  # CUDA_ERROR_NO_DEVICE
            raise unittest.SkipTest("no CUDA device")
        if status != 0:
            raise RuntimeError(f"cuInit failed: CUDA error {status}")
        cls.cuda.cuMemGetInfo_v2.argtypes = [
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)
        ]
        cls.cuda.cuMemAlloc_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t
        ]
        cls.cuda.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        gc.collect()
        if not sapien.physx.is_gpu_enabled():
            sapien.physx.enable_gpu()

    def _build_system(self) -> tuple[sapien.physx.PhysxGpuSystem, sapien.Scene]:
        px = sapien.physx.PhysxGpuSystem()
        scene = sapien.Scene([px])
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.05, 0.05, 0.05], density=1000)
        builder.set_initial_pose(sapien.Pose([0.0, 0.0, 0.5]))
        builder.build(name="box")
        px.gpu_init()
        return px, scene

    def test_creation_oom_raises_and_spares_existing_systems(self) -> None:
        import torch

        # SAPIEN teardown must not reset the primary context shared with callers.
        external = torch.arange(16, device="cuda", dtype=torch.int32)
        self._check_creation_failure_with_live_system()
        gc.collect()
        self.assertTrue(sapien.can_shutdown(), sapien.get_live_resources())
        sapien.shutdown()
        sapien.physx.enable_gpu()
        system, scene = self._build_system()
        try:
            for _ in range(3):
                system.step()
                system.gpu_fetch_rigid_dynamic_data()
            torch.testing.assert_close(external.cpu(), torch.arange(16, dtype=torch.int32))
        finally:
            scene.close()
            system.close()

    def _check_creation_failure_with_live_system(self) -> None:
        sapien.physx.set_gpu_memory_config()
        px, scene = self._build_system()
        reservation = ctypes.c_uint64()
        try:
            # Warm the baseline before inducing failure.
            for _ in range(3):
                px.step()
                px.gpu_fetch_rigid_dynamic_data()
            free, total = ctypes.c_size_t(), ctypes.c_size_t()
            self.assertEqual(self.cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)), 0)
            self.assertGreater(free.value, 512 * 1024**2)
            # Leave headroom for the display/driver, but less than the attempted
            # scene's heap. Never retry or use multi-GiB pinned-host heap requests.
            sapien.physx.set_gpu_memory_config(heap_capacity=256 * 1024**2)
            self.assertEqual(
                self.cuda.cuMemAlloc_v2(ctypes.byref(reservation), free.value - 128 * 1024**2), 0
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "failed to create PhysX scene"):
                    sapien.physx.PhysxGpuSystem()
            finally:
                status = self.cuda.cuMemFree_v2(reservation)
                reservation.value = 0
                sapien.physx.set_gpu_memory_config()
                self.assertEqual(status, 0)

            for _ in range(3):
                px.step()
                px.gpu_fetch_rigid_dynamic_data()
                other_px, other_scene = self._build_system()
                try:
                    other_px.step()
                    other_px.gpu_fetch_rigid_dynamic_data()
                finally:
                    other_scene.close()
                    other_px.close()
        finally:
            sapien.physx.set_gpu_memory_config()
            if reservation.value:
                self.cuda.cuMemFree_v2(reservation)
            scene.close()
            px.close()


if __name__ == "__main__":
    unittest.main()
