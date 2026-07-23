import gc
import unittest

import sapien


def _cuda_device_available() -> bool:
    """Skip only on identifiable no-GPU conditions (missing driver or no CUDA device);
    any other GPU initialization failure must propagate as a test error."""
    import ctypes

    try:
        lib = ctypes.CDLL("libcuda.so")
    except OSError:
        return False
    return lib.cuInit(0) == 0  # nonzero includes CUDA_ERROR_NO_DEVICE


# Bounded VRAM ceiling for the exhaustion loop: 128 systems x ~4 GiB heap = 512 GiB.
_MAX_JUMBO_SYSTEMS = 128
# Largest uint32-representable heap (~4 GiB); PhysX allocates it eagerly at scene creation.
_JUMBO_HEAP_CAPACITY = 2**32 - 1024


class TestGpuSceneCreationFailure(unittest.TestCase):
    """GPU scene creation at VRAM exhaustion must raise a Python error instead of
    crashing the process, must not freeze other GPU systems sharing the device's
    PxCudaContext (PhysX latches it into an out-of-memory abort mode), and must
    allow creating new systems after memory is freed."""

    @classmethod
    def setUpClass(cls):
        if not _cuda_device_available():
            raise unittest.SkipTest("no usable CUDA device")
        # release scenes leaked by earlier tests; a live scene keeps the PhysxEngine
        # singleton alive and enable_gpu() refuses to run with an existing engine
        gc.collect()
        if not sapien.physx.is_gpu_enabled():
            # deliberately unguarded: enable_gpu() failures are real errors, not skips
            sapien.physx.enable_gpu()

    def _build_system(self) -> tuple[sapien.physx.PhysxGpuSystem, sapien.Scene]:
        """Create a GPU system with one scene holding a single dynamic box."""
        px = sapien.physx.PhysxGpuSystem()
        scene = sapien.Scene([px])
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.05, 0.05, 0.05], density=1000)
        builder.set_initial_pose(sapien.Pose([0.0, 0.0, 0.5]))
        builder.build(name="box")
        px.gpu_init()
        return px, scene

    def test_creation_oom_raises_and_spares_existing_systems(self):
        # baseline system standing in for a training simulation
        px, scene = self._build_system()
        px.step()

        # exhaust VRAM: each extra system eagerly allocates a ~4 GiB PhysX GPU heap
        # at scene creation, so creation must fail within the bounded loop
        jumbo = []
        try:
            sapien.physx.set_gpu_memory_config(heap_capacity=_JUMBO_HEAP_CAPACITY)
            with self.assertRaises(RuntimeError) as ctx:
                for _ in range(_MAX_JUMBO_SYSTEMS):
                    jumbo.append(sapien.physx.PhysxGpuSystem())
            self.assertIn("failed to create PhysX scene", str(ctx.exception))
        finally:
            sapien.physx.set_gpu_memory_config()
            jumbo.clear()
            gc.collect()

        # the baseline system must keep stepping: the failed creation must not leave
        # the shared PxCudaContext in abort mode
        px.step()
        px.gpu_fetch_rigid_dynamic_data()

        # after freeing memory, creating a normally-configured system must succeed
        px2, scene2 = self._build_system()
        px2.step()


if __name__ == "__main__":
    unittest.main()
