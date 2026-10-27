"""CUDA interop of PhysxGpuSystem: DLPack stream handshake, borrowed inputs, host-async calls.

Torch supplies CUDA buffers, streams, delay kernels and assertions; the JAX cases run only with a
CUDA-enabled JAX. All simulation is SAPIEN. Delay kernels make every ordering deterministic: a
missing stream dependency reads stale data or trips index validation while the delay runs.
"""

import ctypes
import gc
import os
import threading
import unittest
import weakref

import sapien

# JAX shares the GPU with Torch and PhysX here; never let it grab most of the memory.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

try:
    import torch
except ImportError:  # pragma: no cover - depends on the test environment
    torch = None

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX_CUDA = any(device.platform == "gpu" for device in jax.devices())
except Exception:  # pragma: no cover - depends on the test environment
    jax = None
    jnp = None
    HAS_JAX_CUDA = False

ARTICULATIONS = 32
SLEEP_CYCLES = 400_000_000
TIMESTEP = 0.005

_capsule_is_valid = ctypes.pythonapi.PyCapsule_IsValid
_capsule_is_valid.argtypes = [ctypes.py_object, ctypes.c_char_p]
_capsule_is_valid.restype = ctypes.c_int


class _ForwardingArray:
    """A DLPack producer that only forwards to a tensor; weak-referenceable for lifetime checks."""

    def __init__(self, tensor: "torch.Tensor") -> None:
        self.tensor = tensor

    def __dlpack__(self, **kwargs):
        return self.tensor.__dlpack__(**kwargs)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


@unittest.skipUnless(
    torch is not None and torch.cuda.is_available(), "requires a CUDA device and CUDA-enabled Torch"
)
class TestGpuInterop(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sapien.physx.enable_gpu()

    def setUp(self) -> None:
        self.old_config = sapien.physx.get_scene_config()
        self.system = None

    def tearDown(self) -> None:
        if self.system is not None:
            self.system.wait_idle()
            gc.collect()
            self.scene.close()
            self.system.close()
        sapien.physx.set_scene_config(self.old_config)

    def _build_sliders(self, *, index_validation: bool = True) -> None:
        """Fixed-root articulations with one horizontal prismatic joint and no friction."""
        config = sapien.physx.PhysxSceneConfig()
        config.gravity = [0.0, 0.0, -9.81]
        config.num_scenes = 1
        config.gpu_index_validation = index_validation
        sapien.physx.set_scene_config(config)
        self.system = sapien.physx.PhysxGpuSystem()
        self.system.set_timestep(TIMESTEP)
        self.scene = sapien.Scene([self.system])
        builder = self.scene.create_articulation_builder()
        root = builder.create_link_builder()
        root.add_box_collision(half_size=[0.05, 0.05, 0.05])
        slider = builder.create_link_builder(root)
        slider.add_box_collision(half_size=[0.05, 0.05, 0.05])
        slider.set_joint_properties(
            "prismatic", limits=[[-100.0, 100.0]],
            pose_in_parent=sapien.Pose([0.2, 0.0, 0.0]),
            pose_in_child=sapien.Pose(), friction=0.0, damping=0.0,
        )
        for index in range(ARTICULATIONS):
            builder.set_initial_pose(sapien.Pose([0.0, float(index), 0.0]))
            builder.build(fix_root_link=True)
        self.system.gpu_init()
        self.device = torch.device("cuda", self.system.device.cuda_id)

    def _qvel_after_step(self) -> "torch.Tensor":
        """Joint velocity one step after the currently applied state."""
        self.system.step()
        self.system.gpu_fetch_articulation_qvel()
        qvel = torch.from_dlpack(self.system.cuda_articulation_qvel).clone()
        torch.cuda.synchronize(self.device)
        return qvel[:, 0]

    def test_dlpack_export_is_a_tracked_zero_copy_view(self) -> None:
        self._build_sliders()
        self.assertEqual(
            self.system.cuda_articulation_qpos.__dlpack_device__(), (2, self.system.device.cuda_id)
        )
        gc.collect()
        views = self.system.outstanding_cuda_view_count
        tensor = torch.from_dlpack(self.system.cuda_articulation_qpos)
        gc.collect()
        self.assertEqual(self.system.outstanding_cuda_view_count, views + 1)
        self.assertEqual(tensor.data_ptr(), self.system.cuda_articulation_qpos.ptr)
        del tensor
        gc.collect()
        self.assertEqual(self.system.outstanding_cuda_view_count, views)

    def test_dlpack_capsule_versions(self) -> None:
        self._build_sliders()
        array = self.system.cuda_articulation_qpos
        self.assertTrue(_capsule_is_valid(array.__dlpack__(max_version=(1, 0)), b"dltensor_versioned"))
        self.assertTrue(_capsule_is_valid(array.__dlpack__(), b"dltensor"))
        self.assertTrue(_capsule_is_valid(array.dlpack(), b"dltensor"))
        with self.assertRaises(BufferError):
            array.__dlpack__(copy=True)

    def test_export_orders_the_consumer_after_the_system_stream(self) -> None:
        """from_dlpack on another stream must wait for writes queued on the system's stream."""
        self._build_sliders()
        producer = torch.cuda.Stream(self.device)
        consumer = torch.cuda.Stream(self.device)
        self.system.gpu_set_cuda_stream(producer.cuda_stream)
        self.assertEqual(self.system.gpu_get_cuda_stream(), producer.cuda_stream)
        with torch.cuda.stream(producer):
            view = torch.from_dlpack(self.system.cuda_articulation_qvel)
            view.fill_(1.0)
            torch.cuda._sleep(SLEEP_CYCLES)
            view.fill_(7.0)
        with torch.cuda.stream(consumer):
            seen = torch.from_dlpack(self.system.cuda_articulation_qvel).clone()
        consumer.synchronize()
        torch.testing.assert_close(seen, torch.full_like(seen, 7.0), rtol=0.0, atol=0.0)

    def test_index_buffer_waits_for_its_producer_stream(self) -> None:
        """Index validation reads the buffer only after the producer stream filled it in."""
        self._build_sliders()
        producer = torch.cuda.Stream(self.device)
        valid = torch.arange(ARTICULATIONS, dtype=torch.int32, device=self.device)
        # Load the validation kernel first: a lazily loaded module synchronizes the whole context
        # on its first launch, which would hide a missing stream dependency.
        self.system.gpu_update_articulation_kinematics(valid)
        # Out-of-range until the producer stream overwrites it after the delay.
        indices = torch.full((ARTICULATIONS,), 1 << 20, dtype=torch.int32, device=self.device)
        torch.cuda.synchronize(self.device)
        with torch.cuda.stream(producer):
            torch.cuda._sleep(SLEEP_CYCLES)
            indices.copy_(valid)
            # Torch's DLPack export makes the system's stream wait for the current stream.
            self.system.gpu_update_articulation_kinematics(indices)

    def test_borrowed_input_lives_until_the_system_stream_used_it(self) -> None:
        self._build_sliders(index_validation=False)
        stream = torch.cuda.Stream(self.device)
        self.system.gpu_set_cuda_stream(stream.cuda_stream)
        producer = _ForwardingArray(
            torch.arange(ARTICULATIONS, dtype=torch.int32, device=self.device)
        )
        alive = weakref.ref(producer)
        torch.cuda.synchronize(self.device)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(SLEEP_CYCLES)
            self.system.gpu_apply_articulation_qvel(producer)
        # No reference cycle: the last reference frees it unless SAPIEN still holds it. A full
        # gc.collect() here could outlast the delay kernel.
        del producer
        self.assertFalse(stream.query(), "the delay must still be running")
        self.assertIsNotNone(alive(), "released while the system's stream had not used it")
        self.system.wait_idle()
        gc.collect()
        self.assertIsNone(alive(), "not released after the system's stream went idle")

    def test_copy_from_waits_for_the_source_and_applies(self) -> None:
        self._build_sliders()
        producer = torch.cuda.Stream(self.device)
        with torch.cuda.stream(producer):
            source = torch.full((ARTICULATIONS, 1), -1.0, device=self.device)
            torch.cuda._sleep(SLEEP_CYCLES)
            source.fill_(3.0)
            self.system.cuda_articulation_qvel.copy_from(source)
        self.system.gpu_apply_articulation_qvel()
        qvel = self._qvel_after_step()
        torch.testing.assert_close(qvel, torch.full_like(qvel, 3.0), rtol=0.0, atol=1e-3)

    def test_copy_from_rejects_mismatched_arrays(self) -> None:
        self._build_sliders()
        target = self.system.cuda_articulation_qvel
        with self.assertRaisesRegex(RuntimeError, "shape"):
            target.copy_from(torch.zeros((ARTICULATIONS, 2), device=self.device))
        with self.assertRaisesRegex(RuntimeError, "type"):
            target.copy_from(torch.zeros((ARTICULATIONS, 1), dtype=torch.float64, device=self.device))
        with self.assertRaisesRegex(RuntimeError, "CUDA array"):
            target.copy_from(torch.zeros((ARTICULATIONS, 1)))

    def test_gpu_wait_for_orders_the_system_stream_after_a_consumer(self) -> None:
        """Writes into a SAPIEN buffer on another stream reach the apply after gpu_wait_for."""
        self._build_sliders()
        producer = torch.cuda.Stream(self.device)
        with torch.cuda.stream(producer):
            qvel = torch.from_dlpack(self.system.cuda_articulation_qvel)
            qvel.fill_(-1.0)
            torch.cuda._sleep(SLEEP_CYCLES)
            qvel.fill_(2.0)
            self.system.gpu_wait_for(qvel)
        del qvel
        self.system.gpu_apply_articulation_qvel()
        result = self._qvel_after_step()
        torch.testing.assert_close(result, torch.full_like(result, 2.0), rtol=0.0, atol=1e-3)

    def test_apply_returns_before_the_system_stream_drains(self) -> None:
        """PhysX writes complete on the stream; the host is not blocked behind earlier work."""
        self._build_sliders()
        stream = torch.cuda.Stream(self.device)
        self.system.gpu_set_cuda_stream(stream.cuda_stream)
        for apply in (
            self.system.gpu_apply_articulation_qvel,
            self.system.gpu_apply_articulation_qf,
            self.system.gpu_apply_articulation_target_position,
        ):
            with self.subTest(apply=apply.__name__):
                with torch.cuda.stream(stream):
                    torch.cuda._sleep(SLEEP_CYCLES)
                apply()
                self.assertFalse(stream.query(), "apply waited for the delay kernel")
                self.system.wait_idle()
                self.assertTrue(stream.query())

    def test_wait_idle_ignores_other_streams(self) -> None:
        self._build_sliders()
        other = torch.cuda.Stream(self.device)
        with torch.cuda.stream(other):
            torch.cuda._sleep(SLEEP_CYCLES)
        self.system.wait_idle()
        self.assertFalse(other.query(), "wait_idle waited for an unrelated stream")
        other.synchronize()

    def test_handoffs_do_not_wait_for_unrelated_device_work(self) -> None:
        """Stream handshakes and releasing a borrowed input never wait for the whole device."""
        self._build_sliders()
        busy = torch.cuda.Stream(self.device)
        consumer = torch.cuda.Stream(self.device)
        source = torch.zeros(tuple(self.system.cuda_articulation_qf.shape), device=self.device)
        with torch.cuda.stream(busy):
            torch.cuda._sleep(SLEEP_CYCLES)
        with torch.cuda.stream(consumer):
            view = torch.from_dlpack(self.system.cuda_articulation_qpos)
        del view
        self.system.cuda_articulation_qf.copy_from(source)
        torch.cuda.current_stream(self.device).synchronize()  # the first borrow completes...
        self.system.cuda_articulation_qf.copy_from(source)  # ...and this call releases it
        self.assertFalse(busy.query(), "a SAPIEN call waited for unrelated device work")
        busy.synchronize()

    def test_data_calls_raise_while_a_step_is_in_flight(self) -> None:
        self._build_sliders()
        self.system.step_start()
        for call in (self.system.gpu_fetch_articulation_qpos, self.system.gpu_apply_articulation_qvel):
            with self.subTest(call=call.__name__):
                with self.assertRaisesRegex(RuntimeError, "in flight"):
                    call()
        self.system.step_finish()
        self.system.gpu_fetch_articulation_qpos()

    def test_concurrent_calls_into_one_system_are_serialized(self) -> None:
        self._build_sliders()
        steps = self.system.total_steps
        errors: list[BaseException] = []

        def drive() -> None:
            try:
                for _ in range(100):
                    self.system.gpu_fetch_articulation_qvel()
                    self.system.gpu_apply_articulation_qvel()
                    self.system.step()
            except BaseException as error:  # surfaced in the main thread below
                errors.append(error)

        threads = [threading.Thread(target=drive) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(self.system.total_steps, steps + 200)

    @unittest.skipUnless(HAS_JAX_CUDA, "requires a CUDA-enabled JAX")
    def test_jax_reads_and_writes_through_dlpack(self) -> None:
        self._build_sliders()
        jax_device = next(
            d for d in jax.devices() if d.local_hardware_id == self.system.device.cuda_id
        )
        self.system.gpu_fetch_articulation_qpos()
        qpos = jax.dlpack.from_dlpack(self.system.cuda_articulation_qpos)
        self.assertEqual(qpos.devices(), {jax_device})
        reference = torch.from_dlpack(self.system.cuda_articulation_qpos).cpu().numpy()
        self.assertTrue((jax.device_get(qpos) == reference).all())
        del qpos

        velocity = jax.device_put(jnp.full((ARTICULATIONS, 1), 4.0, dtype=jnp.float32), jax_device)
        self.system.cuda_articulation_qvel.copy_from(velocity * 1.0)
        indices = jax.device_put(jnp.arange(ARTICULATIONS, dtype=jnp.int32), jax_device)
        self.system.gpu_apply_articulation_qvel(indices)
        qvel = self._qvel_after_step()
        torch.testing.assert_close(qvel, torch.full_like(qvel, 4.0), rtol=0.0, atol=1e-3)
        self.assertEqual(self.system.cuda_articulation_qvel.jax().shape, (ARTICULATIONS, 1))
        gc.collect()


if __name__ == "__main__":
    unittest.main()
