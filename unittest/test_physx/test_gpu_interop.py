"""CUDA interop of PhysxGpuSystem: DLPack stream handshake, borrowed inputs, host-async calls.

Torch supplies CUDA buffers, streams, delay kernels and assertions; the JAX cases run only with a
CUDA-enabled JAX, and the two-device case only when Torch and JAX both see two GPUs. All
simulation is SAPIEN. Delay kernels make every ordering deterministic: a missing stream
dependency reads stale data or trips index validation while the delay runs.
"""

import ctypes
import gc
import os
import threading
import time
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

    JAX_GPUS = [device for device in jax.devices() if device.platform == "gpu"]
except Exception:  # pragma: no cover - depends on the test environment
    jax = None
    jnp = None
    JAX_GPUS = []
HAS_JAX_CUDA = bool(JAX_GPUS)

ARTICULATIONS = 32
SLEEP_CYCLES = 400_000_000
TIMESTEP = 0.005
# Closed-loop tracking on two devices: per-step delay on the system stream, controller gain and
# step count (gain * TIMESTEP = 0.1 keeps a one-step-stale trajectory 4% of the target away).
TRACK_DELAY_CYCLES = 50_000_000
TRACK_GAIN = 20.0
TRACK_STEPS = 12

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


class _SwitchingProducer(_ForwardingArray):
    """Switches the system to another stream inside __dlpack__, before its handshake: the
    handshake then orders the stream SAPIEN read before the switch, not the new one."""

    def __init__(self, tensor: "torch.Tensor", system, stream: "torch.cuda.Stream") -> None:
        super().__init__(tensor)
        self.system = system
        self.stream = stream

    def __dlpack__(self, **kwargs):
        self.system.gpu_set_cuda_stream(self.stream.cuda_stream)
        return super().__dlpack__(**kwargs)


def _build_slider_system(
    device: str = "cuda", *, index_validation: bool = True
) -> tuple["sapien.physx.PhysxGpuSystem", "sapien.Scene"]:
    """A GPU system holding fixed-root articulations with one frictionless horizontal slider."""
    config = sapien.physx.PhysxSceneConfig()
    config.gravity = [0.0, 0.0, -9.81]
    config.num_scenes = 1
    config.gpu_index_validation = index_validation
    sapien.physx.set_scene_config(config)
    system = sapien.physx.PhysxGpuSystem(device)
    system.set_timestep(TIMESTEP)
    scene = sapien.Scene([system])
    builder = scene.create_articulation_builder()
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
    system.gpu_init()
    return system, scene


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
        self.system, self.scene = _build_slider_system(index_validation=index_validation)
        self.device = torch.device("cuda", self.system.device.cuda_id)

    def _qvel_after_step(self) -> "torch.Tensor":
        """Joint velocity one step after the currently applied state."""
        self.system.step()
        self.system.gpu_fetch_articulation_qvel()
        qvel = torch.from_dlpack(self.system.cuda_articulation_qvel).clone()
        torch.cuda.synchronize(self.device)
        return qvel[:, 0]

    def _settled_qvel_after_step(self) -> "torch.Tensor":
        """Joint velocity PhysX holds one step after the applied state. Every stream drains before
        the fetch, so a late write into the buffer cannot hide what PhysX received."""
        self.system.step()
        torch.cuda.synchronize(self.device)
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
        # The first indexed apply in a process loads its gather kernel; under CUDA's lazy module
        # loading that load waits for all queued device work, so warm the path up before the delay.
        self.system.gpu_apply_articulation_qvel(producer.tensor)
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

    def test_switching_streams_keeps_queued_work_ordered(self) -> None:
        """Work queued on the previous stream stays ahead of exports and wait_idle() after a
        gpu_set_cuda_stream() switch."""
        self._build_sliders()
        first, second, consumer = (torch.cuda.Stream(self.device) for _ in range(3))
        # Warm up the fetch and delay kernels: a first launch synchronizes the whole context.
        self.system.gpu_fetch_articulation_qvel()
        torch.cuda._sleep(1)
        torch.cuda.synchronize(self.device)
        for check in ("export", "wait_idle"):
            with self.subTest(check=check):
                self.system.gpu_set_cuda_stream(first.cuda_stream)
                with torch.cuda.stream(first):
                    torch.from_dlpack(self.system.cuda_articulation_qvel).fill_(7.0)
                first.synchronize()
                with torch.cuda.stream(first):
                    torch.cuda._sleep(SLEEP_CYCLES)
                    self.system.gpu_fetch_articulation_qvel()  # PhysX holds 0
                self.system.gpu_set_cuda_stream(second.cuda_stream)
                if check == "export":
                    with torch.cuda.stream(consumer):
                        seen = torch.from_dlpack(self.system.cuda_articulation_qvel).clone()
                    consumer.synchronize()
                    torch.testing.assert_close(seen, torch.zeros_like(seen))
                else:
                    self.system.wait_idle()
                    self.assertTrue(first.query(), "wait_idle() returned before the old stream")
                torch.cuda.synchronize(self.device)

    def test_copy_from_after_a_stream_switch_leaves_an_earlier_apply_intact(self) -> None:
        """An apply queued on the previous stream reads its buffer before a copy_from issued after
        a gpu_set_cuda_stream() switch overwrites it."""
        self._build_sliders()
        first, second = torch.cuda.Stream(self.device), torch.cuda.Stream(self.device)
        source = torch.full((ARTICULATIONS, 1), 9.0, device=self.device)
        self.system.gpu_set_cuda_stream(first.cuda_stream)
        with torch.cuda.stream(first):
            torch.from_dlpack(self.system.cuda_articulation_qvel).fill_(5.0)
        # Warm up the apply and delay kernels: a first launch synchronizes the whole context.
        self.system.gpu_apply_articulation_qvel()
        torch.cuda._sleep(1)
        torch.cuda.synchronize(self.device)
        with torch.cuda.stream(first):
            torch.cuda._sleep(SLEEP_CYCLES)
            self.system.gpu_apply_articulation_qvel()  # sends 5, the buffer at this call
        self.system.gpu_set_cuda_stream(second.cuda_stream)
        self.system.cuda_articulation_qvel.copy_from(source)
        qvel = self._settled_qvel_after_step()
        torch.testing.assert_close(qvel, torch.full_like(qvel, 5.0), rtol=0.0, atol=1e-3)

    def test_borrowed_inputs_follow_a_stream_switch_during_the_handshake(self) -> None:
        """A gpu_set_cuda_stream() between SAPIEN reading the system's stream and the producer's
        handshake must not let the call run ahead of the producer on the new stream."""
        self._build_sliders()
        old, new, producer = (torch.cuda.Stream(self.device) for _ in range(3))
        all_rows = torch.arange(ARTICULATIONS, dtype=torch.int32, device=self.device)
        # Warm up every kernel used below: a first launch synchronizes the whole context.
        self.system.gpu_apply_articulation_qvel(all_rows)
        torch.zeros_like(all_rows).copy_(all_rows)
        torch.full((ARTICULATIONS, 1), 1.0, device=self.device).fill_(2.0)
        torch.cuda._sleep(1)
        torch.cuda.synchronize(self.device)
        for call, expected in (("indexed", 5.0), ("copy_from", 3.0), ("gpu_wait_for", 2.0)):
            with self.subTest(call=call):
                self.system.gpu_set_cuda_stream(old.cuda_stream)
                with torch.cuda.stream(old):
                    qvel = torch.from_dlpack(self.system.cuda_articulation_qvel)
                    qvel.zero_()
                self.system.gpu_apply_articulation_qvel()
                with torch.cuda.stream(old):
                    qvel.fill_(expected if call == "indexed" else -1.0)
                del qvel
                torch.cuda.synchronize(self.device)
                with torch.cuda.stream(producer):
                    if call == "indexed":
                        indices = torch.zeros_like(all_rows)
                        torch.cuda._sleep(SLEEP_CYCLES)
                        indices.copy_(all_rows)
                        self.system.gpu_apply_articulation_qvel(
                            _SwitchingProducer(indices, self.system, new)
                        )
                    elif call == "copy_from":
                        source = torch.full((ARTICULATIONS, 1), -1.0, device=self.device)
                        torch.cuda._sleep(SLEEP_CYCLES)
                        source.fill_(expected)
                        self.system.cuda_articulation_qvel.copy_from(
                            _SwitchingProducer(source, self.system, new)
                        )
                    else:
                        qvel = torch.from_dlpack(self.system.cuda_articulation_qvel)
                        torch.cuda._sleep(SLEEP_CYCLES)
                        qvel.fill_(expected)
                        self.system.gpu_wait_for(_SwitchingProducer(qvel, self.system, new))
                        del qvel
                if call != "indexed":
                    self.system.gpu_apply_articulation_qvel()
                result = self._settled_qvel_after_step()
                torch.testing.assert_close(
                    result, torch.full_like(result, expected), rtol=0.0, atol=1e-3
                )

    def test_close_rejects_views_requested_while_it_runs(self) -> None:
        """A view requested from another thread while close() runs raises instead of outliving
        the buffers it views."""
        self._build_sliders()
        self.scene.close()
        delay = torch.cuda.Stream(self.device)
        torch.cuda._sleep(1)  # load the delay kernel: a first launch synchronizes the context
        torch.cuda.synchronize(self.device)
        with torch.cuda.stream(delay):
            torch.cuda._sleep(4 * SLEEP_CYCLES)  # close() waits for it on the device
        errors: list[BaseException] = []

        def close() -> None:
            try:
                self.system.close()
            except BaseException as error:  # surfaced in the main thread below
                errors.append(error)

        closer = threading.Thread(target=close)
        closer.start()
        time.sleep(0.1)  # close() has checked for views and waits for the delay kernel
        with self.assertRaisesRegex(RuntimeError, "closed or being closed"):
            _ = self.system.cuda_articulation_qvel
        closer.join()
        self.assertEqual(errors, [])
        self.assertTrue(self.system.is_closed)
        self.assertEqual(self.system.outstanding_cuda_view_count, 0)

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


@unittest.skipUnless(
    torch is not None
    and torch.cuda.is_available()
    and torch.cuda.device_count() >= 2
    and len(JAX_GPUS) >= 2,
    "requires two CUDA devices seen by both CUDA-enabled Torch and CUDA-enabled JAX",
)
class TestJaxOnTwoDevices(unittest.TestCase):
    """One PhysxGpuSystem per GPU, each driven through JAX from its own thread."""

    @classmethod
    def setUpClass(cls) -> None:
        sapien.physx.enable_gpu()

    def setUp(self) -> None:
        self.old_config = sapien.physx.get_scene_config()
        self.sims: list[tuple["sapien.physx.PhysxGpuSystem", "sapien.Scene"]] = []

    def tearDown(self) -> None:
        for system, _ in self.sims:
            system.wait_idle()
        gc.collect()
        for system, scene in self.sims:
            scene.close()
            system.close()
        sapien.physx.set_scene_config(self.old_config)

    def test_threads_track_targets_through_jax(self) -> None:
        """Each thread reads fetched joint positions into JAX and writes JAX's velocity command
        back. The system stream sleeps before every fetch and the command waits for a matmul
        chain, so skipping the stream handshake in either direction feeds the controller a
        one-step-stale value and moves it off its closed-form trajectory."""
        # Commands (at most TRACK_GAIN * target = 8 m/s) stay below PhysX's default 10 m/s
        # articulation-link speed limit, which would otherwise clip them.
        targets = [0.25, 0.4]
        self.sims = [_build_slider_system(f"cuda:{k}") for k in range(2)]
        devices = [
            next(d for d in JAX_GPUS if d.local_hardware_id == system.device.cuda_id)
            for system, _ in self.sims
        ]
        ballasts = [jax.device_put(jnp.full((1024, 1024), 1e-3, jnp.float32), d) for d in devices]

        @jax.jit
        def track(qpos: "jax.Array", ballast: "jax.Array", target: float) -> "jax.Array":
            for _ in range(8):
                ballast = jnp.tanh(ballast @ ballast)
            ready = jnp.all(jnp.isfinite(ballast))
            return jnp.where(ready, TRACK_GAIN * (target - qpos), jnp.nan)

        # Compile, load the delay kernel and run one zero-velocity step per device up front: first
        # launches synchronize the context and would hide a missing dependency. The joints start
        # at the origin, so tracking target 0 commands zero velocity.
        for (system, _), ballast in zip(self.sims, ballasts):
            with torch.cuda.device(system.device.cuda_id):
                torch.cuda._sleep(1)
            system.gpu_fetch_articulation_qpos()
            command = track(jax.dlpack.from_dlpack(system.cuda_articulation_qpos), ballast, 0.0)
            system.cuda_articulation_qvel.copy_from(command)
            system.gpu_apply_articulation_qvel()
            system.step()
            command.block_until_ready()

        errors: list[BaseException] = []

        def drive(k: int) -> None:
            system = self.sims[k][0]
            try:
                for _ in range(TRACK_STEPS):
                    with torch.cuda.device(system.device.cuda_id):
                        torch.cuda._sleep(TRACK_DELAY_CYCLES)  # on the system's default stream
                    system.gpu_fetch_articulation_qpos()
                    qpos = jax.dlpack.from_dlpack(system.cuda_articulation_qpos)
                    if qpos.devices() != {devices[k]}:
                        raise AssertionError(f"cuda:{k} positions imported to {qpos.devices()}")
                    command = track(qpos, ballasts[k], targets[k])
                    del qpos
                    system.cuda_articulation_qvel.copy_from(command)
                    system.gpu_apply_articulation_qvel()
                    system.step()
                command.block_until_ready()
            except BaseException as error:  # surfaced in the main thread below
                errors.append(error)

        threads = [threading.Thread(target=drive, args=(k,)) for k in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])

        decay = (1.0 - TRACK_GAIN * TIMESTEP) ** TRACK_STEPS
        for (system, _), target in zip(self.sims, targets):
            system.gpu_fetch_articulation_qpos()
            qpos = torch.from_dlpack(system.cuda_articulation_qpos).cpu()
            expected = torch.full_like(qpos, target * (1.0 - decay))
            torch.testing.assert_close(qpos, expected, rtol=0.0, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
