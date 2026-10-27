"""PhysX GPU systems on different CUDA devices in one process.

Every test needs at least two CUDA devices and CUDA-enabled Torch and is skipped otherwise.
Torch only supplies device-side views, streams and assertions; all simulation is SAPIEN.
"""

import gc
import subprocess
import sys
import textwrap
import threading
import unittest
from pathlib import Path
from typing import List

import sapien

try:
    import torch
except ImportError:  # pragma: no cover - depends on the test environment
    torch = None

TIMESTEP = 0.01
GRAVITY = 9.81
STEPS = 50
SCENES = 4
BODIES_PER_SCENE = 4


def _has_two_cuda_devices() -> bool:
    return torch is not None and torch.cuda.is_available() and torch.cuda.device_count() >= 2


class _FallingBoxes:
    """One GPU system whose scenes hold contact-free boxes, so heights follow free fall."""

    def __init__(self, device: str) -> None:
        self.system = sapien.physx.PhysxGpuSystem(device)
        self.system.set_timestep(TIMESTEP)
        self.scenes: List[sapien.Scene] = []
        for i in range(SCENES):
            scene = sapien.Scene([self.system])
            self.system.set_scene_offset(scene, [20.0 * i, 0.0, 0.0])
            builder = scene.create_actor_builder()
            builder.add_box_collision(half_size=[0.05, 0.05, 0.05])
            for j in range(BODIES_PER_SCENE):
                builder.set_initial_pose(sapien.Pose([0.0, 0.5 * j, 10.0]))
                body = builder.build().find_component_by_type(
                    sapien.physx.PhysxRigidDynamicComponent
                )
                body.linear_damping = 0.0
            self.scenes.append(scene)
        self.system.gpu_init()
        self.bodies = self.system.cuda_rigid_dynamic_data.torch()

    def launch(self, vertical_velocity: float) -> "torch.Tensor":
        """Give every box the same vertical velocity; returns the starting heights."""
        self.system.gpu_fetch_rigid_dynamic_data()
        self.bodies[:, 7:13] = 0.0
        self.bodies[:, 9] = vertical_velocity
        self.system.gpu_apply_rigid_dynamic_data()
        return self.heights()

    def heights(self) -> "torch.Tensor":
        self.system.gpu_fetch_rigid_dynamic_data()
        torch.cuda.synchronize(self.bodies.device)
        return self.bodies[:, 2].clone()

    def close(self) -> None:
        self.system.wait_idle()
        del self.bodies
        gc.collect()
        for scene in self.scenes:
            scene.close()
        self.system.close()


def _free_fall(start: "torch.Tensor", vertical_velocity: float, steps: int) -> "torch.Tensor":
    """Semi-implicit Euler heights PhysX produces for undamped, contact-free bodies."""
    return (
        start
        + steps * TIMESTEP * vertical_velocity
        - GRAVITY * TIMESTEP**2 * steps * (steps + 1) / 2
    )


@unittest.skipUnless(_has_two_cuda_devices(), "requires two CUDA devices and CUDA-enabled Torch")
class TestMultiGpuPhysx(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sapien.physx.enable_gpu()

    def setUp(self) -> None:
        self.old_config = sapien.physx.get_scene_config()
        self.sims: List[_FallingBoxes] = []
        torch.cuda.set_device(0)

    def tearDown(self) -> None:
        for sim in self.sims:
            sim.close()
        self.sims.clear()
        sapien.physx.set_scene_config(self.old_config)
        torch.cuda.set_device(0)

    def _sim(self, device: str) -> _FallingBoxes:
        sim = _FallingBoxes(device)
        self.sims.append(sim)
        return sim

    def _close(self, sim: _FallingBoxes) -> None:
        self.sims.remove(sim)
        sim.close()

    def _assert_free_fall(self, sim, start, vertical_velocity, steps) -> None:
        torch.testing.assert_close(
            sim.heights(), _free_fall(start, vertical_velocity, steps), rtol=0.0, atol=1e-3
        )

    def test_systems_on_two_devices_simulate_independently(self) -> None:
        """Interleaved steps keep each system's buffers and state on its own GPU."""
        sims = [self._sim("cuda:0"), self._sim("cuda:1")]
        velocities = [2.0, 5.0]
        for index, sim in enumerate(sims):
            self.assertEqual(sim.system.device.cuda_id, index)
            self.assertEqual(sim.bodies.device, torch.device("cuda", index))
        starts = [sim.launch(v) for sim, v in zip(sims, velocities)]
        for _ in range(STEPS):
            for sim in sims:
                sim.system.step()
        for sim, start, v in zip(sims, starts, velocities):
            self._assert_free_fall(sim, start, v, STEPS)

    def test_calls_restore_the_callers_current_device(self) -> None:
        """A system's device is current only while SAPIEN works on it, not afterwards."""
        for current, device in ((0, "cuda:1"), (1, "cuda:0")):
            torch.cuda.set_device(current)
            sim = self._sim(device)
            self.assertEqual(torch.cuda.current_device(), current)
            sim.launch(1.0)
            sim.system.step()
            sim.system.step_start()
            sim.system.step_finish()
            sim.heights()
            sim.system.wait_idle()
            self.assertEqual(torch.cuda.current_device(), current)
            self._close(sim)
            self.assertEqual(torch.cuda.current_device(), current)

    def test_step_start_on_every_device_before_finishing(self) -> None:
        """Launch-all-then-finish-all stepping matches stepping each system in turn."""
        config = sapien.physx.get_scene_config()
        config.cpu_workers = 1
        sapien.physx.set_scene_config(config)
        sims = [self._sim("cuda:0"), self._sim("cuda:1")]
        starts = [sim.launch(3.0) for sim in sims]
        for _ in range(STEPS):
            for sim in sims:
                sim.system.step_start()
            for sim in sims:
                sim.system.step_finish()
        for sim, start in zip(sims, starts):
            self._assert_free_fall(sim, start, 3.0, STEPS)

    def test_one_thread_per_device_steps_its_system(self) -> None:
        """Fresh threads, which have no current CUDA context, can drive a system each."""
        sims = [self._sim("cuda:0"), self._sim("cuda:1")]
        starts = [sim.launch(4.0) for sim in sims]
        errors: List[BaseException] = []

        def run(sim: _FallingBoxes) -> None:
            try:
                for _ in range(STEPS):
                    sim.system.step()
                sim.system.gpu_fetch_rigid_dynamic_data()
                sim.system.wait_idle()
            except BaseException as error:  # noqa: BLE001 - reported to the main thread
                errors.append(error)

        threads = [threading.Thread(target=run, args=(sim,)) for sim in sims]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        for sim, start in zip(sims, starts):
            self._assert_free_fall(sim, start, 4.0, STEPS)

    def test_threads_fetch_apply_and_step_their_device(self) -> None:
        """Fetch, Torch writes, apply and step run concurrently from one thread per device."""
        sims = [self._sim("cuda:0"), self._sim("cuda:1")]
        velocities = [2.0, 6.0]
        starts: List["torch.Tensor"] = [torch.empty(0), torch.empty(0)]
        errors: List[BaseException] = []

        def run(index: int) -> None:
            try:
                for _ in range(10):
                    starts[index] = sims[index].launch(velocities[index])
                    for _ in range(5):
                        sims[index].system.step()
            except BaseException as error:  # noqa: BLE001 - reported to the main thread
                errors.append(error)

        threads = [threading.Thread(target=run, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        for sim, start, v in zip(sims, starts, velocities):
            self._assert_free_fall(sim, start, v, 5)

    def test_gpu_set_cuda_stream_zero_roundtrip(self) -> None:
        """Stream 0 selects the system's own default stream and later own-device streams work."""
        torch.cuda.set_device(0)
        sim = self._sim("cuda:1")
        start = sim.launch(1.0)
        sim.system.gpu_set_cuda_stream(0)
        sim.system.step()
        self.assertEqual(torch.cuda.current_device(), 0)
        own_stream = torch.cuda.Stream(device=1)
        sim.system.gpu_set_cuda_stream(own_stream.cuda_stream)
        sim.system.step()
        sim.system.gpu_set_cuda_stream(0)
        for _ in range(STEPS - 2):
            sim.system.step()
        self._assert_free_fall(sim, start, 1.0, STEPS)
        self.assertEqual(torch.cuda.current_device(), 0)

    def test_two_systems_on_the_same_device_interleave(self) -> None:
        """Stepping two systems of one device alternately must not disturb the other device."""
        first = self._sim("cuda:0")
        second = self._sim("cuda:0")
        other = self._sim("cuda:1")
        starts = [s.launch(v) for s, v in zip((first, second, other), (1.0, 3.0, 5.0))]
        for _ in range(STEPS):
            first.system.step()
            second.system.step()
            other.system.step()
        for sim, start, v in zip((first, second, other), starts, (1.0, 3.0, 5.0)):
            self._assert_free_fall(sim, start, v, STEPS)

    def test_inputs_from_another_device_are_rejected(self) -> None:
        """Streams and index buffers must live on the system's device."""
        sim = self._sim("cuda:1")
        foreign_stream = torch.cuda.Stream(device=0)
        with self.assertRaisesRegex(RuntimeError, "cuda:1"):
            sim.system.gpu_set_cuda_stream(foreign_stream.cuda_stream)
        own_stream = torch.cuda.Stream(device=1)
        sim.system.gpu_set_cuda_stream(own_stream.cuda_stream)
        sim.system.step()
        sim.system.gpu_fetch_rigid_dynamic_data()
        own_stream.synchronize()
        sim.system.gpu_set_cuda_stream(0)
        foreign_indices = torch.arange(4, dtype=torch.int32, device="cuda:0")
        with self.assertRaisesRegex(RuntimeError, "same CUDA device"):
            sim.system.gpu_apply_rigid_dynamic_data(foreign_indices)

    def test_torch_inputs_on_the_system_device_while_another_is_current(self) -> None:
        """Torch exports only from the current device; SAPIEN makes the input's device current."""
        sim = self._sim("cuda:1")
        sim.system.gpu_fetch_rigid_dynamic_data()
        state = sim.bodies.clone()
        state[:, 7:13] = 0.0
        state[:, 9] = 3.0
        start = state[:, 2].clone()
        sim.system.cuda_rigid_dynamic_data.copy_from(state)
        indices = torch.arange(state.shape[0], dtype=torch.int32, device="cuda:1")
        sim.system.gpu_apply_rigid_dynamic_data(indices)
        self.assertEqual(torch.cuda.current_device(), 0)
        for _ in range(STEPS):
            sim.system.step()
        self._assert_free_fall(sim, start, 3.0, STEPS)
        self.assertEqual(torch.cuda.current_device(), 0)

    def test_closing_one_device_keeps_the_other_running(self) -> None:
        """Closing the systems of one device leaves the systems of another device running."""
        first = self._sim("cuda:0")
        second = self._sim("cuda:1")
        self._close(first)
        start = second.launch(2.0)
        for _ in range(STEPS):
            second.system.step()
        self._assert_free_fall(second, start, 2.0, STEPS)

        recreated = self._sim("cuda:0")
        starts = [recreated.launch(1.0), second.launch(1.0)]
        for _ in range(STEPS):
            recreated.system.step()
            second.system.step()
        for sim, start in zip((recreated, second), starts):
            self._assert_free_fall(sim, start, 1.0, STEPS)

    def test_alternating_devices_after_closing_every_system(self) -> None:
        """A device whose systems were all closed is used again, alternating with another one.

        PhysX caches kernel handles per CUDA context manager address for the lifetime of the
        PhysX engine, which the material keeps alive here, so a device's manager must stay valid
        after its last system closes. Releasing it then let a later manager at a recycled address
        run unloaded kernels; that depends on heap layout, so this test exercises the pattern
        rather than forcing the recycled address."""
        keep_engine_alive = sapien.physx.PhysxMaterial(0.5, 0.5, 0.0)
        for device in ("cuda:0", "cuda:1", "cuda:0", "cuda:1", "cuda:0"):
            sim = self._sim(device)
            start = sim.launch(2.0)
            for _ in range(STEPS):
                sim.system.step()
            self._assert_free_fall(sim, start, 2.0, STEPS)
            self._close(sim)
        del keep_engine_alive

    def test_shared_material_and_mesh_collide_on_both_devices(self) -> None:
        """One cooked convex mesh and material serve scenes on both devices."""
        material = sapien.physx.PhysxMaterial(0.8, 0.8, 0.0)
        cone = str(Path(__file__).resolve().parents[1] / "assets" / "cone.stl")
        for device in ("cuda:0", "cuda:1"):
            system = sapien.physx.PhysxGpuSystem(device)
            system.set_timestep(TIMESTEP)
            scene = sapien.Scene([system])
            scene.add_ground(0.0, render=False, material=material)
            builder = scene.create_actor_builder()
            builder.add_convex_collision_from_file(cone, scale=(0.1, 0.1, 0.1), material=material)
            builder.set_initial_pose(sapien.Pose([0.0, 0.0, 0.3]))
            builder.build()
            system.gpu_init()
            for _ in range(200):
                system.step()
            bodies = system.cuda_rigid_dynamic_data.torch()
            system.gpu_fetch_rigid_dynamic_data()
            torch.cuda.synchronize(bodies.device)
            state = bodies[0].cpu()
            system.wait_idle()
            del bodies
            gc.collect()
            scene.close()
            system.close()
            # Without the mesh on this device's scene the cone would have fallen ~20 m.
            self.assertGreater(float(state[2]), 0.0, f"the cone fell through the ground on {device}")
            self.assertLess(abs(float(state[9])), 0.05, f"the cone is still falling on {device}")

    def test_sync_poses_gpu_to_cpu_on_both_devices(self) -> None:
        """sync_poses_gpu_to_cpu() downloads the poses of each system from its own device."""
        sims = [self._sim("cuda:0"), self._sim("cuda:1")]
        velocities = [2.0, 4.0]
        starts = [sim.launch(v) for sim, v in zip(sims, velocities)]
        for _ in range(STEPS):
            for sim in sims:
                sim.system.step()
        for sim, start, v in zip(sims, starts, velocities):
            sim.system.sync_poses_gpu_to_cpu()
            expected = _free_fall(start, v, STEPS)
            for body, height in zip(sim.system.rigid_dynamic_components, expected.tolist()):
                self.assertAlmostEqual(float(body.entity.pose.p[2]), height, places=3)

    def test_simulating_on_one_device_creates_no_context_on_another(self) -> None:
        """A process that simulates on cuda:1 never claims memory on cuda:0, so per-GPU ranks
        of a multi-process job do not each pay for a stray context on the first GPU."""
        script = textwrap.dedent(
            """
            import ctypes, gc
            import sapien

            sapien.physx.enable_gpu()
            system = sapien.physx.PhysxGpuSystem("cuda:1")
            scene = sapien.Scene([system])
            builder = scene.create_actor_builder()
            builder.add_box_collision(half_size=[0.1, 0.1, 0.1])
            builder.set_initial_pose(sapien.Pose([0.0, 0.0, 1.0]))
            builder.build()
            system.gpu_init()
            data = system.cuda_rigid_dynamic_data
            system.step()
            system.gpu_fetch_rigid_dynamic_data()
            system.wait_idle()
            del data
            gc.collect()
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


if __name__ == "__main__":
    unittest.main()
