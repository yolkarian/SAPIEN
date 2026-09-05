"""GPU stream-order regression tests, independent of policies and rendering.

Run each test in a fresh process when using Compute Sanitizer. Torch only supplies
CUDA buffers, queued delay kernels, and assertions; all simulation is SAPIEN.
"""

import gc
import unittest

import torch
import sapien


class TestGpuStreamOrder(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("a CUDA device and CUDA-enabled Torch are required")
        sapien.physx.enable_gpu()

    def setUp(self) -> None:
        self.old_config = sapien.physx.get_scene_config()
        config = sapien.physx.PhysxSceneConfig()
        config.gravity = [0.0, 0.0, -9.81]
        config.num_scenes = 1
        sapien.physx.set_scene_config(config)
        self.system = sapien.physx.PhysxGpuSystem()
        self.system.set_timestep(0.005)
        self.scene = sapien.Scene([self.system])
        self.views: dict[str, torch.Tensor] = {}
        self.queries: list[sapien.physx.PhysxGpuContactBodyImpulseQuery] = []
        self.stream = torch.cuda.Stream()

    def tearDown(self) -> None:
        self.system.wait_idle()
        self.views.clear()
        self.queries.clear()
        gc.collect()
        self.scene.close()
        self.system.close()
        sapien.physx.set_scene_config(self.old_config)

    def _build_sliders(self) -> None:
        builder = self.scene.create_articulation_builder()
        root = builder.create_link_builder()
        root.add_box_collision(half_size=[0.05, 0.05, 0.05])
        slider = builder.create_link_builder(root)
        slider.add_box_collision(half_size=[0.05, 0.05, 0.05])
        slider.set_joint_properties(
            "prismatic", limits=[[-10.0, 10.0]],
            pose_in_parent=sapien.Pose([0.2, 0.0, 0.0]),
            pose_in_child=sapien.Pose(), friction=0.0, damping=0.0,
        )
        for index in range(32):
            builder.set_initial_pose(sapien.Pose([0.0, float(index), 0.0]))
            builder.build(fix_root_link=True)
        self.system.gpu_init()
        self.views["qpos"] = self.system.cuda_articulation_qpos.torch()
        self.views["qvel"] = self.system.cuda_articulation_qvel.torch()
        self.views["links"] = self.system.cuda_articulation_link_data.torch()
        self.views["qvel"].fill_(1.0)
        self.system.gpu_apply_articulation_qvel()
        for _ in range(4):
            self.system.step()
        torch.cuda.synchronize()
        self.system.gpu_set_cuda_stream(self.stream.cuda_stream)

    def _build_boxes(self, *, contact: bool = False) -> None:
        if contact:
            ground = self.scene.create_actor_builder()
            ground.add_box_collision(half_size=[1.0, 40.0, 0.05])
            ground.set_initial_pose(sapien.Pose([0.0, 15.5, -0.05]))
            ground.build_static()
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.05, 0.05, 0.05])
        bodies = []
        for index in range(32):
            builder.set_initial_pose(sapien.Pose([0.0, float(index), 0.04]))
            entity = builder.build()
            bodies.append(entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent))
        self.system.gpu_init()
        self.views["bodies"] = self.system.cuda_rigid_dynamic_data.torch()
        self.system.gpu_fetch_rigid_dynamic_data()
        torch.cuda.synchronize()
        if contact:
            query = self.system.gpu_create_contact_body_impulse_query(bodies)
            self.queries.append(query)
            self.views["impulses"] = query.cuda_impulses.torch()
        else:
            self.views["bodies"][:, 7] = 1.0
            self.system.gpu_apply_rigid_dynamic_data()
            for _ in range(4):
                self.system.step()
        torch.cuda.synchronize()
        self.system.gpu_set_cuda_stream(self.stream.cuda_stream)

    def test_articulation_fetch_preserves_queued_consumers(self) -> None:
        """A later pose fetch must not overwrite scratch still read by an earlier one."""
        self._build_sliders()
        with torch.cuda.stream(self.stream):
            for iteration in range(16):
                self.system.gpu_fetch_articulation_link_pose()
                reference = self.views["links"][..., :7].clone()
                self.stream.synchronize()
                # Delay the conversion/consumer, but not PhysX's non-blocking stream.
                torch.cuda._sleep(100_000_000)
                self.system.gpu_fetch_articulation_link_pose()
                before = self.views["links"][..., :7].clone()
                self.system.step()
                self.system.gpu_fetch_articulation_link_pose()
                after = self.views["links"][..., :7].clone()
                self.stream.synchronize()
                with self.subTest(iteration=iteration):
                    torch.testing.assert_close(before, reference, rtol=0.0, atol=1e-6)
                    self.assertGreater(float((after - before).abs().max()), 0.001)

    def test_rigid_fetch_preserves_queued_consumers(self) -> None:
        """Pose/velocity scratch cannot be reused before its previous conversion."""
        self._build_boxes()
        with torch.cuda.stream(self.stream):
            for iteration in range(16):
                self.system.gpu_fetch_rigid_dynamic_data()
                reference = self.views["bodies"].clone()
                self.stream.synchronize()
                torch.cuda._sleep(100_000_000)
                self.system.gpu_fetch_rigid_dynamic_data()
                before = self.views["bodies"].clone()
                self.system.step()
                self.system.gpu_fetch_rigid_dynamic_data()
                after = self.views["bodies"].clone()
                self.stream.synchronize()
                with self.subTest(iteration=iteration):
                    torch.testing.assert_close(before, reference, rtol=0.0, atol=1e-6)
                    self.assertGreater(float((after - before).abs().max()), 0.001)

    def test_articulation_fetch_on_default_stream(self) -> None:
        self.stream = torch.cuda.default_stream()
        self.test_articulation_fetch_preserves_queued_consumers()

    def test_rigid_fetch_on_default_stream(self) -> None:
        self.stream = torch.cuda.default_stream()
        self.test_rigid_fetch_preserves_queued_consumers()

    def test_full_state_fetch_preserves_queued_consumers(self) -> None:
        """Exercise the joint-first fetch order used by batched RL environments."""
        self._build_sliders()
        fetches = (
            self.system.gpu_fetch_articulation_qpos,
            self.system.gpu_fetch_articulation_qvel,
            self.system.gpu_fetch_articulation_qacc,
            self.system.gpu_fetch_articulation_link_pose,
            self.system.gpu_fetch_articulation_link_velocity,
            self.system.gpu_fetch_rigid_dynamic_data,
        )
        with torch.cuda.stream(self.stream):
            for iteration in range(16):
                for fetch in fetches:
                    fetch()
                reference = self.views["links"].clone()
                self.stream.synchronize()
                torch.cuda._sleep(100_000_000)
                for fetch in fetches:
                    fetch()
                before = self.views["links"].clone()
                self.system.step()
                for fetch in fetches:
                    fetch()
                after = self.views["links"].clone()
                self.stream.synchronize()
                with self.subTest(iteration=iteration):
                    torch.testing.assert_close(before, reference, rtol=0.0, atol=1e-6)
                    self.assertGreater(float((after - before).abs().max()), 0.001)

    def test_joint_fetch_orders_buffer_writes_and_consumers(self) -> None:
        """Fetch must wait for prior writes and must finish before same-stream reads."""
        self._build_sliders()
        with torch.cuda.stream(self.stream):
            for iteration in range(32):
                self.system.gpu_fetch_articulation_qpos()
                self.system.gpu_fetch_articulation_qvel()
                reference_qpos = self.views["qpos"].clone()
                reference_qvel = self.views["qvel"].clone()
                self.stream.synchronize()
                torch.cuda._sleep(10_000_000)
                self.views["qpos"].fill_(float("nan"))
                self.views["qvel"].fill_(float("nan"))
                self.system.gpu_fetch_articulation_qpos()
                self.system.gpu_fetch_articulation_qvel()
                qpos = self.views["qpos"].clone()
                qvel = self.views["qvel"].clone()
                self.stream.synchronize()
                with self.subTest(iteration=iteration):
                    torch.testing.assert_close(qpos, reference_qpos, rtol=0.0, atol=0.0)
                    torch.testing.assert_close(qvel, reference_qvel, rtol=0.0, atol=0.0)
                self.system.step()

    def test_contact_query_waits_before_next_step(self) -> None:
        """Delay contact reduction on a non-default stream, then immediately step."""
        self._build_boxes(contact=True)
        with torch.cuda.stream(self.stream):
            for iteration in range(32):
                # Reset to overlap so every comparison exercises nonzero contacts.
                self.views["bodies"][:, 2] = 0.04
                self.views["bodies"][:, 7:13] = 0.0
                self.system.gpu_apply_rigid_dynamic_data()
                self.system.step()
                self.system.gpu_query_contact_body_impulses(self.queries[0])
                reference = self.views["impulses"].clone()
                self.stream.synchronize()
                self.assertGreater(float(reference.norm()), 0.0)
                # Contact data is already cached: the next query queues no count wait.
                torch.cuda._sleep(10_000_000)
                self.system.gpu_query_contact_body_impulses(self.queries[0], synchronize=False)
                before = self.views["impulses"].clone()
                self.system.step()
                self.stream.synchronize()
                with self.subTest(iteration=iteration):
                    torch.testing.assert_close(before, reference, rtol=0.0, atol=1e-6)

    def test_contact_impulses_clear_after_separation(self) -> None:
        """No stale contact count/pointers may survive when all pairs disappear."""
        self._build_boxes(contact=True)
        with torch.cuda.stream(self.stream):
            self.system.step()
            self.system.gpu_query_contact_body_impulses(self.queries[0])
            self.assertGreater(float(self.views["impulses"].norm()), 0.0)
            self.views["bodies"][:, 2] = 10.0
            self.views["bodies"][:, 7:13] = 0.0
            self.system.gpu_apply_rigid_dynamic_data()
            for iteration in range(16):
                self.system.step()
                self.system.gpu_query_contact_body_impulses(self.queries[0], synchronize=False)
                result = self.views["impulses"].clone()
                self.stream.synchronize()
                with self.subTest(iteration=iteration):
                    torch.testing.assert_close(result, torch.zeros_like(result), rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
