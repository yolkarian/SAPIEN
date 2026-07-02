import unittest

import numpy as np
import sapien


class TestGpuInverseKinematics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("torch.cuda is not available")
            sapien.physx.enable_gpu()
        except Exception as exc:
            raise unittest.SkipTest(f"GPU IK dependencies are not available: {exc}")

    def setUp(self):
        self._old_config = sapien.physx.get_scene_config()
        config = sapien.physx.PhysxSceneConfig()
        config.gravity = [0.0, 0.0, 0.0]
        sapien.physx.set_scene_config(config)

    def tearDown(self):
        sapien.physx.set_scene_config(self._old_config)

    def _create_scene(self):
        system = sapien.physx.PhysxGpuSystem()
        scene = sapien.Scene([system])
        return system, scene

    def _build_prismatic_articulation(self, scene, y: float, fix_root_link: bool = True):
        builder = scene.create_articulation_builder()
        builder.set_initial_pose(sapien.Pose([0.0, y, 0.0]))

        root = builder.create_link_builder()
        root.set_name("root")
        root.add_box_collision(half_size=[0.05, 0.05, 0.05])

        slider = builder.create_link_builder(root)
        slider.set_name("slider")
        slider.add_box_collision(half_size=[0.05, 0.05, 0.05])
        slider.set_joint_name("slide_x")
        slider.set_joint_properties(
            "prismatic",
            limits=[[0.0, 1.0]],
            pose_in_parent=sapien.Pose(),
            pose_in_child=sapien.Pose(),
            friction=0.0,
            damping=0.0,
        )

        articulation = builder.build(fix_root_link=fix_root_link)
        for link in articulation.links:
            link.disable_gravity = True
        return articulation, articulation.links[1]

    def _build_revolute_articulation(self, scene, cmass_offset=None):
        builder = scene.create_articulation_builder()

        root = builder.create_link_builder()
        root.set_name("root")
        root.add_box_collision(half_size=[0.05, 0.05, 0.05])

        rotor = builder.create_link_builder(root)
        rotor.set_name("rotor")
        rotor.add_box_collision(half_size=[0.05, 0.05, 0.05])
        if cmass_offset is not None:
            rotor.set_mass_and_inertia(1.0, sapien.Pose(cmass_offset), [0.05, 0.05, 0.05])
        rotor.set_joint_name("hinge_x")
        rotor.set_joint_properties(
            "revolute",
            limits=[[-1.0, 1.0]],
            pose_in_parent=sapien.Pose(),
            pose_in_child=sapien.Pose(),
            friction=0.0,
            damping=0.0,
        )

        articulation = builder.build(fix_root_link=True)
        for link in articulation.links:
            link.disable_gravity = True
        return articulation, articulation.links[1]

    def test_batched_prismatic_position_ik(self):
        import torch

        system, scene = self._create_scene()
        ys = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
        target_x = np.array([0.2, 0.45, 0.7], dtype=np.float32)
        articulations = []
        target_links = []
        for y in ys:
            articulation, target_link = self._build_prismatic_articulation(scene, float(y))
            articulations.append(articulation)
            target_links.append(target_link)

        system.gpu_init()
        target_poses = torch.tensor(
            [[x, y, 0.0, 1.0, 0.0, 0.0, 0.0] for x, y in zip(target_x, ys)],
            device="cuda",
            dtype=torch.float32,
        )
        solver = sapien.physx.GpuInverseKinematicsSolver(system, articulations, target_links)
        qpos, success, error = solver.solve(
            target_poses,
            max_iterations=60,
            eps=1e-4,
            rotation_weight=0.0,
            early_stop=True,
        )
        torch.cuda.synchronize()

        self.assertTrue(bool(torch.all(success).cpu()))
        self.assertLess(float(torch.max(error).cpu()), 1e-4)
        np.testing.assert_allclose(qpos[:, 0].detach().cpu().numpy(), target_x, atol=2e-3)

        link_data = system.cuda_articulation_link_data.torch()
        articulation_indices = torch.tensor(
            [articulation.gpu_index for articulation in articulations],
            device=link_data.device,
            dtype=torch.long,
        )
        link_indices = torch.tensor(
            [link.index for link in target_links], device=link_data.device, dtype=torch.long
        )
        actual_positions = link_data[articulation_indices, link_indices, :3].detach().cpu().numpy()
        np.testing.assert_allclose(actual_positions, target_poses[:, :3].cpu().numpy(), atol=2e-3)

    def test_revolute_rotation_ik_fixed_iterations(self):
        import torch

        system, scene = self._create_scene()
        articulation, target_link = self._build_revolute_articulation(scene)
        system.gpu_init()

        target_angle = 0.45
        target_pose = torch.tensor(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    np.cos(target_angle / 2.0),
                    np.sin(target_angle / 2.0),
                    0.0,
                    0.0,
                ]
            ],
            device="cuda",
            dtype=torch.float32,
        )
        solver = sapien.physx.GpuInverseKinematicsSolver(system, [articulation], [target_link])
        qpos, success, error = solver.solve(
            target_pose,
            max_iterations=80,
            eps=1e-4,
            position_weight=0.0,
            rotation_weight=1.0,
            early_stop=False,
        )
        torch.cuda.synchronize()

        self.assertTrue(bool(torch.all(success).cpu()))
        self.assertLess(float(torch.max(error).cpu()), 1e-4)
        np.testing.assert_allclose(
            qpos[:, 0].detach().cpu().numpy(), [target_angle], atol=2e-3
        )

    def test_floating_base_prismatic_position_ik(self):
        import torch

        system, scene = self._create_scene()
        articulation, target_link = self._build_prismatic_articulation(
            scene, 0.25, fix_root_link=False
        )
        self.assertEqual(articulation.get_jacobian_shape()[1] - articulation.dof, 6)

        system.gpu_init()
        target_pose = torch.tensor(
            [[0.35, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32
        )
        solver = sapien.physx.GpuInverseKinematicsSolver(system, [articulation], [target_link])
        qpos, success, error = solver.solve(
            target_pose,
            max_iterations=60,
            eps=1e-4,
            rotation_weight=0.0,
            early_stop=False,
        )
        torch.cuda.synchronize()

        self.assertTrue(bool(torch.all(success).cpu()))
        self.assertLess(float(torch.max(error).cpu()), 1e-4)
        np.testing.assert_allclose(qpos[:, 0].detach().cpu().numpy(), [0.35], atol=2e-3)

    def test_offset_cmass_jacobian_is_shifted_to_link_origin(self):
        import torch

        system, scene = self._create_scene()
        articulation, target_link = self._build_revolute_articulation(
            scene, cmass_offset=[0.0, 0.2, 0.0]
        )
        system.gpu_init()

        solver = sapien.physx.GpuInverseKinematicsSolver(system, [articulation], [target_link])
        system.gpu_fetch_articulation_link_pose()
        system.gpu_compute_articulation_jacobian(solver.index_buffer)
        current_pose = solver.link_data_buffer[solver.articulation_indices, solver.link_indices, :7]
        corrected_jacobian = solver._selected_joint_jacobian(current_pose)

        row_start = max(target_link.index - 1, 0) * 6
        root_columns = articulation.get_jacobian_shape()[1] - articulation.dof
        raw_linear_jacobian = solver.jacobian_buffer[
            articulation.gpu_index, row_start : row_start + 3, root_columns
        ]
        corrected_linear_jacobian = corrected_jacobian[0, :3, 0]
        torch.cuda.synchronize()

        self.assertGreater(
            float(torch.linalg.vector_norm(raw_linear_jacobian).detach().cpu()), 1e-3
        )
        self.assertLess(
            float(torch.linalg.vector_norm(corrected_linear_jacobian).detach().cpu()), 1e-4
        )

    def test_initial_qpos_without_apply_restores_current_gpu_state(self):
        import torch

        system, scene = self._create_scene()
        articulation, target_link = self._build_prismatic_articulation(scene, 0.0)
        system.gpu_init()

        qpos_buffer = system.cuda_articulation_qpos.torch()
        articulation_indices = torch.tensor(
            [articulation.gpu_index], device=qpos_buffer.device, dtype=torch.long
        )
        current_qpos = torch.tensor([[0.6]], device=qpos_buffer.device, dtype=qpos_buffer.dtype)
        stale_qpos = torch.tensor([[0.15]], device=qpos_buffer.device, dtype=qpos_buffer.dtype)

        qpos_buffer[articulation_indices, :1] = current_qpos
        system.gpu_apply_articulation_qpos()
        system.gpu_update_articulation_kinematics()
        system.gpu_fetch_articulation_qpos()
        torch.cuda.synchronize()
        np.testing.assert_allclose(
            qpos_buffer[articulation_indices, :1].detach().cpu().numpy(),
            current_qpos.cpu().numpy(),
            atol=1e-5,
        )

        # Simulate a stale user-side CUDA qpos buffer. solve(..., apply_result=False)
        # must restore the real PhysX state, not this dirty buffer value.
        qpos_buffer[articulation_indices, :1] = stale_qpos

        solver = sapien.physx.GpuInverseKinematicsSolver(system, [articulation], [target_link])
        target_pose = torch.tensor(
            [[0.25, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            device=qpos_buffer.device,
            dtype=qpos_buffer.dtype,
        )
        initial_qpos = torch.tensor([[0.05]], device=qpos_buffer.device, dtype=qpos_buffer.dtype)
        solver.solve(
            target_pose,
            initial_qpos=initial_qpos,
            max_iterations=2,
            rotation_weight=0.0,
            apply_result=False,
            early_stop=False,
        )

        system.gpu_fetch_articulation_qpos()
        torch.cuda.synchronize()
        np.testing.assert_allclose(
            qpos_buffer[articulation_indices, :1].detach().cpu().numpy(),
            current_qpos.cpu().numpy(),
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
