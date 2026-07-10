import os
import unittest
from pathlib import Path

import numpy as np
import sapien
from common import pose_equal, rand_pose


class TestArticulation(unittest.TestCase):
    def _check_dense_jacobian(self, fix_root_link: bool):
        scene = sapien.Scene()
        loader = scene.create_urdf_loader()
        loader.fix_root_link = fix_root_link
        robot = loader.load(str(Path(".") / "assets" / "movo_simple.urdf"))

        for link in robot.links:
            link.disable_gravity = True

        qpos = np.array(
            [
                0.1,
                -0.2,
                0.15,
                -0.1,
                0.12,
                -0.08,
                0.05,
                -0.07,
                0.11,
                -0.09,
                0.06,
                -0.04,
                0.02,
            ],
            dtype=np.float32,
        )
        qvel = np.array(
            [
                -0.25,
                0.1,
                -0.15,
                0.2,
                -0.05,
                0.18,
                -0.12,
                0.08,
                -0.16,
                0.14,
                -0.09,
                0.07,
                -0.03,
            ],
            dtype=np.float32,
        )

        robot.set_qpos(qpos)
        robot.set_qvel(qvel)

        if not fix_root_link:
            # Floating-base root velocities are only propagated to descendant link
            # velocities after stepping once. For fixed-base articulations, stepping
            # would change the joint velocities and invalidate the exact Jacobian check.
            robot.set_root_linear_velocity([0.3, -0.2, 0.1])
            robot.set_root_angular_velocity([-0.4, 0.2, 0.5])
            scene.step()

        jacobian = robot.compute_dense_jacobian()
        rows, cols = map(int, robot.get_jacobian_shape())
        self.assertEqual(jacobian.shape, (rows, cols))

        if fix_root_link:
            generalized_velocity = np.asarray(robot.get_qvel(), dtype=np.float32)
            links = robot.links[1:]
        else:
            generalized_velocity = np.concatenate(
                [
                    np.asarray(robot.get_root_linear_velocity(), dtype=np.float32),
                    np.asarray(robot.get_root_angular_velocity(), dtype=np.float32),
                    np.asarray(robot.get_qvel(), dtype=np.float32),
                ]
            )
            links = robot.links

        expected_velocity = np.concatenate(
            [
                np.concatenate(
                    [
                        np.asarray(link.get_linear_velocity(), dtype=np.float32),
                        np.asarray(link.get_angular_velocity(), dtype=np.float32),
                    ]
                )
                for link in links
            ]
        )
        self.assertTrue(
            np.allclose(
                jacobian @ generalized_velocity,
                expected_velocity,
                rtol=1e-4,
                atol=1e-4,
            )
        )

    def test_drive(self):
        scene = sapien.Scene()
        loader = scene.create_urdf_loader()
        robot = loader.load(str(Path(".") / "assets" / "movo_simple.urdf"))

        q = [
            -0.3283431,
            0.39793425,
            -0.81116733,
            0.14926876,
            -0.67020133,
            -0.17215485,
            -0.6565735,
            0.84940983,
            0.40675576,
            0.73837611,
            -0.93073463,
            -0.7414073,
            0.71490287,
        ]
        for j, v in zip(robot.get_active_joints(), q):
            j.set_drive_velocity_target(v)
        self.assertTrue(
            np.allclose(
                [j.get_drive_velocity_target()[0] for j in robot.get_active_joints()], q
            )
        )

        for j, v in zip(robot.get_active_joints(), q):
            j.set_drive_target(v)
        self.assertTrue(
            np.allclose([j.get_drive_target()[0] for j in robot.get_active_joints()], q)
        )

        for j, v in zip(robot.get_active_joints(), q):
            props = (
                float(np.random.rand()),
                float(np.random.rand()),
                float(np.random.rand()),
                np.random.choice(["force", "acceleration"]),
            )
            j.set_drive_properties(*props)
            for a, b in zip(
                [j.stiffness, j.damping, j.force_limit, j.drive_mode], props
            ):
                self.assertAlmostEqual(a, b)

    def test_max_joint_velocity(self):
        scene = sapien.Scene()
        builder = scene.create_articulation_builder()

        root = builder.create_link_builder()
        root.add_box_collision(half_size=[0.05, 0.05, 0.05])

        child = builder.create_link_builder(root)
        child.add_box_collision(half_size=[0.05, 0.05, 0.05])
        child.set_joint_properties(
            "continuous",
            limits=[[-np.inf, np.inf]],
            pose_in_parent=sapien.Pose(),
            pose_in_child=sapien.Pose(),
            velocity_limit=0.25,
        )

        articulation = builder.build(fix_root_link=True)
        for link in articulation.links:
            link.disable_gravity = True
        joint = articulation.active_joints[0]
        self.assertAlmostEqual(joint.max_joint_velocity[0], 0.25)

        joint.set_max_joint_velocity(0.5)
        self.assertAlmostEqual(joint.get_max_joint_velocity()[0], 0.5)
        joint.max_joint_velocity = [0.25]

        cloned_links = articulation.clone_links()
        self.assertAlmostEqual(cloned_links[1].joint.max_joint_velocity[0], 0.25)

        from sapien.wrapper.urdf_exporter import export_kinematic_chain_urdf

        exported = export_kinematic_chain_urdf(articulation)
        self.assertIn('type="continuous"', exported)
        self.assertIn('velocity="0.25"', exported)

        with self.assertRaises(RuntimeError):
            joint.set_max_joint_velocity(-1.0)
        with self.assertRaises(RuntimeError):
            joint.max_joint_velocity = [np.inf]
        with self.assertRaises(RuntimeError):
            joint.max_joint_velocity = [1.0, 2.0]

        articulation.set_qvel([5.0])
        scene.step()
        self.assertAlmostEqual(abs(articulation.get_qvel()[0]), 0.25, places=5)

        other_builder = scene.create_articulation_builder()
        other_builder.create_link_builder().add_box_collision(
            half_size=[0.05, 0.05, 0.05]
        )
        other = other_builder.build(fix_root_link=True)
        joint.child_link.set_parent(other.root)
        self.assertAlmostEqual(joint.max_joint_velocity[0], 0.25)

    def test_urdf_loader(self):
        scene = sapien.Scene()
        loader = scene.create_urdf_loader()
        robot = loader.load(str(Path(".") / "assets" / "movo_simple.urdf"))

        q = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5]
        robot.set_qpos(q)
        self.assertTrue(np.allclose(robot.get_qpos(), q))

        q = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5]
        robot.set_qvel(q)
        self.assertTrue(np.allclose(robot.get_qvel(), q))

        q = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5]
        robot.set_qf(q)
        self.assertTrue(np.allclose(robot.get_qf(), q))

    def test_kinematics_dynamics(self):
        scene = sapien.Scene()
        loader = scene.create_urdf_loader()
        robot = loader.load(str(Path(".") / "assets" / "movo_simple.urdf"))

        q = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5]
        robot.set_qpos(q)

        model = robot.create_pinocchio_model()
        model.compute_forward_kinematics(q)
        for i, l in enumerate(robot.links):
            self.assertTrue(
                pose_equal(l.pose, model.get_link_pose(i), rtol=1e-5, atol=1e-5)
            )

        self.assertTrue(
            np.allclose(
                model.compute_inverse_dynamics(q, np.zeros_like(q), np.zeros_like(q)),
                robot.compute_passive_force(True, True),
                rtol=1e-5,
                atol=1e-5,
            )
        )

    def test_dense_jacobian(self):
        self._check_dense_jacobian(fix_root_link=True)
        self._check_dense_jacobian(fix_root_link=False)

    def test_joint(self):
        scene = sapien.Scene()
        loader = scene.create_urdf_loader()
        robot = loader.load(str(Path(".") / "assets" / "movo_simple.urdf"))

        q = [
            -0.3283431,
            0.39793425,
            -0.81116733,
            0.14926876,
            -0.67020133,
            -0.17215485,
            -0.6565735,
            0.84940983,
            0.40675576,
            0.73837611,
            -0.93073463,
            -0.7414073,
            0.71490287,
        ]
        for j in robot.active_joints:
            j.armature = [10]
            self.assertAlmostEqual(j.armature[0], 10)

        for j in robot.joints:
            self.assertTrue(
                pose_equal(j.global_pose, j.child_link.entity_pose * j.pose_in_child)
            )
