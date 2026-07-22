import gc
import unittest

import numpy as np
import sapien


def build_slider(scene: sapien.Scene, pos_x: float = 0.0) -> sapien.physx.PhysxArticulation:
    """Build a fixed-base articulation with one horizontal prismatic joint.

    The prismatic axis is world x while gravity is world -z, so gravity produces no
    motion along the DOF and all motion comes from drives; this keeps the behavioral
    tests independent of gravity.
    """
    builder = scene.create_articulation_builder()
    root = builder.create_link_builder()
    root.add_box_collision(half_size=[0.05, 0.05, 0.05], density=1000)
    root.set_name("root")

    child = builder.create_link_builder(root)
    child.add_box_collision(half_size=[0.05, 0.05, 0.05], density=1000)
    child.set_name("slider")
    child.set_joint_properties(
        type="prismatic",
        limits=[[-1.0, 1.0]],
        pose_in_parent=sapien.Pose(),
        pose_in_child=sapien.Pose([-0.2, 0.0, 0.0]),
    )

    builder.set_initial_pose(sapien.Pose([pos_x, 0.0, 0.5]))
    return builder.build(fix_root_link=True)


def build_pendulum(scene: sapien.Scene, pos_x: float = 0.0) -> sapien.physx.PhysxArticulation:
    """Build a fixed-base pendulum with one revolute joint about world y.

    At qpos 0 the child link extends horizontally (+x, 0.3 m from the pivot), so
    gravity torque swings it down with increasing qpos.
    """
    # rotate the joint frame x axis onto world y: rotation about z by 90 degrees (wxyz)
    qz90 = [0.7071068, 0.0, 0.0, 0.7071068]
    builder = scene.create_articulation_builder()
    root = builder.create_link_builder()
    root.add_box_collision(half_size=[0.05, 0.05, 0.05], density=1000)
    root.set_name("root")

    child = builder.create_link_builder(root)
    child.add_box_collision(half_size=[0.05, 0.05, 0.05], density=1000)
    child.set_name("pendulum")
    child.set_joint_properties(
        type="revolute",
        limits=[[-3.14, 3.14]],
        pose_in_parent=sapien.Pose(q=qz90),
        pose_in_child=sapien.Pose([-0.3, 0.0, 0.0], qz90),
    )

    builder.set_initial_pose(sapien.Pose([pos_x, 0.0, 1.0]))
    return builder.build(fix_root_link=True)


class TestBatchedProperties(unittest.TestCase):
    """CPU coverage of the batched property setters (validation + getter round trips)."""

    def test_set_body_masses_scales_inertia(self):
        scene = sapien.Scene()
        art = build_slider(scene)
        links = art.get_links()

        old_masses = np.array([link.mass for link in links])
        old_inertias = [np.array(link.inertia) for link in links]
        masses = np.array([2.0, 4.0], dtype=np.float32)

        sapien.physx.set_body_masses(links, masses)
        for link, mass, old_mass, old_inertia in zip(links, masses, old_masses, old_inertias):
            self.assertAlmostEqual(link.mass, mass, places=5)
            np.testing.assert_allclose(
                np.array(link.inertia), old_inertia * (mass / old_mass), rtol=1e-5
            )

    def test_set_body_masses_keeps_inertia(self):
        scene = sapien.Scene()
        art = build_slider(scene)
        links = art.get_links()

        old_inertias = [np.array(link.inertia) for link in links]
        masses = np.array([3.0, 5.0], dtype=np.float32)

        sapien.physx.set_body_masses(links, masses, scale_inertia=False)
        for link, mass, old_inertia in zip(links, masses, old_inertias):
            self.assertAlmostEqual(link.mass, mass, places=5)
            np.testing.assert_allclose(np.array(link.inertia), old_inertia, rtol=1e-6)

    def test_set_body_masses_rigid_dynamic(self):
        scene = sapien.Scene()
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.1, 0.1, 0.1])
        actor = builder.build()
        body = actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)

        sapien.physx.set_body_masses([body], np.array([3.0], dtype=np.float32))
        self.assertAlmostEqual(body.mass, 3.0, places=5)

    def test_set_body_masses_validation(self):
        scene = sapien.Scene()
        art = build_slider(scene)
        links = art.get_links()

        with self.assertRaises(RuntimeError):
            sapien.physx.set_body_masses(links, np.array([1.0], dtype=np.float32))
        with self.assertRaises(RuntimeError):
            sapien.physx.set_body_masses(links, np.array([1.0, -1.0], dtype=np.float32))
        with self.assertRaises(RuntimeError):
            sapien.physx.set_body_masses(links, np.array([1.0, np.nan], dtype=np.float32))
        # a failed call must not partially apply
        masses_before = [link.mass for link in links]
        with self.assertRaises(RuntimeError):
            sapien.physx.set_body_masses(links, np.array([9.0, -1.0], dtype=np.float32))
        self.assertEqual([link.mass for link in links], masses_before)

    def test_set_joint_frictions(self):
        scene = sapien.Scene()
        art = build_slider(scene)
        joints = art.get_active_joints()

        sapien.physx.set_joint_frictions(joints, np.array([0.5], dtype=np.float32))
        self.assertAlmostEqual(joints[0].friction, 0.5, places=5)

        with self.assertRaises(RuntimeError):
            sapien.physx.set_joint_frictions(joints, np.array([-0.1], dtype=np.float32))
        with self.assertRaises(RuntimeError):
            sapien.physx.set_joint_frictions(joints, np.array([0.1, 0.2], dtype=np.float32))
        # 0-DOF joints (e.g. the root joint) are rejected
        with self.assertRaises(RuntimeError):
            sapien.physx.set_joint_frictions(
                [art.root.joint], np.array([0.1], dtype=np.float32)
            )

    def test_set_joint_drive_properties(self):
        scene = sapien.Scene()
        art = build_slider(scene)
        joints = art.get_active_joints()
        joint = joints[0]
        joint.set_drive_properties(stiffness=10.0, damping=1.0, force_limit=5.0)

        # partial update keeps unspecified fields
        sapien.physx.set_joint_drive_properties(
            joints, stiffness=np.array([100.0], dtype=np.float32)
        )
        self.assertAlmostEqual(joint.stiffness, 100.0, places=5)
        self.assertAlmostEqual(joint.damping, 1.0, places=5)
        self.assertAlmostEqual(joint.force_limit, 5.0, places=5)

        sapien.physx.set_joint_drive_properties(
            joints,
            damping=np.array([2.0], dtype=np.float32),
            force_limit=np.array([np.inf], dtype=np.float32),
        )
        self.assertAlmostEqual(joint.stiffness, 100.0, places=5)
        self.assertAlmostEqual(joint.damping, 2.0, places=5)
        self.assertTrue(np.isinf(joint.force_limit))

        with self.assertRaises(RuntimeError):
            sapien.physx.set_joint_drive_properties(joints)
        with self.assertRaises(RuntimeError):
            sapien.physx.set_joint_drive_properties(
                joints, stiffness=np.array([1.0, 2.0], dtype=np.float32)
            )
        with self.assertRaises(RuntimeError):
            sapien.physx.set_joint_drive_properties(
                joints, damping=np.array([-1.0], dtype=np.float32)
            )

    def test_set_body_inertias(self):
        scene = sapien.Scene()
        art = build_slider(scene)
        links = art.get_links()

        inertias = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        sapien.physx.set_body_inertias(links, inertias)
        for link, inertia in zip(links, inertias):
            np.testing.assert_allclose(np.array(link.inertia), inertia, rtol=1e-5)

        with self.assertRaises(RuntimeError):
            sapien.physx.set_body_inertias(links, np.array([[0.1, 0.2, 0.3]], dtype=np.float32))
        with self.assertRaises(RuntimeError):
            sapien.physx.set_body_inertias(
                links, np.array([[0.1, 0.2, 0.3], [0.4, -0.5, 0.6]], dtype=np.float32)
            )

    def test_set_body_cmass_local_poses(self):
        scene = sapien.Scene()
        art = build_slider(scene)
        links = art.get_links()

        # unnormalized quaternion must be normalized on application
        poses = np.array(
            [[0.1, 0.2, 0.3, 2.0, 0.0, 0.0, 0.0], [-0.1, 0.0, 0.05, 0.0, 0.0, 0.0, -3.0]],
            dtype=np.float32,
        )
        sapien.physx.set_body_cmass_local_poses(links, poses)
        for link, row in zip(links, poses):
            pose = link.cmass_local_pose
            np.testing.assert_allclose(pose.p, row[:3], rtol=1e-5, atol=1e-6)
            q = row[3:] / np.linalg.norm(row[3:])
            self.assertTrue(
                np.allclose(pose.q, q, atol=1e-5) or np.allclose(pose.q, -q, atol=1e-5)
            )

        with self.assertRaises(RuntimeError):
            sapien.physx.set_body_cmass_local_poses(
                links, np.zeros((2, 7), dtype=np.float32)  # zero-norm quaternion
            )
        with self.assertRaises(RuntimeError):
            sapien.physx.set_body_cmass_local_poses(
                links, np.zeros((1, 7), dtype=np.float32)
            )

        # very large finite quaternion components must not overflow during normalization
        big = np.array([[0.0, 0.0, 0.0, 3e30, 0.0, 0.0, 0.0]], dtype=np.float32)
        sapien.physx.set_body_cmass_local_poses([links[0]], big)
        q = np.array(links[0].cmass_local_pose.q)
        self.assertTrue(np.all(np.isfinite(q)))
        self.assertTrue(
            np.allclose(q, [1, 0, 0, 0], atol=1e-5) or np.allclose(q, [-1, 0, 0, 0], atol=1e-5)
        )

    def test_set_joint_armatures(self):
        scene = sapien.Scene()
        art = build_slider(scene)
        joints = art.get_active_joints()

        sapien.physx.set_joint_armatures(joints, np.array([0.25], dtype=np.float32))
        np.testing.assert_allclose(np.array(joints[0].armature), [0.25], rtol=1e-5)

        with self.assertRaises(RuntimeError):
            sapien.physx.set_joint_armatures(joints, np.array([-0.1], dtype=np.float32))
        with self.assertRaises(RuntimeError):
            sapien.physx.set_joint_armatures(
                [art.root.joint], np.array([0.1], dtype=np.float32)
            )

    def test_set_material_properties(self):
        materials = [sapien.physx.PhysxMaterial(0.5, 0.5, 0.1) for _ in range(3)]

        static = np.array([0.2, 0.4, 0.6], dtype=np.float32)
        dynamic = np.array([0.1, 0.3, 0.5], dtype=np.float32)
        restitution = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        sapien.physx.set_material_properties(
            materials, static_friction=static, dynamic_friction=dynamic, restitution=restitution
        )
        for material, s, d, r in zip(materials, static, dynamic, restitution):
            self.assertAlmostEqual(material.static_friction, s, places=5)
            self.assertAlmostEqual(material.dynamic_friction, d, places=5)
            self.assertAlmostEqual(material.restitution, r, places=5)

        # partial update keeps unspecified fields
        sapien.physx.set_material_properties(
            materials, static_friction=np.array([0.9, 0.9, 0.9], dtype=np.float32)
        )
        for material, d, r in zip(materials, dynamic, restitution):
            self.assertAlmostEqual(material.static_friction, 0.9, places=5)
            self.assertAlmostEqual(material.dynamic_friction, d, places=5)
            self.assertAlmostEqual(material.restitution, r, places=5)

        with self.assertRaises(RuntimeError):
            sapien.physx.set_material_properties(materials)
        with self.assertRaises(RuntimeError):
            sapien.physx.set_material_properties(
                materials, restitution=np.array([0.0, 0.5, 1.5], dtype=np.float32)
            )
        with self.assertRaises(RuntimeError):
            sapien.physx.set_material_properties(
                materials, static_friction=np.array([0.1], dtype=np.float32)
            )

    def test_destroyed_articulation_rejected_atomically(self):
        scene0 = sapien.Scene()
        art0 = build_slider(scene0)
        alive = art0.get_active_joints()[0]
        alive.set_drive_properties(stiffness=10.0, damping=1.0, force_limit=5.0)
        alive.friction = 0.05
        armature_before = np.array(alive.armature)

        scene1 = sapien.Scene()
        art1 = build_slider(scene1)
        dead = art1.get_active_joints()[0]
        del art1, scene1
        gc.collect()

        # the dead joint still reports a nonzero cached DOF; validation must reject it
        # before mutating any earlier entry in the batch
        with self.assertRaises(RuntimeError):
            sapien.physx.set_joint_frictions(
                [alive, dead], np.array([0.7, 0.7], dtype=np.float32)
            )
        with self.assertRaises(RuntimeError):
            sapien.physx.set_joint_drive_properties(
                [alive, dead], stiffness=np.array([99.0, 99.0], dtype=np.float32)
            )
        with self.assertRaises(RuntimeError):
            sapien.physx.set_joint_armatures(
                [alive, dead], np.array([0.3, 0.3], dtype=np.float32)
            )

        # the alive joint placed before the dead one must be untouched
        self.assertAlmostEqual(alive.friction, 0.05, places=5)
        self.assertAlmostEqual(alive.stiffness, 10.0, places=5)
        np.testing.assert_allclose(np.array(alive.armature), armature_before, atol=1e-6)


def _cuda_device_available() -> bool:
    """Skip only on identifiable no-GPU conditions (missing driver or no CUDA device);
    any other GPU initialization failure must propagate as a test error."""
    import ctypes

    try:
        lib = ctypes.CDLL("libcuda.so")
    except OSError:
        return False
    return lib.cuInit(0) == 0  # nonzero includes CUDA_ERROR_NO_DEVICE


class TestBatchedPropertiesGpu(unittest.TestCase):
    """GPU coverage: values set after gpu_init() must take effect at the next step
    without disturbing GPU-side simulation state."""

    @classmethod
    def setUpClass(cls):
        if not _cuda_device_available():
            raise unittest.SkipTest("no usable CUDA device")
        # release scenes leaked by earlier CPU tests; a live scene keeps the PhysxEngine
        # singleton alive and enable_gpu() refuses to run with an existing engine
        gc.collect()
        # deliberately unguarded: enable_gpu() failures are real errors, not skips
        sapien.physx.enable_gpu()

    def _build_pair(self):
        px = sapien.physx.PhysxGpuSystem()
        scene = sapien.Scene([px])
        art0 = build_slider(scene, 0.0)
        art1 = build_slider(scene, 5.0)
        for art in (art0, art1):
            for joint in art.get_active_joints():
                joint.set_drive_properties(stiffness=20.0, damping=0.0, force_limit=1e5)
        px.gpu_init()
        for art in (art0, art1):
            px._gpu_upload_articulation_target_qpos(art.gpu_index, [0.5])
        return px, scene, art0, art1

    def _qpos(self, px, art) -> float:
        return px._gpu_download_articulation_qpos(art.gpu_index)[0]

    def test_gpu_mass_update(self):
        px, scene, art0, art1 = self._build_pair()
        links1 = art1.get_links()
        heavy = np.array([link.mass for link in links1], dtype=np.float32) * 10.0
        sapien.physx.set_body_masses(links1, heavy)
        for _ in range(20):
            px.step()
        q0 = self._qpos(px, art0)
        q1 = self._qpos(px, art1)
        # the lighter slider accelerates faster under the same drive force
        self.assertGreater(q0, 0.05)
        self.assertGreater(q0, q1 * 2.0)

    def test_gpu_joint_friction_update(self):
        px, scene, art0, art1 = self._build_pair()
        sapien.physx.set_joint_frictions(
            art1.get_active_joints(), np.array([1000.0], dtype=np.float32)
        )
        for _ in range(20):
            px.step()
        q0 = self._qpos(px, art0)
        q1 = self._qpos(px, art1)
        # enormous joint friction must inhibit motion
        self.assertGreater(q0, 0.05)
        self.assertLess(q1, q0 * 0.5)

    def test_gpu_drive_damping_update(self):
        px, scene, art0, art1 = self._build_pair()
        sapien.physx.set_joint_drive_properties(
            art1.get_active_joints(), damping=np.array([50.0], dtype=np.float32)
        )
        # stiffness is preserved by the partial update
        self.assertAlmostEqual(art1.get_active_joints()[0].stiffness, 20.0, places=4)
        for _ in range(20):
            px.step()
        q0 = self._qpos(px, art0)
        q1 = self._qpos(px, art1)
        # the overdamped slider approaches the target more slowly
        self.assertGreater(q0, 0.05)
        self.assertGreater(q0, q1 * 1.5)

    def test_gpu_state_preserved_across_update(self):
        px, scene, art0, art1 = self._build_pair()
        for _ in range(30):
            px.step()
        q_before = self._qpos(px, art0)
        self.assertGreater(q_before, 0.05)

        links = art0.get_links()
        sapien.physx.set_body_masses(
            links, np.array([link.mass for link in links], dtype=np.float32) * 2.0
        )
        sapien.physx.set_joint_frictions(
            art0.get_active_joints(), np.array([0.1], dtype=np.float32)
        )
        sapien.physx.set_joint_drive_properties(
            art0.get_active_joints(), damping=np.array([1.0], dtype=np.float32)
        )

        # property updates alone must not change GPU joint state
        self.assertAlmostEqual(self._qpos(px, art0), q_before, places=5)

        px.step()
        q_after = self._qpos(px, art0)
        # the state stays continuous; it must not be clobbered by stale CPU state
        self.assertGreater(q_after, 0.5 * q_before)
        self.assertLess(abs(q_after - q_before), 0.1)

    def _build_pendulum_pair(self):
        px = sapien.physx.PhysxGpuSystem()
        scene = sapien.Scene([px])
        art0 = build_pendulum(scene, 0.0)
        art1 = build_pendulum(scene, 5.0)
        px.gpu_init()
        return px, scene, art0, art1

    def test_gpu_inertia_update(self):
        px, scene, art0, art1 = self._build_pendulum_pair()
        links1 = art1.get_links()
        big = np.stack([np.array(link.inertia) for link in links1]).astype(np.float32) * 200.0
        sapien.physx.set_body_inertias(links1, big)
        for _ in range(20):
            px.step()
        q0 = self._qpos(px, art0)
        q1 = self._qpos(px, art1)
        # much larger rotational inertia must slow the gravity-driven swing
        self.assertGreater(q0, 0.1)
        self.assertGreater(q0, q1 * 2.0)

    def test_gpu_cmass_update(self):
        px, scene, art0, art1 = self._build_pendulum_pair()
        pendulum1 = art1.get_links()[1]
        # move the center of mass onto the pivot: no gravity torque, no swing
        sapien.physx.set_body_cmass_local_poses(
            [pendulum1], np.array([[-0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        )
        for _ in range(20):
            px.step()
        q0 = self._qpos(px, art0)
        q1 = self._qpos(px, art1)
        self.assertGreater(q0, 0.1)
        self.assertLess(abs(q1), q0 * 0.2)

    def test_gpu_armature_update(self):
        px, scene, art0, art1 = self._build_pair()
        sapien.physx.set_joint_armatures(
            art1.get_active_joints(), np.array([100.0], dtype=np.float32)
        )
        for _ in range(20):
            px.step()
        q0 = self._qpos(px, art0)
        q1 = self._qpos(px, art1)
        # armature adds joint-space inertia, slowing the driven slider
        self.assertGreater(q0, 0.05)
        self.assertGreater(q0, q1 * 2.0)


if __name__ == "__main__":
    unittest.main()
