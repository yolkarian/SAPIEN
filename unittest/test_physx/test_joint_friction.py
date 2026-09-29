"""Joint Coulomb effort units, URDF loading, and CPU/GPU solver behavior."""

import gc
from pathlib import Path
import tempfile
import unittest

import numpy as np
import sapien

from .test_gpu_contact_queries import _cuda_device_available


def setUpModule() -> None:
    # Enable before CPU tests too: unittest retains failing traceback frames, which
    # may keep CPU PhysX objects alive until after GPU setUpClass would otherwise run.
    if _cuda_device_available() and not sapien.physx.is_gpu_enabled():
        sapien.physx.enable_gpu()


def build_joint(
    scene: sapien.Scene, joint_type: str, friction: float
) -> sapien.physx.PhysxArticulation:
    builder = scene.create_articulation_builder()
    root = builder.create_link_builder()
    root.set_mass_and_inertia(1.0, sapien.Pose(), [0.1, 0.1, 0.1])
    child = builder.create_link_builder(root)
    child.set_mass_and_inertia(1.0, sapien.Pose(), [0.1, 0.1, 0.1])
    child.set_joint_properties(
        joint_type, [[-10, 10]], sapien.Pose(), sapien.Pose(), friction=friction
    )
    child.joint_record.armature = 0.0
    return builder.build(fix_root_link=True)


class TestJointFrictionCpu(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_config = sapien.physx.get_scene_config()
        config = sapien.physx.PhysxSceneConfig()
        config.gravity = [0, 0, 0]
        sapien.physx.set_scene_config(config)
        self.system = sapien.physx.PhysxCpuSystem()
        self.system.timestep = 0.005
        self.scene = sapien.Scene([self.system])

    def tearDown(self) -> None:
        self.scene.close()
        self.system.close()
        del self.scene, self.system
        sapien.physx.set_scene_config(self.previous_config)
        gc.collect()

    def test_builder_clone_and_reparent(self) -> None:
        articulation = build_joint(self.scene, "prismatic", 3.5)
        joint = articulation.active_joints[0]
        self.assertAlmostEqual(joint.friction, 3.5)
        clones = articulation.clone_links()
        self.assertAlmostEqual(clones[1].joint.friction, 3.5)
        other = build_joint(self.scene, "prismatic", 0.0)
        joint.child_link.set_parent(other.root)
        self.assertAlmostEqual(joint.friction, 3.5)
        for invalid in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(RuntimeError):
                joint.set_friction(invalid)
            self.assertAlmostEqual(joint.friction, 3.5)
        joint.friction = 0
        self.assertEqual(joint.friction, 0)

    def test_urdf_friction_is_applied(self) -> None:
        # absent attributes fall back to defaults: friction keeps 0.05, damping is 0
        cases = (('damping="0.5" friction="3.5"', 3.5, 0.5), ("", 0.05, 0.0),
                 ('friction="0"', 0.0, 0.0))
        for dynamics, friction, damping in cases:
            with self.subTest(dynamics=dynamics), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "friction.urdf"
                path.write_text(f'''<robot name="friction">
<link name="a"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
<link name="b"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
<joint name="j" type="revolute"><parent link="a"/><child link="b"/><axis xyz="0 0 1"/>
<limit lower="-1" upper="1" effort="10" velocity="5"/><dynamics {dynamics}/></joint>
</robot>''', encoding="utf-8")
                loader = self.scene.create_urdf_loader()
                builders, _, _ = loader.parse(str(path))
                builder = builders[0]
                builder.set_scene(self.scene)
                for _ in range(2):
                    articulation = builder.build(fix_root_link=True)
                    joint = articulation.active_joints[0]
                    self.assertAlmostEqual(joint.friction, friction)
                    self.assertAlmostEqual(joint.damping, damping)

    def test_builder_without_friction_uses_default(self) -> None:
        builder = self.scene.create_articulation_builder()
        root = builder.create_link_builder()
        child = builder.create_link_builder(root)
        child.set_joint_properties("prismatic", [[-1, 1]], sapien.Pose(), sapien.Pose())
        articulation = builder.build(fix_root_link=True)
        self.assertAlmostEqual(articulation.active_joints[0].friction, 0.05)

    def test_effort_threshold_and_zero(self) -> None:
        for use_tgs in (True, False):
            with self.subTest(use_tgs=use_tgs):
                self.scene.close()
                self.system.close()
                config = sapien.physx.PhysxSceneConfig()
                config.gravity = [0, 0, 0]
                config.enable_tgs = use_tgs
                sapien.physx.set_scene_config(config)
                self.system = sapien.physx.PhysxCpuSystem()
                self.system.timestep = 0.005
                self.scene = sapien.Scene([self.system])
                self._check_effort_threshold_and_zero()

    def _check_effort_threshold_and_zero(self) -> None:
        for joint_type in ("prismatic", "revolute"):
            with self.subTest(joint_type=joint_type):
                articulation = build_joint(self.scene, joint_type, 3.5)
                for _ in range(40):
                    articulation.qf = [2.0]
                    self.system.step()
                # TGS can accumulate tiny positional drift while its friction
                # velocity solve holds the joint still (about 5e-5 m here).
                np.testing.assert_allclose(articulation.qvel, 0, atol=1e-6)
                np.testing.assert_allclose(articulation.qpos, 0, atol=1e-4)
                for _ in range(40):
                    articulation.qf = [6.0]
                    self.system.step()
                self.assertGreater(articulation.qpos[0], 0.01)
                articulation.qpos = [0]
                articulation.qvel = [0]
                articulation.active_joints[0].friction = 0
                for _ in range(40):
                    articulation.qf = [2.0]
                    self.system.step()
                self.assertGreater(articulation.qpos[0], 0.01)


class TestJointFrictionGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not _cuda_device_available():
            raise unittest.SkipTest("no CUDA device")
        gc.collect()
        if not sapien.physx.is_gpu_enabled():
            sapien.physx.enable_gpu()

    def test_gpu_effort_threshold_and_runtime_update(self) -> None:
        for use_tgs in (True, False):
            with self.subTest(use_tgs=use_tgs):
                self._check_effort_threshold_and_runtime_update(use_tgs)

    def _check_effort_threshold_and_runtime_update(self, use_tgs: bool) -> None:
        from .test_gpu_articulation_buffers import _read_values, _write_float_values

        previous = sapien.physx.get_scene_config()
        config = sapien.physx.PhysxSceneConfig()
        config.gravity = [0, 0, 0]
        config.enable_tgs = use_tgs
        sapien.physx.set_scene_config(config)
        system = sapien.physx.PhysxGpuSystem()
        system.timestep = 0.005
        try:
            with sapien.Scene([system]) as scene:
                articulations = [build_joint(scene, kind, 3.5) for kind in ("prismatic", "revolute")]
                system.gpu_init()
                forces = system.cuda_articulation_qf
                try:
                    for articulation in articulations:
                        _write_float_values(forces, (articulation.gpu_index, 0), [2.0])
                finally:
                    del forces
                for _ in range(40):
                    system.gpu_apply_articulation_qf()
                    system.step()
                system.gpu_fetch_articulation_qvel()
                for articulation in articulations:
                    qpos = system._gpu_download_articulation_qpos(articulation.gpu_index)
                    qvel = _read_values(
                        system.cuda_articulation_qvel, (articulation.gpu_index, 0), 1, np.float32
                    )
                    np.testing.assert_allclose(qvel, 0, atol=1e-6)
                    np.testing.assert_allclose(qpos, 0, atol=1e-4)
                # Changing the effort after gpu_init must propagate on the next step.
                sapien.physx.set_joint_frictions(
                    [articulation.active_joints[0] for articulation in articulations], [0, 0]
                )
                for _ in range(40):
                    system.gpu_apply_articulation_qf()
                    system.step()
                for articulation in articulations:
                    qpos = system._gpu_download_articulation_qpos(articulation.gpu_index)
                    self.assertGreater(qpos[0], 0.01)
        finally:
            try:
                system.close()
            finally:
                sapien.physx.set_scene_config(previous)
