"""Collision shape error atomicity and articulation clone properties."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import sapien


class TestShapeRegressions(unittest.TestCase):
    def test_missing_mesh_has_explicit_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for constructor in (
                sapien.physx.PhysxCollisionShapeConvexMesh,
                sapien.physx.PhysxCollisionShapeTriangleMesh,
            ):
                for path in (Path(directory) / "missing.stl", Path(directory)):
                    with self.subTest(constructor=constructor.__name__, path=path):
                        with self.assertRaisesRegex(RuntimeError, "File not found") as error:
                            constructor(str(path), [1, 1, 1], None)
                        self.assertIn(str(path), str(error.exception))

    def test_attached_shape_pose_failure_is_atomic(self) -> None:
        shape = sapien.physx.PhysxCollisionShapeBox([0.1] * 3, None)
        initial = sapien.Pose([0.2, 0.3, 0.4])
        shape.local_pose = initial
        body = sapien.physx.PhysxRigidDynamicComponent()
        body.attach(shape)
        with self.assertRaisesRegex(RuntimeError, "before"):
            shape.local_pose = sapien.Pose([1, 2, 3])
        np.testing.assert_array_equal(shape.local_pose.p, initial.p)
        np.testing.assert_array_equal(shape.local_pose.q, initial.q)

    def test_convex_clone_preserves_shape_properties(self) -> None:
        system = sapien.physx.PhysxCpuSystem()
        with sapien.Scene([system]) as scene:
            builder = scene.create_articulation_builder()
            root = builder.create_link_builder()
            root.add_convex_collision_from_file(
                str(Path(__file__).resolve().parents[1] / "assets" / "cone.stl"),
                pose=sapien.Pose([0.3, 0.4, 0.5]),
                density=750,
            )
            articulation = builder.build(fix_root_link=True)
            original = articulation.links[0].collision_shapes[0]
            original.set_collision_groups([7, 7, 0, 0])
            original.contact_offset = 0.05
            original.rest_offset = 0.002
            original.patch_radius = 0.03
            original.min_patch_radius = 0.01
            clones = articulation.clone_links()
            cloned = clones[0].collision_shapes[0]
            self.assertEqual(cloned.collision_groups, original.collision_groups)
            for property_name in (
                "density", "contact_offset", "rest_offset", "patch_radius", "min_patch_radius"
            ):
                self.assertAlmostEqual(getattr(cloned, property_name), getattr(original, property_name))
            np.testing.assert_allclose(cloned.local_pose.p, original.local_pose.p)
            np.testing.assert_allclose(cloned.local_pose.q, original.local_pose.q)
            np.testing.assert_allclose(cloned.scale, original.scale)
        system.close()
