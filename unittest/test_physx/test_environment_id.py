"""Tests for the PhysX GPU broadphase environment ID API."""

import gc
import pickle
import unittest

import sapien

PX_INVALID_U32 = 0xFFFFFFFF


class TestEnvironmentIDConfig(unittest.TestCase):
    def tearDown(self):
        sapien.physx.set_scene_config(sapien.physx.PhysxSceneConfig())

    def test_default_bits_are_four_on_z_alone(self):
        # Four matches PhysX's own "snap to grid" shift, so the banding costs no coordinate
        # precision; Z alone because the band count is 2 ** max(x, y, z), never the product.
        config = sapien.physx.PhysxSceneConfig()
        self.assertEqual(config.gpu_broadphase_nb_bits_env_id_x, 0)
        self.assertEqual(config.gpu_broadphase_nb_bits_env_id_y, 0)
        self.assertEqual(config.gpu_broadphase_nb_bits_env_id_z, 4)
        self.assertIsNone(config.num_scenes)
        self.assertFalse(config.with_shared_scene)

    def test_bits_are_written_only_through_the_three_axis_setter(self):
        config = sapien.physx.PhysxSceneConfig()
        config.set_gpu_broadphase_env_id_bits(6, 0, 0)
        self.assertEqual(config.gpu_broadphase_nb_bits_env_id_x, 6)
        self.assertEqual(config.gpu_broadphase_nb_bits_env_id_y, 0)
        self.assertEqual(config.gpu_broadphase_nb_bits_env_id_z, 0)

        for axis in ("x", "y", "z"):
            with self.assertRaises(AttributeError):
                setattr(config, f"gpu_broadphase_nb_bits_env_id_{axis}", 8)

        for bits in ((17, 0, 0), (0, 17, 0), (0, 0, 17)):
            with self.assertRaisesRegex(RuntimeError, r"must be in \[0, 16\]"):
                config.set_gpu_broadphase_env_id_bits(*bits)

    def test_declaration_survives_pickle(self):
        config = sapien.physx.PhysxSceneConfig()
        config.set_gpu_broadphase_env_id_bits(0, 0, 9)
        config.num_scenes = 512
        config.with_shared_scene = True

        roundtrip = pickle.loads(pickle.dumps(config))
        self.assertEqual(roundtrip.num_scenes, 512)
        self.assertTrue(roundtrip.with_shared_scene)
        self.assertEqual(roundtrip.gpu_broadphase_nb_bits_env_id_z, 9)


class TestEnvironmentIDGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gc.collect()
        try:
            sapien.physx.enable_gpu()
        except Exception as exc:
            raise unittest.SkipTest(f"GPU PhysX not available: {exc}")

    def tearDown(self):
        sapien.physx.set_scene_config(sapien.physx.PhysxSceneConfig())

    @staticmethod
    def make_system(num_scenes=None, with_shared_scene=False, bits=None):
        """Declare the layout on the config, which is where the system reads it."""
        config = sapien.physx.PhysxSceneConfig()
        if bits is not None:
            config.set_gpu_broadphase_env_id_bits(*bits)
        config.num_scenes = num_scenes
        config.with_shared_scene = with_shared_scene
        sapien.physx.set_scene_config(config)
        return sapien.physx.PhysxGpuSystem()

    @staticmethod
    def axes(system):
        config = system.config
        return (
            config.gpu_broadphase_nb_bits_env_id_x,
            config.gpu_broadphase_nb_bits_env_id_y,
            config.gpu_broadphase_nb_bits_env_id_z,
        )

    @staticmethod
    def add_body(scene, name="body"):
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.1, 0.1, 0.1])
        builder.set_initial_pose(sapien.Pose(p=[0, 0, 1]))
        return builder.build(name=name)

    def test_default_layout_is_z_only_and_shared_widens_every_axis(self):
        # Without a shared object only the widest axis matters, so Z alone is enough. With one,
        # every banded axis must carry the same count: a narrower axis keeps fewer of the
        # shifted ID's low bits and would wrap environments back out of the reachable band.
        self.assertEqual(self.axes(self.make_system()), (0, 0, 4))
        self.assertEqual(self.axes(self.make_system(with_shared_scene=True)), (4, 4, 4))

    def test_explicit_bits_apply_without_a_declared_count(self):
        system = self.make_system(bits=(7, 0, 0))
        self.assertEqual(self.axes(system), (7, 0, 0))

        shared = self.make_system(with_shared_scene=True, bits=(0, 6, 0))
        self.assertEqual(self.axes(shared), (6, 6, 6))

    def test_declared_count_overrides_explicit_bits(self):
        # Only the widest axis decides the band count, so without a shared object one axis is
        # enough and X/Y keep their full broadphase coordinate precision.
        system = self.make_system(num_scenes=4096, bits=(0, 0, 3))
        self.assertEqual(self.axes(system), (0, 0, 12))

        # With a shared scene every banded axis must carry the same count, and the outermost
        # bands are reserved, so the same total needs one more bit than it otherwise would.
        shared = self.make_system(num_scenes=4096, with_shared_scene=True, bits=(0, 0, 3))
        self.assertEqual(self.axes(shared), (13, 13, 13))

    def test_shared_rejects_mixed_non_zero_axis_bits(self):
        # (12, 8, 0): an ID inside the 12-bit axis's safe band wraps to a different band on the
        # 8-bit axis, so that axis stops overlapping the shared bounds and the pair disappears.
        with self.assertRaisesRegex(RuntimeError, "non-zero broadphase env ID bit count"):
            self.make_system(with_shared_scene=True, bits=(12, 8, 0))
        with self.assertRaisesRegex(RuntimeError, "non-zero broadphase env ID bit count"):
            self.make_system(with_shared_scene=True, bits=(2, 0, 5))

        # Mixed counts are fine without a shared object, and equal ones always are.
        self.assertEqual(self.axes(self.make_system(bits=(12, 8, 0))), (12, 8, 0))
        self.assertEqual(
            self.axes(self.make_system(with_shared_scene=True, bits=(5, 5, 5))), (5, 5, 5)
        )

    def test_get_or_assign_requires_a_declared_count(self):
        system = self.make_system()
        scene = sapien.Scene([system])
        with self.assertRaisesRegex(RuntimeError, "did not declare"):
            scene.get_or_assign_environment_id()

        # Undeclared scenes are all environment 0 and nothing is recorded.
        self.assertIsNone(scene.environment_id)
        self.assertEqual(system.get_broadphase_environment_id(scene), 0)

    def test_declared_scenes_get_unique_ids_nobody_can_overwrite(self):
        system = self.make_system(num_scenes=16)
        scenes = [sapien.Scene([system]) for _ in range(3)]

        self.assertEqual([s.get_or_assign_environment_id() for s in scenes], [0, 1, 2])
        self.assertEqual([s.environment_id for s in scenes], [0, 1, 2])
        # Repeating the call is stable, and there is no way to choose a value.
        self.assertEqual(scenes[1].get_or_assign_environment_id(), 1)
        with self.assertRaises(AttributeError):
            scenes[0].environment_id = 5

    def test_set_shared_environment_only_before_bodies(self):
        system = self.make_system(num_scenes=8, with_shared_scene=True)

        shared = sapien.Scene([system])
        shared.set_shared_environment()
        self.assertEqual(shared.environment_id, -1)
        self.assertEqual(system.get_broadphase_environment_id(shared), PX_INVALID_U32)
        shared.set_shared_environment()  # idempotent

        late = sapien.Scene([system])
        self.add_body(late)
        with self.assertRaisesRegex(RuntimeError, "before adding"):
            late.set_shared_environment()

    def test_set_shared_environment_requires_the_declaration(self):
        system = self.make_system(num_scenes=8)
        with self.assertRaisesRegex(RuntimeError, "did not declare"):
            sapien.Scene([system]).set_shared_environment()

    def test_construction_rejects_bad_counts(self):
        with self.assertRaisesRegex(RuntimeError, "num_scenes must be positive"):
            self.make_system(num_scenes=0)

        # A shared scene keeps the scene ID in a 16-bit collision-group field and reserves
        # 0xffff for itself, so 65535 ordinary environments is the ceiling.
        self.make_system(num_scenes=0xFFFF, with_shared_scene=True)
        with self.assertRaisesRegex(RuntimeError, "16-bit collision group field"):
            self.make_system(num_scenes=0x10000, with_shared_scene=True)

    def test_construction_rejects_shared_bits_with_no_usable_band(self):
        # One bit splits the range into two bands and both fall outside the fixed encoded
        # interval PhysX gives shared objects, so no environment could reach the shared scene.
        # The bit count is baked into the PhysX scene at createScene, so this must fail here
        # rather than at the first body.
        with self.assertRaisesRegex(RuntimeError, "leave no band inside"):
            self.make_system(with_shared_scene=True, bits=(1, 1, 1))

    def test_manual_ids_only_without_a_declared_count(self):
        manual = self.make_system()
        a, b = sapien.Scene([manual]), sapien.Scene([manual])
        a.set_environment_id(7)
        b.set_environment_id(2)
        # Read back exactly what was set: the map stores the unshifted ID.
        self.assertEqual((a.environment_id, b.environment_id), (7, 2))
        # No shared scene, so no band offset either -- PhysX gets the same values.
        self.assertEqual(manual.get_broadphase_environment_id(a), 7)
        self.assertEqual(manual.get_broadphase_environment_id(b), 2)

        # Declaring num_scenes makes SAPIEN the sole assigner.
        managed = self.make_system(num_scenes=4)
        with self.assertRaisesRegex(RuntimeError, "declared num_scenes"):
            sapien.Scene([managed]).set_environment_id(1)

    def test_manual_ids_are_offset_only_on_the_way_to_physx(self):
        # With a shared scene the stored ID stays raw and only the broadphase value is shifted
        # into a band that still reaches the shared object.
        system = self.make_system(with_shared_scene=True)
        ground = sapien.Scene([system])
        ground.set_shared_environment()

        scene = sapien.Scene([system])
        scene.set_environment_id(3)
        self.assertEqual(scene.environment_id, 3)
        self.assertGreater(system.get_broadphase_environment_id(scene), 3)
        self.assertEqual(system.get_broadphase_environment_id(ground), PX_INVALID_U32)

    def test_manual_duplicate_ids_put_scenes_in_one_environment(self):
        # Deliberate: several scenes sharing an ID is how they end up colliding with each other.
        system = self.make_system()
        a, b = sapien.Scene([system]), sapien.Scene([system])
        a.set_environment_id(5)
        b.set_environment_id(5)
        self.assertEqual(
            system.get_broadphase_environment_id(a),
            system.get_broadphase_environment_id(b),
        )

    def test_no_offset_without_a_shared_scene_in_either_mode(self):
        # The band offset exists only to keep environments overlapping the fixed encoded
        # interval PhysX gives shared objects. No shared scene, no interval to miss, so PhysX
        # gets the ID untouched whichever mode assigned it.
        managed = self.make_system(num_scenes=8)
        scenes = [sapien.Scene([managed]) for _ in range(3)]
        ids = [s.get_or_assign_environment_id() for s in scenes]
        self.assertEqual(ids, [0, 1, 2])
        self.assertEqual(
            [managed.get_broadphase_environment_id(s) for s in scenes], ids
        )

        manual = self.make_system()
        picked = sapien.Scene([manual])
        picked.set_environment_id(9)
        self.assertEqual(manual.get_broadphase_environment_id(picked), 9)

    def test_collision_group_carries_the_raw_id_not_the_broadphase_one(self):
        # The collision-group scene field is only ever compared for equality by SAPIEN's filter
        # shader, never encoded into bounds, so the band offset that keeps environments
        # overlapping the shared object would be meaningless noise there. Only the value handed
        # to PxActor::setEnvironmentID carries it.
        system = self.make_system(with_shared_scene=True)
        ground = sapien.Scene([system])
        ground.set_shared_environment()

        scene = sapien.Scene([system])
        scene.set_environment_id(3)
        actor = self.add_body(scene)

        broadphase = system.get_broadphase_environment_id(scene)
        self.assertNotEqual(broadphase, 3)  # the offset really is applied on that path

        shape = actor.find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent
        ).collision_shapes[0]
        self.assertEqual(shape.get_collision_groups()[3] >> 16, 3)

        ground_builder = ground.create_actor_builder()
        ground_builder.add_box_collision(half_size=[5.0, 5.0, 0.1])
        ground_builder.set_physx_body_type("static")
        ground_builder.set_initial_pose(sapien.Pose(p=[0, 0, -0.1]))
        ground_actor = ground_builder.build(name="ground")
        ground_shape = ground_actor.find_component_by_type(
            sapien.physx.PhysxRigidStaticComponent
        ).collision_shapes[0]
        self.assertEqual(ground_shape.get_collision_groups()[3] >> 16, 0xFFFF)

    def test_manual_ids_are_rejected_after_bodies_and_out_of_range(self):
        system = self.make_system()
        late = sapien.Scene([system])
        self.add_body(late)
        with self.assertRaisesRegex(RuntimeError, "before adding"):
            late.set_environment_id(1)

        with self.assertRaisesRegex(RuntimeError, "past the maximum"):
            sapien.Scene([system]).set_environment_id(1 << 24)

        # The collision-group field is 16 bits with 0xffff reserved for the shared scene.
        shared = self.make_system(with_shared_scene=True)
        with self.assertRaisesRegex(RuntimeError, "16 bits wide"):
            sapien.Scene([shared]).set_environment_id(0xFFFF)

    def test_declared_count_is_a_hard_cap_on_scenes(self):
        # num_scenes is how many ordinary scenes the simulation holds, not a sizing hint: the
        # derived bit count is frozen at construction, so an extra scene cannot be spread.
        system = self.make_system(num_scenes=4, with_shared_scene=True)
        scenes = [sapien.Scene([system]) for _ in range(4)]
        self.assertEqual([s.get_or_assign_environment_id() for s in scenes], list(range(4)))

        with self.assertRaisesRegex(RuntimeError, "declared num_scenes=4"):
            sapien.Scene([system]).get_or_assign_environment_id()

    def test_destroyed_scenes_return_their_slot(self):
        # IDs are assigned lazily, so a scene that never asked for one costs nothing even if it
        # is built and dropped. One that did ask must give the slot back when it dies, or a
        # setup/teardown loop would exhaust num_scenes without ever holding that many scenes.
        system = self.make_system(num_scenes=3, with_shared_scene=True)

        never_asked = sapien.Scene([system])
        del never_asked
        gc.collect()

        asked = sapien.Scene([system])
        self.assertEqual(asked.get_or_assign_environment_id(), 0)
        del asked
        gc.collect()

        # All three declared environments are still reachable, in some order.
        live = [sapien.Scene([system]) for _ in range(3)]
        self.assertEqual(sorted(s.get_or_assign_environment_id() for s in live), [0, 1, 2])

        # And the cap still bites while they are all alive.
        with self.assertRaisesRegex(RuntimeError, "declared num_scenes=3"):
            sapien.Scene([system]).get_or_assign_environment_id()

    def test_shared_marking_releases_an_already_assigned_id(self):
        # A shared scene is outside the num_scenes budget, so a slot it took before being
        # marked has to go back to the pool rather than being burned.
        system = self.make_system(num_scenes=2, with_shared_scene=True)
        shared = sapien.Scene([system])
        self.assertEqual(shared.get_or_assign_environment_id(), 0)

        shared.set_shared_environment()
        self.assertEqual(shared.environment_id, -1)

        # Slot 0 is free again, and both declared environments are still available.
        ordinary = [sapien.Scene([system]) for _ in range(2)]
        self.assertEqual(sorted(s.get_or_assign_environment_id() for s in ordinary), [0, 1])
        with self.assertRaisesRegex(RuntimeError, "declared num_scenes=2"):
            sapien.Scene([system]).get_or_assign_environment_id()

        # The reused ID is a real environment again, distinct from the shared scene's.
        self.assertEqual(system.get_broadphase_environment_id(shared), PX_INVALID_U32)
        self.assertNotEqual(
            system.get_broadphase_environment_id(ordinary[0]),
            system.get_broadphase_environment_id(ordinary[1]),
        )

    def test_shared_scene_reports_minus_one_from_both_getters(self):
        system = self.make_system(num_scenes=4, with_shared_scene=True)
        shared = sapien.Scene([system])
        shared.set_shared_environment()
        self.assertEqual(shared.environment_id, -1)
        self.assertEqual(shared.get_or_assign_environment_id(), -1)
        # Marking shared consumes no ordinary slot.
        self.assertEqual(sapien.Scene([system]).get_or_assign_environment_id(), 0)

    def test_articulations_get_the_scene_environment_id(self):
        system = self.make_system(num_scenes=4, with_shared_scene=True)
        scenes = [sapien.Scene([system]) for _ in range(2)]
        for index, scene in enumerate(scenes):
            builder = scene.create_articulation_builder()
            root = builder.create_link_builder()
            root.set_name("root")
            root.add_box_collision(half_size=[0.1, 0.1, 0.1])
            child = builder.create_link_builder(root)
            child.set_name("child")
            child.add_box_collision(half_size=[0.1, 0.1, 0.1])
            child.set_joint_properties(
                "revolute",
                [[-1.0, 1.0]],
                sapien.Pose(p=[0.2, 0, 0]),
                sapien.Pose(p=[-0.2, 0, 0]),
            )
            builder.set_initial_pose(sapien.Pose(p=[0, 0, 1]))
            builder.build()

            # Building the articulation is what binds the scene to an ID, exactly like a body.
            self.assertEqual(scene.environment_id, index)
            self.assertEqual(
                system.get_broadphase_environment_id(scene),
                system.get_broadphase_environment_id(scene),
            )

        self.assertNotEqual(
            system.get_broadphase_environment_id(scenes[0]),
            system.get_broadphase_environment_id(scenes[1]),
        )

    def test_broadphase_id_is_shifted_only_with_a_shared_scene(self):
        plain = self.make_system(num_scenes=64)
        scene = sapien.Scene([plain])
        self.assertEqual(scene.get_or_assign_environment_id(), 0)
        self.assertEqual(plain.get_broadphase_environment_id(scene), 0)

        shared_system = self.make_system(num_scenes=64, with_shared_scene=True)
        ground = sapien.Scene([shared_system])
        ground.set_shared_environment()
        env = sapien.Scene([shared_system])
        self.assertEqual(env.get_or_assign_environment_id(), 0)
        # The user-visible ID is untouched; only what PhysX receives moves into a safe band.
        self.assertGreater(shared_system.get_broadphase_environment_id(env), 0)

    def test_shared_scene_marks_the_render_system(self):
        system = self.make_system(num_scenes=4, with_shared_scene=True)
        try:
            render_system = sapien.render.RenderSystem(system.device)
        except Exception as exc:
            raise unittest.SkipTest(f"RenderSystem not available: {exc}")
        scene = sapien.Scene([system, render_system])

        self.assertFalse(render_system.batched_render_shared)
        scene.set_shared_environment()
        self.assertTrue(render_system.batched_render_shared)

    def test_environment_ids_isolate_coincident_scenes(self):
        import numpy as np

        system = self.make_system(num_scenes=8, with_shared_scene=True)
        ground_scene = sapien.Scene([system])
        ground_scene.set_shared_environment()
        builder = ground_scene.create_actor_builder()
        builder.add_box_collision(half_size=[20.0, 20.0, 0.5])
        builder.set_physx_body_type("static")
        builder.set_initial_pose(sapien.Pose(p=[0, 0, -0.5]))
        builder.build(name="ground")

        bodies = []
        keep = [ground_scene]
        for i in range(4):
            scene = sapien.Scene([system])
            scene.get_or_assign_environment_id()
            keep.append(scene)
            b = scene.create_actor_builder()
            b.add_box_collision(half_size=[0.4, 0.4, 0.4])
            b.set_initial_pose(sapien.Pose(p=[0.0, 0.0, 1.0]))  # all coincident
            bodies.append(
                b.build(name=f"b{i}").find_component_by_type(
                    sapien.physx.PhysxRigidDynamicComponent
                )
            )

        system.gpu_init()
        for _ in range(120):
            system.step()
        system.gpu_fetch_rigid_dynamic_data()
        data = system.cuda_rigid_body_data.torch().cpu().numpy()
        positions = np.array([data[b.gpu_index, :3] for b in bodies])

        self.assertTrue((positions[:, 2] > 0.1).all())  # the shared ground holds every one
        self.assertLess(np.abs(positions - positions[0]).max(), 1e-3)  # none pushed the others


if __name__ == "__main__":
    unittest.main()