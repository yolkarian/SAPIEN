"""Tests for PhysX GPU broadphase environment ID API."""

import gc
import pickle
import unittest

import sapien


PX_INVALID_U32 = 0xFFFFFFFF


class TestEnvironmentIDConfig(unittest.TestCase):
    def test_scene_config_gpu_broadphase_bits(self):
        original = sapien.physx.get_scene_config()
        self.addCleanup(sapien.physx.set_scene_config, original)

        config = sapien.physx.PhysxSceneConfig()
        # Five bits on Z alone: the widest count that needs no offset, on the axis whose
        # coordinate precision is cheapest to spend.
        self.assertEqual(config.gpu_broadphase_nb_bits_env_id_x, 0)
        self.assertEqual(config.gpu_broadphase_nb_bits_env_id_y, 0)
        self.assertEqual(config.gpu_broadphase_nb_bits_env_id_z, 5)
        self.assertEqual(config.gpu_broadphase_env_id_bits, 5)
        self.assertIsNone(config.gpu_broadphase_num_scenes)
        self.assertFalse(config.gpu_broadphase_with_shared_scene)

        # The convenience setter keeps writing Z alone; reading reports the widest axis.
        config.gpu_broadphase_env_id_bits = 8
        self.assertEqual(config.gpu_broadphase_nb_bits_env_id_x, 0)
        self.assertEqual(config.gpu_broadphase_nb_bits_env_id_y, 0)
        self.assertEqual(config.gpu_broadphase_nb_bits_env_id_z, 8)
        self.assertEqual(config.gpu_broadphase_env_id_bits, 8)

        config.gpu_broadphase_env_id_bits = -1
        self.assertEqual(config.gpu_broadphase_env_id_bits, 0)

        with self.assertRaises(RuntimeError):
            config.gpu_broadphase_env_id_bits = 17

        # Any other axis works just as well, and gives the same band count.
        other = sapien.physx.PhysxSceneConfig()
        other.gpu_broadphase_nb_bits_env_id_z = 0
        other.gpu_broadphase_nb_bits_env_id_x = 8
        self.assertEqual(other.gpu_broadphase_env_id_bits, 8)
        self.assertEqual(other.gpu_broadphase_env_band_count, 1 << 8)
        self.assertEqual(
            sapien.physx.broadphase_env_id_window(8, 0, 0),
            sapien.physx.broadphase_env_id_window(0, 0, 8),
        )

        config.gpu_broadphase_nb_bits_env_id_x = 4
        config.gpu_broadphase_nb_bits_env_id_y = 4
        config.gpu_broadphase_nb_bits_env_id_z = 8
        sapien.physx.set_scene_config(config)
        retrieved = sapien.physx.get_scene_config()
        self.assertEqual(retrieved.gpu_broadphase_nb_bits_env_id_x, 4)
        self.assertEqual(retrieved.gpu_broadphase_nb_bits_env_id_y, 4)
        self.assertEqual(retrieved.gpu_broadphase_nb_bits_env_id_z, 8)
        self.assertEqual(retrieved.gpu_broadphase_env_id_bits, 8)

        roundtrip = pickle.loads(pickle.dumps(config))
        self.assertEqual(roundtrip.gpu_broadphase_nb_bits_env_id_x, 4)
        self.assertEqual(roundtrip.gpu_broadphase_nb_bits_env_id_y, 4)
        self.assertEqual(roundtrip.gpu_broadphase_nb_bits_env_id_z, 8)

    def test_scene_wrapper_rejects_cpu_system(self):
        system = sapien.physx.PhysxCpuSystem()
        scene = sapien.Scene([system])

        with self.assertRaisesRegex(RuntimeError, "PhysxGpuSystem"):
            scene.get_environment_id()
        with self.assertRaisesRegex(RuntimeError, "PhysxGpuSystem"):
            scene.set_environment_id(0)


class TestEnvironmentIDGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gc.collect()
        try:
            sapien.physx.enable_gpu()
        except Exception as exc:
            raise unittest.SkipTest(f"GPU PhysX not available: {exc}")

    def setUp(self):
        config = sapien.physx.PhysxSceneConfig()
        config.gpu_broadphase_env_id_bits = 4
        sapien.physx.set_scene_config(config)

    def tearDown(self):
        sapien.physx.set_scene_config(sapien.physx.PhysxSceneConfig())

    @staticmethod
    def make_system(num_scenes=None, with_shared_scene=False, bits=None):
        """Declare the layout on the config, which is where the system reads it."""
        config = sapien.physx.PhysxSceneConfig()
        if bits is not None:
            config.gpu_broadphase_env_id_bits = bits
        config.gpu_broadphase_num_scenes = num_scenes
        config.gpu_broadphase_with_shared_scene = with_shared_scene
        sapien.physx.set_scene_config(config)
        return sapien.physx.PhysxGpuSystem()

    def test_gpu_system_auto_assigns_scene_environment_ids_when_managed(self):
        system = self.make_system(num_scenes=16)
        scene0 = sapien.Scene([system])
        scene1 = sapien.Scene([system])
        shared_scene = sapien.Scene([system])
        scene2 = sapien.Scene([system])

        self.assertIsNone(scene0.environment_id)
        self.assertEqual(scene0.get_or_assign_environment_id(), 0)
        self.assertEqual(system.get_or_assign_scene_environment_id(scene1), 1)
        self.assertEqual(scene0.environment_id, 0)
        self.assertEqual(scene1.get_environment_id(), 1)

        shared_scene.set_environment_id(-1)
        self.assertEqual(shared_scene.get_environment_id(), PX_INVALID_U32)

        self.assertEqual(system.get_scene_environment_id(scene2), 2)

    def test_gpu_system_leaves_unmanaged_scenes_at_environment_zero(self):
        # Without `num_scenes` SAPIEN invents nothing: an unset scene is environment 0 and is
        # not recorded, so the caller can still claim any ID without colliding with a phantom.
        system = sapien.physx.PhysxGpuSystem()
        scene0 = sapien.Scene([system])
        scene1 = sapien.Scene([system])

        self.assertFalse(system.has_managed_broadphase_env_ids)
        self.assertEqual(system.get_scene_environment_id(scene0), 0)
        self.assertEqual(system.get_scene_environment_id(scene1), 0)
        self.assertIsNone(system.get_assigned_scene_environment_id(scene0))

        scene0.set_environment_id(0)
        self.assertEqual(scene0.get_environment_id(), 0)
        self.assertIsNone(system.get_assigned_scene_environment_id(scene1))

    def test_gpu_system_manual_scene_environment_id(self):
        system = sapien.physx.PhysxGpuSystem()
        scene = sapien.Scene([system])

        scene.set_environment_id(42)
        self.assertEqual(scene.get_environment_id(), 42)

        scene.set_environment_id(PX_INVALID_U32)
        self.assertEqual(scene.get_environment_id(), PX_INVALID_U32)

        scene.set_environment_id(-1)
        self.assertEqual(scene.get_environment_id(), PX_INVALID_U32)

    def test_manual_mode_still_tracks_and_checks_explicit_ids(self):
        # SAPIEN manages nothing without `num_scenes`, but the IDs the caller does set are
        # recorded and validated -- scenes never touched stay out of the list entirely.
        system = sapien.physx.PhysxGpuSystem()
        tracked = sapien.Scene([system])
        untouched = sapien.Scene([system])
        tracked.set_environment_id(3)

        self.assertEqual(system.get_assigned_scene_environment_id(tracked), 3)
        self.assertIsNone(system.get_assigned_scene_environment_id(untouched))
        self.assertEqual(system.get_broadphase_environment_id(untouched), 0)

        with self.assertRaisesRegex(RuntimeError, "already used by another scene"):
            sapien.Scene([system]).set_environment_id(3)
        system.set_scene_environment_id(sapien.Scene([system]), 3, allow_duplicate=True)

        with self.assertRaises(RuntimeError):
            sapien.Scene([system]).set_environment_id(1 << 24)

    def test_shared_environment_id_sets_render_shared_flag(self):
        system = sapien.physx.PhysxGpuSystem()
        try:
            render_system = sapien.render.RenderSystem(system.device)
        except Exception as exc:
            raise unittest.SkipTest(f"RenderSystem not available: {exc}")
        scene = sapien.Scene([system, render_system])

        self.assertFalse(render_system.batched_render_shared)
        scene.set_environment_id(-1)
        self.assertTrue(render_system.batched_render_shared)
        self.assertEqual(scene.get_or_assign_environment_id(), PX_INVALID_U32)
        self.assertTrue(render_system.batched_render_shared)
        scene.set_environment_id(3)
        self.assertFalse(render_system.batched_render_shared)

        system.set_scene_environment_id(scene, PX_INVALID_U32)
        self.assertTrue(render_system.batched_render_shared)
        system.set_scene_environment_ids([(scene, 5)], allow_duplicate=True)
        self.assertFalse(render_system.batched_render_shared)

    def test_gpu_system_rejects_duplicate_scene_environment_ids_by_default(self):
        system = sapien.physx.PhysxGpuSystem()
        scene0 = sapien.Scene([system])
        scene1 = sapien.Scene([system])
        shared0 = sapien.Scene([system])
        shared1 = sapien.Scene([system])

        scene0.set_environment_id(12)
        with self.assertRaisesRegex(RuntimeError, "already used"):
            scene1.set_environment_id(12)

        scene1.set_environment_id(12, allow_duplicate=True)
        self.assertEqual(scene1.get_environment_id(), 12)

        shared0.set_environment_id(-1)
        shared1.set_environment_id(-1)
        self.assertEqual(shared0.get_environment_id(), PX_INVALID_U32)
        self.assertEqual(shared1.get_environment_id(), PX_INVALID_U32)

    def test_gpu_system_rejects_duplicate_bulk_environment_ids_by_default(self):
        system = sapien.physx.PhysxGpuSystem()
        scene0 = sapien.Scene([system])
        scene1 = sapien.Scene([system])

        with self.assertRaisesRegex(RuntimeError, "already used"):
            system.set_scene_environment_ids([(scene0, 3), (scene1, 3)])

        system = sapien.physx.PhysxGpuSystem()
        scene0 = sapien.Scene([system])
        scene1 = sapien.Scene([system])
        system.set_scene_environment_ids([(scene0, 3), (scene1, 3)], allow_duplicate=True)
        self.assertEqual(scene0.get_environment_id(), 3)
        self.assertEqual(scene1.get_environment_id(), 3)

    def test_gpu_system_ignores_expired_scene_environment_ids(self):
        system = sapien.physx.PhysxGpuSystem()
        scene0 = sapien.Scene([system])
        scene0.set_environment_id(5)
        del scene0
        gc.collect()

        scene1 = sapien.Scene([system])
        scene1.set_environment_id(5)
        self.assertEqual(scene1.get_environment_id(), 5)

    def test_gpu_system_rejects_invalid_scene_environment_ids(self):
        system = sapien.physx.PhysxGpuSystem()
        scene = sapien.Scene([system])

        for env_id in (-2, 1 << 24, 0xFFFFFFFE, 1 << 32):
            with self.subTest(env_id=env_id):
                with self.assertRaises(Exception):
                    scene.set_environment_id(env_id)

    def test_gpu_system_bulk_scene_environment_ids(self):
        system = sapien.physx.PhysxGpuSystem()
        scene0 = sapien.Scene([system])
        scene1 = sapien.Scene([system])
        shared_scene = sapien.Scene([system])

        system.set_scene_environment_ids([
            (scene0, 10),
            (scene1, 11),
            (shared_scene, -1),
        ])

        self.assertEqual(scene0.get_environment_id(), 10)
        self.assertEqual(scene1.get_environment_id(), 11)
        self.assertEqual(shared_scene.get_environment_id(), PX_INVALID_U32)

    def test_gpu_system_rejects_environment_id_change_after_body_add(self):
        system = sapien.physx.PhysxGpuSystem()
        scene = sapien.Scene([system])
        scene.set_environment_id(7)

        material = sapien.physx.PhysxMaterial(0.2, 0.1, 0.05)
        shape = sapien.physx.PhysxCollisionShapeBox([0.1, 0.2, 0.3], material)
        body = sapien.physx.PhysxRigidDynamicComponent()
        body.attach(shape)
        entity = sapien.Entity().add_component(body)
        scene.add_entity(entity)

        self.assertEqual(scene.get_environment_id(), 7)
        scene.set_environment_id(7)  # same value is allowed
        with self.assertRaises(RuntimeError):
            scene.set_environment_id(8)

    def test_gpu_actor_and_articulation_accept_auto_environment_id(self):
        system = self.make_system(num_scenes=16)
        scene = sapien.Scene([system])

        material = sapien.physx.PhysxMaterial(0.2, 0.1, 0.05)

        shape = sapien.physx.PhysxCollisionShapeBox([0.1, 0.2, 0.3], material)
        body = sapien.physx.PhysxRigidDynamicComponent()
        body.attach(shape)
        scene.add_entity(sapien.Entity().add_component(body))

        root = sapien.physx.PhysxArticulationLinkComponent()
        child = sapien.physx.PhysxArticulationLinkComponent(root)
        child.joint.set_type("revolute")
        child.joint.set_pose_in_parent(sapien.Pose([0.5, 0, 0]))
        child.joint.set_pose_in_child(sapien.Pose([0, 0, 0]))
        root.attach(sapien.physx.PhysxCollisionShapeBox([0.1, 0.1, 0.1], material))
        child.attach(sapien.physx.PhysxCollisionShapeBox([0.1, 0.1, 0.1], material))
        scene.add_entity(sapien.Entity().add_component(root))
        scene.add_entity(sapien.Entity().add_component(child))

        self.assertEqual(scene.get_environment_id(), 0)
        system.gpu_init()
        system.step()

    def test_broadphase_env_id_window_reserves_the_unusable_bands(self):
        # PhysX bands environment e into [e << (32 - b), (e + 1) << (32 - b)) but gives shared
        # objects the fixed encoded interval [0x01800000, 0xfe7fffff]. Bands outside it never
        # collide with a shared ground plane, so SAPIEN places every ID inside the window.
        shared_min, shared_max = 0x01800000, 0xFE7FFFFF
        for bits in (4, 8, 10, 12):
            with self.subTest(bits=bits):
                # Declaring exactly what this bit count holds is what sizes it to `bits`.
                _, want = sapien.physx.broadphase_env_id_window(bits, bits, bits)
                system = self.make_system(num_scenes=want, with_shared_scene=True)

                offset, capacity = system.broadphase_env_id_window
                width = 1 << (32 - bits)
                self.assertEqual(capacity, want)
                self.assertEqual(offset, -(-shared_min // width))
                self.assertEqual(offset + capacity - 1, (shared_max + 1) // width - 1)

                # Every band SAPIEN hands out overlaps the shared interval.
                for env_id in (0, capacity - 1):
                    band = (env_id + offset) * width
                    self.assertGreaterEqual(band + width - 1, shared_min)
                    self.assertLessEqual(band, shared_max)

    def test_broadphase_env_id_window_does_not_depend_on_a_shared_scene_existing(self):
        # The window is a function of the declaration alone, never of which scenes happen to
        # exist. PhysX freezes an actor's environment ID at addActor, so the value has to be
        # final when the first body binds -- before the scene set is knowable.
        system = self.make_system(num_scenes=500, with_shared_scene=True)

        self.assertFalse(system.has_shared_environment_scene)
        before = system.broadphase_env_id_window
        self.assertGreater(before[0], 0)

        shared = sapien.Scene([system])
        shared.set_environment_id(-1)
        self.assertTrue(system.has_shared_environment_scene)
        self.assertEqual(system.broadphase_env_id_window, before)

    def test_shared_scene_declaration_alone_still_offsets(self):
        # Declaring only the shared scene earns the offset over whatever bits are configured;
        # sizing from a count is a separate favour.
        system = self.make_system(with_shared_scene=True, bits=10)

        offset, capacity = system.broadphase_env_id_window
        self.assertGreater(offset, 0)
        self.assertLess(capacity, 1 << 10)
        self.assertFalse(system.has_managed_broadphase_env_ids)  # no IDs invented
        self.assertTrue(system.manages_collision_group_scene_ids)

        scene = sapien.Scene([system])
        scene.set_environment_id(3)
        self.assertEqual(system.get_broadphase_environment_id(scene), 3 + offset)

    def test_broadphase_env_id_window_is_full_when_undeclared(self):
        # Nothing declared means nothing reserved: IDs reach PhysX verbatim.
        config = sapien.physx.PhysxSceneConfig()
        config.gpu_broadphase_env_id_bits = 10
        sapien.physx.set_scene_config(config)
        system = sapien.physx.PhysxGpuSystem()

        self.assertEqual(system.broadphase_env_id_window, (0, 1 << 10))
        scene = sapien.Scene([system])
        scene.set_environment_id(3)
        self.assertEqual(system.get_broadphase_environment_id(scene), 3)

    def test_broadphase_env_id_window_is_unrestricted_without_bits(self):
        # Banding turned off entirely: no bands, so nothing to reserve and nothing to miss --
        # PhysX skips the whole environment-ID branch, shared objects included.
        system = self.make_system(bits=0)

        self.assertEqual(system.broadphase_env_band_count, 0)
        self.assertEqual(system.broadphase_env_id_window, (0, 1 << 24))

        shared = self.make_system(bits=0, with_shared_scene=True)
        self.assertEqual(shared.broadphase_env_id_window, (0, 1 << 24))

    def test_broadphase_environment_id_is_shifted_into_the_window(self):
        system = self.make_system(num_scenes=500, with_shared_scene=True)
        shared = sapien.Scene([system])
        shared.set_environment_id(-1)

        offset, _ = system.broadphase_env_id_window
        self.assertGreater(offset, 0)

        scene = sapien.Scene([system])
        scene.set_environment_id(3)
        # The user-visible ID is untouched; only what PhysX receives moves.
        self.assertEqual(scene.get_environment_id(), 3)
        self.assertEqual(system.get_broadphase_environment_id(scene), 3 + offset)
        self.assertEqual(system.get_broadphase_environment_id(shared), PX_INVALID_U32)

    def test_broadphase_environment_id_is_stable_across_calls(self):
        system = sapien.physx.PhysxGpuSystem()
        shared = sapien.Scene([system])
        shared.set_environment_id(-1)
        scene = sapien.Scene([system])
        scene.set_environment_id(2)

        first = system.get_broadphase_environment_id(scene)
        self.assertEqual(system.get_broadphase_environment_id(scene), first)

    def test_broadphase_environment_ids_wrap_past_the_window(self):
        # Only the band -- the low `bits` of the ID -- has to sit inside the window. PhysX
        # discards the higher bits when it places the box but compares the full value when it
        # filters, so environments past the window ride above it instead of being rejected.
        system = sapien.physx.PhysxGpuSystem()
        shared = sapien.Scene([system])
        shared.set_environment_id(-1)
        offset, capacity = system.broadphase_env_id_window
        bands = system.broadphase_env_band_count

        scenes = []
        for env_id in (0, capacity - 1, capacity, capacity + 1, 3 * capacity + 7):
            scene = sapien.Scene([system])
            scene.set_environment_id(env_id)
            scenes.append((env_id, scene))

        seen = set()
        for env_id, scene in scenes:
            physx_id = system.get_broadphase_environment_id(scene)
            wrap, band = divmod(env_id, capacity)
            self.assertEqual(physx_id, offset + band + wrap * bands)
            # The band always lands inside the window, so the shared object stays reachable.
            self.assertGreaterEqual(physx_id % bands, offset)
            self.assertLess(physx_id % bands, offset + capacity)
            seen.add(physx_id)
        self.assertEqual(len(seen), len(scenes))  # still one distinct ID per environment

    def test_broadphase_ids_do_not_depend_on_build_order(self):
        # PhysX freezes an actor's environment ID at addActor, so SAPIEN cannot revisit a
        # decision once a body is bound. The window therefore never depends on which scenes
        # exist yet, and a shared scene created last is as valid as one created first.
        config = sapien.physx.PhysxSceneConfig()
        config.gpu_broadphase_env_id_bits = 10
        sapien.physx.set_scene_config(config)
        system = sapien.physx.PhysxGpuSystem()

        early = sapien.Scene([system])
        early.set_environment_id(0)
        bound = system.get_broadphase_environment_id(early)

        shared = sapien.Scene([system])  # shared scene appears only now
        shared.set_environment_id(-1)
        late = sapien.Scene([system])
        late.set_environment_id(1)

        offset, _ = system.broadphase_env_id_window
        self.assertEqual(bound, 0 + offset)
        self.assertEqual(system.get_broadphase_environment_id(early), bound)
        self.assertEqual(system.get_broadphase_environment_id(late), 1 + offset)

    def test_changing_an_environment_id_drops_the_cached_broadphase_id(self):
        # get_broadphase_environment_id caches so every body of a scene shares a band. Setting
        # a new environment ID beforehand is still legal, and must not leave the stale value
        # behind -- two scenes would otherwise be handed the same ID and collide.
        # Managed by count: SAPIEN sizes the bits and reserves the shared bands.
        system = self.make_system(num_scenes=1024, with_shared_scene=True)
        offset, capacity = system.broadphase_env_id_window
        self.assertGreater(capacity, 900)  # both IDs below get a private band

        scene = sapien.Scene([system])
        scene.set_environment_id(5)
        self.assertEqual(system.get_broadphase_environment_id(scene), 5 + offset)

        scene.set_environment_id(900)
        self.assertEqual(system.get_broadphase_environment_id(scene), 900 + offset)

    def test_one_bit_leaves_no_usable_band(self):
        # One bit puts the boundary between the two bands inside the shared object's fixed
        # encoded interval, so neither band lies entirely within it. Declaring a count never
        # reaches this -- set_gpu_broadphase_env_count always picks a width that holds it -- so
        # this is a property of the arithmetic rather than a state the API can be driven into.
        self.assertEqual(sapien.physx.broadphase_env_id_window(1, 1, 1), (1, 0))
        self.assertGreater(sapien.physx.broadphase_env_id_window(2, 2, 2)[1], 0)
        self.assertEqual(sapien.physx.broadphase_env_id_bits_for_env_count(1), 2)

    def test_env_count_entry_point_sizes_the_bits(self):
        config = sapien.physx.PhysxSceneConfig()

        # 4096 environments need 13 bits, not 12: the guard reserves the outermost bands.
        config.set_gpu_broadphase_env_count(4096)
        self.assertEqual(config.gpu_broadphase_env_id_bits, 13)
        self.assertGreaterEqual(config.gpu_broadphase_max_env_count, 4096)
        self.assertEqual(config.gpu_broadphase_env_band_count, 1 << 13)

        # Without shared objects every band is usable, so 12 bits is enough.
        config.set_gpu_broadphase_env_count(4096, with_shared_objects=False)
        self.assertEqual(config.gpu_broadphase_env_id_bits, 12)

        config.set_gpu_broadphase_env_count(0)
        self.assertEqual(config.gpu_broadphase_env_id_bits, 0)

        # Past the widest band count environments still fit by wrapping, so the widest bit
        # count is handed back rather than an error.
        config.set_gpu_broadphase_env_count(1 << 20)
        self.assertEqual(config.gpu_broadphase_env_id_bits, 16)

        with self.assertRaises(RuntimeError):
            config.set_gpu_broadphase_env_count(sapien.physx.max_broadphase_env_count() + 1)

    def test_band_count_uses_the_widest_axis_not_the_product(self):
        # PhysX shifts the same env ID on every axis, so the axes carry nested subsets of the
        # same low bits rather than independent components.
        config = sapien.physx.PhysxSceneConfig()
        config.gpu_broadphase_nb_bits_env_id_x = 6
        config.gpu_broadphase_nb_bits_env_id_y = 4
        config.gpu_broadphase_nb_bits_env_id_z = 2
        self.assertEqual(config.gpu_broadphase_env_band_count, 1 << 6)

    def test_module_level_sizing_helpers(self):
        self.assertEqual(sapien.physx.broadphase_env_id_bits_for_env_count(4096), 13)
        self.assertEqual(
            sapien.physx.broadphase_env_id_bits_for_env_count(4096, with_shared_objects=False), 12
        )
        # Beyond a private band per environment the widest count still works, via wrapping.
        self.assertEqual(sapien.physx.broadphase_env_id_bits_for_env_count(1 << 20), 16)

        # Wrapping lifts the ceiling far above the band count: only the low bits must land in
        # the window, the rest of the count rides above it.
        self.assertGreater(sapien.physx.max_broadphase_env_count(), 1 << 23)
        self.assertEqual(
            sapien.physx.broadphase_env_id_bits_for_env_count(
                sapien.physx.max_broadphase_env_count() + 1
            ),
            0,
        )

    def test_construction_modes_pick_what_sapien_manages(self):
        # `num_scenes` sizes the bit count and derives the IDs; `with_shared_scene` adds the
        # offset and the collision-group filter on top, and cannot appear on its own.
        for num_scenes, shared, want_managed, want_offset, want_filter in (
            (None, False, False, False, False),
            (64, False, True, False, False),
            (64, True, True, True, True),
        ):
            with self.subTest(num_scenes=num_scenes, with_shared_scene=shared):
                config = sapien.physx.PhysxSceneConfig()
                config.gpu_broadphase_env_id_bits = 10
                sapien.physx.set_scene_config(config)
                kwargs = {} if num_scenes is None else {"num_scenes": num_scenes}
                system = self.make_system(with_shared_scene=shared, **kwargs)

                self.assertEqual(system.has_managed_broadphase_env_ids, want_managed)
                self.assertEqual(system.manages_collision_group_scene_ids, want_filter)
                offset, _ = system.broadphase_env_id_window
                self.assertEqual(offset > 0, want_offset)

                scene = sapien.Scene([system])
                scene.set_environment_id(7)
                self.assertEqual(scene.get_environment_id(), 7)  # never rewritten
                self.assertEqual(system.get_broadphase_environment_id(scene), 7 + offset)

    def test_managed_filter_stamps_the_raw_environment_id(self):
        # The collision-group scene field is compared for equality, never encoded into bounds,
        # so it carries the environment ID with no band offset. Shared scenes get 0xffff.
        system = self.make_system(num_scenes=64, with_shared_scene=True)
        offset, _ = system.broadphase_env_id_window
        self.assertGreater(offset, 0)

        scene = sapien.Scene([system])
        scene.set_environment_id(7)
        material = sapien.physx.PhysxMaterial(0.2, 0.1, 0.05)
        body = sapien.physx.PhysxRigidDynamicComponent()
        body.attach(sapien.physx.PhysxCollisionShapeBox([0.1, 0.1, 0.1], material))
        scene.add_entity(sapien.Entity().add_component(body))

        stamped = body.collision_shapes[0].get_collision_groups()[3] >> 16
        self.assertEqual(stamped, 7)  # the raw ID, not 7 + offset

        shared = sapien.Scene([system])
        shared.set_environment_id(-1)
        ground = sapien.physx.PhysxRigidStaticComponent()
        ground.attach(sapien.physx.PhysxCollisionShapeBox([1.0, 1.0, 0.1], material))
        shared.add_entity(sapien.Entity().add_component(ground))
        self.assertEqual(ground.collision_shapes[0].get_collision_groups()[3] >> 16, 0xFFFF)

    def test_unmanaged_filter_is_left_alone(self):
        system = self.make_system(num_scenes=64)
        self.assertFalse(system.manages_collision_group_scene_ids)

        scene = sapien.Scene([system])
        scene.set_environment_id(7)
        material = sapien.physx.PhysxMaterial(0.2, 0.1, 0.05)
        body = sapien.physx.PhysxRigidDynamicComponent()
        body.attach(sapien.physx.PhysxCollisionShapeBox([0.1, 0.1, 0.1], material))
        scene.add_entity(sapien.Entity().add_component(body))

        self.assertEqual(body.collision_shapes[0].get_collision_groups()[3], 0)

    def test_managed_filter_rejects_ids_past_the_sixteen_bit_field(self):
        system = self.make_system(num_scenes=64, with_shared_scene=True)
        scene = sapien.Scene([system])
        with self.assertRaisesRegex(RuntimeError, "collision-group scene field"):
            scene.set_environment_id(0xFFFF)

    def test_module_level_window_helper_matches_the_system(self):
        # The helper is what a manual caller uses to place IDs without a system; it must agree
        # with what a declaring system arrives at for the same bit count.
        offset, capacity = sapien.physx.broadphase_env_id_window(12, 12, 12)
        system = self.make_system(num_scenes=capacity, with_shared_scene=True)
        self.assertEqual(system.broadphase_env_id_window, (offset, capacity))


if __name__ == "__main__":
    unittest.main()
