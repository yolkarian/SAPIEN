"""Material combination and body speed limits, run on CPU or in an isolated GPU process.

From the repository root, run GPU cases with:
SAPIEN_TEST_GPU=1 SAPIEN_REQUIRE_GPU=1 python -m unittest discover \
    -s unittest -p test_material_velocity_api.py -v
Set SAPIEN_REQUIRE_GPU=1 to fail, rather than skip, if CUDA initialization fails.
"""

from __future__ import annotations

import gc
import itertools
import os
import unittest

import numpy as np
import sapien

GPU = os.environ.get("SAPIEN_TEST_GPU") == "1"
if GPU:
    # Resolve helpers in the test package, not relative to the process's sys.path.
    # Import errors are test infrastructure failures, never GPU-availability skips.
    from .test_gpu_articulation_buffers import (
        _cuda_synchronize,
        _load_cudart,
        _read_values,
        _write_float_values,
    )

MODES = ("average", "min", "multiply", "max")


class TestMaterialVelocityAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if GPU:
            try:
                _load_cudart()
                sapien.physx.enable_gpu()
            except Exception as exc:
                if os.environ.get("SAPIEN_REQUIRE_GPU") == "1":
                    raise
                raise unittest.SkipTest(f"GPU PhysX unavailable: {exc}") from exc

    def setUp(self) -> None:
        self.old_config = sapien.physx.get_scene_config()
        config = sapien.physx.PhysxSceneConfig()
        config.gravity = [0, 0, -10]
        config.bounce_threshold = 0.05
        config.enable_tgs = True
        sapien.physx.set_scene_config(config)
        self.system = (
            sapien.physx.PhysxGpuSystem() if GPU else sapien.physx.PhysxCpuSystem()
        )
        self.system.set_timestep(0.005)
        self.scene = sapien.Scene([self.system])
        self.data = None
        self.forces = None

    def tearDown(self) -> None:
        self.data = None
        self.forces = None
        gc.collect()
        self.scene.close()
        self.system.close()
        sapien.physx.set_scene_config(self.old_config)

    def _body(
        self,
        x: float,
        z: float = 1.0,
        material: sapien.physx.PhysxMaterial | None = None,
        box: bool = False,
    ) -> sapien.physx.PhysxRigidDynamicComponent:
        builder = self.scene.create_actor_builder()
        builder.set_initial_pose(sapien.Pose([x, 0, z]))
        if box:
            builder.add_box_collision(half_size=[0.1] * 3, material=material)
        else:
            builder.add_sphere_collision(radius=0.1, material=material)
        body = builder.build().find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent
        )
        body.linear_damping = 0
        body.angular_damping = 0
        body.disable_gravity = True
        return body

    def _articulation(
        self, x: float, children: int = 0
    ) -> sapien.physx.PhysxArticulation:
        builder = self.scene.create_articulation_builder()
        builder.set_initial_pose(sapien.Pose([x, 0, 1]))
        parent = None
        for index in range(children + 1):
            link = builder.create_link_builder(parent)
            link.set_name(f"link{index}")
            link.add_sphere_collision(radius=0.1)
            link.collision_groups = [0, 0, 0, 0]
            if parent is not None:
                link.set_joint_properties(
                    "fixed",
                    limits=[],
                    pose_in_parent=sapien.Pose([0.3, 0, 0]),
                    pose_in_child=sapien.Pose(),
                )
            parent = link
        art = builder.build(fix_root_link=False)
        for link in art.links:
            link.disable_gravity = True
            link.linear_damping = 0
            link.angular_damping = 0
        return art

    def _init(self) -> None:
        if GPU:
            self.system.gpu_init()
            self.system.gpu_fetch_rigid_dynamic_data()
            self.system.gpu_fetch_articulation_link_pose()
            self.data = self.system.cuda_rigid_body_data

    def _velocity(
        self,
        body: sapien.physx.PhysxRigidBodyComponent,
        linear: list[float],
        angular: list[float],
    ) -> None:
        if GPU:
            _write_float_values(self.data, (body.gpu_pose_index, 7), linear + angular)
            _cuda_synchronize()
            if isinstance(body, sapien.physx.PhysxArticulationLinkComponent):
                self.system.gpu_apply_articulation_root_velocity()
            else:
                self.system.gpu_apply_rigid_dynamic_data()
        elif isinstance(body, sapien.physx.PhysxArticulationLinkComponent):
            body.articulation.root_linear_velocity = linear
            body.articulation.root_angular_velocity = angular
        else:
            body.linear_velocity = linear
            body.angular_velocity = angular

    def _step(self) -> None:
        self.system.step()
        if GPU:
            self.system.gpu_fetch_rigid_dynamic_data()
            self.system.gpu_fetch_articulation_link_velocity()
            _cuda_synchronize()

    def _read_velocity(self, body: sapien.physx.PhysxRigidBodyComponent) -> np.ndarray:
        if GPU:
            return _read_values(self.data, (body.gpu_pose_index, 7), 6, np.float32)
        return np.concatenate([body.linear_velocity, body.angular_velocity])

    def test_material_modes_and_invalid_inputs(self) -> None:
        mat = sapien.physx.PhysxMaterial(0.5, 0.4, 0.3)
        self.assertEqual(mat.friction_combine_mode, "average")
        self.assertEqual(mat.restitution_combine_mode, "average")
        for friction, restitution in itertools.product(MODES, repeat=2):
            mat.friction_combine_mode = friction
            mat.set_restitution_combine_mode(restitution)
            self.assertEqual(mat.get_friction_combine_mode(), friction)
            self.assertEqual(mat.restitution_combine_mode, restitution)
            mat.set_friction_combine_mode(friction)
            mat.restitution_combine_mode = restitution
            self.assertEqual(mat.friction_combine_mode, friction)
            self.assertEqual(mat.get_restitution_combine_mode(), restitution)
        for field in ("friction_combine_mode", "restitution_combine_mode"):
            for invalid in ("invalid", "MAX", "", 1, None, b"max"):
                for method in (False, True):
                    with self.subTest(field=field, invalid=invalid, method=method):
                        with self.assertRaises(TypeError):
                            if method:
                                getattr(mat, f"set_{field}")(invalid)
                            else:
                                setattr(mat, field, invalid)
                        self.assertEqual(mat.friction_combine_mode, "max")
                        self.assertEqual(mat.restitution_combine_mode, "max")

    def test_material_sharing_and_default_lifetime(self) -> None:
        previous = sapien.physx.get_default_material()
        scalars = (
            previous.static_friction,
            previous.dynamic_friction,
            previous.restitution,
        )
        try:
            sapien.physx.set_default_material(*scalars)
            mat = sapien.physx.get_default_material()
            first = sapien.physx.PhysxCollisionShapeSphere(0.1, mat)
            second = sapien.physx.PhysxCollisionShapeSphere(0.2, mat)
            independent = sapien.physx.PhysxMaterial(*scalars)
            mat.friction_combine_mode = "multiply"
            mat.restitution_combine_mode = "min"
            for shape in (first, second):
                self.assertEqual(
                    shape.physical_material.friction_combine_mode, "multiply"
                )
                self.assertEqual(
                    shape.physical_material.restitution_combine_mode, "min"
                )
            self.assertEqual(independent.friction_combine_mode, "average")
            self.assertEqual(
                sapien.physx.get_default_material().friction_combine_mode, "multiply"
            )
            del mat, first, second, shape
            gc.collect()
            self.assertEqual(
                sapien.physx.get_default_material().friction_combine_mode, "average"
            )
        finally:
            sapien.physx.set_default_material(*scalars)

    def test_velocity_defaults_bounds_and_independence(self) -> None:
        dynamic = self._body(0)
        art = self._articulation(3, children=1)
        for body, linear_default, angular_default in (
            (dynamic, float(np.float32(1e16)), 100),
            (art.root, 10, 50),
            (art.links[1], 10, 50),
        ):
            self.assertTrue(
                np.isclose(body.max_linear_velocity, linear_default, rtol=1e-6)
            )
            self.assertEqual(body.max_angular_velocity, angular_default)
            for field, other in (
                ("max_linear_velocity", "max_angular_velocity"),
                ("max_angular_velocity", "max_linear_velocity"),
            ):
                initial = getattr(body, field)
                setattr(body, field, initial)
                self.assertTrue(np.isclose(getattr(body, field), initial, rtol=1e-6))
                unchanged = getattr(body, other)
                for method in (False, True):
                    for value in (0.0, 321.0, 1000.0, float(np.float32(1e16))):
                        if method:
                            getattr(body, f"set_{field}")(value)
                        else:
                            setattr(body, field, value)
                        self.assertTrue(
                            np.isclose(
                                getattr(body, f"get_{field}")(), value, rtol=1e-6
                            )
                        )
                        self.assertEqual(getattr(body, other), unchanged)
                    upper_next = float(
                        np.nextafter(np.float32(1e16), np.float32(np.inf))
                    )
                    for value in (-1.0, -np.inf, np.inf, np.nan, upper_next):
                        before = getattr(body, field)
                        with self.assertRaises(RuntimeError):
                            if method:
                                getattr(body, f"set_{field}")(value)
                            else:
                                setattr(body, field, value)
                        self.assertEqual(getattr(body, field), before)
                        self.assertEqual(getattr(body, other), unchanged)

    def test_clone_and_reparent_subtrees(self) -> None:
        art = self._articulation(0, children=2)
        links = list(art.links)
        for index, link in enumerate(links):
            link.max_linear_velocity = 321 + index
            link.max_angular_velocity = 654 + index
        material = sapien.physx.PhysxMaterial(0.5, 0.4, 0.3)
        links[0].collision_shapes[0].physical_material = material
        clones = art.clone_links()
        # clone_links() returns detached components; attach every clone before
        # reparenting into an articulation that already belongs to this scene.
        for clone in clones:
            entity = sapien.Entity()
            entity.add_component(clone)
            self.scene.add_entity(entity)
        material.friction_combine_mode = "multiply"
        material.restitution_combine_mode = "min"
        self.assertEqual(
            clones[0].collision_shapes[0].physical_material.friction_combine_mode,
            "multiply",
        )
        self.assertEqual(
            clones[0].collision_shapes[0].physical_material.restitution_combine_mode,
            "min",
        )
        other = self._articulation(3)
        links[1].set_parent(other.root)
        for collection in (links, clones):
            for index, link in enumerate(collection):
                self.assertEqual(link.max_linear_velocity, 321 + index)
                self.assertEqual(link.max_angular_velocity, 654 + index)
        self.assertEqual(other.root.max_linear_velocity, 10)
        self.assertEqual(other.root.max_angular_velocity, 50)
        # Rebuild a root subtree, then detach it again; both descendants must survive.
        clones[0].set_parent(other.root)
        clones[0].set_parent(None)
        for index, link in enumerate(clones):
            self.assertEqual(link.max_linear_velocity, 321 + index)
            self.assertEqual(link.max_angular_velocity, 654 + index)
        default_art = self._articulation(6, children=1)
        defaults = default_art.clone_links()
        for clone in defaults:
            entity = sapien.Entity()
            entity.add_component(clone)
            self.scene.add_entity(entity)
        defaults[0].set_parent(other.root)
        for link in defaults:
            self.assertEqual(link.max_linear_velocity, 10)
            self.assertEqual(link.max_angular_velocity, 50)

    def test_velocity_limits_affect_dynamics(self) -> None:
        bodies = [self._body(0), self._body(3)]
        arts = [self._articulation(6), self._articulation(9)]
        bodies += [art.root for art in arts]
        for index, body in enumerate(bodies):
            body.max_linear_velocity = 2 if index % 2 == 0 else 1000
            body.max_angular_velocity = 3 if index % 2 == 0 else 1000
        self._init()
        for body in bodies:
            self._velocity(body, [3, 4, 0], [0, 6, 8])
        self._step()
        for index, body in enumerate(bodies):
            velocity = self._read_velocity(body)
            linear, angular = (2, 3) if index % 2 == 0 else (5, 10)
            np.testing.assert_allclose(
                velocity[:3], np.array([0.6, 0.8, 0]) * linear, atol=1e-3
            )
            np.testing.assert_allclose(
                velocity[3:], np.array([0, 0.6, 0.8]) * angular, atol=1e-3
            )
        if not GPU:
            # CPU setters also take effect between completed steps, not only at startup.
            for body in bodies:
                body.max_linear_velocity = 1
                body.max_angular_velocity = 1
            self._step()
            for body in bodies:
                velocity = self._read_velocity(body)
                self.assertAlmostEqual(np.linalg.norm(velocity[:3]), 1, places=3)
                self.assertAlmostEqual(np.linalg.norm(velocity[3:]), 1, places=3)

    def test_child_limit_affects_articulation(self) -> None:
        arts = [self._articulation(0, 1), self._articulation(3, 1)]
        for index, art in enumerate(arts):
            art.root.max_linear_velocity = 1000
            art.links[1].max_linear_velocity = 1 if index == 0 else 1000
        self._init()
        for art in arts:
            self._velocity(art.root, [5, 0, 0], [0, 0, 0])
        for _ in range(10):
            self._step()
        slow, fast = [self._read_velocity(art.links[1])[0] for art in arts]
        self.assertLess(slow, 2.5)
        self.assertAlmostEqual(fast, 5, places=3)

    def test_child_angular_limit_affects_articulation(self) -> None:
        arts = [self._articulation(0, 1), self._articulation(3, 1)]
        for index, art in enumerate(arts):
            for link in art.links:
                link.max_linear_velocity = 1000
                link.max_angular_velocity = 1000
            art.links[1].max_angular_velocity = 1 if index == 0 else 1000
        self._init()
        for art in arts:
            # Spin along the root-child axis to avoid centrifugal translation.
            self._velocity(art.root, [0, 0, 0], [5, 0, 0])
        for _ in range(10):
            self._step()
        slow, fast = [self._read_velocity(art.links[1])[3] for art in arts]
        self.assertLess(slow, 2.5)
        self.assertAlmostEqual(fast, 5, places=3)

    def test_static_friction_priority(self) -> None:
        cases = []
        for index, (first, second) in enumerate(itertools.product(MODES, repeat=2)):
            floor = sapien.physx.PhysxMaterial(0.1, 0.05, 0)
            box = sapien.physx.PhysxMaterial(1.5, 0.1, 0)
            floor.friction_combine_mode = first
            box.friction_combine_mode = second
            builder = self.scene.create_actor_builder()
            builder.set_initial_pose(sapien.Pose([index * 4, 0, -0.1]))
            builder.add_box_collision(half_size=[1.5, 1, 0.1], material=floor)
            builder.build_static()
            body = self._body(index * 4, z=0.1, material=box, box=True)
            body.disable_gravity = False
            body.set_locked_motion_axes([False, True, False, True, True, True])
            effective = max(MODES.index(first), MODES.index(second))
            cases.append((body, (0.8, 0.1, 0.15, 1.5)[effective]))
        self._init()
        # Establish resting contacts before applying tangential force; otherwise the
        # initial free-flight horizontal velocity would test dynamic, not static, friction.
        for _ in range(20):
            self._step()
        if GPU:
            self.forces = self.system.cuda_rigid_body_force
            for body, _ in cases:
                _write_float_values(
                    self.forces, (body.gpu_index, 0), [4.5 * body.mass, 0, 0]
                )
            _cuda_synchronize()
        for _ in range(60):
            if GPU:
                self.system.gpu_apply_rigid_dynamic_force()
            else:
                for body, _ in cases:
                    body.add_force_torque([4.5 * body.mass, 0, 0], [0, 0, 0])
            self._step()
        # Tangential/normal load ratio is 0.45; only average/max static friction can hold.
        # Dynamic coefficients are all below 0.45, so accidentally using them cannot pass.
        for body, static in cases:
            speed = abs(self._read_velocity(body)[0])
            if static > 0.45:
                self.assertLess(speed, 0.02)
            else:
                self.assertGreater(speed, 0.2)

    def test_restitution_formulas_and_priority(self) -> None:
        cases = []
        materials = []
        for index, (first, second) in enumerate(itertools.product(MODES, repeat=2)):
            floor = sapien.physx.PhysxMaterial(0, 0, 0.2)
            sphere = sapien.physx.PhysxMaterial(0, 0, 0.8)
            floor.restitution_combine_mode = first
            sphere.restitution_combine_mode = second
            materials.extend([floor, sphere])
            builder = self.scene.create_actor_builder()
            builder.set_initial_pose(sapien.Pose([index * 3, 0, -0.1]))
            builder.add_box_collision(half_size=[1, 1, 0.1], material=floor)
            ground = builder.build_static().find_component_by_type(
                sapien.physx.PhysxRigidStaticComponent
            )
            body = self._body(index * 3, z=0.15, material=sphere)
            # Keep contact shells thinner than one step of incoming travel (0.01).
            # Large speculative contact distances can dissipate speed before impact,
            # so restitution alone no longer predicts the outgoing velocity on GPU.
            ground.collision_shapes[0].contact_offset = 0.001
            body.collision_shapes[0].contact_offset = 0.001
            effective = max(MODES.index(first), MODES.index(second))
            expected = (0.5, 0.2, 0.16, 0.8)[effective] * 2
            cases.append((body, expected))
        self._init()
        for body, _ in cases:
            self._velocity(body, [0, 0, -2], [0, 0, 0])
        rebound = np.full(len(cases), -np.inf)
        for _ in range(20):
            self._step()
            rebound = np.maximum(
                rebound, [self._read_velocity(body)[2] for body, _ in cases]
            )
        # Contact offset and solver tolerances may perturb bounce; all four predictions remain distinct.
        np.testing.assert_allclose(
            rebound, [expected for _, expected in cases], atol=0.04, rtol=0
        )
        if not GPU:
            # Changing a shared material between CPU steps must affect later contacts.
            for material in materials:
                material.restitution_combine_mode = "max"
            for index, (body, _) in enumerate(cases):
                body.entity.pose = sapien.Pose([index * 3, 0, 0.15])
                body.wake_up()
                self._velocity(body, [0, 0, -2], [0, 0, 0])
            rebound.fill(-np.inf)
            for _ in range(20):
                self._step()
                rebound = np.maximum(
                    rebound, [self._read_velocity(body)[2] for body, _ in cases]
                )
            np.testing.assert_allclose(rebound, 1.6, atol=0.04, rtol=0)

    def test_dynamic_friction_formulas_and_priority(self) -> None:
        cases = []
        for index, (first, second) in enumerate(itertools.product(MODES, repeat=2)):
            floor = sapien.physx.PhysxMaterial(0.4, 0.4, 0)
            box = sapien.physx.PhysxMaterial(0.8, 0.8, 0)
            floor.friction_combine_mode = first
            box.friction_combine_mode = second
            builder = self.scene.create_actor_builder()
            builder.set_initial_pose(sapien.Pose([index * 4, 0, -0.1]))
            builder.add_box_collision(half_size=[1.5, 1, 0.1], material=floor)
            builder.build_static()
            body = self._body(index * 4, z=0.1, material=box, box=True)
            body.disable_gravity = False
            body.set_locked_motion_axes([False, True, False, True, True, True])
            effective = max(MODES.index(first), MODES.index(second))
            cases.append((body, (0.6, 0.4, 0.32, 0.8)[effective]))
        self._init()
        for _ in range(10):
            self._step()
        for body, _ in cases:
            self._velocity(body, [2, 0, 0], [0, 0, 0])
        for _ in range(20):
            self._step()
        speeds = [self._read_velocity(body)[0] for body, _ in cases]
        # Coulomb sliding: delta-v = mu * g * dt, g=10, dt=20*0.005.
        np.testing.assert_allclose(
            speeds, [2 - friction for _, friction in cases], atol=0.05, rtol=0
        )

