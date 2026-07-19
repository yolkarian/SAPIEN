"""Tests for PhysX Direct GPU articulation link force/torque buffers."""

import ctypes
import ctypes.util
import gc
import unittest

import numpy as np
import sapien

_CUDART = None
_CUDA_MEMCPY_HOST_TO_DEVICE = 1
_CUDA_MEMCPY_DEVICE_TO_HOST = 2
_CUDA_STREAM_NON_BLOCKING = 1


def _load_cudart():
    global _CUDART
    if _CUDART is not None:
        return _CUDART

    library = ctypes.util.find_library("cudart") or "libcudart.so.12"
    _CUDART = ctypes.CDLL(library)
    _CUDART.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    _CUDART.cudaFree.argtypes = [ctypes.c_void_p]
    _CUDART.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    _CUDART.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    _CUDART.cudaMemsetAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    _CUDART.cudaStreamCreateWithFlags.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
    ]
    _CUDART.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    _CUDART.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    _CUDART.cudaDeviceSynchronize.argtypes = []
    _CUDART.cudaGetErrorString.argtypes = [ctypes.c_int]
    _CUDART.cudaGetErrorString.restype = ctypes.c_char_p
    return _CUDART


def _check_cuda(code: int) -> None:
    if code != 0:
        cudart = _load_cudart()
        message = cudart.cudaGetErrorString(code).decode("utf-8", errors="replace")
        raise RuntimeError(f"CUDA error {code}: {message}")


def _cuda_synchronize() -> None:
    _check_cuda(_load_cudart().cudaDeviceSynchronize())


class _DeviceArray:
    def __init__(self, host: np.ndarray):
        self.host = np.ascontiguousarray(host)
        self.ptr = ctypes.c_void_p()
        cudart = _load_cudart()
        _check_cuda(cudart.cudaMalloc(ctypes.byref(self.ptr), self.host.nbytes))
        _check_cuda(
            cudart.cudaMemcpy(
                self.ptr,
                ctypes.c_void_p(self.host.ctypes.data),
                self.host.nbytes,
                _CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        )

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": self.host.shape,
            "typestr": self.host.dtype.str,
            "data": (self.ptr.value, False),
            "version": 2,
        }

    def close(self) -> None:
        if self.ptr.value:
            _check_cuda(_load_cudart().cudaFree(self.ptr))
            self.ptr = ctypes.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _memset_cuda(array, value: int = 0) -> None:
    nbytes = int(np.prod(array.shape)) * np.dtype(array.typestr).itemsize
    _check_cuda(_load_cudart().cudaMemset(ctypes.c_void_p(array.ptr), value, nbytes))


def _write_float_values(array, indices: tuple[int, ...], values) -> None:
    host = np.asarray(values, dtype=np.float32)
    offset = sum(index * stride for index, stride in zip(indices, array.strides))
    _check_cuda(
        _load_cudart().cudaMemcpy(
            ctypes.c_void_p(array.ptr + offset),
            ctypes.c_void_p(host.ctypes.data),
            host.nbytes,
            _CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    )


def _read_float_values(array, indices: tuple[int, ...], count: int) -> np.ndarray:
    host = np.empty(count, dtype=np.float32)
    offset = sum(index * stride for index, stride in zip(indices, array.strides))
    _check_cuda(
        _load_cudart().cudaMemcpy(
            ctypes.c_void_p(host.ctypes.data),
            ctypes.c_void_p(array.ptr + offset),
            host.nbytes,
            _CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    )
    return host


class TestGpuArticulationLinkForce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gc.collect()
        try:
            _load_cudart()
            sapien.physx.enable_gpu()
        except Exception as exc:
            raise unittest.SkipTest(f"GPU PhysX not available: {exc}")

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

    def _build_rigid_dynamic(self, scene, x: float = 0.0):
        material = sapien.physx.PhysxMaterial(0.2, 0.1, 0.05)
        body = sapien.physx.PhysxRigidDynamicComponent()
        body.disable_gravity = True
        body.attach(
            sapien.physx.PhysxCollisionShapeBox([0.1, 0.1, 0.1], material)
        )
        entity = sapien.Entity().add_component(body)
        entity.pose = sapien.Pose([x, 0.0, 0.0])
        scene.add_entity(entity)
        return entity, body

    def _build_two_link_articulation(
        self, scene, x: float = 0.0, joint_type: str = "fixed"
    ):
        material = sapien.physx.PhysxMaterial(0.2, 0.1, 0.05)

        root = sapien.physx.PhysxArticulationLinkComponent()
        child = sapien.physx.PhysxArticulationLinkComponent(root)
        root.name = "root"
        child.name = "child"
        root.disable_gravity = True
        child.disable_gravity = True

        root.attach(sapien.physx.PhysxCollisionShapeBox([0.1, 0.1, 0.1], material))
        child.attach(sapien.physx.PhysxCollisionShapeBox([0.1, 0.1, 0.1], material))
        child.joint.set_type(joint_type)
        if joint_type == "revolute":
            child.joint.set_limits([[-1.0, 1.0]])
        child.joint.set_pose_in_parent(sapien.Pose([0.3, 0.0, 0.0]))
        child.joint.set_pose_in_child(sapien.Pose([0.0, 0.0, 0.0]))

        root_entity = sapien.Entity().add_component(root)
        root_entity.pose = sapien.Pose([x, 0.0, 0.0])
        scene.add_entity(root_entity)
        scene.add_entity(sapien.Entity().add_component(child))
        return root.articulation, root, child

    def _cuda_index_buffer(self, *indices: int):
        owner = _DeviceArray(np.asarray(indices, dtype=np.int32))
        return owner, sapien.CudaArray(owner)

    def test_shape_and_rigid_body_aliasing(self):
        system, scene = self._create_scene()
        art, root, child = self._build_two_link_articulation(scene)
        system.gpu_init()

        force = system.cuda_articulation_link_force
        torque = system.cuda_articulation_link_torque
        rigid_force = system.cuda_rigid_body_force
        rigid_torque = system.cuda_rigid_body_torque

        self.assertEqual(force.shape, [1, 2, 4])
        self.assertEqual(torque.shape, [1, 2, 4])
        self.assertEqual(force.ptr, rigid_force.ptr + rigid_force.strides[0] * root.gpu_pose_index)
        self.assertEqual(torque.ptr, rigid_torque.ptr + rigid_torque.strides[0] * root.gpu_pose_index)

        _memset_cuda(force)
        _memset_cuda(torque)
        _write_float_values(force, (art.gpu_index, child.index, 0), [1.25, -2.5, 3.75])
        _write_float_values(torque, (art.gpu_index, child.index, 0), [-4.0, 5.0, -6.0])
        _cuda_synchronize()

        self.assertTrue(
            np.allclose(
                _read_float_values(rigid_force, (child.gpu_pose_index, 0), 3),
                [1.25, -2.5, 3.75],
            )
        )
        self.assertTrue(
            np.allclose(
                _read_float_values(rigid_torque, (child.gpu_pose_index, 0), 3),
                [-4.0, 5.0, -6.0],
            )
        )

    def test_indexed_and_all_force_apply(self):
        system, scene = self._create_scene()
        art0, _, child0 = self._build_two_link_articulation(scene, x=-1.0)
        art1, _, child1 = self._build_two_link_articulation(scene, x=1.0)
        system.gpu_init()

        force = system.cuda_articulation_link_force
        _memset_cuda(force)
        _write_float_values(force, (art0.gpu_index, child0.index, 0), [1000.0, 0.0, 0.0])
        _write_float_values(force, (art1.gpu_index, child1.index, 0), [1000.0, 0.0, 0.0])
        _cuda_synchronize()

        owner, index_buffer = self._cuda_index_buffer(art0.gpu_index)
        system.gpu_apply_articulation_link_force(index_buffer)
        system.step()
        system.gpu_fetch_articulation_link_velocity()
        _cuda_synchronize()
        link_data = system.cuda_articulation_link_data
        v0 = _read_float_values(link_data, (art0.gpu_index, child0.index, 7), 3)
        v1 = _read_float_values(link_data, (art1.gpu_index, child1.index, 7), 3)
        owner.close()

        self.assertGreater(v0[0], 1e-3)
        self.assertLess(abs(v1[0]), 1e-4)

        system.gpu_apply_articulation_link_force()
        system.step()
        system.gpu_fetch_articulation_link_velocity()
        _cuda_synchronize()
        v1_after_all = _read_float_values(link_data, (art1.gpu_index, child1.index, 7), 3)

        self.assertGreater(v1_after_all[0], 1e-3)

    def test_link_torque_apply(self):
        system, scene = self._create_scene()
        art, _, child = self._build_two_link_articulation(scene)
        system.gpu_init()

        force = system.cuda_articulation_link_force
        torque = system.cuda_articulation_link_torque
        _memset_cuda(force)
        _memset_cuda(torque)
        _write_float_values(torque, (art.gpu_index, child.index, 0), [0.0, 0.0, 1000.0])
        _cuda_synchronize()

        system.gpu_apply_articulation_link_torque()
        system.step()
        system.gpu_fetch_articulation_link_velocity()
        _cuda_synchronize()
        link_data = system.cuda_articulation_link_data
        angular_velocity = _read_float_values(link_data, (art.gpu_index, child.index, 10), 3)
        self.assertGreater(abs(angular_velocity[2]), 1e-3)

    def test_indexed_rigid_force_and_torque_apply(self):
        system, scene = self._create_scene()
        _, body0 = self._build_rigid_dynamic(scene, x=-1.0)
        _, body1 = self._build_rigid_dynamic(scene, x=1.0)
        system.gpu_init()

        force = system.cuda_rigid_body_force
        torque = system.cuda_rigid_body_torque
        _memset_cuda(force)
        _memset_cuda(torque)
        for body in (body0, body1):
            _write_float_values(
                force, (body.gpu_pose_index, 0), [100.0, 0.0, 0.0]
            )
            _write_float_values(
                torque, (body.gpu_pose_index, 0), [0.0, 0.0, 100.0]
            )
        _cuda_synchronize()

        owner, index_buffer = self._cuda_index_buffer(body0.gpu_index)
        system.gpu_apply_rigid_dynamic_force(index_buffer)
        system.gpu_apply_rigid_dynamic_torque(index_buffer)
        system.step()
        system.gpu_fetch_rigid_dynamic_data()
        _cuda_synchronize()
        data0 = _read_float_values(
            system.cuda_rigid_body_data, (body0.gpu_pose_index, 7), 6
        )
        data1 = _read_float_values(
            system.cuda_rigid_body_data, (body1.gpu_pose_index, 7), 6
        )
        owner.close()

        self.assertGreater(data0[0], 1e-4)
        self.assertGreater(data0[5], 1e-4)
        self.assertTrue(np.allclose(data1, 0.0, atol=1e-4))

    def test_viewer_rigid_wrench_composes_without_modifying_application_buffer(self):
        system, scene = self._create_scene()
        _, body = self._build_rigid_dynamic(scene)
        system.gpu_init()

        force = system.cuda_rigid_body_force
        torque = system.cuda_rigid_body_torque
        _memset_cuda(force)
        _memset_cuda(torque)
        _write_float_values(force, (body.gpu_pose_index, 0), [10.0, 0.0, 0.0])
        _cuda_synchronize()

        system._gpu_apply_viewer_rigid_dynamic_wrench(
            body.gpu_index,
            [0.0, 0.0, 0.0],
            body.cmass_local_pose.p,
            [0.0, 1.0, 0.0],
            body.mass,
            100.0,
            0.0,
            1000.0,
        )
        system.step()
        system.gpu_fetch_rigid_dynamic_data()
        _cuda_synchronize()

        velocity = _read_float_values(
            system.cuda_rigid_body_data, (body.gpu_pose_index, 7), 3
        )
        self.assertGreater(velocity[0], 1e-4)
        self.assertGreater(velocity[1], 1e-4)
        self.assertTrue(
            np.allclose(
                _read_float_values(force, (body.gpu_pose_index, 0), 3),
                [10.0, 0.0, 0.0],
            )
        )

    def test_viewer_articulation_wrench_composes_application_buffer(self):
        system, scene = self._create_scene()
        art, _, child = self._build_two_link_articulation(scene)
        system.gpu_init()

        force = system.cuda_articulation_link_force
        torque = system.cuda_articulation_link_torque
        _memset_cuda(force)
        _memset_cuda(torque)
        _write_float_values(
            force, (art.gpu_index, child.index, 0), [10.0, 0.0, 0.0]
        )
        _cuda_synchronize()

        system._gpu_apply_viewer_articulation_link_wrench(
            art.gpu_index,
            child.index,
            child.gpu_pose_index,
            [0.0, 0.0, 0.0],
            child.cmass_local_pose.p,
            [0.3, 1.0, 0.0],
            child.mass,
            100.0,
            0.0,
            1000.0,
        )
        system.step()
        system.gpu_fetch_articulation_link_velocity()
        _cuda_synchronize()

        velocity = _read_float_values(
            system.cuda_articulation_link_data,
            (art.gpu_index, child.index, 7),
            3,
        )
        self.assertGreater(abs(float(velocity[0])), 1e-4)
        self.assertGreater(velocity[1], 1e-4)
        self.assertTrue(
            np.allclose(
                _read_float_values(force, (art.gpu_index, child.index, 0), 3),
                [10.0, 0.0, 0.0],
            )
        )

    def test_viewer_gpu_teleport_preserves_or_zeros_velocity(self):
        system, scene = self._create_scene()
        _, body = self._build_rigid_dynamic(scene)
        system.gpu_init()
        system.gpu_fetch_rigid_dynamic_data()
        _write_float_values(
            system.cuda_rigid_body_data,
            (body.gpu_pose_index, 7),
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.5],
        )
        system.gpu_apply_rigid_dynamic_data()
        system._gpu_set_viewer_rigid_dynamic_pose(
            body.gpu_index, sapien.Pose([2.0, 0.0, 1.0])
        )
        system.step()
        system.gpu_fetch_rigid_dynamic_data()
        _cuda_synchronize()
        data = _read_float_values(
            system.cuda_rigid_body_data, (body.gpu_pose_index, 0), 13
        )
        self.assertTrue(np.allclose(data[7:10], [1.0, 2.0, 3.0], atol=1e-4))
        self.assertAlmostEqual(float(data[12]), 0.5, delta=1e-3)

        system._gpu_set_viewer_rigid_dynamic_pose(
            body.gpu_index, sapien.Pose([4.0, 0.0, 1.0]), True
        )
        system.step()
        system.gpu_fetch_rigid_dynamic_data()
        _cuda_synchronize()
        stopped = _read_float_values(
            system.cuda_rigid_body_data, (body.gpu_pose_index, 0), 13
        )
        self.assertTrue(np.allclose(stopped[7:13], 0.0, atol=1e-4))
        self.assertAlmostEqual(float(stopped[0]), 4.0, places=3)

    def test_viewer_selected_articulation_state_roundtrip(self):
        system, scene = self._create_scene()
        art, _, _ = self._build_two_link_articulation(
            scene, joint_type="revolute"
        )
        system.gpu_init()

        system._gpu_upload_articulation_qpos(
            art.gpu_index, np.asarray([0.0], dtype=np.float32)
        )
        system._gpu_upload_articulation_target_qpos(
            art.gpu_index, np.asarray([0.25], dtype=np.float32)
        )
        system._gpu_upload_articulation_target_qvel(
            art.gpu_index, np.asarray([-0.5], dtype=np.float32)
        )
        qpos = system._gpu_download_articulation_qpos(art.gpu_index)
        target_qpos = system._gpu_download_articulation_target_qpos(art.gpu_index)
        target_qvel = system._gpu_download_articulation_target_qvel(art.gpu_index)

        self.assertAlmostEqual(qpos[0], 0.0, places=5)
        self.assertAlmostEqual(target_qpos[0], 0.25, places=5)
        self.assertAlmostEqual(target_qvel[0], -0.5, places=5)
        with self.assertRaisesRegex(RuntimeError, "invalid articulation GPU index"):
            system._gpu_download_articulation_qpos(-1)
        with self.assertRaisesRegex(RuntimeError, "invalid index or shape"):
            system._gpu_upload_articulation_qpos(
                art.gpu_index, np.asarray([0.0, 0.1], dtype=np.float32)
            )

    def test_selected_articulation_transfers_wait_for_custom_stream(self):
        system, scene = self._create_scene()
        art, _, _ = self._build_two_link_articulation(
            scene, joint_type="revolute"
        )
        system.gpu_init()

        cudart = _load_cudart()
        stream = ctypes.c_void_p()
        scratch = ctypes.c_void_p()
        scratch_size = 64 * 1024 * 1024

        try:
            _check_cuda(
                cudart.cudaStreamCreateWithFlags(
                    ctypes.byref(stream), _CUDA_STREAM_NON_BLOCKING
                )
            )
            _check_cuda(cudart.cudaMalloc(ctypes.byref(scratch), scratch_size))
            system.gpu_set_cuda_stream(stream.value)

            def enqueue_delayed_zero(buffer):
                for _ in range(256):
                    _check_cuda(
                        cudart.cudaMemsetAsync(scratch, 0, scratch_size, stream)
                    )
                _check_cuda(
                    cudart.cudaMemsetAsync(
                        ctypes.c_void_p(buffer.ptr),
                        0,
                        int(np.prod(buffer.shape))
                        * np.dtype(buffer.typestr).itemsize,
                        stream,
                    )
                )

            enqueue_delayed_zero(system.cuda_articulation_qpos)
            system._gpu_upload_articulation_qpos(
                art.gpu_index, np.asarray([0.2], dtype=np.float32)
            )
            enqueue_delayed_zero(system.cuda_articulation_target_qpos)
            system._gpu_upload_articulation_target_qpos(
                art.gpu_index, np.asarray([0.35], dtype=np.float32)
            )
            enqueue_delayed_zero(system.cuda_articulation_target_qvel)
            system._gpu_upload_articulation_target_qvel(
                art.gpu_index, np.asarray([-0.4], dtype=np.float32)
            )

            enqueue_delayed_zero(system.cuda_articulation_qpos)
            self.assertAlmostEqual(
                system._gpu_download_articulation_qpos(art.gpu_index)[0],
                0.2,
                places=4,
            )
            enqueue_delayed_zero(system.cuda_articulation_target_qpos)
            self.assertAlmostEqual(
                system._gpu_download_articulation_target_qpos(art.gpu_index)[0],
                0.35,
                places=4,
            )
            enqueue_delayed_zero(system.cuda_articulation_target_qvel)
            self.assertAlmostEqual(
                system._gpu_download_articulation_target_qvel(art.gpu_index)[0],
                -0.4,
                places=4,
            )
        finally:
            if stream.value:
                _check_cuda(cudart.cudaStreamSynchronize(stream))
            system.gpu_set_cuda_stream(0)
            if scratch.value:
                _check_cuda(cudart.cudaFree(scratch))
            if stream.value:
                _check_cuda(cudart.cudaStreamDestroy(stream))

    def test_viewer_gpu_articulation_root_teleport(self):
        system, scene = self._create_scene()
        art, root, _ = self._build_two_link_articulation(scene)
        system.gpu_init()
        system._gpu_set_viewer_articulation_root_pose(
            art.gpu_index, root.gpu_pose_index, sapien.Pose([2.0, 0.0, 1.0])
        )
        system.step()
        system.gpu_fetch_articulation_link_pose()
        _cuda_synchronize()
        root_pose = _read_float_values(
            system.cuda_articulation_link_data,
            (art.gpu_index, root.index, 0),
            7,
        )
        self.assertTrue(np.allclose(root_pose[:3], [2.0, 0.0, 1.0], atol=1e-3))

    def test_force_writes_replace_instead_of_accumulate(self):
        system, scene = self._create_scene()
        art0, _, child0 = self._build_two_link_articulation(scene, x=-1.0)
        art1, _, child1 = self._build_two_link_articulation(scene, x=1.0)
        system.gpu_init()

        force = system.cuda_articulation_link_force
        _memset_cuda(force)
        _write_float_values(force, (art0.gpu_index, child0.index, 0), [1000.0, 0.0, 0.0])
        _write_float_values(force, (art1.gpu_index, child1.index, 0), [1000.0, 0.0, 0.0])
        _cuda_synchronize()

        owner0, index0 = self._cuda_index_buffer(art0.gpu_index)
        owner1, index1 = self._cuda_index_buffer(art1.gpu_index)
        system.gpu_apply_articulation_link_force(index0)
        system.gpu_apply_articulation_link_force(index0)
        system.gpu_apply_articulation_link_force(index1)
        system.step()
        system.gpu_fetch_articulation_link_velocity()
        _cuda_synchronize()

        link_data = system.cuda_articulation_link_data
        velocity0 = _read_float_values(link_data, (art0.gpu_index, child0.index, 7), 3)
        velocity1 = _read_float_values(link_data, (art1.gpu_index, child1.index, 7), 3)
        owner0.close()
        owner1.close()

        self.assertGreater(velocity0[0], 1e-3)
        self.assertAlmostEqual(velocity0[0], velocity1[0], delta=1e-3)

    def test_force_is_cleared_after_one_step(self):
        system, scene = self._create_scene()
        art, _, child = self._build_two_link_articulation(scene)
        system.gpu_init()

        force = system.cuda_articulation_link_force
        _memset_cuda(force)
        _write_float_values(force, (art.gpu_index, child.index, 0), [1000.0, 0.0, 0.0])
        _cuda_synchronize()
        system.gpu_apply_articulation_link_force()
        system.step()
        system.gpu_fetch_articulation_link_velocity()
        _cuda_synchronize()
        link_data = system.cuda_articulation_link_data
        velocity_after_force = _read_float_values(link_data, (art.gpu_index, child.index, 7), 3)

        # Do not clear or re-apply the exposed force buffer. PhysX consumes the SET write for one
        # simulation step because SAPIEN does not enable eRETAIN_ACCELERATIONS.
        system.step()
        system.gpu_fetch_articulation_link_velocity()
        _cuda_synchronize()
        velocity_after_next_step = _read_float_values(
            link_data, (art.gpu_index, child.index, 7), 3
        )

        self.assertGreater(velocity_after_force[0], 1e-3)
        self.assertLess(
            abs(velocity_after_next_step[0] - velocity_after_force[0]), 1e-3
        )


if __name__ == "__main__":
    unittest.main()
