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

    def _build_two_link_articulation(self, scene, x: float = 0.0):
        material = sapien.physx.PhysxMaterial(0.2, 0.1, 0.05)

        root = sapien.physx.PhysxArticulationLinkComponent()
        child = sapien.physx.PhysxArticulationLinkComponent(root)
        root.name = "root"
        child.name = "child"
        root.disable_gravity = True
        child.disable_gravity = True

        root.attach(sapien.physx.PhysxCollisionShapeBox([0.1, 0.1, 0.1], material))
        child.attach(sapien.physx.PhysxCollisionShapeBox([0.1, 0.1, 0.1], material))
        child.joint.set_type("fixed")
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
