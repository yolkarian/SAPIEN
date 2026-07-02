"""Tests for low-level PhysX GPU articulation buffers."""

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


def _read_values(array, indices: tuple[int, ...], count: int, dtype) -> np.ndarray:
    host = np.empty(count, dtype=dtype)
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


class TestPhysxGpuIkHelperRemoval(unittest.TestCase):
    def test_gpu_ik_helper_is_not_exported(self):
        self.assertFalse(hasattr(sapien.physx, "GpuInverseKinematicsSolver"))
        self.assertFalse(hasattr(sapien.physx, "gpu_inverse_kinematics"))


class TestGpuArticulationBuffers(unittest.TestCase):
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

    def _build_prismatic_articulation(self, scene, y: float):
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

        articulation = builder.build(fix_root_link=True)
        for link in articulation.links:
            link.disable_gravity = True
        return articulation, articulation.links[1]

    def test_cuda_array_protocol_indices_for_apply_update_and_jacobian(self):
        system, scene = self._create_scene()
        art0, slider0 = self._build_prismatic_articulation(scene, y=-0.25)
        art1, slider1 = self._build_prismatic_articulation(scene, y=0.25)
        system.gpu_init()

        qpos = system.cuda_articulation_qpos
        _write_float_values(qpos, (art0.gpu_index, 0), [0.4])
        _write_float_values(qpos, (art1.gpu_index, 0), [0.8])
        _cuda_synchronize()

        index_owner = _DeviceArray(np.asarray([art0.gpu_index], dtype=np.int32))
        try:
            system.gpu_apply_articulation_qpos(index_owner)
            system.gpu_update_articulation_kinematics(index_owner)
            system.gpu_fetch_articulation_link_pose()
            system.gpu_compute_articulation_jacobian(index_owner)
            _cuda_synchronize()
        finally:
            index_owner.close()

        link_data = system.cuda_articulation_link_data
        root0_pos = _read_values(link_data, (art0.gpu_index, 0, 0), 3, np.float32)
        slider0_pos = _read_values(link_data, (art0.gpu_index, slider0.index, 0), 3, np.float32)
        root1_pos = _read_values(link_data, (art1.gpu_index, 0, 0), 3, np.float32)
        slider1_pos = _read_values(link_data, (art1.gpu_index, slider1.index, 0), 3, np.float32)

        self.assertGreater(slider0_pos[0] - root0_pos[0], 0.35)
        self.assertLess(abs(slider1_pos[0] - root1_pos[0]), 0.05)

        jacobian_shape = system.cuda_articulation_jacobian_shape
        self.assertEqual(jacobian_shape.shape, [2, 2])
        shape0 = _read_values(jacobian_shape, (art0.gpu_index, 0), 2, np.uint32)
        self.assertEqual(tuple(int(x) for x in shape0), tuple(art0.get_jacobian_shape()))

        jacobian = system.cuda_articulation_jacobian
        self.assertEqual(jacobian.shape, [2, 12, 7])


if __name__ == "__main__":
    unittest.main()
