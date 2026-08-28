"""Synchronization contracts for PhysX GPU contact impulse queries."""

import ctypes
import ctypes.util
import gc
import unittest

import numpy as np
import sapien

_CUDA_MEMCPY_DEVICE_TO_HOST = 2
_CUDA_STREAM_NON_BLOCKING = 1


def _cuda_device_available() -> bool:
    try:
        lib = ctypes.CDLL("libcuda.so")
    except OSError:
        return False
    return lib.cuInit(0) == 0


def _load_cudart():
    library = ctypes.util.find_library("cudart") or "libcudart.so.12"
    cudart = ctypes.CDLL(library)
    cudart.cudaMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    cudart.cudaMemsetAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    cudart.cudaStreamCreateWithFlags.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
    ]
    cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    cudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    cudart.cudaGetErrorString.argtypes = [ctypes.c_int]
    cudart.cudaGetErrorString.restype = ctypes.c_char_p
    return cudart


def _check_cuda(code: int) -> None:
    if code:
        message = _load_cudart().cudaGetErrorString(code).decode("utf-8", errors="replace")
        raise RuntimeError(f"CUDA error {code}: {message}")


def _copy_cuda_array(array) -> np.ndarray:
    interface = array.__cuda_array_interface__
    result = np.empty(interface["shape"], dtype=np.dtype(interface["typestr"]))
    cudart = _load_cudart()
    code = cudart.cudaMemcpy(
        ctypes.c_void_p(result.ctypes.data),
        ctypes.c_void_p(interface["data"][0]),
        result.nbytes,
        _CUDA_MEMCPY_DEVICE_TO_HOST,
    )
    _check_cuda(code)
    return result


class TestGpuContactQueries(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not _cuda_device_available():
            raise unittest.SkipTest("no usable CUDA device")

    def tearDown(self) -> None:
        gc.collect()
        if sapien.physx.can_shutdown():
            sapien.physx.shutdown()

    def test_optional_synchronization_and_explicit_wait(self) -> None:
        sapien.physx.enable_gpu()
        config = sapien.physx.PhysxSceneConfig()
        config.num_scenes = 1
        sapien.physx.set_scene_config(config)
        system = sapien.physx.PhysxGpuSystem()
        scene = sapien.Scene([system])

        ground_builder = scene.create_actor_builder()
        ground_builder.add_box_collision(half_size=[1.0, 1.0, 0.05])
        ground_builder.set_physx_body_type("static")
        ground_builder.set_initial_pose(sapien.Pose([0.0, 0.0, -0.05]))
        ground = ground_builder.build(name="ground")

        box_builder = scene.create_actor_builder()
        box_builder.add_box_collision(half_size=[0.05, 0.05, 0.05], density=1000)
        box_builder.set_initial_pose(sapien.Pose([0.0, 0.0, 0.04]))
        box = box_builder.build(name="box")
        ground_body = ground.find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
        box_body = box.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)

        system.gpu_init()
        pair_query = system.gpu_create_contact_pair_impulse_query([(box_body, ground_body)])
        body_query = system.gpu_create_contact_body_impulse_query([box_body])
        system.step()

        system.gpu_query_contact_pair_impulses(pair_query)
        expected_pair = _copy_cuda_array(pair_query.cuda_impulses)
        system.gpu_query_contact_body_impulses(body_query)
        expected_body = _copy_cuda_array(body_query.cuda_impulses)
        self.assertGreater(float(np.linalg.norm(expected_pair)), 0.0)
        self.assertGreater(float(np.linalg.norm(expected_body)), 0.0)

        system.gpu_query_contact_pair_impulses(pair_query, synchronize=False)
        system.gpu_query_contact_body_impulses(body_query, synchronize=False)
        system.gpu_wait_contact_queries()
        np.testing.assert_array_equal(_copy_cuda_array(pair_query.cuda_impulses), expected_pair)
        np.testing.assert_array_equal(_copy_cuda_array(body_query.cuda_impulses), expected_body)

        # A non-default stream with queued work exposes ordering bugs hidden by
        # legacy-default-stream synchronization. The next step must wait before
        # PhysX invalidates contact pointers or overwrites the shared contact buffer.
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
            for _ in range(128):
                _check_cuda(
                    cudart.cudaMemsetAsync(scratch, 0, scratch_size, stream)
                )
            system.gpu_query_contact_pair_impulses(pair_query, synchronize=False)
            system.gpu_query_contact_body_impulses(body_query, synchronize=False)
            system.step()
            np.testing.assert_array_equal(
                _copy_cuda_array(pair_query.cuda_impulses), expected_pair
            )
            np.testing.assert_array_equal(
                _copy_cuda_array(body_query.cuda_impulses), expected_body
            )
        finally:
            if stream.value:
                _check_cuda(cudart.cudaStreamSynchronize(stream))
            system.gpu_set_cuda_stream(0)
            if scratch.value:
                _check_cuda(cudart.cudaFree(scratch))
            if stream.value:
                _check_cuda(cudart.cudaStreamDestroy(stream))

        del pair_query, body_query, ground_body, box_body, ground, box
        del ground_builder, box_builder
        scene.close()
        system.close()
        del scene, system
