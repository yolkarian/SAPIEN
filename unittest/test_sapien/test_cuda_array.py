import gc
import subprocess
import sys
import textwrap
import unittest

import sapien


class TestCudaArray(unittest.TestCase):
    def test_dlpack_capsule_release(self) -> None:
        """Release unconsumed exports; ASan checks metadata allocation pairing."""
        import torch

        backing = torch.arange(24, dtype=torch.float32, device="cuda").reshape(2, 3, 4)
        for owner in (backing, backing[:, ::2, :]):
            with self.subTest(shape=tuple(owner.shape), stride=owner.stride()):
                array = sapien.CudaArray(owner)
                for _ in range(32):
                    capsule = array.dlpack()
                    del capsule
                gc.collect()
                self.assertEqual(array.ptr, owner.data_ptr())
                torch.testing.assert_close(backing.flatten(), torch.arange(24, device="cuda").float())

    def test_dlpack_torch_consumer_release(self) -> None:
        """Free consumed exports without releasing their external backing storage."""
        import torch

        backing = torch.arange(24, dtype=torch.float32, device="cuda").reshape(2, 3, 4)
        owner = backing[:, ::2, :]
        array = sapien.CudaArray(owner)
        for _ in range(32):
            consumer = array.torch()
            self.assertEqual(consumer.data_ptr(), owner.data_ptr())
            self.assertEqual(consumer.stride(), owner.stride())
            torch.testing.assert_close(consumer, owner)
            del consumer
        gc.collect()
        torch.testing.assert_close(backing.flatten(), torch.arange(24, device="cuda").float())

    def test_torch(self):
        import torch

        tensor = torch.tensor([[0, 1, 2], [2, 3, 4]]).float().cuda()
        array = sapien.CudaArray(tensor)

        self.assertEqual(
            array.typestr,
            array.__cuda_array_interface__["typestr"],
        )
        self.assertEqual(
            tuple(array.shape),
            array.__cuda_array_interface__["shape"],
        )
        self.assertEqual(
            tuple(array.strides),
            array.__cuda_array_interface__["strides"],
        )
        self.assertEqual(tuple(array.strides), (12, 4))
        self.assertEqual(
            array.ptr,
            array.__cuda_array_interface__["data"][0],
        )

        self.assertEqual(
            tensor.__cuda_array_interface__["typestr"],
            array.__cuda_array_interface__["typestr"],
        )
        self.assertEqual(
            tensor.__cuda_array_interface__["shape"],
            array.__cuda_array_interface__["shape"],
        )
        self.assertEqual(
            tensor.__cuda_array_interface__["data"],
            array.__cuda_array_interface__["data"],
        )

        self.assertEqual(
            array.torch().__cuda_array_interface__["data"],
            array.__cuda_array_interface__["data"],
        )

    def test_cupy(self):
        import torch
        import cupy

        tensor = torch.tensor([[0, 1, 2], [2, 3, 4]]).float().cuda()
        array = sapien.CudaArray(tensor)

        cp_array = array.cupy()
        self.assertIsInstance(cp_array, cupy.ndarray)
        self.assertEqual(cp_array.shape, tuple(array.shape))
        self.assertEqual(cp_array.dtype, cupy.float32)
        self.assertEqual(
            cp_array.__cuda_array_interface__["data"],
            array.__cuda_array_interface__["data"],
        )

    def test_cupy_slice(self):
        import torch
        import cupy

        tensor = torch.tensor([[0, 1, 2], [2, 3, 4], [3, 4, 5]]).float().cuda()
        tensor = tensor[1:, :-1]
        array = sapien.CudaArray(tensor)

        cp_array = array.cupy()
        self.assertIsInstance(cp_array, cupy.ndarray)
        self.assertEqual(cp_array.shape, tuple(array.shape))
        self.assertEqual(
            cp_array.__cuda_array_interface__["data"],
            array.__cuda_array_interface__["data"],
        )
        self.assertEqual(
            cp_array.__cuda_array_interface__["strides"],
            array.__cuda_array_interface__["strides"],
        )

    def test_slice(self):
        import torch

        tensor = torch.tensor([[0, 1, 2], [2, 3, 4], [3, 4, 5]]).float().cuda()
        tensor = tensor[1:, :-1]
        array = sapien.CudaArray(tensor)

        self.assertEqual(
            array.typestr,
            array.__cuda_array_interface__["typestr"],
        )
        self.assertEqual(
            tuple(array.shape),
            array.__cuda_array_interface__["shape"],
        )
        self.assertEqual(
            tuple(array.strides),
            array.__cuda_array_interface__["strides"],
        )
        self.assertEqual(
            array.ptr,
            array.__cuda_array_interface__["data"][0],
        )

        self.assertEqual(
            tensor.__cuda_array_interface__["typestr"],
            array.__cuda_array_interface__["typestr"],
        )
        self.assertEqual(
            tensor.__cuda_array_interface__["shape"],
            array.__cuda_array_interface__["shape"],
        )
        self.assertEqual(
            tensor.__cuda_array_interface__["data"],
            array.__cuda_array_interface__["data"],
        )
        self.assertEqual(
            tensor.__cuda_array_interface__["strides"],
            array.__cuda_array_interface__["strides"],
        )

        self.assertEqual(
            array.torch().__cuda_array_interface__["data"],
            array.__cuda_array_interface__["data"],
        )

    def test_interface_pointer_in_a_fresh_process(self) -> None:
        """Wrap a device pointer of PhysX through __cuda_array_interface__ before any library
        has made a CUDA context current on the Python thread. The pointer query ran without a
        current context there and called every device pointer invalid; earlier tests that use
        Torch made the context current and hid it, so this runs in a new interpreter."""
        script = textwrap.dedent(
            """
            import sapien

            sapien.physx.enable_gpu()
            physx = sapien.physx.PhysxGpuSystem(sapien.Device("cuda"))
            scene = sapien.Scene([physx])
            builder = scene.create_actor_builder()
            builder.add_box_collision(half_size=[0.1, 0.1, 0.1])
            builder.build()
            physx.gpu_init()
            data = physx.cuda_rigid_body_data

            class View:
                __cuda_array_interface__ = {
                    "shape": tuple(data.shape),
                    "strides": tuple(data.strides),
                    "typestr": "<f4",
                    "data": (data.ptr, False),
                    "version": 2,
                }

            array = sapien.CudaArray(View())
            assert array.ptr == data.ptr and array.cuda_id == data.cuda_id
            del array, data
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
        )
        self.assertEqual(result.returncode, 0, result.stderr)
