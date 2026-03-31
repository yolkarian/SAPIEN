import unittest
import sapien


class TestCudaArray(unittest.TestCase):
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
