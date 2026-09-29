"""Empty external CUDA views retain a device through DLPack exports."""

import unittest

import sapien


class TestCudaArray(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch is required for CUDA interoperability tests")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("no CUDA device")
        cls.torch = torch

    def test_empty_shapes_and_exports(self) -> None:
        torch = self.torch
        for shape in ((0,), (2, 0, 3)):
            with self.subTest(shape=shape):
                owner = torch.empty(shape, dtype=torch.int32, device="cuda")
                view = sapien.CudaArray(owner)
                self.assertEqual(view.shape, list(shape))
                self.assertEqual(view.cuda_id, owner.device.index)
                for tensor in (view.torch(), torch.from_dlpack(view.dlpack())):
                    self.assertEqual(tuple(tensor.shape), shape)
                    self.assertEqual(tensor.dtype, owner.dtype)
                    self.assertEqual(tensor.device, owner.device)

    def test_empty_interface_without_device_uses_current_device(self) -> None:
        class Empty:
            __cuda_array_interface__ = {
                "shape": (0,), "typestr": "<i4", "data": (0, False), "version": 3
            }

        view = sapien.CudaArray(Empty())
        self.assertEqual(view.cuda_id, self.torch.cuda.current_device())
        self.assertEqual(view.torch().numel(), 0)

    def test_nonempty_null_pointer_is_rejected(self) -> None:
        class Invalid:
            __cuda_array_interface__ = {
                "shape": (1,), "typestr": "<i4", "data": (0, False), "version": 3
            }

        with self.assertRaises(RuntimeError):
            sapien.CudaArray(Invalid())

    def test_empty_cupy_export(self) -> None:
        try:
            import cupy
        except ImportError:
            self.skipTest("CuPy is required for this interoperability test")
        owner = cupy.empty((0,), dtype=cupy.float32)
        view = sapien.CudaArray(owner)
        result = view.cupy()
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, owner.dtype)
        self.assertEqual(result.device.id, owner.device.id)

    def test_empty_source_device_is_not_current_device(self) -> None:
        torch = self.torch
        if torch.cuda.device_count() < 2:
            self.skipTest("requires two CUDA devices")
        owner = torch.empty(0, device="cuda:1")
        with torch.cuda.device(0):
            view = sapien.CudaArray(owner)
            self.assertEqual(view.cuda_id, 1)
            self.assertEqual(view.torch().device.index, 1)
