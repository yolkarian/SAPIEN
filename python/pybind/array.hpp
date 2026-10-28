#pragma once
#include "sapien/array.h"
#include <algorithm>
#include <deque>
#ifdef SAPIEN_CUDA
#include "sapien/utils/cuda.h"
#include <dlpack/dlpack.h>
#endif
#include <pybind11/numpy.h>
#include <pybind11/smart_holder.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace sapien;

namespace sapien {

#ifdef SAPIEN_CUDA
inline py::capsule DLPackToCapsule(DLManagedTensor *tensor) {
  return py::capsule(tensor, "dltensor", [](PyObject *data) {
    auto tensor = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(data, "dltensor"));
    if (tensor) {
      tensor->deleter(tensor);
    } else {
      PyErr_Clear();
    }
  });
}

inline py::capsule DLPackVersionedToCapsule(DLManagedTensorVersioned *tensor) {
  return py::capsule(tensor, "dltensor_versioned", [](PyObject *data) {
    auto tensor = static_cast<DLManagedTensorVersioned *>(
        PyCapsule_GetPointer(data, "dltensor_versioned"));
    if (tensor) {
      tensor->deleter(tensor);
    } else {
      PyErr_Clear();
    }
  });
}

/** DLPack/array-API value of a CUDA stream handle: the legacy default stream is 1, not 0. */
inline intptr_t DLPackStreamValue(uintptr_t stream) {
  return stream == 0 ? 1 : static_cast<intptr_t>(stream);
}

/** The array-API `__dlpack__` protocol for a SAPIEN CUDA array. `stream` is the consumer's
 *  stream (None: legacy default stream, -1: no synchronization); it is ordered after the work
 *  the owner submitted to its stream, so the consumer never reads a buffer still being written.
 *  Exports are zero-copy views that keep the owner's view guard. */
inline py::object ExportCudaArrayDLPack(CudaArrayHandle const &array, py::object stream,
                                        py::object maxVersion, py::object dlDevice,
                                        py::object copy) {
  if (!dlDevice.is_none()) {
    auto device = dlDevice.cast<std::pair<int, int>>();
    if (device.first != kDLCUDA || device.second != array.cudaId) {
      throw py::buffer_error("a SAPIEN CUDA array can only be exported to its own device, cuda:" +
                             std::to_string(array.cudaId));
    }
  }
  if (!copy.is_none() && copy.cast<bool>()) {
    throw py::buffer_error("SAPIEN CUDA arrays are exported without copying");
  }
  intptr_t consumer = stream.is_none() ? 1 : stream.cast<intptr_t>();
  if (consumer != -1 && array.cudaId >= 0) {
    cudaStreamWaitStream(array.cudaId, static_cast<uintptr_t>(consumer), array.producerStream());
  }
  if (!maxVersion.is_none() && maxVersion.cast<std::pair<int, int>>().first >= 1) {
    return DLPackVersionedToCapsule(array.toDLPackVersioned());
  }
  return DLPackToCapsule(array.toDLPack());
}
#endif

inline CudaArrayHandle CudaArrayHandleFromCudaArrayInterface(py::handle obj) {
  auto interface = obj.attr("__cuda_array_interface__").cast<py::dict>();

  auto shape = interface["shape"].cast<py::tuple>().cast<std::vector<int>>();
  auto type = interface["typestr"].cast<std::string>();

  std::vector<int> strides;
  if (interface.contains("strides") && !interface["strides"].is_none()) {
    strides = interface["strides"].cast<py::tuple>().cast<std::vector<int>>();
  } else {
    strides = ShapeToStrides(shape, py::dtype(type).itemsize());
  }

  auto data = interface["data"].cast<py::tuple>();
  void *ptr = reinterpret_cast<void *>(data[0].cast<uintptr_t>());

  int cudaId = -1;
#ifdef SAPIEN_CUDA
  bool empty = std::find(shape.begin(), shape.end(), 0) != shape.end();
  if (ptr || !empty) {
    cudaId = getCudaPtrDevice(ptr);
  } else if (py::hasattr(obj, "__dlpack_device__")) {
    cudaId = obj.attr("__dlpack_device__")().cast<std::pair<int, int>>().second;
  } else {
    // a null empty view has no address to infer a device from; use the current one
    cudaId = getCudaCurrentDevice();
  }
#endif

  return CudaArrayHandle{.shape = shape,
                         .strides = strides,
                         .type = type,
                         .cudaId = cudaId,
                         .ptr = ptr};
}

#ifdef SAPIEN_CUDA
inline std::string DLDataTypeToTypestr(DLDataType dtype) {
  if (dtype.lanes != 1 || dtype.bits % 8 != 0) {
    throw std::runtime_error("unsupported DLPack dtype: only scalar types with whole bytes");
  }
  switch (dtype.code) {
  case kDLInt:
    return "i" + std::to_string(dtype.bits / 8);
  case kDLUInt:
    return "u" + std::to_string(dtype.bits / 8);
  case kDLFloat:
    return "f" + std::to_string(dtype.bits / 8);
  default:
    throw std::runtime_error("unsupported DLPack dtype code " + std::to_string(dtype.code));
  }
}

inline CudaArrayHandle CudaArrayHandleFromDLTensor(DLTensor const &tensor) {
  if (tensor.device.device_type != kDLCUDA) {
    throw std::runtime_error("expected a CUDA array, got an array on DLPack device type " +
                             std::to_string(tensor.device.device_type));
  }
  std::string type = DLDataTypeToTypestr(tensor.dtype);
  int itemsize = tensor.dtype.bits / 8;
  std::vector<int> shape(tensor.shape, tensor.shape + tensor.ndim);
  std::vector<int> strides;
  if (tensor.strides) {
    for (int i = 0; i < tensor.ndim; ++i) {
      strides.push_back(tensor.strides[i] * itemsize);
    }
  } else {
    strides = ShapeToStrides(shape, itemsize);
  }
  return CudaArrayHandle{.shape = shape,
                         .strides = strides,
                         .type = type,
                         .cudaId = tensor.device.device_id,
                         .ptr = static_cast<char *>(tensor.data) + tensor.byte_offset};
}

/** A CUDA array borrowed from a Python producer for asynchronous device work. `keepAlive` holds
 *  the producer (and its DLPack capsule); release it with ReleaseAfterStream, never earlier. */
struct BorrowedCudaArray {
  CudaArrayHandle handle;
  py::object keepAlive;
};

/** Borrow `obj` for work on `stream` of device `cudaId`, ordering that stream after the
 *  producer: a sapien.CudaArray through its owner's stream, an object implementing `__dlpack__`
 *  (torch, JAX, CuPy) through the DLPack stream handshake, and a `__cuda_array_interface__`
 *  object through its v3 `stream` entry when present. An array on another device is returned
 *  without synchronization so the caller's device check reports it. */
inline BorrowedCudaArray BorrowCudaArray(py::handle obj, int cudaId, uintptr_t stream) {
  auto owner = py::reinterpret_borrow<py::object>(obj);
  if (py::isinstance<CudaArrayHandle>(obj)) {
    auto handle = obj.cast<CudaArrayHandle>();
    if (handle.cudaId == cudaId) {
      cudaStreamWaitStream(cudaId, stream, handle.producerStream());
    }
    return {std::move(handle), std::move(owner)};
  }

  if (py::hasattr(obj, "__dlpack__")) {
    intptr_t consumer = DLPackStreamValue(stream);
    int producerDevice = cudaId;
    if (py::hasattr(obj, "__dlpack_device__")) {
      auto device = obj.attr("__dlpack_device__")().cast<std::pair<int, int>>();
      if (device.first != kDLCUDA) {
        throw std::runtime_error("expected a CUDA array, got an array on DLPack device type " +
                                 std::to_string(device.first));
      }
      producerDevice = device.second;
      if (producerDevice != cudaId) {
        consumer = -1;
      }
    }
    py::object capsule;
    {
      // Exporters such as torch export only from their own device, which must be current, and
      // read the consumer stream as a stream of that device.
      CudaDeviceScope scope(producerDevice);
      try {
        capsule = obj.attr("__dlpack__")(py::arg("stream") = consumer,
                                         py::arg("max_version") = py::make_tuple(1, 0));
      } catch (py::error_already_set &e) {
        if (!e.matches(PyExc_TypeError)) {
          throw;
        }
        capsule = obj.attr("__dlpack__")(py::arg("stream") = consumer);
      }
    }
    DLTensor *tensor{};
    if (PyCapsule_IsValid(capsule.ptr(), "dltensor_versioned")) {
      auto managed = static_cast<DLManagedTensorVersioned *>(
          PyCapsule_GetPointer(capsule.ptr(), "dltensor_versioned"));
      if (managed->version.major != DLPACK_MAJOR_VERSION) {
        throw std::runtime_error("unsupported DLPack major version " +
                                 std::to_string(managed->version.major));
      }
      tensor = &managed->dl_tensor;
    } else if (PyCapsule_IsValid(capsule.ptr(), "dltensor")) {
      tensor = &static_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule.ptr(), "dltensor"))
                    ->dl_tensor;
    } else {
      throw std::runtime_error("__dlpack__ did not return a DLPack capsule");
    }
    // The capsule stays unconsumed: its destructor hands the tensor back to the producer when
    // keepAlive is dropped.
    return {CudaArrayHandleFromDLTensor(*tensor), py::make_tuple(owner, capsule)};
  }

  if (py::hasattr(obj, "__cuda_array_interface__")) {
    auto handle = CudaArrayHandleFromCudaArrayInterface(obj);
    auto interface = obj.attr("__cuda_array_interface__").cast<py::dict>();
    if (handle.cudaId == cudaId && interface.contains("stream") &&
        !interface["stream"].is_none()) {
      cudaStreamWaitStream(cudaId, stream, interface["stream"].cast<uintptr_t>());
    }
    return {std::move(handle), std::move(owner)};
  }

  throw std::runtime_error("expected a sapien.CudaArray, an object implementing __dlpack__, or "
                           "an object exposing __cuda_array_interface__");
}

/** Borrowed producers whose device work may still run, with the event that marks its end. */
struct PendingBorrow {
  CudaEvent done;
  py::object keepAlive;
};

/** Process-wide; only touched with the GIL held. Leaked on purpose so no Python object is
 *  destroyed during static destruction, after the interpreter has finalized. */
inline std::deque<PendingBorrow> &PendingBorrows() {
  static auto *pending = new std::deque<PendingBorrow>();
  return *pending;
}

/** Drop the borrowed producers whose device work has completed. Dropping a producer can release
 *  the GIL (JAX does while it frees an array) and let another thread use the queue, so the queue
 *  is rebuilt first and the completed producers are dropped only once it is consistent again. */
inline void ReleaseCompletedBorrows() {
  auto &pending = PendingBorrows();
  std::vector<bool> done;
  done.reserve(pending.size());
  for (auto const &borrow : pending) {
    done.push_back(borrow.done.query());
  }
  if (std::find(done.begin(), done.end(), true) == done.end()) {
    return;
  }
  std::vector<PendingBorrow> completed;
  std::deque<PendingBorrow> running;
  for (size_t i = 0; i < pending.size(); ++i) {
    if (done[i]) {
      completed.push_back(std::move(pending[i]));
    } else {
      running.push_back(std::move(pending[i]));
    }
  }
  pending.swap(running);
}

/** Keep `keepAlive` until the work submitted to `stream` of device `cudaId` so far completed:
 *  a producer such as JAX or torch may otherwise reuse the memory while it is still read. */
inline void ReleaseAfterStream(py::object keepAlive, int cudaId, uintptr_t stream) {
  PendingBorrow borrow{CudaEvent(), std::move(keepAlive)};
  {
    CudaDeviceScope scope(cudaId);
    borrow.done.record(reinterpret_cast<cudaStream_t>(stream));
  }
  auto &pending = PendingBorrows();
  pending.push_back(std::move(borrow));
  ReleaseCompletedBorrows();
  // Bound the queue when the device falls far behind the host; as above, drop the producers
  // only after they left the queue.
  constexpr size_t kMaxPendingBorrows = 4096;
  std::vector<PendingBorrow> overflow;
  while (pending.size() > kMaxPendingBorrows) {
    pending.front().done.synchronize();
    overflow.push_back(std::move(pending.front()));
    pending.pop_front();
  }
}

/** Run `work(handle, stream)` (C++ GPU calls, without the GIL) on an array borrowed from `obj`
 *  for `stream`, then keep the producer alive until the stream `work` submitted to ran past it,
 *  also when `work` throws after submitting device work. The borrow's handshake calls into
 *  Python before the owner's lock can be taken, so the owner may have switched streams since
 *  `stream` was read: under its lock, `work` orders the owner's current stream after `stream`
 *  with FollowStream() and submits there. */
template <typename Work>
void RunWithBorrowedCudaArray(py::handle obj, int cudaId, uintptr_t stream, Work &&work) {
  auto borrowed = BorrowCudaArray(obj, cudaId, stream);
  try {
    py::gil_scoped_release release;
    work(borrowed.handle, stream);
  } catch (...) {
    ReleaseAfterStream(std::move(borrowed.keepAlive), cudaId, stream);
    throw;
  }
  ReleaseAfterStream(std::move(borrowed.keepAlive), cudaId, stream);
}

/** Order `current`, the owner's stream read under its lock, after `borrowed`, the stream a borrow
 *  was ordered against, and make it the stream to submit to. */
inline void FollowStream(int cudaId, uintptr_t &borrowed, uintptr_t current) {
  cudaStreamWaitStream(cudaId, current, borrowed);
  borrowed = current;
}
#endif

} // namespace sapien

namespace pybind11::detail {

template <> struct type_caster<CpuArrayHandle> {
  PYBIND11_TYPE_CASTER(CpuArrayHandle, _("numpy.ndarray"));

  bool load(py::handle src, bool convert) { return false; }

  static py::handle cast(CpuArrayHandle const &src, py::return_value_policy policy,
                         py::handle parent) {

    std::vector<py::ssize_t> shape;
    std::vector<py::ssize_t> strides;
    for (auto s : src.shape) {
      shape.push_back(s);
    }
    for (auto s : src.strides) {
      strides.push_back(s);
    }

    // this makes a copy
    auto array = py::array(py::dtype(src.type), shape, strides, src.ptr);

    // this makes array in-place
    // assert(static_cast<const void *>(array.request().ptr) == static_cast<const void
    // *>(src.ptr));

    return array.release();
  }
};

template <> struct type_caster<CpuArray> {
  PYBIND11_TYPE_CASTER(CpuArray, _("numpy.ndarray"));

  bool load(py::handle src, bool convert) { return false; }

  static py::handle cast(CpuArray const &src, py::return_value_policy policy, py::handle parent) {

    std::vector<py::ssize_t> shape;
    for (auto s : src.shape) {
      shape.push_back(s);
    }

    auto array = py::array(py::dtype(src.type), shape, src.data.data());
    return array.release();
  }
};

} // namespace pybind11::detail
