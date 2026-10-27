#ifdef SAPIEN_CUDA
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

namespace sapien {

struct CudaLib {
  static CudaLib &Get();
  CudaLib();
  void *libcuda{};

  decltype(::cuInit) *cuInit{};
  decltype(::cuDeviceGet) *cuDeviceGet{};
  decltype(::cuDevicePrimaryCtxRetain) *cuDevicePrimaryCtxRetain{};
  decltype(::cuCtxGetCurrent) *cuCtxGetCurrent{};
  decltype(::cuCtxPushCurrent) *cuCtxPushCurrent{};
  decltype(::cuCtxPopCurrent) *cuCtxPopCurrent{};
  CUresult (*cuStreamGetCtx)(CUstream, CUcontext *){};
  decltype(::cuEventCreate) *cuEventCreate{};
  decltype(::cuEventDestroy) *cuEventDestroy{};
  decltype(::cuEventRecord) *cuEventRecord{};
  decltype(::cuStreamWaitEvent) *cuStreamWaitEvent{};
  decltype(::cuEventSynchronize) *cuEventSynchronize{};
  decltype(::cuEventQuery) *cuEventQuery{};
};

/** Primary context of CUDA runtime device `cudaId`: the context the CUDA runtime, PyTorch, CuPy
 *  and JAX share on that device. Retained once and kept for the process lifetime, matching the
 *  runtime's own retain; SAPIEN never resets a device. Creates the context on first use. */
CUcontext cudaPrimaryContext(int cudaId);

} // namespace sapien
#endif
