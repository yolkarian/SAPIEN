#ifdef SAPIEN_CUDA
#include "./cuda_lib.h"
#include <array>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>

#if _WIN64
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace sapien {

template <typename F> static void bindSymbol(F *&function, void *symbol) {
  function = reinterpret_cast<F *>(symbol);
}

CudaLib::CudaLib() {
#if _WIN64
  HMODULE handle = LoadLibrary("nvcuda.dll");
  libcuda = (void *)handle;
  auto load = [handle](char const *name) { return (void *)GetProcAddress(handle, name); };
#else
  libcuda = dlopen("libcuda.so", RTLD_LAZY);
  auto load = [this](char const *name) { return dlsym(libcuda, name); };
#endif
  if (!libcuda) {
    return;
  }

  // Driver-only initialization: unlike cudaFree(0) it creates no context, so loading the
  // library never claims memory on whichever device happens to be current.
  bindSymbol(cuInit, load("cuInit"));
  bindSymbol(cuDeviceGet, load("cuDeviceGet"));
  bindSymbol(cuDevicePrimaryCtxRetain, load("cuDevicePrimaryCtxRetain"));
  bindSymbol(cuCtxGetCurrent, load("cuCtxGetCurrent"));
  bindSymbol(cuCtxPushCurrent, load("cuCtxPushCurrent_v2"));
  bindSymbol(cuCtxPopCurrent, load("cuCtxPopCurrent_v2"));
  bindSymbol(cuStreamGetCtx, load("cuStreamGetCtx"));
  bindSymbol(cuEventCreate, load("cuEventCreate"));
  bindSymbol(cuEventDestroy, load("cuEventDestroy"));
  bindSymbol(cuEventRecord, load("cuEventRecord"));
  bindSymbol(cuStreamWaitEvent, load("cuStreamWaitEvent"));
  bindSymbol(cuEventSynchronize, load("cuEventSynchronize"));
  if (cuInit) {
    cuInit(0);
  }
}

CudaLib &CudaLib::Get() {
  static CudaLib lib;
  if (!lib.libcuda) {
    throw std::runtime_error("failed to load libcuda");
  }
  return lib;
}

static constexpr int kMaxCudaDevices = 64;

CUcontext cudaPrimaryContext(int cudaId) {
  static std::array<std::atomic<CUcontext>, kMaxCudaDevices> contexts{};
  // Leaked so device-scoped frees that run during static destruction can still lock it.
  static std::mutex *mutex = new std::mutex;
  if (cudaId < 0 || cudaId >= kMaxCudaDevices) {
    throw std::runtime_error("invalid CUDA device id " + std::to_string(cudaId));
  }
  if (CUcontext context = contexts[cudaId].load(std::memory_order_acquire)) {
    return context;
  }
  std::lock_guard lock(*mutex);
  if (CUcontext context = contexts[cudaId].load(std::memory_order_relaxed)) {
    return context;
  }
  auto &lib = CudaLib::Get();
  CUdevice device{};
  if (lib.cuDeviceGet(&device, cudaId) != CUDA_SUCCESS) {
    throw std::runtime_error("failed to get CUDA device cuda:" + std::to_string(cudaId));
  }
  CUcontext context{};
  if (lib.cuDevicePrimaryCtxRetain(&context, device) != CUDA_SUCCESS || !context) {
    throw std::runtime_error("failed to retain the primary CUDA context of cuda:" +
                             std::to_string(cudaId));
  }
  contexts[cudaId].store(context, std::memory_order_release);
  return context;
}

} // namespace sapien
#endif
