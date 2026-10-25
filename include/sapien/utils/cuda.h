#ifdef SAPIEN_CUDA
#pragma once
#include <stdexcept>
#include <string>

#define checkCudaErrors(call)                                                                     \
  do {                                                                                            \
    cudaError_t err = call;                                                                       \
    if (err != cudaSuccess) {                                                                     \
      throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));            \
    }                                                                                             \
  } while (0)

#define checkCudaDriverErrors(call)                                                               \
  do {                                                                                            \
    CUresult err = call;                                                                          \
    if (err != CUDA_SUCCESS) {                                                                    \
      throw std::runtime_error("CUDA failed");                                                    \
    }                                                                                             \
  } while (0)

using cudaStream_t = struct CUstream_st *;
using CUevent = struct CUevent_st *;

namespace sapien {

int getCudaPtrDevice(void *ptr);
int getCudaCurrentDevice();

/** Makes the primary CUDA context of runtime device `cudaId` current on this thread for the
 *  lifetime of the scope by pushing it onto the driver context stack, and pops it on exit.
 *
 *  CUDA runtime calls and kernel launches inside the scope target `cudaId`. Unlike a bare
 *  cudaSetDevice, leaving the scope restores the caller's current device exactly, so frameworks
 *  such as PyTorch keep their current device, and it never creates a context on the caller's
 *  previous device. Entering a scope for the already-current device costs no push. */
class CudaDeviceScope {
public:
  explicit CudaDeviceScope(int cudaId);
  ~CudaDeviceScope();
  CudaDeviceScope(CudaDeviceScope const &) = delete;
  CudaDeviceScope &operator=(CudaDeviceScope const &) = delete;

private:
  bool mPushed{false};
};

struct CudaEvent {
  CudaEvent() {}
  void init();
  CudaEvent(CudaEvent const &) = delete;
  CudaEvent &operator=(CudaEvent const &) = delete;
  CudaEvent(CudaEvent &&other);
  CudaEvent &operator=(CudaEvent &&other);

  /** record cuda event on the current cuda runtime device
   *  @param stream cuda stream, must be created from the same cuda runtime device */
  void record(cudaStream_t stream);

  void wait(cudaStream_t stream) const;

  void synchronize() const;

  ~CudaEvent();

  int cudaId{-1};
  CUevent event{nullptr};
};

}; // namespace sapien
#endif