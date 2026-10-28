#ifdef SAPIEN_CUDA
#pragma once
#include <cstdint>
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

  /** @return whether all work captured by the last record() has completed; true when the event
   *  was never recorded */
  bool query() const;

  ~CudaEvent();

  int cudaId{-1};
  CUevent event{nullptr};
};

/** Make `consumer` wait for all work submitted to `producer` so far, both streams of device
 *  `cudaId`. Streams are raw handles in the DLPack convention: 0 and 1 (CU_STREAM_LEGACY) are the
 *  legacy default stream, 2 is the per-thread default stream. Equal streams need no wait. */
void cudaStreamWaitStream(int cudaId, uintptr_t consumer, uintptr_t producer);

/** Throw std::runtime_error unless `stream` (a raw handle; 0 and the special handles resolve to
 *  this device's primary context) belongs to the primary CUDA context of `cudaId`. `owner` names
 *  the object whose stream is being set in the message, e.g. "PhysxGpuSystem". */
void checkCudaStreamDevice(int cudaId, uintptr_t stream, char const *owner);

}; // namespace sapien
#endif