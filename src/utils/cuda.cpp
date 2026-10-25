#ifdef SAPIEN_CUDA
#include "sapien/utils/cuda.h"

#include "./cuda_lib.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace sapien {

CudaDeviceScope::CudaDeviceScope(int cudaId) {
  auto &lib = CudaLib::Get();
  CUcontext target = cudaPrimaryContext(cudaId);
  CUcontext current{};
  checkCudaDriverErrors(lib.cuCtxGetCurrent(&current));
  if (current == target) {
    return;
  }
  checkCudaDriverErrors(lib.cuCtxPushCurrent(target));
  mPushed = true;
}

CudaDeviceScope::~CudaDeviceScope() {
  if (mPushed) {
    CUcontext popped{};
    CudaLib::Get().cuCtxPopCurrent(&popped);
  }
}

void CudaEvent::init() {
  checkCudaErrors(cudaGetDevice(&cudaId));
  checkCudaDriverErrors(CudaLib::Get().cuEventCreate(&event, 0));
}

CudaEvent::CudaEvent(CudaEvent &&other) {
  // copy
  cudaId = other.cudaId;
  event = other.event;

  // destory other
  other.event = nullptr;
}

CudaEvent &CudaEvent::operator=(CudaEvent &&other) {
  // destroy self
  if (event) {
    CudaDeviceScope scope(cudaId);
    checkCudaDriverErrors(CudaLib::Get().cuEventDestroy(event));
  }

  // copy
  cudaId = other.cudaId;
  event = other.event;

  // destroy other
  other.event = nullptr;

  return *this;
}

/** record cuda event on the current cuda runtime device
 *  @param stream cuda stream, must be created from the same cuda runtime device */
void CudaEvent::record(cudaStream_t stream) {
  if (!event) {
    init();
  }
  checkCudaDriverErrors(CudaLib::Get().cuEventRecord(event, stream));
}

void CudaEvent::wait(cudaStream_t stream) const {
  if (!event) {
    // no need to wait if no one records to this event
    return;
  }
  checkCudaDriverErrors(CudaLib::Get().cuStreamWaitEvent(stream, event, 0));
}

void CudaEvent::synchronize() const {
  if (!event) {
    // no need to wait if no one records to this event
    return;
  }
  checkCudaDriverErrors(CudaLib::Get().cuEventSynchronize(event));
}

CudaEvent::~CudaEvent() {
  if (event) {
    try {
      CudaDeviceScope scope(cudaId);
      CudaLib::Get().cuEventDestroy(event);
    } catch (...) {
    }
  }
}

int getCudaCurrentDevice() {
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  return device;
}

int getCudaPtrDevice(void *ptr) {
  cudaPointerAttributes attr{};
  checkCudaErrors(cudaPointerGetAttributes(&attr, ptr));
  if (!attr.devicePointer) {
    throw std::runtime_error("invalid cuda pointer");
  }
  return attr.device;
}

} // namespace sapien
#endif
