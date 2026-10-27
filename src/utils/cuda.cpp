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
  // Events only order streams; skipping timestamps makes record/wait/query cheapest.
  checkCudaDriverErrors(CudaLib::Get().cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
}

CudaEvent::CudaEvent(CudaEvent &&other) {
  // copy
  cudaId = other.cudaId;
  event = other.event;

  // destory other
  other.cudaId = -1;
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

bool CudaEvent::query() const {
  if (!event) {
    return true;
  }
  CUresult result = CudaLib::Get().cuEventQuery(event);
  if (result == CUDA_ERROR_NOT_READY) {
    return false;
  }
  checkCudaDriverErrors(result);
  return true;
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

void cudaStreamWaitStream(int cudaId, uintptr_t consumer, uintptr_t producer) {
  auto legacyAsOne = [](uintptr_t stream) { return stream == 0 ? uintptr_t(1) : stream; };
  if (legacyAsOne(consumer) == legacyAsOne(producer)) {
    return;
  }
  CudaDeviceScope scope(cudaId);
  // A recorded event may be destroyed right away: the driver frees it once it completes.
  CudaEvent produced;
  produced.record(reinterpret_cast<cudaStream_t>(producer));
  produced.wait(reinterpret_cast<cudaStream_t>(consumer));
}

} // namespace sapien
#endif
