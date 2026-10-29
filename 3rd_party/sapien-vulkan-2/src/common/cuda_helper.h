#ifdef SVULKAN2_CUDA_INTEROP
#pragma once

#include "svulkan2/common/vk.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#include <stdexcept>
#include <string>
#ifndef _WIN32
#include <unistd.h>
#endif

// Raises std::runtime_error, which the Python bindings turn into RuntimeError. Destructors must
// catch it: an exception leaving a destructor terminates the process.
#define checkCudaErrors(call)                                                                     \
  do {                                                                                            \
    cudaError_t svulkan2CudaError_ = (call);                                                      \
    if (svulkan2CudaError_ != cudaSuccess) {                                                      \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" +                  \
                               std::to_string(__LINE__) + ": " +                                 \
                               cudaGetErrorString(svulkan2CudaError_));                          \
    }                                                                                             \
  } while (0)

/** Close a file descriptor exported for a CUDA import that failed: CUDA takes ownership only on
 *  success, and an open descriptor keeps the exported Vulkan memory alive. */
inline void closeUnimportedFd(int fd) {
#ifdef _WIN32
  (void)fd;
#else
  ::close(fd);
#endif
}

inline std::array<uint32_t, 4> parsePCIString(std::string s) {
  if (s.length() == 12) {
    return {static_cast<uint32_t>(std::stoi(s.substr(0, 4), 0, 16)),
            static_cast<uint32_t>(std::stoi(s.substr(5, 2), 0, 16)),
            static_cast<uint32_t>(std::stoi(s.substr(8, 2), 0, 16)),
            static_cast<uint32_t>(std::stoi(s.substr(11, 1), 0, 16))};
  }
  if (s.length() == 7) {
    return {0u, static_cast<uint32_t>(std::stoi(s.substr(0, 2), 0, 16)),
            static_cast<uint32_t>(std::stoi(s.substr(3, 2), 0, 16)),
            static_cast<uint32_t>(std::stoi(s.substr(6, 1), 0, 16))};
  }
  throw std::runtime_error("invalid PCI string");
}

inline std::array<uint32_t, 4> getPCIFromCudaDeviceId(int cudaDeviceId) {
  std::string pciBus(20, '\0');
  if (cudaDeviceGetPCIBusId(pciBus.data(), 20, cudaDeviceId) != cudaSuccess) {
    throw std::runtime_error("invalid cuda device id");
  }
  return parsePCIString(std::string(pciBus.c_str()));
}

inline std::string PCIToString(std::array<uint32_t, 4> pci) {
  std::string pciBus(20, '\0');
  sprintf(pciBus.data(), "%04x:%02x:%02x.%1x", pci[0], pci[1], pci[2], pci[3]);
  return std::string(pciBus.c_str());
}

inline int getCudaDeviceIdFromPhysicalDevice(const vk::PhysicalDevice &device) {

  auto props = device.getProperties2<vk::PhysicalDeviceProperties2,
                                     vk::PhysicalDevicePCIBusInfoPropertiesEXT>();
  std::array<uint32_t, 4> pci{
      props.get<vk::PhysicalDevicePCIBusInfoPropertiesEXT>().pciDomain,
      props.get<vk::PhysicalDevicePCIBusInfoPropertiesEXT>().pciBus,
      props.get<vk::PhysicalDevicePCIBusInfoPropertiesEXT>().pciDevice,
      props.get<vk::PhysicalDevicePCIBusInfoPropertiesEXT>().pciFunction,
  };

  int count;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    return -1;
  }

  for (int cudaDeviceId = 0; cudaDeviceId < count; cudaDeviceId++) {
    if (pci == getPCIFromCudaDeviceId(cudaDeviceId))
      return cudaDeviceId;
  }

  return -1;
}

namespace svulkan2 {

/** Make `device` the current CUDA device for this scope, then restore exactly the context
 *  the caller had current (including none). Restoring through cudaSetDevice would create a
 *  context on device 0 for callers that never used CUDA. */
class CudaDeviceScope {
public:
  explicit CudaDeviceScope(int device) {
    checkCudaDriverErrors(api().getCurrent(&mPrevious));
    checkCudaErrors(cudaSetDevice(device));
  }
  ~CudaDeviceScope() { api().setCurrent(mPrevious); }
  CudaDeviceScope(CudaDeviceScope const &) = delete;
  CudaDeviceScope &operator=(CudaDeviceScope const &) = delete;

private:
  // driver functions resolved through the runtime, so svulkan2 does not link libcuda
  struct Api {
    CUresult (*getCurrent)(CUcontext *){};
    CUresult (*setCurrent)(CUcontext){};
  };
  static Api const &api() {
    static Api const api = [] {
      Api api;
      checkCudaErrors(cudaGetDriverEntryPoint(
          "cuCtxGetCurrent", reinterpret_cast<void **>(&api.getCurrent), cudaEnableDefault));
      checkCudaErrors(cudaGetDriverEntryPoint(
          "cuCtxSetCurrent", reinterpret_cast<void **>(&api.setCurrent), cudaEnableDefault));
      return api;
    }();
    return api;
  }
  static void checkCudaDriverErrors(CUresult result) {
    if (result != CUDA_SUCCESS) {
      throw std::runtime_error("CUDA driver error " + std::to_string(static_cast<int>(result)));
    }
  }

  CUcontext mPrevious{};
};

} // namespace svulkan2

#endif
