#include "sapien/device.h"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <svulkan2/core/context.h>
#include <svulkan2/core/instance.h>
#include <svulkan2/core/physical_device.h>

#ifdef SAPIEN_CUDA
#include <cuda_runtime.h>
#endif

namespace sapien {

static std::vector<std::shared_ptr<Device>> vulkanFindDevices() {
  auto level = svulkan2::logger::getLogLevel();
  svulkan2::logger::setLogLevel("off");

  std::vector<std::shared_ptr<Device>> res;
#ifdef _WIN32
  // Win32 extension-name macros require VK_USE_PLATFORM_WIN32_KHR, but this code only needs
  // the platform-independent extension strings.
  constexpr auto externalMemoryExtensionName = "VK_KHR_external_memory_win32";
  constexpr auto externalSemaphoreExtensionName = "VK_KHR_external_semaphore_win32";
#else
  constexpr auto externalMemoryExtensionName = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
  constexpr auto externalSemaphoreExtensionName = VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME;
#endif
  try {
    std::shared_ptr<svulkan2::core::Instance> instance;
    try {
      auto context = svulkan2::core::Context::Get();
      instance = context->getInstance2();
    } catch (std::runtime_error &) {
      instance = std::make_shared<svulkan2::core::Instance>(
          VK_MAKE_VERSION(0, 0, 1), VK_MAKE_VERSION(0, 0, 1), VK_API_VERSION_1_2, false);
    }

    if (instance) {
      auto devices = instance->summarizePhysicalDevices();
      for (auto &d : devices) {
        int priority = 0;

        bool externalMemory = false;
        bool externalSemaphore = false;
        for (auto const &extension : d.device.enumerateDeviceExtensionProperties()) {
          externalMemory |=
              std::strcmp(extension.extensionName, externalMemoryExtensionName) == 0;
          externalSemaphore |=
              std::strcmp(extension.extensionName, externalSemaphoreExtensionName) == 0;
        }

        auto properties =
            d.device
                .getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceIDProperties>();
        std::array<uint8_t, 16> uuid{};
        std::memcpy(uuid.data(),
                    properties.get<vk::PhysicalDeviceIDProperties>().deviceUUID.data(),
                    uuid.size());

        if (d.supported) {
          priority = 1;
          if (d.cudaId >= 0) {
            priority += 1000;
          }
          if (d.present) {
            priority += 100;
          }
          if (d.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
            priority += 10;
          }
          if (d.rayTracing) {
            priority += 1;
          }
        }

        Device::Type type = Device::Type::GPU;
        res.push_back(std::make_shared<Device>(Device{.type = type,
                                                      .name = d.name,
                                                      .render = d.supported,
                                                      .present = d.present,
                                                      .cudaId = d.cudaId,
                                                      .pci = d.pci,
                                                      .uuid = uuid,
                                                      .vulkanExternalMemory = externalMemory,
                                                      .vulkanExternalSemaphore = externalSemaphore,
                                                      .renderPriority = priority}));
      }
    }
  } catch (std::runtime_error &e) {
  }
  svulkan2::logger::setLogLevel(level);
  return res;
}

#ifdef SAPIEN_CUDA

static std::array<uint32_t, 4> parsePCIString(std::string s) {
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

static std::vector<std::shared_ptr<Device>> cudaFindDevices() {
  std::vector<std::shared_ptr<Device>> res;

  int nDevices;
  if (cudaGetDeviceCount(&nDevices) != cudaSuccess) {
    return res;
  }
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) {
      continue;
    }
    char pci[20] = {0};
    if (cudaDeviceGetPCIBusId(pci, 20, i) != cudaSuccess) {
      continue;
    }

    int driverVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    bool externalInterop = prop.unifiedAddressing && driverVersion >= 10000;
    std::array<uint8_t, 16> uuid{};
    std::memcpy(uuid.data(), prop.uuid.bytes, uuid.size());

    res.push_back(std::make_shared<Device>(Device{.type = Device::Type::GPU,
                                                  .name = prop.name,
                                                  .render = false,
                                                  .present = false,
                                                  .cudaId = i,
                                                  .pci = parsePCIString(std::string(pci)),
                                                  .uuid = uuid,
                                                  .cudaExternalMemory = externalInterop,
                                                  .cudaExternalSemaphore = externalInterop}));
  }
  return res;
}
#endif

static std::vector<std::shared_ptr<Device>> findDevices() {
  std::vector<std::shared_ptr<Device>> devices;
  devices.push_back(std::make_shared<Device>(Device{
      .type = Device::Type::CPU, .name = "cpu", .render = false, .present = false, .cudaId = -1}));

  auto gpuDevices = vulkanFindDevices();
#ifdef SAPIEN_CUDA
  auto cudaDevices = cudaFindDevices();
  // Merge CUDA capabilities into the matching Vulkan physical device. A CUDA-only entry is
  // retained only when no Vulkan device has the same PCI identity.
  for (auto const &cd : cudaDevices) {
    bool merged = false;
    for (auto const &vd : gpuDevices) {
      if (cd->pci != vd->pci && vd->cudaId != cd->cudaId) {
        continue;
      }
      vd->cudaId = cd->cudaId;
      vd->cudaExternalMemory = cd->cudaExternalMemory;
      vd->cudaExternalSemaphore = cd->cudaExternalSemaphore;
      if (std::all_of(vd->uuid.begin(), vd->uuid.end(),
                      [](uint8_t value) { return value == 0; })) {
        vd->uuid = cd->uuid;
      }
      merged = true;
      break;
    }
    if (!merged) {
      gpuDevices.push_back(cd);
    }
  }
#endif

  devices.insert(devices.end(), gpuDevices.begin(), gpuDevices.end());
  return devices;
}

std::optional<std::string> Device::getUuidString() const {
  if (isCpu() || std::all_of(uuid.begin(), uuid.end(), [](uint8_t value) { return value == 0; })) {
    return {};
  }
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (auto value : uuid) {
    stream << std::setw(2) << static_cast<int>(value);
  }
  return stream.str();
}

bool Device::canAccessPeer(Device const &peer) const {
#ifdef SAPIEN_CUDA
  if (!isCuda() || !peer.isCuda() || cudaId == peer.cudaId) {
    return false;
  }
  int access = 0;
  return cudaDeviceCanAccessPeer(&access, cudaId, peer.cudaId) == cudaSuccess && access != 0;
#else
  return false;
#endif
}

std::string Device::getAlias() const {
  if (isCpu()) {
    return "cpu";
  }
  if (cudaId >= 0) {
    return "cuda:" + std::to_string(cudaId);
  }
  if (isGpu()) {
    char pciString[20] = {0};
    sprintf(pciString, "%04x:%02x:%02x.%1x", pci[0], pci[1], pci[2], pci[3]);
    return "pci:" + std::string(pciString);
  }
  return "unknown";
}

static std::vector<std::shared_ptr<Device>> gDevices;

std::shared_ptr<Device> findDevice(std::string alias) {
  if (gDevices.empty()) {
    gDevices = findDevices();
  }

  // CPU
  if (alias == "cpu") {
    for (auto d : gDevices) {
      if (d->isCpu()) {
        return d;
      }
    }
    throw std::runtime_error("failed to find device \"" + alias + "\"");
  }

  // CUDA
  if (alias.starts_with("cuda")) {
    // any cuda
    if (alias == "cuda") {
      for (auto d : gDevices) {
        if (d->isCuda()) {
          return d;
        }
      }
    }

    // cuda with id
    if (alias.starts_with("cuda:")) {
      int id = std::stoi(alias.substr(5));
      for (auto d : gDevices) {
        if (d->cudaId == id) {
          return d;
        }
      }
    }

    throw std::runtime_error("failed to find device \"" + alias + "\"");
  }

  // PCI
  if (alias.starts_with("pci:")) {
    std::array<uint32_t, 4> pci = {};
#ifdef SAPIEN_CUDA
    try {
      pci = parsePCIString(alias.substr(4));
      for (auto d : gDevices) {
        if (d->isGpu() && d->pci == pci) {
          return d;
        }
      }
    } catch (std::runtime_error const &) {
    }
#endif
    try {
      uint32_t busId = std::stoi(alias.substr(4), 0, 16);
      for (auto d : gDevices) {
        if (d->isGpu() && d->pci[1] == busId) {
          return d;
        }
      }
    } catch (std::runtime_error const &) {
    }

    throw std::runtime_error("failed to find device \"" + alias + "\"");
  }

  throw std::runtime_error("failed to find device \"" + alias + "\"");
}

std::shared_ptr<Device> findBestRenderDevice() {
  if (gDevices.empty()) {
    gDevices = findDevices();
  }
  std::shared_ptr<Device> bestDevice;
  int priority = 0;
  for (auto d : gDevices) {
    if (d->renderPriority > priority) {
      priority = d->renderPriority;
      bestDevice = d;
    }
  }
  return bestDevice;
}

} // namespace sapien
