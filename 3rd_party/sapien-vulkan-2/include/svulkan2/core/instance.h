#pragma once
#include "svulkan2/common/vk.h"
#include <memory>

namespace svulkan2 {
namespace core {

class PhysicalDevice;

struct PhysicalDeviceInfo {
  std::string name;
  vk::PhysicalDevice device{};
  bool present{};
  bool supported{};
  int cudaId{-1};
  std::array<uint32_t, 4> pci;
  int queueIndex{-1};
  bool rayTracing{};
  int cudaComputeMode{-1};
  vk::PhysicalDeviceType deviceType{};
  uint32_t subgroupSize{0};

  struct Features {
    bool wideLines{false};
    bool geometryShader{false};
  } features;
};

class Instance : public std::enable_shared_from_this<Instance> {
public:
  /** The process-wide instance shared by every Context, created on first call.
   *  `enableVR` only takes effect when this call creates the instance. Only one instance
   *  may exist at a time because it initializes the global Vulkan dispatcher. */
  static std::shared_ptr<Instance> Get(bool enableVR = false);
  ~Instance();

  inline bool isGLFWEnabled() const { return mGLFWSupported; }
  vk::Instance getInternal() const { return mInstance.get(); }
  inline uint32_t getApiVersion() const { return mApiVersion; }

  auto getProcAddr(char const *name) const { return mInstance->getProcAddr(name); }

  /** enumerate physical devices again (creates a hidden test window for present support) */
  std::vector<PhysicalDeviceInfo> summarizePhysicalDevices() const;
  /** physical devices enumerated when the instance was created */
  inline std::vector<PhysicalDeviceInfo> const &getPhysicalDeviceInfo() const {
    return mPhysicalDeviceInfo;
  }
  std::shared_ptr<PhysicalDevice> createPhysicalDevice(std::string const &hint);

  // HACK: shutdown VR should be called before device shutdown
  void shutdownVR() const;

  Instance(Instance const &other) = delete;
  Instance(Instance const &&other) = delete;
  Instance &operator=(Instance const &other) = delete;
  Instance &operator=(Instance const &&other) = delete;

private:
  Instance(uint32_t appVersion, uint32_t engineVersion, uint32_t apiVersion, bool enableVR);

  uint32_t mApiVersion;
  std::unique_ptr<vk::DynamicLoader> mDynamicLoader;
  vk::UniqueInstance mInstance{};
  bool mGLFWSupported{};
  bool mVRRequested{};
  bool mVRSupported{};
  std::vector<PhysicalDeviceInfo> mPhysicalDeviceInfo;

#ifdef VK_VALIDATION
  vk::DebugReportCallbackEXT mDebugCallbackHandle;
#endif
};

} // namespace core
} // namespace svulkan2
