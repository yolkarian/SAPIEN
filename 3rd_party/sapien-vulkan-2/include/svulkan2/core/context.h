#pragma once
#include "allocator.h"
#include "command_pool.h"
#include "descriptor_pool.h"
#include "queue.h"
#include "svulkan2/common/vk.h"
#include "svulkan2/renderer/gui.h"
#include <map>
#include <mutex>

namespace svulkan2 {

namespace logger {

void setLogLevel(std::string_view level);
std::string getLogLevel();

}; // namespace logger

namespace core {
#ifdef SAPIEN_MACOS
struct VkSamplerCreateInfoCompare {
  bool operator()(const vk::SamplerCreateInfo& lhs, const vk::SamplerCreateInfo& rhs) const {
    return std::tie(
      lhs.sType, lhs.pNext, lhs.flags, lhs.magFilter, lhs.minFilter, lhs.mipmapMode,
      lhs.addressModeU, lhs.addressModeV, lhs.addressModeW, lhs.mipLodBias, lhs.anisotropyEnable,
      lhs.maxAnisotropy, lhs.compareEnable, lhs.compareOp, lhs.minLod, lhs.maxLod, lhs.borderColor,
      lhs.unnormalizedCoordinates) <
      std::tie(rhs.sType, rhs.pNext, rhs.flags, rhs.magFilter, rhs.minFilter, rhs.mipmapMode,
      rhs.addressModeU, rhs.addressModeV, rhs.addressModeW, rhs.mipLodBias, rhs.anisotropyEnable,
      rhs.maxAnisotropy, rhs.compareEnable, rhs.compareOp, rhs.minLod, rhs.maxLod, rhs.borderColor,
      rhs.unnormalizedCoordinates);
  }
};
#endif
class Instance;
class PhysicalDevice;
class Device;
class Allocator;

/** Vulkan device state of one physical device.
 *
 *  A process holds at most one live Context per physical device; every Context shares the
 *  process-wide Instance. Objects that record or own Vulkan work keep the Context they were
 *  created for; device-independent resources keep per-Context state (see PerContext). */
class Context : public std::enable_shared_from_this<Context> {
public:
  /** Return the live context of the physical device selected by `device` ("", "cuda",
   *  "cuda:N" or "pci:..."), creating it when none is alive. The descriptor pool sizes only
   *  apply when the context is created. */
  static std::shared_ptr<Context> Create(uint32_t maxNumMaterials = 5000,
                                         uint32_t maxNumTextures = 5000, std::string device = "",
                                         bool enableVR = false);

  ~Context();
  inline bool isVulkanAvailable() const { return mInstance && mPhysicalDevice; }

  /** unique per Context object and never reused within the process */
  inline uint64_t getId() const { return mId; }
  /** CUDA ordinal of this device, -1 when it is not a CUDA device */
  int getCudaDeviceId() const;

  bool isPresentAvailable() const;
  bool isRayTracingAvailable() const;

  Queue &getQueue() const;
  Allocator &getAllocator();

  std::unique_ptr<CommandPool> createCommandPool() const;

  uint32_t getGraphicsQueueFamilyIndex() const;

  vk::Instance getInstance() const;
  vk::Device getDevice() const;
  vk::PhysicalDevice getPhysicalDevice() const;

  std::shared_ptr<Device> getDevice2() const { return mDevice; };
  std::shared_ptr<Instance> getInstance2() const { return mInstance; }
  std::shared_ptr<PhysicalDevice> getPhysicalDevice2() const { return mPhysicalDevice; }

  vk::PhysicalDeviceLimits const &getPhysicalDeviceLimits() const;

  inline std::mutex &getGlobalLock() { return mGlobalLock; }

  inline DynamicDescriptorPool &getDescriptorPool() const { return *mDescriptorPool; }

  inline vk::DescriptorSetLayout getMetallicDescriptorSetLayout() const {
    return mMetallicDescriptorSetLayout.get();
  }

  std::unique_ptr<renderer::GuiWindow> createWindow(uint32_t width, uint32_t height);
  vk::UniqueSemaphore createTimelineSemaphore(uint64_t initialValue);

  vk::Sampler createSampler(vk::SamplerCreateInfo const &info);

private:
  uint64_t mId;
  std::shared_ptr<Instance> mInstance;
  std::shared_ptr<PhysicalDevice> mPhysicalDevice;
  std::shared_ptr<Device> mDevice;

  std::mutex mGlobalLock{};
  std::unique_ptr<DynamicDescriptorPool> mDescriptorPool;

  uint32_t mMaxNumMaterials;
  uint32_t mMaxNumTextures;

  vk::UniqueDescriptorSetLayout mMetallicDescriptorSetLayout;

  Context(std::shared_ptr<Instance> instance, std::shared_ptr<PhysicalDevice> physicalDevice,
          uint32_t maxNumMaterials, uint32_t maxNumTextures);

  void createDescriptorPool();

  std::mutex mSamplerLock{};
#ifdef SAPIEN_MACOS
  std::map<vk::SamplerCreateInfo, vk::UniqueSampler, VkSamplerCreateInfoCompare> mSamplerRegistry;
#else
  std::map<vk::SamplerCreateInfo, vk::UniqueSampler> mSamplerRegistry;
#endif
};

} // namespace core
} // namespace svulkan2
