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

  /** Block until all work submitted to this context's device has completed. Callable from any
   *  thread: it holds the device's queue lock, so concurrent submissions wait for it. Use it
   *  instead of vk::Device::waitIdle, which needs the queue externally synchronized. */
  void waitIdle() const;

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

  /** Number of slots of the textures[] array of a raster scene descriptor set. */
  inline uint32_t getMaxNumTextures() const { return mMaxNumTextures; }
  /** Initial capacity of the materials[] table of a raster scene descriptor set. */
  inline uint32_t getMaxNumMaterials() const { return mMaxNumMaterials; }


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

  Context(std::shared_ptr<Instance> instance, std::shared_ptr<PhysicalDevice> physicalDevice,
          uint32_t maxNumMaterials, uint32_t maxNumTextures);

  void createDescriptorPool();

  std::mutex mSamplerLock{};
  std::map<vk::SamplerCreateInfo, vk::UniqueSampler> mSamplerRegistry;
};

} // namespace core
} // namespace svulkan2
