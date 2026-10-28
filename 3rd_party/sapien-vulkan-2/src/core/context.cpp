#include "svulkan2/core/context.h"
#include "../common/cuda_helper.h"
#include "../common/logger.h"
#include "svulkan2/common/launch_policy.h"
#include "svulkan2/common/profiler.h"
#include "svulkan2/core/allocator.h"
#include "svulkan2/core/device.h"
#include "svulkan2/core/instance.h"
#include "svulkan2/core/physical_device.h"
#include "svulkan2/shader/glsl_compiler.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <atomic>
#include <iomanip>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace svulkan2 {
namespace core {

// live contexts, at most one per physical device. Recursive: a strong reference taken
// while scanning may be the last one, so ~Context can run with the lock already held.
static std::recursive_mutex gContextLock;
static std::vector<std::weak_ptr<Context>> gContexts;
static std::atomic<uint64_t> gNextContextId{1};

std::shared_ptr<Context> Context::Create(uint32_t maxNumMaterials, uint32_t maxNumTextures,
                                         std::string device, bool enableVR) {
  auto instance = Instance::Get(enableVR);
  auto physicalDevice = instance->createPhysicalDevice(device);

  std::scoped_lock lock(gContextLock);
  std::erase_if(gContexts, [](auto const &context) { return context.expired(); });
  for (auto const &weak : gContexts) {
    auto context = weak.lock();
    if (context && context->getPhysicalDevice() == physicalDevice->getInternal()) {
      return context;
    }
  }
  auto context = std::shared_ptr<Context>(new Context(
      std::move(instance), std::move(physicalDevice), maxNumMaterials, maxNumTextures));
  gContexts.push_back(context);
  return context;
}

// called with gContextLock held
Context::Context(std::shared_ptr<Instance> instance,
                 std::shared_ptr<PhysicalDevice> physicalDevice, uint32_t maxNumMaterials,
                 uint32_t maxNumTextures)
    : mId(gNextContextId++), mInstance(std::move(instance)),
      mPhysicalDevice(std::move(physicalDevice)), mMaxNumMaterials(maxNumMaterials),
      mMaxNumTextures(maxNumTextures) {
  mDevice = mPhysicalDevice->createDevice();
  createDescriptorPool();
  GLSLCompiler::InitializeProcess();
}

Context::~Context() {
  waitIdle();

  std::scoped_lock lock(gContextLock);
  GLSLCompiler::FinalizeProcess();
  // VR must shut down before the last device using it is destroyed
  if (std::none_of(gContexts.begin(), gContexts.end(),
                   [](auto const &context) { return !context.expired(); })) {
    mInstance->shutdownVR();
  }
  logger::info("Vulkan device finished");
}

int Context::getCudaDeviceId() const { return mPhysicalDevice->getPickedDeviceInfo().cudaId; }

void Context::createDescriptorPool() {
  if (!isVulkanAvailable()) {
    return;
  }

  mDescriptorPool = std::make_unique<DynamicDescriptorPool>(
      mDevice,
      std::vector{
          vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, mMaxNumTextures},
          vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, mMaxNumMaterials},
          vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 10}});

  {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    bindings.push_back(vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                                                      vk::ShaderStageFlagBits::eFragment));
    for (uint32_t i = 0; i < 6; ++i) {
      // 6 textures
      bindings.push_back(vk::DescriptorSetLayoutBinding(i + 1,
                                                        vk::DescriptorType::eCombinedImageSampler,
                                                        1, vk::ShaderStageFlagBits::eFragment));
    }
    mMetallicDescriptorSetLayout = getDevice().createDescriptorSetLayoutUnique(
        vk::DescriptorSetLayoutCreateInfo({}, bindings.size(), bindings.data()));
  }
}

std::unique_ptr<CommandPool> Context::createCommandPool() const {
  SVULKAN2_PROFILE_FUNCTION;
  return mDevice->createCommandPool();
}

vk::UniqueSemaphore Context::createTimelineSemaphore(uint64_t initialValue = 0) {
  vk::SemaphoreTypeCreateInfo timelineCreateInfo(vk::SemaphoreType::eTimeline, initialValue);
  vk::SemaphoreCreateInfo createInfo{};
  createInfo.setPNext(&timelineCreateInfo);
  return getDevice().createSemaphoreUnique(createInfo);
}

vk::Sampler Context::createSampler(vk::SamplerCreateInfo const &info) {
  std::lock_guard<std::mutex> lock(mSamplerLock);
  auto it = mSamplerRegistry.find(info);
  if (it != mSamplerRegistry.end()) {
    return it->second.get();
  }
  auto samplerUnique = getDevice().createSamplerUnique(info);
  auto sampler = samplerUnique.get();
  mSamplerRegistry[info] = std::move(samplerUnique);
  return sampler;
}

std::unique_ptr<renderer::GuiWindow> Context::createWindow(uint32_t width, uint32_t height) {
  if (!isVulkanAvailable()) {
    throw std::runtime_error("Vulkan is not initialized");
  }

  if (!isPresentAvailable()) {
    throw std::runtime_error("Create window failed: Renderer does not support display.");
  }
  if (width == 0 || height == 0) {
    throw std::runtime_error("Create window failed: width and height must be positive.");
  }
  return std::make_unique<renderer::GuiWindow>(
      shared_from_this(),
      std::vector<vk::Format>{vk::Format::eB8G8R8A8Unorm, vk::Format::eR8G8B8A8Unorm,
                              vk::Format::eB8G8R8Unorm, vk::Format::eR8G8B8Unorm},
      vk::ColorSpaceKHR::eSrgbNonlinear, width, height,
      std::vector<vk::PresentModeKHR>{vk::PresentModeKHR::eMailbox}, 2);
};

vk::Instance Context::getInstance() const {
  if (!mInstance) {
    return {};
  }
  return mInstance->getInternal();
}
bool Context::isPresentAvailable() const {
  if (!mInstance || !mPhysicalDevice) {
    return false;
  }
  return mInstance->isGLFWEnabled() && mPhysicalDevice->getPickedDeviceInfo().present;
}
bool Context::isRayTracingAvailable() const {
  if (!mPhysicalDevice) {
    return false;
  }
  return mPhysicalDevice->getPickedDeviceInfo().rayTracing;
}
uint32_t Context::getGraphicsQueueFamilyIndex() const {
  if (!mPhysicalDevice) {
    return -1;
  }
  return mPhysicalDevice->getPickedDeviceInfo().queueIndex;
}
vk::PhysicalDevice Context::getPhysicalDevice() const {
  if (!mPhysicalDevice) {
    return {};
  }
  return mPhysicalDevice->getPickedDeviceInfo().device;
}

vk::PhysicalDeviceLimits const &Context::getPhysicalDeviceLimits() const {
  if (!mPhysicalDevice) {
    throw std::runtime_error("invalid physical device");
  }
  return mPhysicalDevice->getPickedDeviceLimits();
}
vk::Device Context::getDevice() const { return mDevice->getInternal(); }
Queue &Context::getQueue() const { return mDevice->getQueue(); }
Allocator &Context::getAllocator() { return mDevice->getAllocator(); }

void Context::waitIdle() const {
  // vkDeviceWaitIdle needs every queue of the device externally synchronized; Device creates
  // exactly one queue and all submissions go through core::Queue.
  auto lock = mDevice->getQueue().lock();
  mDevice->getInternal().waitIdle();
}

} // namespace core
} // namespace svulkan2
