#include "svulkan2/core/allocator.h"
#include "svulkan2/core/device.h"
#include "svulkan2/core/instance.h"
#include "svulkan2/core/physical_device.h"
#include <bit>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wunused-variable"
#define VMA_BUFFER_DEVICE_ADDRESS 1
#define VMA_EXTERNAL_MEMORY 1
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_IMPLEMENTATION
#include "svulkan2/common/vk_mem_alloc.h"
#pragma GCC diagnostic pop

namespace svulkan2 {
namespace core {

Allocator::Allocator(Device &device) {
  VmaAllocatorCreateInfo allocatorInfo = {};

  auto physicalDevice = device.getPhysicalDevice();
  auto instance = physicalDevice->getInstance();

  allocatorInfo.vulkanApiVersion = instance->getApiVersion();
  allocatorInfo.instance = instance->getInternal();
  allocatorInfo.physicalDevice = physicalDevice->getInternal();
  allocatorInfo.device = device.getInternal();

  // VMA fetches every other function through vkGetDeviceProcAddr for this device. The global
  // dispatcher holds instance-level loader trampolines, whose promoted KHR aliases (e.g.
  // vkGetImageMemoryRequirements2KHR) are null on devices that did not enable the extension.
  VmaVulkanFunctions vulkanFunctions{};
  vulkanFunctions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
  vulkanFunctions.vkGetDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
  allocatorInfo.pVulkanFunctions = &vulkanFunctions;

  if (physicalDevice->getPickedDeviceInfo().rayTracing) {
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  }

  if (vmaCreateAllocator(&allocatorInfo, &mMemoryAllocator) != VK_SUCCESS) {
    throw std::runtime_error("failed to create VmaAllocator");
  }

  // find GPU memory suitable for external buffer
  // VkPhysicalDeviceMemoryProperties properties;
  vk::PhysicalDeviceMemoryProperties properties;
  physicalDevice->getInternal().getMemoryProperties(&properties);

  vk::BufferCreateInfo bufferInfo({}, 64,
                                  vk::BufferUsageFlagBits::eVertexBuffer |
                                      vk::BufferUsageFlagBits::eIndexBuffer |
                                      vk::BufferUsageFlagBits::eTransferDst);
#if !defined(VK_USE_PLATFORM_MACOS_MVK)
  vk::ExternalMemoryBufferCreateInfo externalMemoryBufferInfo(
      vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);
  bufferInfo.setPNext(&externalMemoryBufferInfo);
#endif

  vk::MemoryRequirements memReq;
  {
    auto buffer = device.getInternal().createBufferUnique(bufferInfo);
    memReq = device.getInternal().getBufferMemoryRequirements(buffer.get());
  }

  int index = -1;
  for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
    if ((memReq.memoryTypeBits & (1 << i)) &&
        (properties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
      index = i;
      break;
    }
  }
  if (index == -1) {
    throw std::runtime_error("Failed to find a suitable memory type for external memory pool");
  }

  {
    VmaPoolCreateInfo poolInfo{};
    poolInfo.memoryTypeIndex = static_cast<uint32_t>(index);
    // auto limits = physicalDevice->getInternal().getProperties().limits;
    // poolInfo.minAllocationAlignment = std::max(
    //     std::max(limits.minStorageBufferOffsetAlignment, limits.minTexelBufferOffsetAlignment),
    //     limits.minUniformBufferOffsetAlignment);
#if !defined(VK_USE_PLATFORM_MACOS_MVK)
    mExternalAllocInfo =
        vk::ExportMemoryAllocateInfo(vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);
    poolInfo.pMemoryAllocateNext = &mExternalAllocInfo;
#endif
    vmaCreatePool(mMemoryAllocator, &poolInfo, &mExternalMemoryPool);
  }

  if (physicalDevice->getPickedDeviceInfo().rayTracing) {
    auto rtprops = physicalDevice->getRayTracingProperties();
    vk::DeviceSize sbtAlign = rtprops.shaderGroupBaseAlignment;
    auto asprops = physicalDevice->getASProperties();
    vk::DeviceSize asAlign = asprops.minAccelerationStructureScratchOffsetAlignment;
    vk::DeviceSize alignment = std::bit_ceil(std::max(sbtAlign, asAlign));

    VmaPoolCreateInfo poolInfo{};
    poolInfo.memoryTypeIndex = static_cast<uint32_t>(index);
    poolInfo.minAllocationAlignment = alignment;
    vmaCreatePool(mMemoryAllocator, &poolInfo, &mRTPool);
  }
}

Allocator::~Allocator() {
  if (mRTPool) {
    vmaDestroyPool(mMemoryAllocator, mRTPool);
  }
  if (mExternalMemoryPool) {
    vmaDestroyPool(mMemoryAllocator, mExternalMemoryPool);
  }
  vmaDestroyAllocator(mMemoryAllocator);
}

} // namespace core
} // namespace svulkan2
