#include "svulkan2/core/descriptor_pool.h"
#include "svulkan2/core/device.h"
#include <utility>

namespace svulkan2 {
namespace core {

DescriptorSet::DescriptorSet(DescriptorSet &&other) noexcept
    : mOwner(std::exchange(other.mOwner, nullptr)), mPool(std::exchange(other.mPool, {})),
      mSet(std::exchange(other.mSet, {})) {}

DescriptorSet &DescriptorSet::operator=(DescriptorSet &&other) noexcept {
  if (this != &other) {
    reset();
    mOwner = std::exchange(other.mOwner, nullptr);
    mPool = std::exchange(other.mPool, {});
    mSet = std::exchange(other.mSet, {});
  }
  return *this;
}

void DescriptorSet::reset() {
  if (mSet) {
    mOwner->free(mPool, mSet);
    mOwner = nullptr;
    mPool = vk::DescriptorPool{};
    mSet = vk::DescriptorSet{};
  }
}

DynamicDescriptorPool::DynamicDescriptorPool(std::shared_ptr<Device> device,
                                             std::vector<vk::DescriptorPoolSize> const &sizes)
    : mDevice(device), mSizes(sizes) {
  expand();
}

void DynamicDescriptorPool::expand() {
  std::vector<vk::DescriptorPoolSize> sizes = mSizes;
  uint32_t count = 0;
  for (auto &size : mSizes) {
    size.setDescriptorCount(size.descriptorCount * (mPools.size() + 1));
    count += size.descriptorCount;
  }
  auto info = vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                                           count, sizes.size(), sizes.data());
  mPools.push_back(mDevice->getInternal().createDescriptorPoolUnique(info));
}

DescriptorSet DynamicDescriptorPool::allocateSet(vk::DescriptorSetLayout const &layout) {
  std::scoped_lock lock(mMutex);
  auto device = mDevice->getInternal();
  for (auto &pool : mPools) {
    try {
      auto set = device.allocateDescriptorSets({pool.get(), 1, &layout}).front();
      return DescriptorSet(this, pool.get(), set);
    } catch (vk::OutOfPoolMemoryError const &) {
    }
  }
  expand();
  auto pool = mPools.back().get();
  return DescriptorSet(this, pool, device.allocateDescriptorSets({pool, 1, &layout}).front());
}

void DynamicDescriptorPool::free(vk::DescriptorPool pool, vk::DescriptorSet set) {
  std::scoped_lock lock(mMutex);
  mDevice->getInternal().freeDescriptorSets(pool, set);
}

} // namespace core
} // namespace svulkan2
