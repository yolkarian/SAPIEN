#pragma once
#include "svulkan2/common/vk.h"
#include <memory>
#include <mutex>
#include <vector>

namespace svulkan2 {
namespace core {

class Device;
class DynamicDescriptorPool;

/** A descriptor set allocated from a DynamicDescriptorPool and freed back to it when destroyed
 *  or reset, under the pool's lock. The pool must outlive the set. */
class DescriptorSet {
public:
  DescriptorSet() = default;
  DescriptorSet(DescriptorSet const &) = delete;
  DescriptorSet &operator=(DescriptorSet const &) = delete;
  DescriptorSet(DescriptorSet &&other) noexcept;
  DescriptorSet &operator=(DescriptorSet &&other) noexcept;
  ~DescriptorSet() { reset(); }

  vk::DescriptorSet get() const { return mSet; }
  explicit operator bool() const { return static_cast<bool>(mSet); }
  void reset();

private:
  friend class DynamicDescriptorPool;
  DescriptorSet(DynamicDescriptorPool *owner, vk::DescriptorPool pool, vk::DescriptorSet set)
      : mOwner(owner), mPool(pool), mSet(set) {}

  DynamicDescriptorPool *mOwner{};
  vk::DescriptorPool mPool{};
  vk::DescriptorSet mSet{};
};

/** Descriptor pools that grow on demand.
 *
 *  Allocating and freeing sets is thread-safe: Vulkan requires a descriptor pool to be
 *  externally synchronized, and a Context's pool is shared by every thread rendering on that
 *  device (material uploads, environment-map prefiltering, compute modules). */
class DynamicDescriptorPool {
public:
  DynamicDescriptorPool(std::shared_ptr<Device> device,
                        std::vector<vk::DescriptorPoolSize> const &sizes);
  // sets point back to their pool, so the pool never moves
  DynamicDescriptorPool(DynamicDescriptorPool const &) = delete;
  DynamicDescriptorPool &operator=(DynamicDescriptorPool const &) = delete;

  DescriptorSet allocateSet(vk::DescriptorSetLayout const &layout);

private:
  friend class DescriptorSet;
  void expand();
  void free(vk::DescriptorPool pool, vk::DescriptorSet set);

  std::shared_ptr<Device> mDevice;
  std::vector<vk::DescriptorPoolSize> mSizes;
  std::vector<vk::UniqueDescriptorPool> mPools;
  std::mutex mMutex; // guards mPools, mSizes and the Vulkan pools' allocate/free
};

} // namespace core
} // namespace svulkan2
