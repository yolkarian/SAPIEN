#pragma once
#include "svulkan2/common/layout.h"
#include "svulkan2/common/vk.h"
#include "svulkan2/core/as.h"
#include "svulkan2/core/buffer.h"
#include "svulkan2/core/per_context.h"
#include <mutex>

namespace svulkan2 {
namespace resource {

/** Line or point set. Host attributes are uploaded per device (Context); a device whose
 *  buffer is older than the host attributes re-uploads on the next upload. */
class SVPrimitiveSet {

public:
  SVPrimitiveSet(uint32_t capacity = 0);
  void setVertexAttribute(std::string const &name, std::vector<float> const &attrib);
  std::vector<float> const &getVertexAttribute(std::string const &name) const;
  /** this device's vertex buffer, uploaded first when absent or stale */
  core::Buffer &getVertexBuffer(core::Context &context);

  uint32_t getVertexCapacity() const { return mVertexCapacity; }

  size_t getVertexSize();
  uint32_t getVertexCount() const { return mVertexCount; }

  void uploadToDevice(core::Context &context);
  /** release the buffers of every device */
  virtual void removeFromDevice();

  virtual ~SVPrimitiveSet() = default;

protected:
  struct DeviceState {
    std::shared_ptr<core::Context> context; // re-upload target for attribute edits
    std::unique_ptr<core::Buffer> vertexBuffer;
    uint64_t version{}; // mVersion of the uploaded attributes
  };

  uint32_t mVertexCapacity;

  std::unordered_map<std::string, std::vector<float>> mAttributes;
  /** bumped whenever host attributes change */
  uint64_t mVersion{1};

  uint32_t mVertexCount{};
  core::PerContext<DeviceState> mDeviceStates;

  std::mutex mUploadingMutex;
};

class SVLineSet : public SVPrimitiveSet {
public:
  using SVPrimitiveSet::SVPrimitiveSet;
};
class SVPointSet : public SVPrimitiveSet {
public:
  using SVPrimitiveSet::SVPrimitiveSet;
  void buildBLAS(core::Context &context, bool update);
  /** this device's BLAS and AABB buffer, nullptr before buildBLAS on that device */
  core::BLAS *getBLAS(core::Context const &context);
  core::Buffer *getAabbBuffer(core::Context const &context) const;
  void recordUpdateBLAS(core::Context &context, vk::CommandBuffer commandBuffer);
  /** also release the BLAS and AABB buffer of every device */
  void removeFromDevice() override;

private:
  struct RTState {
    std::unique_ptr<core::BLAS> blas;
    std::unique_ptr<core::Buffer> aabbBuffer;
  };
  core::PerContext<RTState> mRTStates;
};

} // namespace resource
} // namespace svulkan2
