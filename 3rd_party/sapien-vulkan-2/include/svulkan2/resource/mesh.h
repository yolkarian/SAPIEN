#pragma once
#include "svulkan2/common/layout.h"
#include "svulkan2/common/vk.h"
#include "svulkan2/core/buffer.h"
#include "svulkan2/core/per_context.h"
#include <mutex>
#include <unordered_map>

namespace svulkan2 {
namespace resource {
class SVMeshRigid;

/** Triangle mesh. Vertex and index buffers live per device (Context) and are created on
 *  first use there. */
class SVMesh {
public:
  uint32_t getVertexSize() const;
  uint32_t getVertexCount() const { return mVertexCount; }
  uint32_t getTriangleCount() const { return mTriangleCount; }

  bool isOnDevice(core::Context const &context) const;
  /** this device's buffers, uploaded first when absent */
  core::Buffer &getVertexBuffer(core::Context &context);
  core::Buffer &getIndexBuffer(core::Context &context);

  virtual void setIndices(std::vector<uint32_t> const &indices) = 0;
  virtual std::vector<uint32_t> const &getIndices() const = 0;

  virtual void setVertexAttribute(std::string const &name, std::vector<float> const &attrib) = 0;
  virtual std::vector<float> const &getVertexAttribute(std::string const &name) const = 0;
  virtual bool hasVertexAttribute(std::string const &name) const = 0;

  virtual void uploadToDevice(core::Context &context) = 0;
  /** release the buffers of every device */
  virtual void removeFromDevice() = 0;

  vk::AccelerationStructureGeometryKHR getASGeometry(core::Context &context);

  static std::shared_ptr<SVMesh> CreateUVSphere(int segments, int rings);
  static std::shared_ptr<SVMesh> CreateCapsule(float radius, float halfLength, int segments,
                                               int halfRings);
  static std::shared_ptr<SVMesh> CreateCone(int segments);
  static std::shared_ptr<SVMesh> CreateCube();
  static std::shared_ptr<SVMesh> CreateYZPlane();
  static std::shared_ptr<SVMesh> CreateCylinder(int segments);
  static std::shared_ptr<SVMesh> Create(std::vector<float> const &position,
                                        std::vector<uint32_t> const &index);

  virtual ~SVMesh() = default;

protected:
  struct DeviceState {
    std::unique_ptr<core::Buffer> vertexBuffer;
    std::unique_ptr<core::Buffer> indexBuffer;
  };

  size_t mTriangleCount{};
  size_t mVertexCount{};

  core::PerContext<DeviceState> mDeviceStates;
  std::mutex mUploadingMutex;
};

class SVMeshRigid : public SVMesh {
public:
  SVMeshRigid();

  void setIndices(std::vector<uint32_t> const &indices) override;
  std::vector<uint32_t> const &getIndices() const override;

  void setVertexAttribute(std::string const &name, std::vector<float> const &attrib) override;
  std::vector<float> const &getVertexAttribute(std::string const &name) const override;
  bool hasVertexAttribute(std::string const &name) const override;

  void uploadToDevice(core::Context &context) override;
  void removeFromDevice() override;

  uint32_t getTriangleCount();

private:
  std::vector<uint32_t> mIndices;
  std::unordered_map<std::string, std::vector<float>> mAttributes;
};

/** Mesh written in place on the GPU (e.g. by CUDA). It has no host data, so its buffers
 *  exist on the single device it was first uploaded to; uploading it to another device
 *  throws. */
class SVMeshDeformable : public SVMesh {
public:
  SVMeshDeformable(uint32_t maxVertexCount = 0, uint32_t maxTriangleCount = 0);

  uint32_t getMaxVertexCount() { return mMaxVertexCount; }
  uint32_t getMaxTriangleCount() { return mMaxTriangleCount; }

  void setVertexCount(uint32_t vertexCount);
  void setTriangleCount(uint32_t triangleCount);

  void setIndices(std::vector<uint32_t> const &indices) override;
  std::vector<uint32_t> const &getIndices() const override;

  void setVertexAttribute(std::string const &name, std::vector<float> const &attrib) override;
  std::vector<float> const &getVertexAttribute(std::string const &name) const override;
  bool hasVertexAttribute(std::string const &name) const override;

  void uploadToDevice(core::Context &context) override;
  void removeFromDevice() override;

private:
  uint32_t mMaxVertexCount{};
  uint32_t mMaxTriangleCount{};
  /** id of the context holding the buffers; 0 before the first upload */
  uint64_t mContextId{};

  std::vector<uint32_t> mIndices;
  std::unordered_map<std::string, std::vector<float>> mAttributes;
};

} // namespace resource
} // namespace svulkan2
