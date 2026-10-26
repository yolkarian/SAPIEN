#include "svulkan2/resource/mesh.h"
#include "svulkan2/core/context.h"
#include "svulkan2/resource/manager.h"
#include <memory>
#include <stdexcept>

namespace svulkan2 {
namespace resource {

uint32_t SVMesh::getVertexSize() const {
  return SVResourceManager::Get()->getVertexLayout()->getSize();
}

bool SVMesh::isOnDevice(core::Context const &context) const {
  auto state = mDeviceStates.find(context);
  return state && state->vertexBuffer;
}

core::Buffer &SVMesh::getVertexBuffer(core::Context &context) {
  uploadToDevice(context);
  auto state = mDeviceStates.find(context);
  if (!state || !state->vertexBuffer) {
    throw std::runtime_error("failed to get vertex buffer: the mesh is not on this device");
  }
  return *state->vertexBuffer;
}

core::Buffer &SVMesh::getIndexBuffer(core::Context &context) {
  uploadToDevice(context);
  auto state = mDeviceStates.find(context);
  if (!state || !state->indexBuffer) {
    throw std::runtime_error("failed to get index buffer: the mesh is not on this device");
  }
  return *state->indexBuffer;
}

vk::AccelerationStructureGeometryKHR SVMesh::getASGeometry(core::Context &context) {
  auto &vertexBuffer = getVertexBuffer(context);
  auto &indexBuffer = getIndexBuffer(context);

  vk::DeviceAddress vertexAddress =
      context.getDevice().getBufferAddress({vertexBuffer.getVulkanBuffer()});
  vk::DeviceAddress indexAddress =
      context.getDevice().getBufferAddress({indexBuffer.getVulkanBuffer()});

  vk::AccelerationStructureGeometryTrianglesDataKHR trianglesData(
      vk::Format::eR32G32B32Sfloat, vertexAddress, getVertexSize(), mVertexCount,
      vk::IndexType::eUint32, indexAddress, {});

  vk::AccelerationStructureGeometryKHR geom(vk::GeometryTypeKHR::eTriangles, trianglesData,
                                            {}); // TODO: add opaque flag?
  return geom;
}

SVMeshRigid::SVMeshRigid() {}

void SVMeshRigid::setIndices(std::vector<uint32_t> const &indices) {
  bool onDevice = false;
  mDeviceStates.forEach([&](DeviceState const &state) {
    if (state.vertexBuffer) {
      onDevice = true;
    }
  });
  if (onDevice) {
    throw std::runtime_error("set indices failed: mesh already on device");
  }

  if (indices.size() % 3 != 0) {
    throw std::runtime_error("set indices failed: provided number is not a multiple of 3");
  }
  mIndices = indices;
  mTriangleCount = indices.size() / 3;
}

std::vector<uint32_t> const &SVMeshRigid::getIndices() const { return mIndices; }

void SVMeshRigid::setVertexAttribute(std::string const &name, std::vector<float> const &attrib) {
  bool onDevice = false;
  mDeviceStates.forEach([&](DeviceState const &state) {
    if (state.vertexBuffer) {
      onDevice = true;
    }
  });
  if (onDevice) {
    throw std::runtime_error("set vertex attribute failed: mesh already on device");
  }

  mAttributes[name] = attrib;

  if (name == "position") {
    if (attrib.size() % 3 != 0) {
      throw std::runtime_error("failed to set attribute position");
    }
    mVertexCount = attrib.size() / 3;
  }
}

std::vector<float> const &SVMeshRigid::getVertexAttribute(std::string const &name) const {
  if (mAttributes.find(name) == mAttributes.end()) {
    throw std::runtime_error("failed to get vertex attribute: attribute " + name +
                             " does not exist");
  }
  return mAttributes.at(name);
}

bool SVMeshRigid::hasVertexAttribute(std::string const &name) const {
  return mAttributes.find(name) != mAttributes.end();
}

void SVMeshRigid::uploadToDevice(core::Context &context) {
  std::scoped_lock lock(mUploadingMutex);
  auto existing = mDeviceStates.find(context);
  if (existing && existing->vertexBuffer) {
    return;
  }

  auto layout = SVResourceManager::Get()->getVertexLayout();

  if (!mAttributes.contains("position") || mAttributes["position"].size() == 0) {
    throw std::runtime_error("mesh upload failed: no vertex positions");
  }
  if (!mIndices.size()) {
    throw std::runtime_error("mesh upload failed: empty vertex indices");
  }

  size_t vertexSize = layout->getSize();
  size_t bufferSize = vertexSize * mVertexCount;
  size_t indexBufferSize = sizeof(uint32_t) * mTriangleCount * 3;

  std::vector<char> buffer(bufferSize, 0);
  auto elements = layout->getElementsSorted();
  uint32_t offset = 0;
  for (auto &elem : elements) {
    if (mAttributes.find(elem.name) != mAttributes.end()) {
      if (mAttributes[elem.name].size() * sizeof(float) != mVertexCount * elem.getSize()) {
        throw std::runtime_error("vertex attribute " + elem.name + " has incorrect size");
      }
      strided_memcpy(buffer.data() + offset, mAttributes[elem.name].data(), elem.getSize(),
                     mVertexCount, vertexSize);
    }
    offset += elem.getSize();
  }

  vk::BufferUsageFlags deviceAddressFlag =
      context.isRayTracingAvailable()
          ? vk::BufferUsageFlagBits::eShaderDeviceAddress |
                vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                vk::BufferUsageFlagBits::eStorageBuffer
          : vk::BufferUsageFlags{};

  auto &state = mDeviceStates.get(context);
  if (!state.vertexBuffer) {
    state.vertexBuffer = core::Buffer::Create(
        context, bufferSize,
        deviceAddressFlag | vk::BufferUsageFlagBits::eVertexBuffer |
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, VmaAllocationCreateFlags{}, true);
    state.indexBuffer = core::Buffer::Create(
        context, indexBufferSize,
        deviceAddressFlag | vk::BufferUsageFlagBits::eIndexBuffer |
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, VmaAllocationCreateFlags{}, true);
  }
  state.vertexBuffer->upload(buffer.data(), vertexSize * mVertexCount);
  state.indexBuffer->upload<uint32_t>(mIndices);
}

void SVMeshRigid::removeFromDevice() { mDeviceStates.clear(); }

uint32_t SVMeshRigid::getTriangleCount() { return static_cast<uint32_t>(mTriangleCount); }

//==================== Deformable ====================//

SVMeshDeformable::SVMeshDeformable(uint32_t maxVertexCount, uint32_t maxTriangleCount)
    : mMaxVertexCount(maxVertexCount), mMaxTriangleCount(maxTriangleCount) {}

void SVMeshDeformable::setVertexCount(uint32_t vertexCount) {
  if (mMaxVertexCount == 0) {
    mMaxVertexCount = vertexCount;
  }
  if (vertexCount > mMaxVertexCount) {
    throw std::runtime_error("failed to set vertex count: it exceeds max vertex count.");
  }
  mVertexCount = vertexCount;
}

void SVMeshDeformable::setTriangleCount(uint32_t triangleCount) {
  if (mMaxTriangleCount == 0) {
    mMaxTriangleCount = triangleCount;
  }
  if (triangleCount > mMaxTriangleCount) {
    throw std::runtime_error("failed to set triangle count: it exceeds max triangle count.");
  }
  mTriangleCount = triangleCount;
}

void SVMeshDeformable::setIndices(std::vector<uint32_t> const &indices) {
  throw std::runtime_error("not implemented");
}
std::vector<uint32_t> const &SVMeshDeformable::getIndices() const {
  throw std::runtime_error("not implemented");
}

void SVMeshDeformable::setVertexAttribute(std::string const &name,
                                          std::vector<float> const &attrib) {
  throw std::runtime_error("not implemented");
}
std::vector<float> const &SVMeshDeformable::getVertexAttribute(std::string const &name) const {
  throw std::runtime_error("not implemented");
}
bool SVMeshDeformable::hasVertexAttribute(std::string const &name) const {
  throw std::runtime_error("not implemented");
}

void SVMeshDeformable::uploadToDevice(core::Context &context) {
  std::scoped_lock lock(mUploadingMutex);

  if (mContextId != 0 && mContextId != context.getId()) {
    throw std::runtime_error("deformable mesh is bound to another device");
  }

  auto existing = mDeviceStates.find(context);
  if (existing && existing->vertexBuffer) {
    mContextId = context.getId();
    return;
  }

  auto layout = SVResourceManager::Get()->getVertexLayout();
  size_t vertexSize = layout->getSize();

  if (mMaxVertexCount == 0 || mMaxTriangleCount == 0) {
    throw std::runtime_error("failed to upload mesh: vertex or index capacity is 0");
  }

  size_t bufferSize = vertexSize * mMaxVertexCount;
  size_t indexBufferSize = sizeof(uint32_t) * mMaxTriangleCount * 3;

  vk::BufferUsageFlags deviceAddressFlag =
      context.isRayTracingAvailable()
          ? vk::BufferUsageFlagBits::eShaderDeviceAddress |
                vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                vk::BufferUsageFlagBits::eStorageBuffer
          : vk::BufferUsageFlags{};

  auto &state = mDeviceStates.get(context);
  if (!state.vertexBuffer) {
    state.vertexBuffer = core::Buffer::Create(
        context, bufferSize,
        deviceAddressFlag | vk::BufferUsageFlagBits::eVertexBuffer |
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, VmaAllocationCreateFlags{}, true);
    state.indexBuffer = core::Buffer::Create(
        context, indexBufferSize,
        deviceAddressFlag | vk::BufferUsageFlagBits::eIndexBuffer |
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, VmaAllocationCreateFlags{}, true);
  }
  mContextId = context.getId();
}

void SVMeshDeformable::removeFromDevice() {
  mDeviceStates.clear();
  mContextId = 0;
}

} // namespace resource
} // namespace svulkan2
