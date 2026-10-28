#include "svulkan2/resource/primitive_set.h"
#include "svulkan2/core/context.h"
#include "svulkan2/resource/manager.h"
#include <stdexcept>

namespace svulkan2 {
namespace resource {

SVPrimitiveSet::SVPrimitiveSet(uint32_t capacity) : mVertexCapacity(capacity) {}

void SVPrimitiveSet::setVertexAttribute(std::string const &name,
                                        std::vector<float> const &attrib) {
  mAttributes[name] = attrib;
  if (name == "position") {
    mVertexCount = attrib.size() / 3;
  }
  ++mVersion;

  // Edits reach every device already holding the set: recorded draws bind its buffer
  // directly. Collect first, uploading inside forEach would deadlock on the PerContext lock.
  std::vector<std::shared_ptr<core::Context>> contexts;
  mDeviceStates.forEach([&](DeviceState const &state) {
    if (state.context) {
      contexts.push_back(state.context);
    }
  });
  for (auto const &context : contexts) {
    uploadToDevice(*context);
  }
}

std::vector<float> const &SVPrimitiveSet::getVertexAttribute(std::string const &name) const {
  if (mAttributes.find(name) == mAttributes.end()) {
    throw std::runtime_error("attribute " + name + " does not exist on vertex");
  }
  return mAttributes.at(name);
}

size_t SVPrimitiveSet::getVertexSize() {
  return SVResourceManager::Get()->getLineVertexLayout()->getSize();
}

core::Buffer &SVPrimitiveSet::getVertexBuffer(core::Context &context) {
  uploadToDevice(context);
  auto state = mDeviceStates.find(context);
  if (!state || !state->vertexBuffer) {
    throw std::runtime_error(
        "failed to get vertex buffer: the primitive set is not on this device");
  }
  return *state->vertexBuffer;
}

void SVPrimitiveSet::uploadToDevice(core::Context &context) {
  std::scoped_lock lock(mUploadingMutex);

  auto layout = SVResourceManager::Get()->getLineVertexLayout();

  if (mAttributes.find("position") == mAttributes.end() || mAttributes["position"].size() == 0) {
    throw std::runtime_error("primitive set upload failed: no vertex positions");
  }

  size_t vertexSize = layout->getSize();

  if (mVertexCapacity == 0) {
    mVertexCapacity = mVertexCount;
  }

  if (mVertexCapacity < mVertexCount) {
    throw std::runtime_error("failed to upload primitive set: vertex count exceeds capacity");
  }

  auto &state = mDeviceStates.get(context);
  if (state.vertexBuffer && state.version == mVersion) {
    return;
  }

  size_t bufferSize = vertexSize * mVertexCapacity;

  std::vector<char> buffer(bufferSize, 0);
  auto elements = layout->getElementsSorted();
  uint32_t offset = 0;
  for (auto &elem : elements) {
    if (mAttributes.find(elem.name) != mAttributes.end()) {
      if (mAttributes[elem.name].size() * sizeof(float) > mVertexCapacity * elem.getSize()) {
        throw std::runtime_error("vertex attribute " + elem.name + " has incorrect size");
      }
      auto count = mAttributes[elem.name].size() * sizeof(float) / elem.getSize();
      strided_memcpy(buffer.data() + offset, mAttributes[elem.name].data(), elem.getSize(), count,
                     vertexSize);
    }
    offset += elem.getSize();
  }

  if (!state.vertexBuffer || state.vertexBuffer->getSize() != bufferSize) {
    state.vertexBuffer = core::Buffer::Create(
        context, bufferSize,
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, VmaAllocationCreateFlags{}, true);
    state.context = context.shared_from_this();
  }
  state.vertexBuffer->upload(buffer.data(), bufferSize);
  state.version = mVersion;
}

void SVPrimitiveSet::removeFromDevice() { mDeviceStates.clear(); }

//==================== Point Set ====================//

void SVPointSet::removeFromDevice() {
  mRTStates.clear();
  SVPrimitiveSet::removeFromDevice();
}

void SVPointSet::buildBLAS(core::Context &context, bool update) {
  if (getVertexCapacity() == 0) {
    throw std::runtime_error("failed to build BLAS: unspecified vertex capacity");
  }
  struct Aabb {
    glm::vec3 min;
    glm::vec3 max;
  };

  auto &state = mRTStates.get(context);

  if (!state.aabbBuffer) {
    state.aabbBuffer = core::Buffer::Create(
        context, getVertexCapacity() * sizeof(Aabb),
        vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
            vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, VmaAllocationCreateFlags{}, true);
  }

  std::vector<Aabb> aabbs;
  auto position = mAttributes.at("position");
  auto scale = mAttributes.at("scale");
  int size = position.size() / 3;
  aabbs.reserve(size);
  for (int i = 0; i < size; ++i) {
    glm::vec3 center = glm::vec3(position[3 * i], position[3 * i + 1], position[3 * i + 2]);
    aabbs.push_back({center - glm::vec3(scale.at(i)),
                     center + glm::vec3(scale.at(i))}); // TODO use a proper scale
  }
  state.aabbBuffer->upload(aabbs);

  static_assert(sizeof(Aabb) == sizeof(float) * 6);

  vk::AccelerationStructureGeometryAabbsDataKHR data(state.aabbBuffer->getAddress(), sizeof(Aabb));
  vk::AccelerationStructureGeometryKHR geom(vk::GeometryTypeKHR::eAabbs, data,
                                            vk::GeometryFlagBitsKHR::eOpaque);
  vk::AccelerationStructureBuildRangeInfoKHR range(getVertexCount(), 0, 0, 0);
  state.blas =
      std::make_unique<core::BLAS>(context, std::vector{geom}, std::vector{range},
                                   std::vector<uint32_t>{getVertexCapacity()}, false, update);
  state.blas->build();
}

core::BLAS *SVPointSet::getBLAS(core::Context const &context) {
  auto state = mRTStates.find(context);
  return state && state->blas ? state->blas.get() : nullptr;
}

core::Buffer *SVPointSet::getAabbBuffer(core::Context const &context) const {
  auto state = mRTStates.find(context);
  return state && state->aabbBuffer ? state->aabbBuffer.get() : nullptr;
}

void SVPointSet::recordUpdateBLAS(core::Context &context, vk::CommandBuffer commandBuffer) {
  auto state = mRTStates.find(context);
  if (!state || !state->blas) {
    throw std::runtime_error("failed to update BLAS: the point set has no BLAS on this device");
  }
  vk::AccelerationStructureBuildRangeInfoKHR range(getVertexCount(), 0, 0, 0);
  state->blas->recordUpdate(commandBuffer, {range});
}

} // namespace resource
} // namespace svulkan2
