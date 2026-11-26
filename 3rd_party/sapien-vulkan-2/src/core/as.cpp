#include "svulkan2/core/as.h"
#include "../common/logger.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace core {

namespace {

// Scratch memory of one build submission. Builds are recorded together until their scratch
// reaches this size; more at once would only hold more memory without saving submissions.
constexpr vk::DeviceSize BUILD_SCRATCH_BUDGET = vk::DeviceSize(256) << 20;

std::unique_ptr<Buffer> createScratchBuffer(Context &context, vk::DeviceSize size) {
  return Buffer::Create(context, size,
                        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                            vk::BufferUsageFlagBits::eShaderDeviceAddress |
                            vk::BufferUsageFlagBits::eStorageBuffer,
                        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, VmaAllocationCreateFlags{},
                        false, context.getAllocator().getRTPool());
}

/** Split `count` builds into submissions by scratch size; `scratchSize(i)` prepares build `i`
 *  and returns its scratch size, `submit(begin, end)` records and submits builds [begin, end). */
template <typename Prepare, typename Submit>
void forEachBuildBatch(size_t count, Prepare scratchSize, Submit submit) {
  size_t begin = 0;
  while (begin < count) {
    vk::DeviceSize total = 0;
    size_t end = begin;
    while (end < count && (end == begin || total < BUILD_SCRATCH_BUDGET)) {
      total += scratchSize(end++);
    }
    submit(begin, end);
    begin = end;
  }
}

} // namespace

BLAS::BLAS(Context &context, std::vector<vk::AccelerationStructureGeometryKHR> const &geometries,
           std::vector<vk::AccelerationStructureBuildRangeInfoKHR> const &buildRanges,
           std::vector<uint32_t> const &maxPrimitiveCounts, bool update)
    : mContext(context.shared_from_this()), mGeometries(geometries), mBuildRanges(buildRanges),
      mUpdate(update) {
  if (maxPrimitiveCounts.empty()) {
    for (auto &range : mBuildRanges) {
      mMaxPrimitiveCounts.push_back(range.primitiveCount);
    }
  } else {
    // sanity checks
    if (maxPrimitiveCounts.size() != buildRanges.size()) {
      throw std::runtime_error("buildRanges and maxPrimitiveCounts must have equal size");
    }
    for (size_t i = 0; i < maxPrimitiveCounts.size(); ++i) {
      if (maxPrimitiveCounts[i] < buildRanges[i].primitiveCount) {
        throw std::runtime_error(
            "maxPrimitiveCounts must be greater or equal to primitive count in buildRanges");
      }
    }
    mMaxPrimitiveCounts = maxPrimitiveCounts;
  }
}

vk::AccelerationStructureBuildGeometryInfoKHR
BLAS::buildInfo(vk::BuildAccelerationStructureModeKHR mode) const {
  vk::BuildAccelerationStructureFlagsKHR flags{};
  if (mUpdate) {
    flags |= vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
  }
  return vk::AccelerationStructureBuildGeometryInfoKHR(
      vk::AccelerationStructureTypeKHR::eBottomLevel, flags, mode, nullptr, nullptr, mGeometries,
      nullptr, vk::DeviceOrHostAddressKHR{});
}

vk::DeviceSize BLAS::createStorage() {
  auto device = mContext->getDevice();
  // the sizes do not depend on the build mode: one query gives the update scratch size too
  auto sizes = device.getAccelerationStructureBuildSizesKHR(
      vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo(vk::BuildAccelerationStructureModeKHR::eBuild),
      mMaxPrimitiveCounts);
  mBuffer = Buffer::Create(*mContext, sizes.accelerationStructureSize,
                           vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                               vk::BufferUsageFlagBits::eShaderDeviceAddress,
                           VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
  mAS = device.createAccelerationStructureKHRUnique(vk::AccelerationStructureCreateInfoKHR(
      {}, mBuffer->getVulkanBuffer(), {}, sizes.accelerationStructureSize,
      vk::AccelerationStructureTypeKHR::eBottomLevel, {}));
  if (mUpdate) {
    mUpdateScratchBuffer = createScratchBuffer(*mContext, sizes.updateScratchSize);
    mUpdateScratchBufferAddress = mUpdateScratchBuffer->getAddress();
  }
  return sizes.buildScratchSize;
}

void BLAS::build() { BuildAll(*mContext, {this}); }

void BLAS::BuildAll(Context &context, std::vector<BLAS *> const &blases) {
  auto commandPool = context.createCommandPool();
  std::vector<std::unique_ptr<Buffer>> scratch(blases.size());
  forEachBuildBatch(
      blases.size(),
      [&](size_t i) {
        vk::DeviceSize size = blases[i]->createStorage();
        scratch[i] = createScratchBuffer(context, size);
        return size;
      },
      [&](size_t begin, size_t end) {
        // independent structures with their own scratch: one build command, no barriers
        std::vector<vk::AccelerationStructureBuildGeometryInfoKHR> infos;
        std::vector<vk::AccelerationStructureBuildRangeInfoKHR const *> ranges;
        for (size_t i = begin; i < end; ++i) {
          auto info = blases[i]->buildInfo(vk::BuildAccelerationStructureModeKHR::eBuild);
          info.setDstAccelerationStructure(blases[i]->mAS.get());
          info.scratchData.setDeviceAddress(scratch[i]->getAddress());
          infos.push_back(info);
          ranges.push_back(blases[i]->mBuildRanges.data());
        }
        auto commandBuffer = commandPool->allocateCommandBuffer();
        commandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        commandBuffer->buildAccelerationStructuresKHR(infos, ranges);
        commandBuffer->end();
        context.getQueue().submitAndWait(commandBuffer.get());
        for (size_t i = begin; i < end; ++i) {
          scratch[i].reset();
        }
      });
}

void BLAS::recordUpdate(
    vk::CommandBuffer commandBuffer,
    std::vector<vk::AccelerationStructureBuildRangeInfoKHR> const &buildRanges) {
  if (!mUpdate) {
    logger::error("BLAS is not built to allow update");
  }
  vk::MemoryBarrier barrier(vk::AccessFlagBits::eTransferWrite,
                            vk::AccessFlagBits::eAccelerationStructureWriteKHR);
  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {},
                                barrier, nullptr, nullptr);
  vk::AccelerationStructureBuildGeometryInfoKHR buildInfo(
      vk::AccelerationStructureTypeKHR::eBottomLevel,
      vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
      vk::BuildAccelerationStructureModeKHR::eUpdate, mAS.get(), mAS.get(), mGeometries, nullptr,
      mUpdateScratchBufferAddress);
  commandBuffer.buildAccelerationStructuresKHR(buildInfo, buildRanges.data());
}

vk::DeviceAddress BLAS::getAddress() {
  return mContext->getDevice().getAccelerationStructureAddressKHR({mAS.get()});
}

TLAS::TLAS(Context &context, std::vector<vk::AccelerationStructureInstanceKHR> const &instances,
           bool externalInstanceBuffer)
    : mContext(context.shared_from_this()), mInstances(instances),
      mInstanceBufferExternal(externalInstanceBuffer) {}

void TLAS::createInstanceBuffer(bool external) {
#ifndef SVULKAN2_CUDA_INTEROP
  if (external) {
    throw std::runtime_error("external TLAS instance buffers require CUDA interop");
  }
#endif

  auto memoryUsage = external ? VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY
                              : VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU;
  auto buffer = core::Buffer::Create(
      *mContext,
      sizeof(vk::AccelerationStructureInstanceKHR) * std::max<size_t>(1, mInstances.size()),
      vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
          vk::BufferUsageFlagBits::eStorageBuffer |
          vk::BufferUsageFlagBits::eShaderDeviceAddress |
          vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
      memoryUsage, VmaAllocationCreateFlags{}, external);
  mInstanceBuffer = std::move(buffer);
  mInstanceBufferExternal = external;
}

void TLAS::enableExternalInstanceBuffer() {
  if (!mInstanceBufferExternal) {
    createInstanceBuffer(true);
    mInstanceBuffer->upload(mInstances);
  }
}

vk::AccelerationStructureGeometryKHR TLAS::geometry() const {
  return vk::AccelerationStructureGeometryKHR(
      vk::GeometryTypeKHR::eInstances,
      vk::AccelerationStructureGeometryInstancesDataKHR(VK_FALSE, mInstanceBufferAddress), {});
}

vk::DeviceSize TLAS::createStorage() {
  createInstanceBuffer(mInstanceBufferExternal);
  mInstanceBufferAddress = mInstanceBuffer->getAddress();
  auto instances = geometry();
  vk::AccelerationStructureBuildGeometryInfoKHR info(
      vk::AccelerationStructureTypeKHR::eTopLevel,
      vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
          vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
      vk::BuildAccelerationStructureModeKHR::eBuild, nullptr, nullptr, instances, {}, {});
  uint32_t count = static_cast<uint32_t>(mInstances.size());
  auto device = mContext->getDevice();
  // the sizes do not depend on the build mode: one query gives the update scratch size too
  auto sizes = device.getAccelerationStructureBuildSizesKHR(
      vk::AccelerationStructureBuildTypeKHR::eDevice, info, count);
  mUpdateScratchBuffer = createScratchBuffer(*mContext, sizes.updateScratchSize);
  mUpdateScratchBufferAddress = mUpdateScratchBuffer->getAddress();
  mBuffer = Buffer::Create(*mContext, sizes.accelerationStructureSize,
                           vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                               vk::BufferUsageFlagBits::eShaderDeviceAddress,
                           VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
  mAS = device.createAccelerationStructureKHRUnique(vk::AccelerationStructureCreateInfoKHR(
      {}, mBuffer->getVulkanBuffer(), {}, sizes.accelerationStructureSize,
      vk::AccelerationStructureTypeKHR::eTopLevel, {}));
  return sizes.buildScratchSize;
}

void TLAS::build() { BuildAll(*mContext, {this}); }

void TLAS::BuildAll(Context &context, std::vector<TLAS *> const &tlases) {
  auto commandPool = context.createCommandPool();
  std::vector<std::unique_ptr<Buffer>> scratch(tlases.size());
  forEachBuildBatch(
      tlases.size(),
      [&](size_t i) {
        vk::DeviceSize size = tlases[i]->createStorage();
        scratch[i] = createScratchBuffer(context, size);
        return size;
      },
      [&](size_t begin, size_t end) {
        // the instance rows of this submission go through one staging buffer
        size_t rowSize = sizeof(vk::AccelerationStructureInstanceKHR);
        size_t rows = 0;
        for (size_t i = begin; i < end; ++i) {
          rows += tlases[i]->mInstances.size();
        }
        auto staging = Buffer::Create(context, std::max<size_t>(rows, 1) * rowSize,
                                      vk::BufferUsageFlagBits::eTransferSrc,
                                      VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_ONLY);
        auto commandBuffer = commandPool->allocateCommandBuffer();
        commandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        size_t offset = 0;
        for (size_t i = begin; i < end; ++i) {
          auto const &instances = tlases[i]->mInstances;
          if (instances.empty()) {
            continue;
          }
          size_t bytes = instances.size() * rowSize;
          staging->upload(instances.data(), bytes, offset);
          commandBuffer->copyBuffer(staging->getVulkanBuffer(),
                                    tlases[i]->mInstanceBuffer->getVulkanBuffer(),
                                    vk::BufferCopy(offset, 0, bytes));
          offset += bytes;
        }
        vk::MemoryBarrier barrier(vk::AccessFlagBits::eTransferWrite,
                                  vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                      vk::AccessFlagBits::eShaderRead);
        commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                       vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                                       {}, barrier, nullptr, nullptr);

        // sized up front: every build info points at its geometry and every range pointer at
        // its range
        size_t count = end - begin;
        std::vector<vk::AccelerationStructureGeometryKHR> geometries(count);
        std::vector<vk::AccelerationStructureBuildGeometryInfoKHR> infos(count);
        std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges(count);
        std::vector<vk::AccelerationStructureBuildRangeInfoKHR const *> rangePointers(count);
        for (size_t j = 0; j < count; ++j) {
          TLAS &tlas = *tlases[begin + j];
          geometries[j] = tlas.geometry();
          infos[j] = vk::AccelerationStructureBuildGeometryInfoKHR(
              vk::AccelerationStructureTypeKHR::eTopLevel,
              vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
                  vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
              vk::BuildAccelerationStructureModeKHR::eBuild, nullptr, tlas.mAS.get(),
              geometries[j], {}, scratch[begin + j]->getAddress());
          ranges[j] = vk::AccelerationStructureBuildRangeInfoKHR(
              static_cast<uint32_t>(tlas.mInstances.size()), 0, 0, 0);
          rangePointers[j] = &ranges[j];
        }
        commandBuffer->buildAccelerationStructuresKHR(infos, rangePointers);
        commandBuffer->end();
        context.getQueue().submitAndWait(commandBuffer.get());
        for (size_t i = begin; i < end; ++i) {
          scratch[i].reset();
        }
      });
}

void TLAS::recordUpdate(vk::CommandBuffer commandBuffer,
                        std::vector<vk::TransformMatrixKHR> const &transforms) {
  if (mInstances.size() != transforms.size()) {
    throw std::runtime_error("failed to update TLAS: length of transforms does "
                             "not match instance count");
  }

  for (size_t i = 0; i < transforms.size(); ++i) {
    mInstances[i].setTransform(transforms[i]);
  }
  mInstanceBuffer->upload(mInstances);
  recordUpdate(commandBuffer);
}

void TLAS::recordUpdate(vk::CommandBuffer commandBuffer) {
  mInstanceBufferAddress = mInstanceBuffer->getAddress();
  vk::AccelerationStructureGeometryInstancesDataKHR instancesData(VK_FALSE,
                                                                  mInstanceBufferAddress);
  vk::AccelerationStructureGeometryKHR tlasGeometry(vk::GeometryTypeKHR::eInstances, instancesData,
                                                    {});

  vk::AccelerationStructureBuildGeometryInfoKHR buildInfo(
      vk::AccelerationStructureTypeKHR::eTopLevel,
      vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
          vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
      vk::BuildAccelerationStructureModeKHR::eUpdate, mAS.get(), mAS.get(), tlasGeometry, {},
      mUpdateScratchBufferAddress);

  vk::MemoryBarrier barrier;
  vk::PipelineStageFlags sourceStage;
  if (mInstanceBufferExternal) {
    // Wait for CUDA external-memory writes and BLAS builds.
    barrier = vk::MemoryBarrier(vk::AccessFlagBits::eMemoryWrite |
                                    vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                                vk::AccessFlagBits::eAccelerationStructureWriteKHR |
                                    vk::AccessFlagBits::eAccelerationStructureReadKHR);
    sourceStage = vk::PipelineStageFlagBits::eAllCommands;
  } else {
    barrier = vk::MemoryBarrier(vk::AccessFlagBits::eHostWrite |
                                    vk::AccessFlagBits::eTransferWrite |
                                    vk::AccessFlagBits::eShaderWrite |
                                    vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                                vk::AccessFlagBits::eAccelerationStructureWriteKHR |
                                    vk::AccessFlagBits::eAccelerationStructureReadKHR);
    sourceStage = vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eTransfer |
                  vk::PipelineStageFlagBits::eComputeShader |
                  vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR;
  }
  commandBuffer.pipelineBarrier(sourceStage,
                                vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {},
                                barrier, nullptr, nullptr);

  vk::AccelerationStructureBuildRangeInfoKHR buildRange(static_cast<uint32_t>(mInstances.size()),
                                                        0, 0, 0);
  commandBuffer.buildAccelerationStructuresKHR(buildInfo, &buildRange);
}

void TLAS::RecordUpdates(vk::CommandBuffer commandBuffer, std::vector<TLAS *> const &tlases) {
  // sized up front: every build info points at its geometry and every range pointer at its range
  std::vector<vk::AccelerationStructureGeometryKHR> geometries(tlases.size());
  std::vector<vk::AccelerationStructureBuildGeometryInfoKHR> infos(tlases.size());
  std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges(tlases.size());
  std::vector<vk::AccelerationStructureBuildRangeInfoKHR const *> rangePointers(tlases.size());
  for (size_t i = 0; i < tlases.size(); ++i) {
    TLAS &tlas = *tlases[i];
    tlas.mInstanceBufferAddress = tlas.mInstanceBuffer->getAddress();
    geometries[i] = vk::AccelerationStructureGeometryKHR(
        vk::GeometryTypeKHR::eInstances,
        vk::AccelerationStructureGeometryInstancesDataKHR(VK_FALSE, tlas.mInstanceBufferAddress),
        {});
    infos[i] = vk::AccelerationStructureBuildGeometryInfoKHR(
        vk::AccelerationStructureTypeKHR::eTopLevel,
        vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
            vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
        vk::BuildAccelerationStructureModeKHR::eUpdate, tlas.mAS.get(), tlas.mAS.get(),
        geometries[i], {}, tlas.mUpdateScratchBufferAddress);
    ranges[i] = vk::AccelerationStructureBuildRangeInfoKHR(
        static_cast<uint32_t>(tlas.mInstances.size()), 0, 0, 0);
    rangePointers[i] = &ranges[i];
  }
  if (!infos.empty()) {
    commandBuffer.buildAccelerationStructuresKHR(infos, rangePointers);
  }
}

vk::DeviceAddress TLAS::getAddress() {
  return mContext->getDevice().getAccelerationStructureAddressKHR({mAS.get()});
}

} // namespace core
} // namespace svulkan2
