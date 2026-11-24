#pragma once
#include "buffer.h"
#include "command_pool.h"

namespace svulkan2 {
namespace core {

class Context;

class BLAS {
public:
  /** `update`: the BLAS can be refit later (deformable meshes, point sets) */
  BLAS(Context &context, std::vector<vk::AccelerationStructureGeometryKHR> const &geometries,
       std::vector<vk::AccelerationStructureBuildRangeInfoKHR> const &buildRanges,
       std::vector<uint32_t> const &maxPrimitiveCounts = {}, bool update = false);

  void build();
  /** Build every BLAS of `blases` (all on `context`) in as few queue submissions as the scratch
   *  memory budget allows, and wait for them. One submission per structure dominated the start-up
   *  of scenes with thousands of objects. */
  static void BuildAll(Context &context, std::vector<BLAS *> const &blases);
  void recordUpdate(vk::CommandBuffer commandBuffer,
                    std::vector<vk::AccelerationStructureBuildRangeInfoKHR> const &buildRanges);

  vk::DeviceAddress getAddress();

private:
  vk::AccelerationStructureBuildGeometryInfoKHR
  buildInfo(vk::BuildAccelerationStructureModeKHR mode) const;
  /** create the structure and its update scratch; returns the build scratch size */
  vk::DeviceSize createStorage();

  std::shared_ptr<Context> mContext; // first member: outlives the device objects below
  std::vector<vk::AccelerationStructureGeometryKHR> mGeometries;
  std::vector<vk::AccelerationStructureBuildRangeInfoKHR> mBuildRanges;
  std::vector<uint32_t> mMaxPrimitiveCounts;
  bool mUpdate;

  std::unique_ptr<Buffer> mUpdateScratchBuffer;
  vk::DeviceAddress mUpdateScratchBufferAddress;

  std::unique_ptr<Buffer> mBuffer;
  vk::UniqueAccelerationStructureKHR mAS;
};

class TLAS {

public:
  /** `externalInstanceBuffer`: the instance rows live in device memory exported to CUDA */
  TLAS(Context &context, std::vector<vk::AccelerationStructureInstanceKHR> const &instances,
       bool externalInstanceBuffer = false);
  void build();
  /** Build every TLAS of `tlases` (all on `context`, their BLASes built) in as few queue
   *  submissions as the scratch memory budget allows, and wait for them. */
  static void BuildAll(Context &context, std::vector<TLAS *> const &tlases);
  void recordUpdate(vk::CommandBuffer commandBuffer,
                    std::vector<vk::TransformMatrixKHR> const &transforms);
  void recordUpdate(vk::CommandBuffer commandBuffer);
  /** Record the refit of every TLAS of `tlases` from its instance buffer in one build command.
   *  Records no barrier: the caller orders the instance and BLAS writes before it and the
   *  readers after it. Each TLAS refits into its own scratch buffer, so the builds are
   *  independent. */
  static void RecordUpdates(vk::CommandBuffer commandBuffer, std::vector<TLAS *> const &tlases);

  void enableExternalInstanceBuffer();
  bool isInstanceBufferExternal() const { return mInstanceBufferExternal; }

  vk::DeviceAddress getAddress();

  vk::AccelerationStructureKHR getVulkanAS() const { return mAS.get(); };
  Buffer &getInstanceBuffer() const { return *mInstanceBuffer; }

private:
  /** an instance buffer without contents */
  void createInstanceBuffer(bool external);
  vk::AccelerationStructureGeometryKHR geometry() const;
  /** create the instance buffer, the structure and its update scratch; returns the build scratch
   *  size */
  vk::DeviceSize createStorage();

  std::shared_ptr<Context> mContext; // first member: outlives the device objects below
  std::vector<vk::AccelerationStructureInstanceKHR> mInstances;

  std::unique_ptr<Buffer> mInstanceBuffer;
  bool mInstanceBufferExternal{};
  vk::DeviceAddress mInstanceBufferAddress;

  std::unique_ptr<Buffer> mUpdateScratchBuffer;
  vk::DeviceAddress mUpdateScratchBufferAddress;

  std::unique_ptr<Buffer> mBuffer;
  vk::UniqueAccelerationStructureKHR mAS;
};

} // namespace core
} // namespace svulkan2
