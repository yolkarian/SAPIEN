#pragma once

#ifdef SAPIEN_CUDA

#include "sapien/array.h"
#include "sapien/utils/cuda.h"
#include <memory>
#include <svulkan2/common/vk.h>
#include <vector>

struct CUstream_st;

namespace svulkan2::core {
class Buffer;
class CommandPool;
} // namespace svulkan2::core

namespace svulkan2::scene {
class Scene;
} // namespace svulkan2::scene

namespace svulkan2::shader {
class ComputeModule;
class ComputeModuleInstance;
} // namespace svulkan2::shader

namespace sapien::sapien_renderer {

class SapienRendererSystem;
class SapienRenderBodyComponent;

class StagedRenderSystem {
public:
  StagedRenderSystem(
      std::vector<std::shared_ptr<SapienRendererSystem>> systems,
      std::shared_ptr<svulkan2::scene::Scene> renderScene,
      std::vector<std::shared_ptr<SapienRenderBodyComponent>> gpuSourcedBodies,
      CudaArrayHandle poseSource, CUstream_st *cudaStream);
  ~StagedRenderSystem();

  void update();
  uint64_t getTransferredBytes() const { return mTransferredBytes; }

private:
  struct Slot {
    CudaHostArray hostPoses;
    CudaEvent cudaReady;
    std::unique_ptr<svulkan2::core::Buffer> stagingBuffer;
    vk::UniqueCommandBuffer commandBuffer;
    vk::UniqueFence fence;
  };

  std::vector<std::shared_ptr<SapienRendererSystem>> mSystems;
  std::shared_ptr<svulkan2::scene::Scene> mRenderScene;
  std::vector<uint64_t> mSceneVersions;
  uint64_t mRenderSceneVersion{};
  std::vector<std::shared_ptr<SapienRenderBodyComponent>> mGpuSourcedBodies;

  CudaArrayHandle mPoseSource;
  CUstream_st *mCudaStream{};
  CudaArray mSourcePoseIndices;
  CudaArray mCompactPoses;

  std::unique_ptr<svulkan2::core::Buffer> mPoseBuffer;
  std::unique_ptr<svulkan2::core::Buffer> mShapeBuffer;
  std::unique_ptr<svulkan2::core::CommandPool> mCommandPool;
  std::shared_ptr<svulkan2::shader::ComputeModule> mComputeModule;
  std::unique_ptr<svulkan2::shader::ComputeModuleInstance> mComputeInstance;
  std::vector<Slot> mSlots;

  uint32_t mPoseCount{};
  uint32_t mShapeCount{};
  uint32_t mTransformStride{};
  uint32_t mNextSlot{};
  uint64_t mTransferredBytes{};
};

} // namespace sapien::sapien_renderer

#endif
