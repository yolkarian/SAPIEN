#include "staged_render_system.h"

#ifdef SAPIEN_CUDA

#include "staged_render_system.cuh"
#include "sapien/profiler.h"
#include "sapien/sapien_renderer/render_body_component.h"
#include "sapien/sapien_renderer/sapien_renderer_default.h"
#include "sapien/sapien_renderer/sapien_renderer_system.h"
#include "sapien/utils/typestr.h"
#include <algorithm>
#include <array>
#include <cuda_runtime.h>
#include <filesystem>
#include <svulkan2/core/buffer.h>
#include <svulkan2/core/command_pool.h>
#include <svulkan2/core/context.h>
#include <svulkan2/scene/scene.h>
#include <svulkan2/shader/compute_module.h>
#include <unordered_map>
#include <unordered_set>

namespace sapien::sapien_renderer {
namespace {

struct alignas(16) StagedShapeData {
  std::array<uint32_t, 4> indices;
  std::array<float, 4> localPosition;
  std::array<float, 4> localQuaternion;
  std::array<float, 4> scale;
};
static_assert(sizeof(StagedShapeData) == 64);

} // namespace

StagedRenderSystem::StagedRenderSystem(
    std::vector<std::shared_ptr<SapienRendererSystem>> systems,
    std::shared_ptr<svulkan2::scene::Scene> renderScene,
    std::vector<std::shared_ptr<SapienRenderBodyComponent>> gpuSourcedBodies,
    CudaArrayHandle poseSource, CUstream_st *cudaStream)
    : mSystems(std::move(systems)), mRenderScene(std::move(renderScene)),
      mPoseSource(std::move(poseSource)), mCudaStream(cudaStream) {
  if (!mRenderScene || mSystems.empty()) {
    throw std::runtime_error("staged rendering requires a non-empty render scene selection");
  }
  mPoseSource.checkShape({-1, -1});
  if (typestrCode(mPoseSource.type) != 'f' ||
      typestrBytes(mPoseSource.type) != sizeof(float) || mPoseSource.shape[1] < 7 ||
      mPoseSource.strides[1] != sizeof(float)) {
    throw std::runtime_error("staged pose source must be a CUDA float32 array with 7+ channels");
  }

  for (auto const &system : mSystems) {
    mSceneVersions.push_back(system->getScene()->getVersion());
  }
  mRenderSceneVersion = mRenderScene->getVersion();
  mRenderScene->prepareObjectTransformBuffer();
  mTransformStride = mRenderScene->getGpuTransformBufferSize() / sizeof(float);

  std::unordered_map<int, uint32_t> compactIndexByPose;
  std::vector<int> sourcePoseIndices;
  std::vector<StagedShapeData> shapeData;
  std::unordered_set<SapienRenderBodyComponent *> acquiredBodies;
  for (auto const &body : gpuSourcedBodies) {
    bool acquired = false;
    for (auto const &shape : body->getRenderShapes()) {
      int sourcePoseIndex = shape->getGpuBatchedPoseIndex();
      if (sourcePoseIndex < 0) {
        continue;
      }
      if (sourcePoseIndex >= mPoseSource.shape[0]) {
        throw std::runtime_error("staged GPU pose index is outside the source pose buffer");
      }
      auto [iterator, inserted] =
          compactIndexByPose.emplace(sourcePoseIndex, compactIndexByPose.size());
      if (inserted) {
        sourcePoseIndices.push_back(sourcePoseIndex);
      }

      Pose localPose = shape->getLocalPose();
      Vec3 scale = shape->getGpuScale();
      int transformIndex = shape->getInternalGpuTransformIndex(*mRenderScene);
      shapeData.push_back(StagedShapeData{
          .indices = {iterator->second, static_cast<uint32_t>(transformIndex), 0, 0},
          .localPosition = {localPose.p.x, localPose.p.y, localPose.p.z, 0.f},
          .localQuaternion = {localPose.q.w, localPose.q.x, localPose.q.y, localPose.q.z},
          .scale = {scale.x, scale.y, scale.z, 0.f},
      });
      acquired = true;
    }
    if (acquired && acquiredBodies.insert(body.get()).second) {
      body->internalAcquireGpuPoseSource();
      mGpuSourcedBodies.push_back(body);
    }
  }
  mPoseCount = sourcePoseIndices.size();
  mShapeCount = shapeData.size();
  mCompactIndexByPose = std::move(compactIndexByPose);

  checkCudaErrors(cudaSetDevice(mPoseSource.cudaId));
  mSourcePoseIndices = CudaArray({static_cast<int>(mPoseCount)}, "i4");
  mCompactPoses = CudaArray({static_cast<int>(mPoseCount), 7}, "f4");
  if (mPoseCount) {
    checkCudaErrors(cudaMemcpy(mSourcePoseIndices.ptr, sourcePoseIndices.data(),
                               sourcePoseIndices.size() * sizeof(int), cudaMemcpyHostToDevice));
  }

  auto context = SapienRenderEngine::Get()->getContext();
  auto poseBytes = std::max<vk::DeviceSize>(mPoseCount * 7 * sizeof(float), sizeof(float));
  auto shapeBytes = std::max<vk::DeviceSize>(shapeData.size() * sizeof(StagedShapeData),
                                             sizeof(StagedShapeData));
  mPoseBuffer = svulkan2::core::Buffer::Create(
      poseBytes, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
      VMA_MEMORY_USAGE_GPU_ONLY);
  mShapeBuffer = svulkan2::core::Buffer::Create(
      shapeBytes, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
      VMA_MEMORY_USAGE_GPU_ONLY);
  mDummyRTInstanceBuffer = svulkan2::core::Buffer::Create(
      sizeof(vk::AccelerationStructureInstanceKHR), vk::BufferUsageFlagBits::eStorageBuffer,
      VMA_MEMORY_USAGE_GPU_ONLY);
  if (!shapeData.empty()) {
    mShapeBuffer->upload(shapeData);
  }

  auto shaderPath = std::filesystem::path(SapienRendererDefault::internalGetShaderSearchPath()) /
                    "internal" / "staged_pose.comp";
  mComputeModule = std::make_shared<svulkan2::shader::ComputeModule>(shaderPath.string(), 64);
  mComputeInstance =
      std::make_unique<svulkan2::shader::ComputeModuleInstance>(mComputeModule);
  mComputeInstance->setBuffer("PoseBuffer", mPoseBuffer.get());
  mComputeInstance->setBuffer("ShapeBuffer", mShapeBuffer.get());
  mComputeInstance->setBuffer("TransformBuffer",
                              mRenderScene->getObjectTransformBuffer().get());
  mComputeInstance->setBuffer("RTInstanceBuffer", mDummyRTInstanceBuffer.get());
  mComputeInstance->setPushConstant<uint32_t>("shapeCount", mShapeCount);
  mComputeInstance->setPushConstant<uint32_t>("transformStride", mTransformStride);
  mComputeInstance->setPushConstant<uint32_t>("rtEnabled", 0);

  mCommandPool = context->createCommandPool();
  mSlots.reserve(2);
  for (int index = 0; index < 2; ++index) {
    Slot slot;
    slot.hostPoses = CudaHostArray({static_cast<int>(mPoseCount), 7}, "f4");
    slot.stagingBuffer = svulkan2::core::Buffer::CreateStaging(poseBytes);
    slot.commandBuffer = mCommandPool->allocateCommandBuffer();
    slot.fence = context->getDevice().createFenceUnique({vk::FenceCreateFlagBits::eSignaled});
    mSlots.push_back(std::move(slot));
  }
}

void StagedRenderSystem::update() {
  SAPIEN_PROFILE_FUNCTION;
  for (uint32_t index = 0; index < mSystems.size(); ++index) {
    if (mSystems[index]->getScene()->getVersion() != mSceneVersions[index]) {
      throw std::runtime_error(
          "staged render mappings are stale after a source scene topology change");
    }
  }
  if (mRenderScene->getVersion() != mRenderSceneVersion) {
    throw std::runtime_error("staged render mappings are stale after an output topology change");
  }
  if (!mPoseCount) {
    return;
  }

  uint32_t slotIndex = mNextSlot;
  Slot &slot = mSlots[slotIndex];
  mNextSlot = (mNextSlot + 1) % mSlots.size();

  SAPIEN_PROFILE_BLOCK_BEGIN("compact staged poses");
  checkCudaErrors(cudaSetDevice(mPoseSource.cudaId));
  pack_staged_poses(static_cast<float *>(mCompactPoses.ptr),
                    static_cast<float const *>(mPoseSource.ptr), mPoseSource.shape[1],
                    static_cast<int const *>(mSourcePoseIndices.ptr), mPoseCount, mCudaStream);
  auto transferBytes = mPoseCount * 7 * sizeof(float);
  checkCudaErrors(cudaMemcpyAsync(slot.hostPoses.ptr, mCompactPoses.ptr, transferBytes,
                                  cudaMemcpyDeviceToHost, mCudaStream));
  slot.cudaReady.record(mCudaStream);
  SAPIEN_PROFILE_BLOCK_END;

  SAPIEN_PROFILE_BLOCK_BEGIN("wait for staged poses");
  slot.cudaReady.synchronize();
  SAPIEN_PROFILE_BLOCK_END;

  auto context = SapienRenderEngine::Get()->getContext();
  auto device = context->getDevice();
  auto rtInstanceBuffer = mRenderScene->getTLAS()
                              ? &mRenderScene->getTLAS()->getInstanceBuffer()
                              : nullptr;
  if (rtInstanceBuffer != mRTInstanceBuffer) {
    mRTInstanceBuffer = rtInstanceBuffer;
    mComputeInstance->setBuffer("RTInstanceBuffer",
                                mRTInstanceBuffer ? mRTInstanceBuffer
                                                  : mDummyRTInstanceBuffer.get());
    mComputeInstance->setPushConstant<uint32_t>("rtEnabled", mRTInstanceBuffer ? 1 : 0);
  }
  if (device.waitForFences(slot.fence.get(), VK_TRUE, UINT64_MAX) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to reuse staged Vulkan upload slot");
  }
  device.resetFences(slot.fence.get());
  slot.stagingBuffer->upload(slot.hostPoses.ptr, transferBytes);

  SAPIEN_PROFILE_BLOCK_BEGIN("staged Vulkan transform update");
  auto commandBuffer = slot.commandBuffer.get();
  commandBuffer.reset();
  commandBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  commandBuffer.copyBuffer(slot.stagingBuffer->getVulkanBuffer(), mPoseBuffer->getVulkanBuffer(),
                           vk::BufferCopy(0, 0, transferBytes));
  vk::MemoryBarrier poseBarrier(vk::AccessFlagBits::eTransferWrite,
                                vk::AccessFlagBits::eShaderRead);
  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eComputeShader, {}, poseBarrier, {}, {});
  mComputeInstance->record(commandBuffer, (mShapeCount + 63) / 64, 1, 1);
  vk::MemoryBarrier transformBarrier(vk::AccessFlagBits::eShaderWrite,
                                     vk::AccessFlagBits::eUniformRead);
  commandBuffer.pipelineBarrier(
      vk::PipelineStageFlagBits::eComputeShader,
      vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eFragmentShader, {},
      transformBarrier, {}, {});
  if (mRTInstanceBuffer) {
    vk::MemoryBarrier instanceBarrier(vk::AccessFlagBits::eShaderWrite,
                                      vk::AccessFlagBits::eAccelerationStructureReadKHR);
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                  vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {},
                                  instanceBarrier, {}, {});
  }
  commandBuffer.end();
  context->getQueue().submit(commandBuffer, slot.fence.get());
  if (mRTInstanceBuffer) {
    mRenderScene->updateRenderVersion();
  }
  SAPIEN_PROFILE_BLOCK_END;

  mLastCompletedSlot = slotIndex;
  mTransferredBytes += transferBytes;
}

std::optional<Pose> StagedRenderSystem::getPose(int sourcePoseIndex) const {
  if (mLastCompletedSlot < 0) {
    return std::nullopt;
  }
  auto iterator = mCompactIndexByPose.find(sourcePoseIndex);
  if (iterator == mCompactIndexByPose.end()) {
    return std::nullopt;
  }
  auto const *poses = static_cast<float const *>(mSlots[mLastCompletedSlot].hostPoses.ptr);
  auto const *pose = poses + iterator->second * 7;
  return Pose({pose[0], pose[1], pose[2]}, {pose[3], pose[4], pose[5], pose[6]});
}

StagedRenderSystem::~StagedRenderSystem() {
  SapienRenderEngine::Get()->getContext()->getDevice().waitIdle();
  for (auto const &body : mGpuSourcedBodies) {
    body->internalReleaseGpuPoseSource();
  }
  cudaSetDevice(mPoseSource.cudaId);
}

} // namespace sapien::sapien_renderer

#endif
