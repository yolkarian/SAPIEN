#include "sapien/sapien_renderer/batched_render_system.h"
#include "./batched_render_system.cuh"
#include "render_scene_resolver.h"
#include "sapien/entity.h"
#include "sapien/physx/articulation_link_component.h"
#include "sapien/physx/rigid_component.h"
#include "sapien/profiler.h"
#include "sapien/sapien_renderer/camera_component.h"
#include "sapien/sapien_renderer/render_body_component.h"
#include "sapien/scene.h"
#include "sapien/utils/typestr.h"
#include <algorithm>
#include <svulkan2/renderer/renderer.h>
#include <svulkan2/renderer/renderer_base.h>
#include <svulkan2/renderer/rt_renderer.h>
#include <svulkan2/scene/scene_group.h>
#include <unordered_set>

#ifdef SAPIEN_CUDA
#include "sapien/utils/cuda.h"
#include <cuda_runtime.h>
#endif

namespace sapien {
namespace sapien_renderer {

namespace {

CudaArrayHandle getTransformCudaArray(std::shared_ptr<svulkan2::scene::Scene> const &scene) {
  scene->prepareObjectTransformBuffer();
  int offset = scene->getGpuTransformBufferSize();

  auto buffer = scene->getObjectTransformBuffer();
#ifdef SAPIEN_CUDA
  return CudaArrayHandle{.shape = {static_cast<int>(buffer->getSize() / offset), 4, 4},
                         .strides = {offset, 16, 4},
                         .type = "f4",
                         .cudaId = buffer->getCudaDeviceId(),
                         .ptr = buffer->getCudaPtr()};
#else
  return CudaArrayHandle{.shape = {static_cast<int>(buffer->getSize() / offset), 4, 4},
                         .strides = {offset, 16, 4},
                         .type = "f4"};
#endif
}

} // namespace

BatchedCamera::BatchedCamera(std::vector<std::shared_ptr<SapienRenderCameraComponent>> cameras,
                             std::vector<std::string> renderTargets)
    : mCameras(cameras) {
  if (cameras.empty()) {
    throw std::runtime_error("failed to create BatchedCamera: empty cameras");
  }
  uint32_t width = cameras.at(0)->getWidth();
  uint32_t height = cameras.at(0)->getHeight();

  for (auto &cam : cameras) {
    if (!cam->getScene()) {
      throw std::runtime_error(
          "failed to create BatchedCamera: some camera is not added to scene");
    }
    if (cam->getWidth() != width || cam->getHeight() != height) {
      throw std::runtime_error(
          "failed to create BatchedCamera: the cameras must have the same width and height");
    }
  }

  auto context = SapienRenderEngine::Get()->getContext();
  mCommandPool = context->createCommandPool();
  mCommandBuffer = mCommandPool->allocateCommandBuffer();

  for (auto &cam : cameras) {
    cam->gpuInit();
  }

  mCommandBuffer->begin(vk::CommandBufferBeginInfo());
  for (uint32_t i = 0; i < cameras.size(); ++i) {
    auto &cam = cameras[i];

    for (auto &name : renderTargets) {
      auto &image = cam->getInternalImage(name);
      auto extent = image.getExtent();
      vk::Format format = image.getFormat();
      vk::DeviceSize imageSize =
          extent.width * extent.height * extent.depth * svulkan2::getFormatSize(format);
      if (!mCudaImageBuffers.contains(name)) {
        mCudaImageBuffers[name] = svulkan2::core::Buffer::Create(
            imageSize * cameras.size(),
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_GPU_ONLY, VmaAllocationCreateFlags{}, true);
        {
          CudaArrayHandle array;
          int channels = getFormatChannels(format);
          int itemsize = getFormatChannelSize(format);
          array.shape = {static_cast<int>(cameras.size()), static_cast<int>(extent.height),
                         static_cast<int>(extent.width)};
          array.strides = {static_cast<int>(itemsize * channels * width * height),
                           static_cast<int>(itemsize * channels * width),
                           static_cast<int>(itemsize * channels)};
          if (channels != 1) {
            array.shape.push_back(channels);
            array.strides.push_back(itemsize);
          }
          array.type = getFormatTypestr(format);
#ifdef SAPIEN_CUDA
          array.ptr = mCudaImageBuffers[name]->getCudaPtr();
          array.cudaId = mCudaImageBuffers[name]->getCudaDeviceId();
#endif
          mCudaImageHandles[name] = array;
        }
      }
      image.recordCopyToBuffer(mCommandBuffer.get(),
                               cam->getInternalRenderer().getRenderTargetImageLayout(name),
                               mCudaImageBuffers[name]->getVulkanBuffer(), i * imageSize,
                               imageSize, {0, 0, 0}, extent, 0);
    }
  }
  mCommandBuffer->end();

  // setup semaphore for CUDA
  mFrameCounter = 0;
  vk::SemaphoreTypeCreateInfo timelineCreateInfo(vk::SemaphoreType::eTimeline, 0);
  vk::SemaphoreCreateInfo createInfo{};
  vk::ExportSemaphoreCreateInfo exportCreateInfo(
      vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd);
  createInfo.setPNext(&exportCreateInfo);
  exportCreateInfo.setPNext(&timelineCreateInfo);
  auto device = context->getDevice();
  mSemaphore = device.createSemaphoreUnique(createInfo);

  int fd = device.getSemaphoreFdKHR(
      {mSemaphore.get(), vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd});
#ifdef SAPIEN_CUDA
  cudaExternalSemaphoreHandleDesc desc = {};
  desc.flags = 0;
  desc.handle.fd = fd;
  desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
  checkCudaErrors(cudaImportExternalSemaphore(&mCudaSem, &desc));
#endif
  // TODO clean up cudaSem
}

void BatchedCamera::takePicture() {
  auto context = SapienRenderEngine::Get()->getContext();

  // make sure previous takePicture has finished
  auto result = context->getDevice().waitSemaphores(
      vk::SemaphoreWaitInfo({}, mSemaphore.get(), mFrameCounter), UINT64_MAX);

  if (result != vk::Result::eSuccess) {
    throw std::runtime_error("take picture failed: wait for fence failed");
  }

  for (auto &cam : mCameras) {
    cam->getInternalRenderer().render(cam->getInternalCamera(), {}, {}, {}, {});
  }
  mFrameCounter++;
  context->getQueue().submit(mCommandBuffer.get(), {}, {}, {}, mSemaphore.get(), mFrameCounter,
                             {});
#ifdef SAPIEN_CUDA
  cudaExternalSemaphoreWaitParams waitParams{};
  waitParams.params.fence.value = mFrameCounter;
  cudaWaitExternalSemaphoresAsync(&mCudaSem, &waitParams, 1, mCudaStream);
#endif
}

CudaArrayHandle BatchedCamera::getPictureCuda(std::string const &name) {
  if (!mCudaImageHandles.contains(name)) {
    throw std::runtime_error("Failed to get image with name :" + name +
                             ". Did you forget to specify it in create_camera_group?");
  }
  return mCudaImageHandles.at(name);
}

BatchedCamera::~BatchedCamera() {
  SapienRenderEngine::Get()->getContext()->getDevice().waitIdle();
#ifdef SAPIEN_CUDA
  cudaDestroyExternalSemaphore(mCudaSem);
#endif
}

BatchedRenderSystem::BatchedRenderSystem(
    std::vector<std::shared_ptr<SapienRendererSystem>> systems)
    : mSystems(RenderSceneResolver::resolve(systems, {})) {
  if (mSystems.empty()) {
    throw std::runtime_error("systems must not be empty");
  }
  init();
}

BatchedRenderSystem::BatchedRenderSystem(
    std::vector<std::shared_ptr<SapienRendererSystem>> systems,
    std::shared_ptr<svulkan2::scene::Scene> renderScene,
    std::vector<std::shared_ptr<SapienRenderBodyComponent>> gpuSourcedBodies)
    : mSystems(RenderSceneResolver::resolve(systems, {})), mFixedRenderScene(renderScene),
      mAutoBindPhysxGpuPoses(false), mFixedGpuSourcedBodies(gpuSourcedBodies) {
  if (mSystems.empty()) {
    throw std::runtime_error("systems must not be empty");
  }
  if (!mFixedRenderScene) {
    throw std::runtime_error("render scene must not be null");
  }
  init();
}

void BatchedRenderSystem::init() {
  for (auto const &body : mGpuSourcedBodies) {
    body->internalReleaseGpuPoseSource();
  }
  mGpuSourcedBodies.clear();

  std::vector<RenderShapeData> allShapeData;
  std::vector<void *> sceneTransformRefs;

  mSceneVersions = {};
  mRenderScenes = {};
  mRenderSceneVersions = {};
  mRenderSceneSystems = {};
  mTransformBufferElementByteOffset = 0;
  mMaximumPoseIndex = -1;

  // TODO ensure all cameras are valid
  for (auto &system : mSystems) {
    // run a step
    system->step();

    // cache current versions
    mSceneVersions.push_back(system->getScene()->getVersion());
  }

  if (mFixedRenderScene) {
    mRenderScenes.push_back(mFixedRenderScene);
    mRenderSceneSystems.push_back(mSystems);
  } else {
    for (auto const &system : mSystems) {
      auto resolvedSystems = RenderSceneResolver::resolve({system}, mSystems);
      auto renderScene = RenderSceneResolver::build(resolvedSystems);
      mRenderScenes.push_back(renderScene);
      mRenderSceneSystems.push_back(resolvedSystems);
    }
    for (auto const &resolvedSystems : mAdditionalRenderSelections) {
      mRenderScenes.push_back(RenderSceneResolver::build(resolvedSystems));
      mRenderSceneSystems.push_back(resolvedSystems);
    }
  }

  // PhysX GPU assigns sibling body/link pose indices during gpu_init(). Bind those indices
  // automatically while preserving explicit indices for non-PhysX pose sources. The Viewer
  // performs a stricter binding against its configured PhysX system before using a fixed output.
  if (mAutoBindPhysxGpuPoses) {
    for (auto const &system : mSystems) {
      for (auto const &body : system->getRenderBodyComponents()) {
        auto entity = body->getEntity();
        int poseIndex = -1;
        if (auto rigid = entity->getComponent<physx::PhysxRigidDynamicComponent>();
            rigid && rigid->isUsingDirectGPUAPI()) {
          poseIndex = rigid->getGpuPoseIndex();
        } else if (auto link = entity->getComponent<physx::PhysxArticulationLinkComponent>();
                   link && link->isUsingDirectGPUAPI()) {
          poseIndex = link->getGpuPoseIndex();
        }
        if (poseIndex < 0) {
          continue;
        }
        for (auto const &shape : body->getRenderShapes()) {
          if (shape->getGpuBatchedPoseIndex() < 0) {
            shape->setGpuBatchedPoseIndex(poseIndex);
          }
        }
      }
    }
  }

  std::unordered_set<SapienRenderBodyComponent *> fixedGpuSourcedBodySet;
  for (auto const &body : mFixedGpuSourcedBodies) {
    fixedGpuSourcedBodySet.insert(body.get());
  }
  std::unordered_set<SapienRenderBodyComponent *> gpuSourcedBodySet;
  auto appendSystemShapeData = [&](std::shared_ptr<SapienRendererSystem> const &shapeSystem,
                                   uint32_t renderSceneIndex,
                                   std::shared_ptr<svulkan2::scene::Scene> const &renderScene) {
    renderScene->prepareObjectTransformBuffer();
    for (auto &body : shapeSystem->getRenderBodyComponents()) {
      if (mFixedRenderScene && !fixedGpuSourcedBodySet.contains(body.get())) {
        continue;
      }
      for (auto &shape : body->getRenderShapes()) {
        int poseIndex = shape->getGpuBatchedPoseIndex();

        if (poseIndex < 0) {
          continue;
        }

        mMaximumPoseIndex = std::max(mMaximumPoseIndex, poseIndex);
        if (gpuSourcedBodySet.insert(body.get()).second) {
          body->internalAcquireGpuPoseSource();
          mGpuSourcedBodies.push_back(body);
        }
        Pose localPose = shape->getLocalPose();
        Vec3 scale = shape->getGpuScale();
        int transformIndex = shape->getInternalGpuTransformIndex(*renderScene);

        static_assert(sizeof(RenderShapeData) == 4 * 13);
        RenderShapeData data;
        data.localPose = localPose;
        data.scale = scale;

        data.poseIndex = poseIndex;
        data.sceneIndex = renderSceneIndex;
        data.transformIndex = transformIndex;

        allShapeData.push_back(data);
      }
    }
  };

  for (uint32_t sceneIndex = 0; sceneIndex < mRenderScenes.size(); ++sceneIndex) {
    auto renderScene = mRenderScenes[sceneIndex];
    auto transformArray = getTransformCudaArray(renderScene);
    sceneTransformRefs.push_back(transformArray.ptr);

    if (mTransformBufferElementByteOffset == 0) {
      mTransformBufferElementByteOffset = transformArray.strides.at(0);
      if (mTransformBufferElementByteOffset % 4 != 0) {
        throw std::runtime_error("corrupted transform array buffer");
      }
    } else if (mTransformBufferElementByteOffset != transformArray.strides.at(0)) {
      throw std::runtime_error("corrupted transform array buffer");
    }

    for (auto const &system : mRenderSceneSystems[sceneIndex]) {
      appendSystemShapeData(system, sceneIndex, renderScene);
    }
  }
  mShapeCount = allShapeData.size();

#ifdef SAPIEN_CUDA
  checkCudaErrors(cudaSetDevice(SapienRenderEngine::Get()->getDevice()->cudaId));
#endif
  mCudaShapeDataBuffer = CudaArray::FromData(allShapeData);
  mCudaSceneTransformRefBuffer = CudaArray::FromData(sceneTransformRefs);
  mCudaRTInstanceRefBuffer =
      CudaArray::FromData(std::vector<void *>(mRenderScenes.size(), nullptr));
  mRTSceneEnabled.assign(mRenderScenes.size(), false);
  for (auto const &scene : mRenderScenes) {
    mRenderSceneVersions.push_back(scene->getVersion());
  }

  // create semaphore
  auto context = SapienRenderEngine::Get()->getContext();
  if (!mSem) {
    vk::SemaphoreTypeCreateInfo timelineCreateInfo(vk::SemaphoreType::eTimeline, 0);
    vk::SemaphoreCreateInfo createInfo{};
    vk::ExportSemaphoreCreateInfo exportCreateInfo(
        vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd);
    createInfo.setPNext(&exportCreateInfo);
    exportCreateInfo.setPNext(&timelineCreateInfo);
    mSem = context->getDevice().createSemaphoreUnique(createInfo);

    int fd = context->getDevice().getSemaphoreFdKHR(
        {mSem.get(), vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd});
#ifdef SAPIEN_CUDA
    cudaExternalSemaphoreHandleDesc desc = {};
    desc.flags = 0;
    desc.handle.fd = fd;
    desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    checkCudaErrors(cudaImportExternalSemaphore(&mCudaSem, &desc));
#endif
  }
}

void BatchedRenderSystem::setPoseSource(CudaArrayHandle const &poses) {
  poses.checkShape({-1, -1});
  if (typestrCode(poses.type) != 'f' || typestrBytes(poses.type) != sizeof(float)) {
    throw std::runtime_error("pose buffer must be a CUDA float32 array");
  }
  if (poses.shape.at(1) < 7) {
    throw std::runtime_error("pose buffer must have at least 7 channels per row");
  }
  poses.checkCongiguous();
  poses.checkStride({-1, sizeof(float)});
  if (mMaximumPoseIndex >= poses.shape.at(0)) {
    throw std::runtime_error("GPU pose batch index is outside the pose buffer");
  }

  mCudaPoseHandle = poses;
}

void BatchedRenderSystem::ensureCameraRenderScenes(
    std::vector<std::shared_ptr<SapienRenderCameraComponent>> const &additionalCameras) {
  std::vector<std::shared_ptr<SapienRenderCameraComponent>> cameras = additionalCameras;
  for (auto const &batch : mCameraBatches) {
    auto const &batchCameras = batch->getCameras();
    cameras.insert(cameras.end(), batchCameras.begin(), batchCameras.end());
  }

  bool rebuildRenderScenes = false;
  for (auto const &camera : cameras) {
    auto resolvedSystems = camera->internalResolveRenderSystems(mSystems);
    bool exists = std::find(mRenderSceneSystems.begin(), mRenderSceneSystems.end(),
                            resolvedSystems) != mRenderSceneSystems.end();
    exists |= std::find(mAdditionalRenderSelections.begin(), mAdditionalRenderSelections.end(),
                        resolvedSystems) != mAdditionalRenderSelections.end();
    if (!exists) {
      mAdditionalRenderSelections.push_back(resolvedSystems);
      rebuildRenderScenes = true;
    }
  }
  if (rebuildRenderScenes) {
    SapienRenderEngine::Get()->getContext()->getDevice().waitIdle();
    init();
    if (mCudaPoseHandle.ptr) {
      setPoseSource(mCudaPoseHandle);
    }
  }

  for (auto const &camera : cameras) {
    auto resolvedSystems = camera->internalResolveRenderSystems(mSystems);
    auto it = std::find(mRenderSceneSystems.begin(), mRenderSceneSystems.end(), resolvedSystems);
    if (it == mRenderSceneSystems.end()) {
      throw std::runtime_error("failed to resolve camera output render scene");
    }
    camera->internalSetRenderScene(
        mRenderScenes.at(std::distance(mRenderSceneSystems.begin(), it)), resolvedSystems);
  }
}

std::shared_ptr<BatchedCamera> BatchedRenderSystem::createCameraBatch(
    std::vector<std::shared_ptr<SapienRenderCameraComponent>> cameras,
    std::vector<std::string> renderTargets) {
  for (auto const &camera : cameras) {
    if (camera->getGpuBatchedPoseIndex() >= 0) {
      continue;
    }
    auto entity = camera->getEntity();
    if (auto rigid = entity->getComponent<physx::PhysxRigidDynamicComponent>();
        rigid && rigid->isUsingDirectGPUAPI()) {
      camera->setGpuBatchedPoseIndex(rigid->getGpuPoseIndex());
    } else if (auto link = entity->getComponent<physx::PhysxArticulationLinkComponent>();
               link && link->isUsingDirectGPUAPI()) {
      camera->setGpuBatchedPoseIndex(link->getGpuPoseIndex());
    }
  }
  ensureCameraRenderScenes(cameras);

#ifdef SAPIEN_CUDA
  for (auto &camera : cameras) {
    if (auto renderer =
            dynamic_cast<svulkan2::renderer::RTRenderer *>(&camera->getInternalRenderer())) {
      renderer->setExternalTransformUpdatesEnabled(true);
      renderer->setExternalCameraUpdatesEnabled(camera->getGpuBatchedPoseIndex() >= 0);
    }
  }
#endif

  auto cameraBatch = std::make_shared<BatchedCamera>(cameras, renderTargets);
  cameraBatch->setCudaStream(mCudaStream);

  static_assert(sizeof(vk::AccelerationStructureInstanceKHR) == sizeof(float) * 16);
  std::vector<void *> rtInstanceRefs(mRenderScenes.size(), nullptr);
  mRTSceneEnabled.assign(mRenderScenes.size(), false);
  for (uint32_t i = 0; i < mRenderScenes.size(); ++i) {
    auto tlas = mRenderScenes[i]->getTLAS();
    if (!tlas) {
      continue;
    }
#ifdef SAPIEN_CUDA
    auto &buffer = tlas->getInstanceBuffer();
    rtInstanceRefs[i] = buffer.getCudaPtr();
    mRTSceneEnabled[i] = true;
#endif
  }
  mCudaRTInstanceRefBuffer = CudaArray::FromData(rtInstanceRefs);

  mCameraBatches.push_back(cameraBatch);

  std::vector<CameraData> allCamData;
  mCameraCount = 0;
  for (auto &cb : mCameraBatches) {
    for (auto &cam : cb->getCameras()) {
      int index = cam->getGpuBatchedPoseIndex();
      if (index < 0) {
        // this camera does not need to be updated
        continue;
      }
      mMaximumPoseIndex = std::max(mMaximumPoseIndex, index);
      CameraData data;
      data.buffer = cam->getCudaBuffer().ptr;
      auto pose = cam->getLocalPose() * POSE_GL_TO_ROS;
      data.localPose = pose;

      data.poseIndex = index;

      allCamData.push_back(data);
      mCameraCount++;
    }
  }
#ifdef SAPIEN_CUDA
  checkCudaErrors(cudaSetDevice(SapienRenderEngine::Get()->getDevice()->cudaId));
#endif
  mCudaCameraDataBuffer = CudaArray::FromData(allCamData);
  if (mCudaPoseHandle.ptr && mMaximumPoseIndex >= mCudaPoseHandle.shape.at(0)) {
    throw std::runtime_error("GPU pose batch index is outside the pose buffer");
  }
  return cameraBatch;
}

void BatchedRenderSystem::update() {
  SAPIEN_PROFILE_FUNCTION;
  // check pose handle
  if (!mCudaPoseHandle.ptr) {
    throw std::runtime_error("the data source for pose has not been set.");
  }

  ensureCameraRenderScenes();

  // check scene versions
  for (uint32_t i = 0; i < mSystems.size(); ++i) {
    if (mSystems.at(i)->getScene()->getVersion() != mSceneVersions.at(i)) {
      throw std::runtime_error("Modifying a scene (add/remove object/camera) is not allowed after "
                               "creating the batched render system.");
    }
  }

  for (uint32_t i = 0; i < mRenderScenes.size(); ++i) {
    if (mRenderScenes[i]->getVersion() != mRenderSceneVersions[i]) {
      throw std::runtime_error(
          "Modifying an output render scene is not allowed after creating the render system "
          "group.");
    }
  }

  if (mMaximumPoseIndex >= mCudaPoseHandle.shape.at(0)) {
    throw std::runtime_error("GPU pose batch index is outside the pose buffer");
  }

  if (mCudaSceneTransformRefBuffer.cudaId != mCudaPoseHandle.cudaId) {
    throw std::runtime_error("failed to update render system group: cuda pose buffer (cuda:" +
                             std::to_string(mCudaPoseHandle.cudaId) +
                             ") and the "
                             "renderer (cuda:" +
                             std::to_string(mCudaSceneTransformRefBuffer.cudaId) +
                             ") are on different cuda devices.");
  }

  // An RT renderer may create its TLAS after this transport was initialized (notably the Viewer
  // on its first draw). Bind newly available instance buffers before the next transform update.
  bool refreshRTReferences = false;
  for (uint32_t i = 0; i < mRenderScenes.size(); ++i) {
    refreshRTReferences |= !mRTSceneEnabled[i] && mRenderScenes[i]->getTLAS() != nullptr;
  }
  if (refreshRTReferences) {
    std::vector<void *> rtInstanceRefs(mRenderScenes.size(), nullptr);
    for (uint32_t i = 0; i < mRenderScenes.size(); ++i) {
      auto tlas = mRenderScenes[i]->getTLAS();
      if (!tlas) {
        continue;
      }
#ifdef SAPIEN_CUDA
      rtInstanceRefs[i] = tlas->getInstanceBuffer().getCudaPtr();
      mRTSceneEnabled[i] = true;
#endif
    }
    mCudaRTInstanceRefBuffer = CudaArray::FromData(rtInstanceRefs);
  }

  // Upload CPU-owned RT transforms first, then order CUDA after the Vulkan update on the same
  // timeline semaphore. The CUDA kernel below remains the final writer for GPU-bound instances.
#ifdef SAPIEN_CUDA
  if (std::any_of(mRTSceneEnabled.begin(), mRTSceneEnabled.end(),
                  [](bool value) { return value; })) {
    SAPIEN_PROFILE_BLOCK_BEGIN("CPU-owned RT transform update");
    for (uint32_t i = 0; i < mRenderScenes.size(); ++i) {
      if (mRTSceneEnabled[i]) {
        mRenderScenes[i]->updateRTResources(false);
      }
    }
    SAPIEN_PROFILE_BLOCK_END;

    ++mSemValue;
    SapienRenderEngine::Get()->getContext()->getQueue().submit(
        {}, {}, {}, {}, mSem.get(), mSemValue, {});
    cudaExternalSemaphoreWaitParams waitParams{};
    waitParams.flags = 0;
    waitParams.params.fence.value = mSemValue;
    checkCudaErrors(cudaWaitExternalSemaphoresAsync(&mCudaSem, &waitParams, 1, mCudaStream));
  }

  SAPIEN_PROFILE_BLOCK_BEGIN("render object and camera transform update");
  update_object_transforms(
      (float **)mCudaSceneTransformRefBuffer.ptr, (float **)mCudaRTInstanceRefBuffer.ptr,
      mTransformBufferElementByteOffset / 4, (RenderShapeData *)mCudaShapeDataBuffer.ptr,
      (float *)mCudaPoseHandle.ptr, mCudaPoseHandle.shape.at(1), mShapeCount, mCudaStream);

  update_camera_transforms((CameraData *)mCudaCameraDataBuffer.ptr, (float *)mCudaPoseHandle.ptr,
                           mCudaPoseHandle.shape.at(1), mCameraCount, mCudaStream);
  SAPIEN_PROFILE_BLOCK_END;
#endif

  SAPIEN_PROFILE_BLOCK_BEGIN("RT render version update");
  for (uint32_t i = 0; i < mRenderScenes.size(); ++i) {
    if (mRTSceneEnabled[i]) {
      mRenderScenes[i]->updateRenderVersion();
    }
  }
  SAPIEN_PROFILE_BLOCK_END;

  // sync with renderer
  SAPIEN_PROFILE_BLOCK_BEGIN("CUDA Vulkan synchronization");
  notifyUpdate();
  SAPIEN_PROFILE_BLOCK_END;
}

void BatchedRenderSystem::notifyUpdate() {
#ifdef SAPIEN_CUDA
  cudaExternalSemaphoreSignalParams sigParams{};
  sigParams.flags = 0;
  sigParams.params.fence.value = ++mSemValue;
  cudaSignalExternalSemaphoresAsync(&mCudaSem, &sigParams, 1, (cudaStream_t)mCudaStream);
#endif
  vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands;
  SapienRenderEngine::Get()->getContext()->getQueue().submit({}, mSem.get(), stage, mSemValue, {},
                                                             {}, {});
}

void BatchedRenderSystem::setCudaStream(uintptr_t stream) {
  mCudaStream = (cudaStream_t)stream;
  for (auto &c : mCameraBatches) {
    c->setCudaStream(mCudaStream);
  }
}

BatchedRenderSystem ::~BatchedRenderSystem() {
  for (auto const &body : mGpuSourcedBodies) {
    body->internalReleaseGpuPoseSource();
  }
  SapienRenderEngine::Get()->getContext()->getDevice().waitIdle();
#ifdef SAPIEN_CUDA
  cudaDestroyExternalSemaphore(mCudaSem);
#endif
}

} // namespace sapien_renderer
} // namespace sapien
