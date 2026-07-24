#include "sapien/sapien_renderer/batched_render_system.h"
#include "./batched_render_system.cuh"
#include "render_scene_resolver.h"
#include "sapien/entity.h"
#include "sapien/physx/articulation_link_component.h"
#include "sapien/physx/physx_system.h"
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

  mRenderTargets = renderTargets;

  size_t registeredCount = 0;
  try {
    for (auto &cam : mCameras) {
      cam->internalRegisterGpuOwnership(this);
      ++registeredCount;
    }
  } catch (...) {
    for (size_t i = 0; i < registeredCount; ++i) {
      mCameras[i]->internalReleaseGpuOwnership(this);
    }
    throw;
  }
}

void BatchedCamera::checkGpuInitialized() const {
  if (!mGpuInitialized) {
    throw std::runtime_error(
        "the camera group is not initialized: call RenderSystemGroup.gpu_init() first");
  }
}

void BatchedCamera::internalGpuInit() {
  if (mGpuInitialized) {
    return;
  }

  auto context = SapienRenderEngine::Get()->getContext();
  mCommandPool = context->createCommandPool();
  mCommandBuffer = mCommandPool->allocateCommandBuffer();

  uint32_t width = mCameras.at(0)->getWidth();
  uint32_t height = mCameras.at(0)->getHeight();

#ifdef SAPIEN_CUDA
  if (mOwnedRenderScenes.empty()) {
    std::unordered_set<svulkan2::scene::Scene *> acquiredScenes;
    for (auto const &camera : mCameras) {
      auto scene = camera->getInternalRenderScene();
      if (acquiredScenes.insert(scene.get()).second) {
        scene->acquireExternalTransformOwnership(this);
        mOwnedRenderScenes.push_back(std::move(scene));
      }
    }
  }

  // Enter grouped ownership before resource preparation. prepareResources() must
  // never perform an implicit CPU transform upload; gpuInit() takes the one explicit
  // CPU snapshot after resources exist.
  for (auto &cam : mCameras) {
    auto &renderer = cam->getInternalRenderer();
    renderer.setExecutionMode(svulkan2::renderer::RenderExecutionMode::eGroupedGpu);
    if (auto raster = dynamic_cast<svulkan2::renderer::Renderer *>(&renderer)) {
      raster->setExternalTransformUpdatesEnabled(true);
    } else if (auto rt = dynamic_cast<svulkan2::renderer::RTRenderer *>(&renderer)) {
      rt->setExternalTransformUpdatesEnabled(true);
      rt->setExternalCameraUpdatesEnabled(true);
    }
  }
#endif

  // prepare pipelines/targets/records and take the one-time CPU snapshot per camera
  for (auto &cam : mCameras) {
    cam->gpuInit();
  }

  for (uint32_t i = 0; i < mCameras.size(); ++i) {
    auto &cam = mCameras[i];

    for (auto &name : mRenderTargets) {
      auto &image = cam->getInternalImage(name);
      auto extent = image.getExtent();
      vk::Format format = image.getFormat();
      vk::DeviceSize imageSize =
          extent.width * extent.height * extent.depth * svulkan2::getFormatSize(format);
      if (!mCudaImageBuffers.contains(name)) {
        mCudaImageBuffers[name] = svulkan2::core::Buffer::Create(
            imageSize * mCameras.size(),
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_GPU_ONLY, VmaAllocationCreateFlags{}, true);
        {
          CudaArrayHandle array;
          int channels = getFormatChannels(format);
          int itemsize = getFormatChannelSize(format);
          array.shape = {static_cast<int>(mCameras.size()), static_cast<int>(extent.height),
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
    }
  }

#ifdef SAPIEN_CUDA
  // Cameras without a GPU pose batch index become group-owned free cameras. Their world pose
  // lives in a CUDA row [px, py, pz, qw, qx, qy, qz], seeded once from the current CPU global
  // pose; update_render() derives the camera matrices from that row. The CPU pose is sealed
  // below, so grouped capture involves no CPU camera upload.
  checkCudaErrors(cudaSetDevice(SapienRenderEngine::Get()->getDevice()->cudaId));
  std::vector<float> seedPoses;
  std::vector<CameraData> freeCameraData;
  for (auto &cam : mCameras) {
    if (cam->getGpuBatchedPoseIndex() >= 0) {
      continue;
    }
    Pose pose = cam->getGlobalPose();
    seedPoses.insert(seedPoses.end(),
                     {pose.p.x, pose.p.y, pose.p.z, pose.q.w, pose.q.x, pose.q.y, pose.q.z});
    CameraData data;
    data.buffer = cam->getCudaBuffer().ptr;
    data.localPose = POSE_GL_TO_ROS;
    data.poseIndex = static_cast<int>(mFreeCameras.size());
    freeCameraData.push_back(data);
    mFreeCameras.push_back(cam);
  }
  if (!mFreeCameras.empty()) {
    mCudaFreeCameraPoseBuffer = CudaArray({static_cast<int>(mFreeCameras.size()), 7}, "f4");
    checkCudaErrors(cudaMemcpy(mCudaFreeCameraPoseBuffer.ptr, seedPoses.data(),
                               seedPoses.size() * sizeof(float), cudaMemcpyHostToDevice));
    mCudaFreeCameraDataBuffer = CudaArray::FromData(freeCameraData);
  }
#endif

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

  // From here on every camera transform in this group is GPU-owned: mounted cameras
  // follow their parent CUDA pose row and free cameras follow their group-owned row.
  for (auto &cam : mCameras) {
    cam->internalSealGpuOwnership(this);
  }
  mGpuInitialized = true;
}

void BatchedCamera::recordCopyCommands() {
  mRecordedCopyImages.clear();
  mCommandBuffer->reset();
  mCommandBuffer->begin(vk::CommandBufferBeginInfo());
  for (uint32_t i = 0; i < mCameras.size(); ++i) {
    auto &cam = mCameras[i];
    for (auto &name : mRenderTargets) {
      auto &image = cam->getInternalImage(name);
      auto extent = image.getExtent();
      vk::DeviceSize imageSize = extent.width * extent.height * extent.depth *
                                 svulkan2::getFormatSize(image.getFormat());
      image.recordCopyToBuffer(mCommandBuffer.get(),
                               cam->getInternalRenderer().getRenderTargetImageLayout(name),
                               mCudaImageBuffers.at(name)->getVulkanBuffer(), i * imageSize,
                               imageSize, {0, 0, 0}, extent, 0);
      mRecordedCopyImages.push_back(image.getVulkanImage());
    }
  }
  mCommandBuffer->end();
}

void BatchedCamera::takePicture() {
  checkGpuInitialized();
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
  // Re-record the copies only when a renderer rebuild replaced the render target images;
  // recorded copy commands against destroyed vk::Image handles silently read reused images.
  std::vector<vk::Image> currentImages;
  for (auto &cam : mCameras) {
    for (auto &name : mRenderTargets) {
      currentImages.push_back(cam->getInternalImage(name).getVulkanImage());
    }
  }
  if (currentImages != mRecordedCopyImages) {
    recordCopyCommands();
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

CudaArrayHandle BatchedCamera::getFreeCameraPoseHandle() const {
  checkGpuInitialized();
#ifdef SAPIEN_CUDA
  if (mFreeCameras.empty()) {
    throw std::runtime_error("this camera group has no free cameras: every camera derives its "
                             "pose from a GPU pose batch index");
  }
  return CudaArrayHandle{.shape = {static_cast<int>(mFreeCameras.size()), 7},
                         .strides = {28, 4},
                         .type = "f4",
                         .cudaId = mCudaFreeCameraPoseBuffer.cudaId,
                         .ptr = mCudaFreeCameraPoseBuffer.ptr};
#else
  throw std::runtime_error("sapien is not compiled with CUDA support");
#endif
}

int BatchedCamera::getFreeCameraPoseIndex(
    std::shared_ptr<SapienRenderCameraComponent> const &camera) const {
  checkGpuInitialized();
  for (size_t i = 0; i < mFreeCameras.size(); ++i) {
    if (mFreeCameras[i] == camera) {
      return static_cast<int>(i);
    }
  }
  if (camera->getGpuBatchedPoseIndex() >= 0 &&
      std::find(mCameras.begin(), mCameras.end(), camera) != mCameras.end()) {
    throw std::runtime_error("this camera derives its pose from its mounted GPU parent "
                             "body/link; it has no free-camera pose row");
  }
  throw std::runtime_error("the camera is not a free camera of this camera group");
}

void BatchedCamera::setFreeCameraPose(
    std::shared_ptr<SapienRenderCameraComponent> const &camera, Pose const &pose) {
#ifdef SAPIEN_CUDA
  int row = getFreeCameraPoseIndex(camera);
  checkCudaErrors(cudaSetDevice(SapienRenderEngine::Get()->getDevice()->cudaId));
  float data[7] = {pose.p.x, pose.p.y, pose.p.z, pose.q.w, pose.q.x, pose.q.y, pose.q.z};
  // Synchronous convenience path. Performance-sensitive callers should write
  // cuda_free_camera_poses directly on the group's configured CUDA stream.
  checkCudaErrors(cudaMemcpy(static_cast<float *>(mCudaFreeCameraPoseBuffer.ptr) +
                                 static_cast<size_t>(row) * 7,
                             data, sizeof(data), cudaMemcpyHostToDevice));
#else
  throw std::runtime_error("sapien is not compiled with CUDA support");
#endif
}

CudaArrayHandle BatchedCamera::getPictureCuda(std::string const &name) {
  checkGpuInitialized();
  if (!mCudaImageHandles.contains(name)) {
    throw std::runtime_error("Failed to get image with name :" + name +
                             ". Did you forget to specify it in create_camera_group?");
  }
  return mCudaImageHandles.at(name);
}

BatchedCamera::~BatchedCamera() {
  SapienRenderEngine::Get()->getContext()->getDevice().waitIdle();
#ifdef SAPIEN_CUDA
  if (mCudaSem) {
    cudaDestroyExternalSemaphore(mCudaSem);
  }
#endif
  for (auto &camera : mCameras) {
    camera->internalReleaseGpuOwnership(this);
  }
  for (auto const &scene : mOwnedRenderScenes) {
    scene->releaseExternalTransformOwnership(this);
  }
}

BatchedRenderSystem::BatchedRenderSystem(
    std::vector<std::shared_ptr<SapienRendererSystem>> systems)
    : mSystems(RenderSceneResolver::resolve(systems, {})) {
  if (mSystems.empty()) {
    throw std::runtime_error("systems must not be empty");
  }
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
  std::unordered_set<SapienRendererSystem *> steppedSystems;
  for (auto &system : mSystems) {
    // run a step
    system->step();
    steppedSystems.insert(system.get());

    // cache current versions
    mSceneVersions.push_back(system->getScene()->getVersion());
  }
  // camera-selected shared systems also need current model matrices before the
  // one-time CPU transform seed below
  for (auto const &selection : mAdditionalRenderSelections) {
    for (auto const &system : selection) {
      if (steppedSystems.insert(system.get()).second) {
        system->step();
      }
    }
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

    // Seed every CPU-owned transform (static bodies, ground planes) once per render scene.
    // A scene already leased by another camera group retains its existing static seed and
    // CUDA-authored rows; initializing an additional group must not overwrite either.
    if (!renderScene->hasExternalTransformOwnership()) {
      renderScene->uploadObjectTransforms();
    }

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
  if (mGpuInitialized) {
    throw std::runtime_error(
        "failed to set CUDA poses: the render system group is already initialized");
  }
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
    // Resolve against every live render system so batched-render-shared scenes
    // (for example shared ground/terrain) join the camera's output render scene,
    // matching the resolution direct capture would compute.
    auto resolvedSystems =
        camera->internalResolveRenderSystems(SapienRenderEngine::Get()->getRenderSystems());
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
    auto resolvedSystems =
        camera->internalResolveRenderSystems(SapienRenderEngine::Get()->getRenderSystems());
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
  if (mGpuInitialized) {
    throw std::runtime_error(
        "failed to create camera group: the render system group is already initialized; create "
        "all camera groups before calling gpu_init()");
  }
  // Camera ownership is reserved here, but mounted PhysX bindings are resolved at
  // gpu_init() after all referenced PhysX GPU systems have final pose indices.
  auto cameraBatch = std::make_shared<BatchedCamera>(cameras, renderTargets);
  cameraBatch->setCudaStream(mCudaStream);
  mCameraBatches.push_back(cameraBatch);
  return cameraBatch;
}

void BatchedRenderSystem::gpuInit() {
  if (mGpuInitialized) {
    throw std::runtime_error("the render system group is already initialized");
  }

#ifdef SAPIEN_CUDA
  // PhysX GPU owns the authoritative body/link pose indices. Every referenced
  // system must be initialized before mounted cameras and render shapes are bound.
  std::unordered_set<physx::PhysxSystemGpu *> validatedSystems;
  auto validatePhysxGpu = [&](std::shared_ptr<Scene> const &scene) {
    auto physxSystem =
        std::dynamic_pointer_cast<physx::PhysxSystemGpu>(scene->getPhysxSystem());
    if (!physxSystem || !validatedSystems.insert(physxSystem.get()).second) {
      return;
    }
    if (!physxSystem->isInitialized()) {
      throw std::runtime_error(
          "failed to initialize render system group: every referenced PhysxGpuSystem must "
          "complete gpu_init() first");
    }
  };
  for (auto const &system : mSystems) {
    for (auto const &body : system->getRenderBodyComponents()) {
      validatePhysxGpu(body->getEntity()->getScene());
    }
  }
  for (auto const &batch : mCameraBatches) {
    for (auto const &camera : batch->getCameras()) {
      validatePhysxGpu(camera->getEntity()->getScene());
    }
  }

  // Resolve mounted-camera bindings only after PhysX assigned final pose indices.
  for (auto const &batch : mCameraBatches) {
    for (auto const &camera : batch->getCameras()) {
      if (camera->getGpuBatchedPoseIndex() >= 0) {
        continue;
      }
      auto entity = camera->getEntity();
      auto gpuSystem = std::dynamic_pointer_cast<physx::PhysxSystemGpu>(
          entity->getScene()->getPhysxSystem());
      if (!gpuSystem) {
        continue;
      }
      int poseIndex = -1;
      bool hasDynamicParent = false;
      if (auto rigid = entity->getComponent<physx::PhysxRigidDynamicComponent>()) {
        poseIndex = rigid->getGpuPoseIndex();
        hasDynamicParent = true;
      } else if (auto link = entity->getComponent<physx::PhysxArticulationLinkComponent>()) {
        poseIndex = link->getGpuPoseIndex();
        hasDynamicParent = true;
      }
      if (hasDynamicParent && poseIndex < 0) {
        throw std::runtime_error(
            "failed to initialize mounted camera: its PhysX GPU parent has no pose index");
      }
      if (poseIndex >= 0) {
        camera->setGpuBatchedPoseIndex(poseIndex);
      }
    }
  }
#endif

  // Resolve base and camera-selected output render scenes, freeze topology, and
  // seed every CPU-owned object transform once.
  init();
  ensureCameraRenderScenes();

  // Finish deterministic source validation before preparing or sealing any camera.
  for (auto const &batch : mCameraBatches) {
    for (auto const &camera : batch->getCameras()) {
      mMaximumPoseIndex = std::max(mMaximumPoseIndex, camera->getGpuBatchedPoseIndex());
    }
  }
  mRequiresPrimaryPoseSource = mShapeCount > 0 || mMaximumPoseIndex >= 0;
  if (mRequiresPrimaryPoseSource && !mCudaPoseHandle.ptr) {
    throw std::runtime_error(
        "failed to initialize render system group: GPU objects or mounted cameras require "
        "set_cuda_poses() before gpu_init()");
  }
  if (mCudaPoseHandle.ptr) {
    if (mMaximumPoseIndex >= mCudaPoseHandle.shape.at(0)) {
      throw std::runtime_error("GPU pose batch index is outside the pose buffer");
    }
    if (mCudaSceneTransformRefBuffer.cudaId != mCudaPoseHandle.cudaId) {
      throw std::runtime_error(
          "failed to initialize render system group: CUDA pose buffer and renderer are on "
          "different CUDA devices");
    }
  }

  // Prepare per-camera resources, take the CPU snapshots, create image and
  // free-camera pose buffers, and seal camera ownership.
  for (auto &batch : mCameraBatches) {
    batch->internalGpuInit();
  }

  // Consolidated update descriptors for cameras with GPU pose batch indices; free
  // cameras use their per-group registered pose source instead.
  std::vector<CameraData> allCamData;
  mCameraCount = 0;
  for (auto &cb : mCameraBatches) {
    for (auto &cam : cb->getCameras()) {
      int index = cam->getGpuBatchedPoseIndex();
      if (index < 0) {
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
  mCudaCameraDataBuffer = CudaArray::FromData(allCamData);
#endif

  // Pose-source registry tracks only valid sources: the optional primary
  // object/mounted-camera source followed by each free-camera source.
  mCudaPoseSources.clear();
  if (mCudaPoseHandle.ptr) {
    mCudaPoseSources.push_back({mCudaPoseHandle});
  }
  for (auto &cb : mCameraBatches) {
    if (cb->getFreeCameraCount() > 0) {
      mCudaPoseSources.push_back({cb->getFreeCameraPoseHandle()});
    }
  }

  // RT instance buffers exist now if camera resource preparation built a TLAS.
  static_assert(sizeof(vk::AccelerationStructureInstanceKHR) == sizeof(float) * 16);
  std::vector<void *> rtInstanceRefs(mRenderScenes.size(), nullptr);
  mRTSceneEnabled.assign(mRenderScenes.size(), false);
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

  // Seed CPU-owned RT instance transforms once. Sealed groups never repeat this per
  // frame; the Viewer's fixed-scene group stays CPU-managed and re-uploads per update.
  for (uint32_t i = 0; i < mRenderScenes.size(); ++i) {
    if (mRTSceneEnabled[i]) {
      mRenderScenes[i]->updateRTResources(false);
    }
  }

  // Seal every shape's pose-source binding. The update descriptors now own the
  // selected row indices until the group is destroyed.
  if (!mFixedRenderScene) {
    std::unordered_set<RenderShape *> sealedShapes;
    for (auto const &selection : mRenderSceneSystems) {
      for (auto const &system : selection) {
        for (auto const &body : system->getRenderBodyComponents()) {
          for (auto const &shape : body->getRenderShapes()) {
            if (sealedShapes.insert(shape.get()).second) {
              shape->internalSealGpuBatchedPoseIndex();
              mSealedShapes.push_back(shape);
            }
          }
        }
      }
    }
  }

  // Seal static snapshots: bodies without a CUDA pose source were copied to the GPU
  // exactly once; later CPU pose changes are errors. The Viewer's fixed-scene group
  // is CPU-managed by design and does not seal.
  if (!mFixedRenderScene) {
    std::unordered_set<SapienRenderBodyComponent *> gpuSourced;
    for (auto const &body : mGpuSourcedBodies) {
      gpuSourced.insert(body.get());
    }
    std::unordered_set<SapienRenderBodyComponent *> sealed;
    for (auto const &selection : mRenderSceneSystems) {
      for (auto const &system : selection) {
        for (auto const &body : system->getRenderBodyComponents()) {
          if (!gpuSourced.contains(body.get()) && sealed.insert(body.get()).second) {
            body->internalSealStaticPose();
            mSealedStaticBodies.push_back(body);
          }
        }
      }
    }
  }

  mGpuInitialized = true;
}

void BatchedRenderSystem::update() {
  SAPIEN_PROFILE_FUNCTION;
  if (!mGpuInitialized) {
    throw std::runtime_error(
        "failed to update render system group: call gpu_init() after configuring pose sources "
        "and camera groups");
  }
  if (mRequiresPrimaryPoseSource && !mCudaPoseHandle.ptr) {
    throw std::runtime_error("the primary CUDA pose source has not been set");
  }

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

  if (mCudaPoseHandle.ptr) {
    if (mMaximumPoseIndex >= mCudaPoseHandle.shape.at(0)) {
      throw std::runtime_error("GPU pose batch index is outside the pose buffer");
    }
    if (mCudaSceneTransformRefBuffer.cudaId != mCudaPoseHandle.cudaId) {
      throw std::runtime_error("failed to update render system group: cuda pose buffer (cuda:" +
                               std::to_string(mCudaPoseHandle.cudaId) +
                               ") and the renderer (cuda:" +
                               std::to_string(mCudaSceneTransformRefBuffer.cudaId) +
                               ") are on different cuda devices.");
    }
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
    // seed CPU-owned instance transforms for buffers that just appeared
    for (uint32_t i = 0; i < mRenderScenes.size(); ++i) {
      if (mRTSceneEnabled[i]) {
        mRenderScenes[i]->updateRTResources(false);
      }
    }
  }

  // The Viewer's fixed-scene group is CPU-managed: its CPU-moved helpers (gizmos,
  // axes) re-upload RT instance transforms every update, and the CUDA kernel below
  // remains the final writer for GPU-bound instances. Sealed groups seeded CPU
  // instances once at gpu_init() and never repeat this per frame.
#ifdef SAPIEN_CUDA
  if (mFixedRenderScene && std::any_of(mRTSceneEnabled.begin(), mRTSceneEnabled.end(),
                                       [](bool value) { return value; })) {
    SAPIEN_PROFILE_BLOCK_BEGIN("CPU-owned RT transform update");
    for (uint32_t i = 0; i < mRenderScenes.size(); ++i) {
      if (mRTSceneEnabled[i]) {
        mRenderScenes[i]->updateRTResources(false);
      }
    }
    SAPIEN_PROFILE_BLOCK_END;
  }

  // Order all prior Vulkan queue work (raster and RT renders that may still read the shared
  // object transform, camera, and RT instance buffers, plus the RT resource upload above)
  // before CUDA rewrites those buffers on mCudaStream. Together with notifyUpdate() this forms
  // the bidirectional handoff
  //   prior Vulkan read -> semaphore -> CUDA write -> semaphore -> next Vulkan read.
  // Restricting the first direction to RT scenes let raster renders race the CUDA transform
  // writes and observe torn or stale link poses.
  ++mSemValue;
  SapienRenderEngine::Get()->getContext()->getQueue().submit(
      {}, {}, {}, {}, mSem.get(), mSemValue, {});
  cudaExternalSemaphoreWaitParams waitParams{};
  waitParams.flags = 0;
  waitParams.params.fence.value = mSemValue;
  checkCudaErrors(cudaWaitExternalSemaphoresAsync(&mCudaSem, &waitParams, 1, mCudaStream));

  SAPIEN_PROFILE_BLOCK_BEGIN("render object and camera transform update");
  if (mShapeCount > 0) {
    update_object_transforms(
        (float **)mCudaSceneTransformRefBuffer.ptr, (float **)mCudaRTInstanceRefBuffer.ptr,
        mTransformBufferElementByteOffset / 4, (RenderShapeData *)mCudaShapeDataBuffer.ptr,
        (float *)mCudaPoseHandle.ptr, mCudaPoseHandle.shape.at(1), mShapeCount, mCudaStream);
  }

  if (mCameraCount > 0) {
    update_camera_transforms((CameraData *)mCudaCameraDataBuffer.ptr,
                             (float *)mCudaPoseHandle.ptr, mCudaPoseHandle.shape.at(1),
                             mCameraCount, mCudaStream);
  }

  // free cameras read their group-owned world-pose rows
  for (auto &batch : mCameraBatches) {
    if (int freeCameraCount = batch->getFreeCameraCount()) {
      update_camera_transforms((CameraData *)batch->getFreeCameraDataPtr(),
                               (float *)batch->getFreeCameraPosePtr(), 7, freeCameraCount,
                               mCudaStream);
    }
  }
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
  for (auto const &shape : mSealedShapes) {
    shape->internalReleaseGpuBatchedPoseIndexSeal();
  }
  for (auto const &body : mSealedStaticBodies) {
    body->internalReleaseStaticPoseSeal();
  }
  SapienRenderEngine::Get()->getContext()->getDevice().waitIdle();
#ifdef SAPIEN_CUDA
  if (mCudaSem) {
    cudaDestroyExternalSemaphore(mCudaSem);
  }
#endif
}

} // namespace sapien_renderer
} // namespace sapien
