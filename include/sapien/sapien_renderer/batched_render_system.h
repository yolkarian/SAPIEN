#pragma once

#include "./camera_component.h"
#include "./sapien_renderer_system.h"
#include "sapien/math/pose.h"
#include <map>
struct CUstream_st;

namespace sapien {
namespace sapien_renderer {

class RenderShape;

class BatchedCamera {
public:
  BatchedCamera(std::vector<std::shared_ptr<SapienRenderCameraComponent>> cameras,
                std::vector<std::string> renderTargets);
  std::vector<std::shared_ptr<SapienRenderCameraComponent>> const &getCameras() const {
    return mCameras;
  }

  void setCudaStream(CUstream_st *stream) { mCudaStream = stream; }

  /** performs all GPU work deferred from construction; called by
   *  BatchedRenderSystem::gpuInit() */
  void internalGpuInit();
  bool isGpuInitialized() const { return mGpuInitialized; }

  void takePicture();
  CudaArrayHandle getPictureCuda(std::string const &name);

  /** configure a member camera's grouped pose source before RenderSystemGroup.gpu_init() */
  void setPoseMode(std::shared_ptr<SapienRenderCameraComponent> const &camera,
                   CameraPoseMode mode);

  /** world pose rows [px, py, pz, qw, qx, qy, qz] for the cameras with pose mode 'cuda' */
  CudaArrayHandle getCudaPoseHandle() const;
  /** row index of a cuda-mode camera in the group's CUDA pose buffer */
  int getCudaPoseIndex(std::shared_ptr<SapienRenderCameraComponent> const &camera) const;
  /** synchronous host-to-device pose copy ordered on the configured CUDA stream */
  void setCudaPose(std::shared_ptr<SapienRenderCameraComponent> const &camera,
                   Pose const &pose);

  int getCudaRowCameraCount() const { return static_cast<int>(mCudaRowCameras.size()); }
  void *getCudaRowCameraDataPtr() const { return mCudaRowCameraDataBuffer.ptr; }
  void *getCudaRowPosePtr() const { return mCudaRowPoseBuffer.ptr; }

  /** host-wait until every submitted render of this camera group completed */
  void internalWaitForRendersIdle();
  /** true when any member camera's CPU state version moved past its uploaded state */
  bool internalHasDirtyCameraState() const;
  /** upload dirty CPU camera state (projection and cpu-mode poses); the caller must
   *  have waited for in-flight renders */
  void internalUploadDirtyCameraState();

  ~BatchedCamera();

private:
  void checkGpuInitialized() const;

  std::vector<std::shared_ptr<SapienRenderCameraComponent>> mCameras;
  bool mGpuInitialized{false};
  std::vector<std::string> mRenderTargets;
  std::map<std::string, std::shared_ptr<svulkan2::core::Buffer>> mCudaImageBuffers;
  std::map<std::string, CudaArrayHandle> mCudaImageHandles;

  // Cameras with pose mode 'cuda' and no GPU pose batch index own a row of
  // mCudaRowPoseBuffer [px, py, pz, qw, qx, qy, qz], seeded once from the CPU pose at
  // gpu_init() and consumed by update_camera_transforms().
  std::vector<std::shared_ptr<SapienRenderCameraComponent>> mCudaRowCameras;
  // Scene-level leases prevent a released camera's CPU renderer from overwriting
  // transform buffers that remain CUDA-owned by another live camera group.
  std::vector<std::shared_ptr<svulkan2::scene::Scene>> mOwnedRenderScenes;
  CudaArray mCudaRowPoseBuffer;
  CudaArray mCudaRowCameraDataBuffer;
  // camera state versions covered by the last CPU upload, parallel to mCameras
  std::vector<uint64_t> mUploadedStateVersions;

  // re-records mCommandBuffer against the cameras' current render target images
  void recordCopyCommands();
  // images referenced by the recorded copy commands; a renderer rebuild replaces the
  // render targets and requires re-recording
  std::vector<vk::Image> mRecordedCopyImages;

  std::unique_ptr<svulkan2::core::CommandPool> mCommandPool;
  vk::UniqueCommandBuffer mCommandBuffer;

  CUstream_st *mCudaStream{nullptr};

  vk::UniqueSemaphore mSemaphore;
  uint64_t mFrameCounter{0};
  cudaExternalSemaphore_t mCudaSem{nullptr};
};

class BatchedRenderSystem {
public:
  BatchedRenderSystem(std::vector<std::shared_ptr<SapienRendererSystem>> systems);
  BatchedRenderSystem(
      std::vector<std::shared_ptr<SapienRendererSystem>> systems,
      std::shared_ptr<svulkan2::scene::Scene> renderScene,
      std::vector<std::shared_ptr<SapienRenderBodyComponent>> gpuSourcedBodies);

  void init();

  /** One-time initialization and seal: resolves output render scenes, parses camera
   *  and light pose modes, prepares camera resources, takes the CPU snapshots, seeds
   *  cuda-camera pose rows, seals pose ownership, and freezes topology. Configure
   *  pose sources, pose modes, and camera groups before calling; steady state
   *  afterwards is update() + capture only. */
  void gpuInit();
  bool isGpuInitialized() const { return mGpuInitialized; }

  void setPoseSource(CudaArrayHandle const &poses);

  std::shared_ptr<BatchedCamera>
  createCameraBatch(std::vector<std::shared_ptr<SapienRenderCameraComponent>> cameras,
                    std::vector<std::string> renderTargets);

  void update();
  void setCudaStream(uintptr_t);

  ~BatchedRenderSystem();

private:
  std::vector<std::shared_ptr<SapienRendererSystem>> mSystems;
  std::shared_ptr<svulkan2::scene::Scene> mFixedRenderScene;
  bool mAutoBindPhysxGpuPoses{true};
  std::vector<std::shared_ptr<SapienRenderBodyComponent>> mFixedGpuSourcedBodies;
  std::vector<std::shared_ptr<SapienRendererSystem>> mTrackedSystems;
  std::vector<uint64_t> mSceneVersions;
  std::vector<std::shared_ptr<svulkan2::scene::Scene>> mRenderScenes;
  std::vector<uint64_t> mRenderSceneVersions;
  std::vector<std::vector<std::shared_ptr<SapienRendererSystem>>> mRenderSceneSystems;
  std::vector<std::vector<std::shared_ptr<SapienRendererSystem>>> mAdditionalRenderSelections;

  /** external poses array */
  CudaArrayHandle mCudaPoseHandle;

  std::vector<std::shared_ptr<BatchedCamera>> mCameraBatches;

  // size of a mat4 element in the transform buffer
  int mTransformBufferElementByteOffset{0};

  int mShapeCount{0};
  int mMaximumPoseIndex{-1};
  CudaArray mCudaSceneTransformRefBuffer;
  CudaArray mCudaRTInstanceRefBuffer;
  CudaArray mCudaShapeDataBuffer;
  std::vector<bool> mRTSceneEnabled;
  std::vector<std::shared_ptr<SapienRenderBodyComponent>> mGpuSourcedBodies;
  // Pose bindings and static snapshots sealed at gpuInit(); released on destruction.
  std::vector<std::shared_ptr<RenderShape>> mSealedShapes;
  std::vector<std::shared_ptr<SapienRenderBodyComponent>> mSealedStaticBodies;
  std::vector<std::shared_ptr<PointCloudComponent>> mSealedPointClouds;
  std::vector<std::shared_ptr<SapienRenderLightComponent>> mSealedLights;
  // scene light state versions covered by the last CPU upload, parallel to
  // mRenderSceneSystems
  std::vector<std::vector<uint64_t>> mLightStateVersions;
  bool mGpuInitialized{false};

  int mCameraCount{0};
  bool mRequiresPrimaryPoseSource{false};
  CudaArray mCudaCameraDataBuffer;

  CUstream_st *mCudaStream{nullptr};

  bool collectCameraRenderSelections();
  void assignCameraRenderScenes();
  // refresh CPU components (static tamper checks + dirty version accumulation), then
  // upload dirty CPU camera and scene/light state; zero uploads when nothing changed
  void refreshAndUploadCpuState();

  // semaphore to notify scene update
  void notifyUpdate();
  vk::UniqueSemaphore mSem{};
  cudaExternalSemaphore_t mCudaSem{nullptr};
  uint64_t mSemValue{0};
};

} // namespace sapien_renderer
} // namespace sapien
