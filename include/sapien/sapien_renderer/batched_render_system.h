#pragma once

#include "./camera_component.h"
#include "./sapien_renderer_system.h"
#include "sapien/math/pose.h"
#include <atomic>
#include <map>
#include <mutex>
struct CUstream_st;

namespace sapien {
namespace sapien_renderer {

class RenderShape;

class BatchedCamera {
public:
  /** every camera must render on the same device; the group's GPU work runs there */
  BatchedCamera(std::vector<std::shared_ptr<SapienRenderCameraComponent>> cameras,
                std::vector<std::string> renderTargets);
  std::vector<std::shared_ptr<SapienRenderCameraComponent>> const &getCameras() const {
    return mCameras;
  }

  /** Stream for image/pose CUDA work; DLPack exports of this group's buffers order on it. Work
   *  queued on the previous stream stays ordered before work queued on the new one. */
  void setCudaStream(CUstream_st *stream);
  /** The configured CUDA stream as a raw handle (0 is the legacy default stream). */
  uintptr_t getCudaStreamHandle() const;
  /** CUDA device of the group; throws once the group is closed. */
  int getCudaDeviceId() const;
  /** The lock every entry point of this camera group and of its RenderSystemGroup takes
   *  (recursively). Holding it keeps the group from switching streams or closing. */
  std::unique_lock<std::recursive_mutex> lockGpuApi() const {
    return std::unique_lock(*mApiMutex);
  }

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
  /** Terminal close: wait for GPU/Vulkan work, destroy exported semaphores and release
   *  camera/render-scene ownership. Idempotent; refuses while exported CUDA views live. */
  void close();
  bool isClosed() const { return mClosed; }
  int64_t outstandingCudaViewCount() const { return mViewLifecycle->viewCount(); }
  /** Attach this camera group's lifecycle to an exported CUDA handle. */
  CudaArrayHandle trackView(CudaArrayHandle handle) const;
  /** true when any member camera's CPU state version moved past its uploaded state */
  bool internalHasDirtyCameraState() const;
  /** upload dirty CPU camera state (projection and cpu-mode poses); the caller must
   *  have waited for in-flight renders */
  void internalUploadDirtyCameraState();

  ~BatchedCamera();

private:
  friend class BatchedRenderSystem;
  void checkGpuInitialized() const;
  void closeImpl();
  void closeNoThrow();

  std::shared_ptr<SapienRenderEngine> mEngine;
  std::shared_ptr<Device> mDevice;
  std::shared_ptr<svulkan2::core::Context> mContext;
  std::vector<std::shared_ptr<SapienRenderCameraComponent>> mCameras;
  /** Shared with the RenderSystemGroup that created this camera group: its update_render()
   *  drives this group's state, so one lock serializes both against other threads. */
  std::shared_ptr<std::recursive_mutex> mApiMutex{std::make_shared<std::recursive_mutex>()};
  bool mGpuInitialized{false};
  std::atomic<bool> mClosed{false};
  std::shared_ptr<CudaArrayLifecycle> mViewLifecycle{std::make_shared<CudaArrayLifecycle>()};
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
  // Signalled on the group's CUDA stream by take_picture() and waited on by the copy into the
  // exported image buffers, so the copy cannot overwrite an image while work ordered before
  // that stream (consumers on it, or fenced with gpu_wait_for) still reads the previous frame.
  vk::UniqueSemaphore mConsumerSemaphore;
  uint64_t mConsumerValue{0};
  cudaExternalSemaphore_t mCudaConsumerSem{nullptr};
};

class BatchedRenderSystem {
public:
  /** every system must render on the same device, otherwise std::runtime_error names them */
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

  /** Terminal close: closes camera groups, releases all GPU pose seals/sources and
   *  destroys Vulkan/CUDA synchronization resources. Idempotent. */
  void close();
  bool isClosed() const { return mClosed; }

  ~BatchedRenderSystem();

private:
  void checkNotClosed() const;
  void closeImpl();
  void closeNoThrow();
  /** resolve the single device/context of `systems`; throws when they differ */
  void resolveDevice(std::vector<std::shared_ptr<SapienRendererSystem>> const &systems);

  std::shared_ptr<SapienRenderEngine> mEngine;
  std::shared_ptr<Device> mDevice;
  std::shared_ptr<svulkan2::core::Context> mContext;
  std::vector<std::shared_ptr<SapienRendererSystem>> mSystems;
  /** Serializes this group's and its camera groups' entry points across threads; recursive
   *  because entry points call each other. Never held while acquiring the GIL. */
  std::shared_ptr<std::recursive_mutex> mApiMutex{std::make_shared<std::recursive_mutex>()};
  std::atomic<bool> mClosed{false};
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
