#pragma once

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

  /** world pose rows [px, py, pz, qw, qx, qy, qz] for the group-owned free cameras */
  CudaArrayHandle getFreeCameraPoseHandle() const;
  /** row index of a free camera in the group's CUDA pose buffer */
  int getFreeCameraPoseIndex(std::shared_ptr<SapienRenderCameraComponent> const &camera) const;
  /** explicit host-to-device copy of one CPU-authored pose into a free camera's CUDA row */
  void setFreeCameraPose(std::shared_ptr<SapienRenderCameraComponent> const &camera,
                         Pose const &pose);

  int getFreeCameraCount() const { return static_cast<int>(mFreeCameras.size()); }
  void *getFreeCameraDataPtr() const { return mCudaFreeCameraDataBuffer.ptr; }
  void *getFreeCameraPosePtr() const { return mCudaFreeCameraPoseBuffer.ptr; }

  ~BatchedCamera();

private:
  void checkGpuInitialized() const;

  std::vector<std::shared_ptr<SapienRenderCameraComponent>> mCameras;
  bool mGpuInitialized{false};
  std::vector<std::string> mRenderTargets;
  std::map<std::string, std::shared_ptr<svulkan2::core::Buffer>> mCudaImageBuffers;
  std::map<std::string, CudaArrayHandle> mCudaImageHandles;

  // Cameras without a GPU pose batch index are group-owned free cameras: their world
  // pose lives in a row of mCudaFreeCameraPoseBuffer, seeded once from the CPU pose at
  // gpu_init() and consumed by update_camera_transforms().
  std::vector<std::shared_ptr<SapienRenderCameraComponent>> mFreeCameras;
  // Scene-level leases prevent a released camera's CPU renderer from overwriting
  // transform buffers that remain CUDA-owned by another live camera group.
  std::vector<std::shared_ptr<svulkan2::scene::Scene>> mOwnedRenderScenes;
  CudaArray mCudaFreeCameraPoseBuffer;
  CudaArray mCudaFreeCameraDataBuffer;

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

  /** One-time initialization and seal: resolves output render scenes, prepares
   *  camera resources, takes the CPU snapshots, seeds free-camera pose rows,
   *  seals camera and static-body ownership, and freezes topology. Configure
   *  pose sources and camera groups before calling; steady state afterwards is
   *  update() + capture only. */
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
  std::vector<uint64_t> mSceneVersions;
  std::vector<std::shared_ptr<svulkan2::scene::Scene>> mRenderScenes;
  std::vector<uint64_t> mRenderSceneVersions;
  std::vector<std::vector<std::shared_ptr<SapienRendererSystem>>> mRenderSceneSystems;
  std::vector<std::vector<std::shared_ptr<SapienRendererSystem>>> mAdditionalRenderSelections;

  /** external poses array */
  CudaArrayHandle mCudaPoseHandle;

  /** Pose-source registry: the optional primary object/mounted-camera source is
   *  followed by one free-camera source per applicable camera group. Future
   *  producers register additional sources without changing transform ownership. */
  struct CudaPoseSource {
    CudaArrayHandle poses;
  };
  std::vector<CudaPoseSource> mCudaPoseSources;

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
  bool mGpuInitialized{false};

  int mCameraCount{0};
  bool mRequiresPrimaryPoseSource{false};
  CudaArray mCudaCameraDataBuffer;

  CUstream_st *mCudaStream{nullptr};

  void ensureCameraRenderScenes(
      std::vector<std::shared_ptr<SapienRenderCameraComponent>> const &additionalCameras = {});

  // semaphore to notify scene update
  void notifyUpdate();
  vk::UniqueSemaphore mSem{};
  cudaExternalSemaphore_t mCudaSem{nullptr};
  uint64_t mSemValue{0};
};

} // namespace sapien_renderer
} // namespace sapien
