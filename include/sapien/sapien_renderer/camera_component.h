#pragma once
#include "../component.h"
#include "image.h"
#include "sapien/math/mat.h"
#include "sapien/math/pose.h"
#include <map>
#include <svulkan2/renderer/renderer_base.h>
#include <svulkan2/scene/camera.h>
#include <variant>

namespace sapien {
class Entity;
namespace sapien_renderer {
struct SapienRenderCameraInternal;
class SapienRenderTexture;
class SapienRendererSystem;

enum class CameraMode { ePerspective, eOrthographic };

/** Pose source of a grouped camera after RenderSystemGroup.gpu_init().
 *  eStatic: the pose is a one-time CPU snapshot; CPU pose setters raise after seal.
 *  eCpu: the CPU component pose stays authoritative and uploads when dirty at
 *  RenderSystemGroup.update_render().
 *  eCuda: the pose lives in a group-owned CUDA row [p, q(wxyz)]; CPU pose setters raise.
 *  Cameras mounted on GPU bodies are automatically CUDA-attached (they reuse the PhysX
 *  pose row without allocating a group row) and cannot be configured eCpu or eStatic. */
enum class CameraPoseMode { eStatic, eCpu, eCuda };

class SapienRenderCameraComponent : public Component {
public:
  SapienRenderCameraComponent(uint32_t width, uint32_t height, std::string const &shaderDir);

  virtual void onAddToScene(Scene &scene) override;
  virtual void onRemoveFromScene(Scene &scene) override;

  void setLocalPose(Pose const &);
  Pose getLocalPose() const;
  Pose getGlobalPose() const;

  // getters
  inline uint32_t getWidth() const { return mWidth; }
  inline uint32_t getHeight() const { return mHeight; }
  inline float getFocalLengthX() const { return mFx; }
  inline float getFocalLengthY() const { return mFy; }
  inline float getFovX() const { return std::atan(mWidth / 2.f / mFx) * 2.f; }
  inline float getFovY() const { return std::atan(mHeight / 2.f / mFy) * 2.f; }
  inline float getNear() const { return mNear; }
  inline float getFar() const { return mFar; }
  inline float getPrincipalPointX() const { return mCx; }
  inline float getPrincipalPointY() const { return mCy; }
  inline float getSkew() const { return mSkew; }

  inline float getOrthoLeft() const { return mLeft; }
  inline float getOrthoRight() const { return mRight; }
  inline float getOrthoBottom() const { return mBottom; }
  inline float getOrthoTop() const { return mTop; }

  Mat4 getModelMatrix() const;
  Mat4 getProjectionMatrix() const;
  Mat3 getIntrinsicMatrix() const;
  Mat34 getExtrinsicMatrix() const;

  // setters
  void setPerspectiveParameters(float near, float far, float fx, float fy, float cx, float cy,
                                float skew);
  void setFocalLengths(float fx, float fy);
  void setFovX(float fovx, bool computeY = true);
  void setFovY(float fovy, bool computeX = true);
  void setNear(float near);
  void setFar(float far);
  void setPrincipalPoint(float cx, float cy);
  void setSkew(float s);

  void setOrthographicParameters(float near, float far, float top);
  void setOrthographicParameters(float near, float far, float left, float right, float bottom,
                                 float top);

  inline CameraMode getMode() const { return mMode; }

  void takePicture();
  void setScenes(std::vector<std::shared_ptr<Scene>> const &scenes);
  std::vector<std::shared_ptr<SapienRendererSystem>> internalResolveRenderSystems(
      std::vector<std::shared_ptr<SapienRendererSystem>> const &contextSystems);
  std::vector<std::string> getImageNames() const;
  SapienRenderImageCpu getImage(std::string const &name);
  SapienRenderImageCuda getImageCuda(std::string const &name);

  // TODO: make the following serializable
  void setProperty(std::string const &name, int property);
  void setProperty(std::string const &name, float property);
  void setTexture(std::string const &name, std::shared_ptr<SapienRenderTexture> texture);
  void setTextureArray(std::string const &name,
                       std::vector<std::shared_ptr<SapienRenderTexture>> texture);

  void internalUpdate();

  // GPU apis
  void gpuInit();
  CudaArrayHandle getCudaBuffer();
  void setGpuBatchedPoseIndex(int);
  int getGpuBatchedPoseIndex() const;
  /** reserve this camera for exactly one RenderCameraGroup before GPU initialization */
  void internalRegisterGpuOwnership(void const *owner);
  /** configure the grouped pose mode; only valid for the registered owner before seal */
  void internalSetPoseMode(void const *owner, CameraPoseMode mode);
  CameraPoseMode getPoseMode() const { return mPoseMode; }
  /** true when a RenderCameraGroup explicitly configured the pose mode */
  bool isPoseModeConfigured() const { return mPoseModeConfigured; }
  /** seal the registered camera pose after its one-time CPU snapshot */
  void internalSealGpuOwnership(void const *owner);
  /** release the group's ownership and restore ordinary CPU-managed rendering */
  void internalReleaseGpuOwnership(void const *owner);
  bool isGpuOwnershipSealed() const { return mGpuOwnershipSealed; }
  /** coarse dirty version covering projection/intrinsics and (for eCpu mode) pose */
  uint64_t getCameraStateVersion() const { return mCameraStateVersion; }
  void internalSetRenderScene(
      std::shared_ptr<svulkan2::scene::Scene> scene,
      std::vector<std::shared_ptr<SapienRendererSystem>> const &resolvedSystems = {});
  svulkan2::core::Image &getInternalImage(std::string const &name);
  svulkan2::renderer::RendererBase &getInternalRenderer();
  svulkan2::scene::Camera &getInternalCamera();
  std::shared_ptr<svulkan2::scene::Scene> getInternalRenderScene();

  ~SapienRenderCameraComponent();
  SapienRenderCameraComponent(SapienRenderCameraComponent const &) = delete;
  SapienRenderCameraComponent &operator=(SapienRenderCameraComponent const &) = delete;
  SapienRenderCameraComponent(SapienRenderCameraComponent const &&) = delete;
  SapienRenderCameraComponent &operator=(SapienRenderCameraComponent const &&) = delete;

private:
  CameraMode mMode{CameraMode::ePerspective};
  void checkMode(CameraMode mode) const;

  uint32_t mWidth{};
  uint32_t mHeight{};
  float mFx{};
  float mFy{};
  float mCx{};
  float mCy{};
  float mNear{};
  float mFar{};
  float mSkew{};

  // ortho only
  float mLeft{};
  float mRight{};
  float mBottom{};
  float mTop{};

  std::string mShaderDir;

  std::map<std::string, std::variant<int, float>> mProperties;

  std::unique_ptr<SapienRenderCameraInternal> mCamera;
  Pose mLocalPose;

  // this is set to true when the camera is updated but take picture has not
  // been called to produce a warning for get picture
  bool mUpdatedWithoutTakingPicture{true};

  // this is set to true when GPU resources is available
  bool mGpuInitialized{false};
  int mGpuPoseIndex{-1};
  // One camera may belong to exactly one RenderCameraGroup. Registration happens
  // at camera-group creation; pose sealing happens at RenderSystemGroup.gpu_init().
  void const *mGpuOwnershipOwner{};
  bool mGpuOwnershipSealed{false};
  CameraPoseMode mPoseMode{CameraPoseMode::eStatic};
  bool mPoseModeConfigured{false};
  Pose mSealedCpuGlobalPose;
  // last CPU pose folded into mCameraStateVersion for a sealed eCpu camera
  Pose mLastCpuStatePose;
  // bumped by projection/intrinsic setters and eCpu pose changes; consumed by
  // RenderSystemGroup.update_render() for dirty CPU camera uploads
  uint64_t mCameraStateVersion{1};

  void checkGpuOwnershipMutable(char const *operation) const;

  bool mHasSceneSelectionOverride{false};
  std::vector<std::weak_ptr<Scene>> mSelectedScenes;
  std::vector<std::shared_ptr<SapienRendererSystem>> mResolvedRenderSystems;
  std::vector<uint64_t> mResolvedRenderSceneVersions;
  std::shared_ptr<svulkan2::scene::Scene> mResolvedRenderScene;
  void refreshRenderScene();
};

} // namespace sapien_renderer
} // namespace sapien
