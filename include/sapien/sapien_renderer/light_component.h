#pragma once
#include "../component.h"
#include "material.h"
#include "sapien/math/pose.h"
#include <numbers>
#include <svulkan2/scene/light.h>

namespace sapien {
class Entity;
namespace sapien_renderer {

/** Pose source of a light after RenderSystemGroup.gpu_init().
 *  eStatic: the pose is a one-time CPU snapshot; changing it after seal raises.
 *  eCpu: the CPU component pose stays authoritative; per-frame entity/local pose
 *  updates propagate and upload when dirty at RenderSystemGroup.update_render().
 *  Non-pose properties (color, FOV, shape, shadow near/far/half-size) stay CPU
 *  real-time in both modes. */
enum class LightPoseMode { eStatic, eCpu };

class SapienRenderLightComponent : public Component {
public:
  Vec3 getColor() const { return mColor; }
  virtual void setColor(Vec3 color);

  bool getShadowEnabled() const { return mShadowEnabled; }
  void setShadowEnabled(bool enabled) {
    checkSetupMutable("set light shadow state");
    mShadowEnabled = enabled;
  }
  void enableShadow() { setShadowEnabled(true); }
  void disableShadow() { setShadowEnabled(false); }

  float getShadowNear() const { return mShadowNear; }
  void setShadowNear(float near) {
    mShadowNear = near;
    internalApplyShadowParameters();
    markLightStateDirty();
  }

  float getShadowFar() const { return mShadowFar; }
  void setShadowFar(float far) {
    mShadowFar = far;
    internalApplyShadowParameters();
    markLightStateDirty();
  }

  uint32_t getShadowMapSize() const { return mShadowMapSize; }
  void setShadowMapSize(uint32_t size) {
    checkSetupMutable("set light shadow map size");
    mShadowMapSize = size;
  }

  LightPoseMode getPoseMode() const { return mPoseMode; }
  /** configure the grouped pose source; sealed at RenderSystemGroup.gpu_init() */
  void setPoseMode(LightPoseMode mode);

  void setLocalPose(Pose const &);
  Pose getLocalPose() const;
  Pose getGlobalPose() const;

  virtual void internalUpdate() = 0;
  /** seal group-managed state at RenderSystemGroup.gpu_init(); static-mode lights
   *  additionally snapshot their pose */
  void internalSealGroupState();
  void internalReleaseGroupStateSeal();
  bool isGroupStateSealed() const { return mGroupSealCount > 0; }

protected:
  void checkSetupMutable(char const *operation) const;
  /** static-mode tamper check plus cpu-mode dirty propagation for the CPU pose */
  void internalNotePoseUpdate(Pose const &globalPose);
  /** propagate shadow near/far/half-size to the svulkan2 light object */
  virtual void internalApplyShadowParameters() {}
  /** bump the owning render system's scene light state version */
  void markLightStateDirty();

  Vec3 mColor{1.f, 1.f, 1.f};
  bool mShadowEnabled{true};
  float mShadowNear{0.01f};
  float mShadowFar{10.f};
  uint32_t mShadowMapSize{2048};
  Pose mLocalPose{};
  LightPoseMode mPoseMode{LightPoseMode::eStatic};
  uint32_t mGroupSealCount{0};
  Pose mSealedGlobalPose;
  Pose mLastCpuStatePose;
};

class SapienRenderPointLightComponent : public SapienRenderLightComponent {
public:
  void onAddToScene(Scene &scene) override;
  void onRemoveFromScene(Scene &scene) override;

  void internalUpdate() override;
  void setColor(Vec3 color) override;

protected:
  void internalApplyShadowParameters() override;

private:
  svulkan2::scene::PointLight *mPointLight{};
};

class SapienRenderDirectionalLightComponent : public SapienRenderLightComponent {
public:
  float getShadowHalfSize() const { return mShadowHalfSize; }
  void setShadowHalfSize(float size) {
    mShadowHalfSize = size;
    internalApplyShadowParameters();
    markLightStateDirty();
  }

  void onAddToScene(Scene &scene) override;
  void onRemoveFromScene(Scene &scene) override;

  void internalUpdate() override;
  void setColor(Vec3 color) override;

protected:
  void internalApplyShadowParameters() override;

private:
  svulkan2::scene::DirectionalLight *mDirectionalLight{};
  float mShadowHalfSize{10.f};
};

class SapienRenderSpotLightComponent : public SapienRenderLightComponent {
public:
  float getFovInner() const { return mFovInner; }
  void setFovInner(float fov) {
    mFovInner = fov;
    if (mSpotLight) {
      mSpotLight->setFovSmall(fov);
    }
    markLightStateDirty();
  }
  float getFovOuter() const { return mFovOuter; }
  void setFovOuter(float fov) {
    mFovOuter = fov;
    if (mSpotLight) {
      mSpotLight->setFov(fov);
    }
    markLightStateDirty();
  }

  void onAddToScene(Scene &scene) override;
  void onRemoveFromScene(Scene &scene) override;

  void internalUpdate() override;
  void setColor(Vec3 color) override;

protected:
  void internalApplyShadowParameters() override;

  float mFovInner{0.f};
  float mFovOuter{0.f};
  svulkan2::scene::SpotLight *mSpotLight{};
};

class SapienRenderTexturedLightComponent : public SapienRenderSpotLightComponent {

public:
  void onAddToScene(Scene &scene) override;
  void onRemoveFromScene(Scene &scene) override;

  void setTexture(std::shared_ptr<SapienRenderTexture2D> texture) {
    checkSetupMutable("set textured-light texture");
    mTexture = texture;
  }
  std::shared_ptr<SapienRenderTexture2D> getTexture() const { return mTexture; }

  void internalUpdate() override;

private:
  svulkan2::scene::TexturedLight *getLight() const {
    return static_cast<svulkan2::scene::TexturedLight *>(mSpotLight);
  }
  std::shared_ptr<SapienRenderTexture2D> mTexture;
};

class SapienRenderParallelogramLightComponent : public SapienRenderLightComponent {
public:
  void onAddToScene(Scene &scene) override;
  void onRemoveFromScene(Scene &scene) override;

  void setShape(float halfWidth, float halfHeight, float angle);
  float getHalfWidth() const { return mHalfWidth; }
  float getHalfHeight() const { return mHalfHeight; }
  float getAngle() const { return mAngle; }

  void internalUpdate() override;
  void setColor(Vec3 color) override;

private:
  float mHalfWidth = 1.f;
  float mHalfHeight = 1.f;
  float mAngle = std::numbers::pi_v<float> / 2.f;

  svulkan2::scene::ParallelogramLight *mParallelogramLight{};
};

} // namespace sapien_renderer
} // namespace sapien
