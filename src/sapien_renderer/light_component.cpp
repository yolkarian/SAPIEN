#include "sapien/sapien_renderer/light_component.h"
#include "sapien/sapien_renderer/sapien_renderer_system.h"
#include "sapien/scene.h"
#include <svulkan2/scene/light.h>
#include <svulkan2/scene/scene.h>

namespace sapien {
namespace sapien_renderer {

void SapienRenderPointLightComponent::onAddToScene(Scene &scene) {
  auto system = scene.getSapienRendererSystem();
  auto s = system->getScene();
  mPointLight = &s->addPointLight();
  mPointLight->setColor({mColor.x, mColor.y, mColor.z});
  mPointLight->enableShadow(mShadowEnabled);
  mPointLight->setShadowParameters(mShadowNear, mShadowFar, mShadowMapSize);
  system->registerComponent(
      std::static_pointer_cast<SapienRenderLightComponent>(shared_from_this()));
}

void SapienRenderPointLightComponent::onRemoveFromScene(Scene &scene) {
  auto system = scene.getSapienRendererSystem();
  auto s = system->getScene();
  s->removeNode(*mPointLight);
  // The node is destroyed at the next forceRemove(); drop the back-pointer so the
  // real-time color/shadow setters cannot dereference it after removal.
  mPointLight = nullptr;
  system->unregisterComponent(
      std::static_pointer_cast<SapienRenderLightComponent>(shared_from_this()));
}

void SapienRenderDirectionalLightComponent::onAddToScene(Scene &scene) {
  auto system = scene.getSapienRendererSystem();
  auto s = system->getScene();
  mDirectionalLight = &s->addDirectionalLight();
  mDirectionalLight->setColor({mColor.x, mColor.y, mColor.z});
  mDirectionalLight->enableShadow(mShadowEnabled);
  mDirectionalLight->setShadowParameters(mShadowNear, mShadowFar, mShadowHalfSize, mShadowMapSize);
  system->registerComponent(
      std::static_pointer_cast<SapienRenderLightComponent>(shared_from_this()));
}

void SapienRenderDirectionalLightComponent::onRemoveFromScene(Scene &scene) {
  auto system = scene.getSapienRendererSystem();
  auto s = system->getScene();
  s->removeNode(*mDirectionalLight);
  mDirectionalLight = nullptr;
  system->unregisterComponent(
      std::static_pointer_cast<SapienRenderLightComponent>(shared_from_this()));
}

void SapienRenderSpotLightComponent::onAddToScene(Scene &scene) {
  auto system = scene.getSapienRendererSystem();
  auto s = system->getScene();
  mSpotLight = &s->addSpotLight();
  mSpotLight->setColor({mColor.x, mColor.y, mColor.z});
  mSpotLight->enableShadow(mShadowEnabled);
  mSpotLight->setShadowParameters(mShadowNear, mShadowFar, mShadowMapSize);
  mSpotLight->setFovSmall(mFovInner);
  mSpotLight->setFov(mFovOuter);
  system->registerComponent(
      std::static_pointer_cast<SapienRenderLightComponent>(shared_from_this()));
}

void SapienRenderSpotLightComponent::onRemoveFromScene(Scene &scene) {
  auto system = scene.getSapienRendererSystem();
  auto s = system->getScene();
  s->removeNode(*mSpotLight);
  mSpotLight = nullptr;
  system->unregisterComponent(
      std::static_pointer_cast<SapienRenderLightComponent>(shared_from_this()));
}

void SapienRenderTexturedLightComponent::onAddToScene(Scene &scene) {
  auto system = scene.getSapienRendererSystem();
  auto s = system->getScene();
  mSpotLight = &s->addTexturedLight();
  mSpotLight->setColor({mColor.x, mColor.y, mColor.z});
  mSpotLight->enableShadow(mShadowEnabled);
  mSpotLight->setShadowParameters(mShadowNear, mShadowFar, mShadowMapSize);
  mSpotLight->setFovSmall(mFovInner);
  mSpotLight->setFov(mFovOuter);
  getLight()->setTexture(mTexture->getTexture());
  system->registerComponent(
      std::static_pointer_cast<SapienRenderLightComponent>(shared_from_this()));
}

void SapienRenderTexturedLightComponent::onRemoveFromScene(Scene &scene) {
  auto system = scene.getSapienRendererSystem();
  auto s = system->getScene();
  s->removeNode(*mSpotLight);
  mSpotLight = nullptr;
  system->unregisterComponent(
      std::static_pointer_cast<SapienRenderLightComponent>(shared_from_this()));
}

void SapienRenderParallelogramLightComponent::onAddToScene(Scene &scene) {
  auto system = scene.getSapienRendererSystem();
  auto s = system->getScene();
  mParallelogramLight = &s->addParallelogramLight();
  mParallelogramLight->setColor({mColor.x, mColor.y, mColor.z});
  mParallelogramLight->setShape({mHalfWidth, mHalfHeight}, mAngle);
  system->registerComponent(
      std::static_pointer_cast<SapienRenderLightComponent>(shared_from_this()));
}

void SapienRenderParallelogramLightComponent::onRemoveFromScene(Scene &scene) {
  auto system = scene.getSapienRendererSystem();
  auto s = system->getScene();
  s->removeNode(*mParallelogramLight);
  mParallelogramLight = nullptr;
  system->unregisterComponent(
      std::static_pointer_cast<SapienRenderLightComponent>(shared_from_this()));
}

void SapienRenderParallelogramLightComponent::setShape(float halfWidth, float halfHeight,
                                                       float angle) {
  mHalfWidth = halfWidth;
  mHalfHeight = halfHeight;
  mAngle = angle;
  if (mParallelogramLight) {
    mParallelogramLight->setShape({mHalfWidth, mHalfHeight}, mAngle);
  }
  markLightStateDirty();
}

void SapienRenderPointLightComponent::internalUpdate() {
  Pose globalPose = getGlobalPose();
  internalNotePoseUpdate(globalPose);
  if (!mPointLight) {
    return;
  }
  auto pose = globalPose * POSE_GL_TO_ROS;
  mPointLight->setTransform({.position = {pose.p.x, pose.p.y, pose.p.z},
                             .rotation = {pose.q.w, pose.q.x, pose.q.y, pose.q.z}});
}
void SapienRenderDirectionalLightComponent::internalUpdate() {
  Pose globalPose = getGlobalPose();
  internalNotePoseUpdate(globalPose);
  if (!mDirectionalLight) {
    return;
  }
  auto pose = globalPose * POSE_GL_TO_ROS;
  mDirectionalLight->setTransform({.position = {pose.p.x, pose.p.y, pose.p.z},
                                   .rotation = {pose.q.w, pose.q.x, pose.q.y, pose.q.z}});
}
void SapienRenderSpotLightComponent::internalUpdate() {
  Pose globalPose = getGlobalPose();
  internalNotePoseUpdate(globalPose);
  if (!mSpotLight) {
    return;
  }
  auto pose = globalPose * POSE_GL_TO_ROS;
  mSpotLight->setTransform({.position = {pose.p.x, pose.p.y, pose.p.z},
                            .rotation = {pose.q.w, pose.q.x, pose.q.y, pose.q.z}});
}
void SapienRenderTexturedLightComponent::internalUpdate() {
  Pose globalPose = getGlobalPose();
  internalNotePoseUpdate(globalPose);
  if (!mSpotLight) {
    return;
  }
  auto pose = globalPose * POSE_GL_TO_ROS;
  mSpotLight->setTransform({.position = {pose.p.x, pose.p.y, pose.p.z},
                            .rotation = {pose.q.w, pose.q.x, pose.q.y, pose.q.z}});
}
void SapienRenderParallelogramLightComponent::internalUpdate() {
  Pose globalPose = getGlobalPose();
  internalNotePoseUpdate(globalPose);
  if (!mParallelogramLight) {
    return;
  }
  auto pose = globalPose * POSE_GL_TO_ROS;
  mParallelogramLight->setTransform({.position = {pose.p.x, pose.p.y, pose.p.z},
                                     .rotation = {pose.q.w, pose.q.x, pose.q.y, pose.q.z}});
}

void SapienRenderLightComponent::setLocalPose(Pose const &pose) {
  if (mGroupSealCount > 0 && mPoseMode == LightPoseMode::eStatic) {
    throw std::runtime_error(
        "failed to set light local pose: the light pose mode is 'static' (default) and its pose "
        "is a snapshot sealed by a render system group; set pose mode 'cpu' before "
        "RenderSystemGroup.gpu_init() to move it");
  }
  mLocalPose = pose;
}
Pose SapienRenderLightComponent::getLocalPose() const { return mLocalPose; }
Pose SapienRenderLightComponent::getGlobalPose() const { return getPose() * mLocalPose; }

void SapienRenderLightComponent::setPoseMode(LightPoseMode mode) {
  if (mGroupSealCount > 0) {
    throw std::runtime_error(
        "failed to set light pose mode: the mode is sealed by a render system group; configure "
        "it before RenderSystemGroup.gpu_init()");
  }
  mPoseMode = mode;
}

void SapienRenderLightComponent::checkSetupMutable(char const *operation) const {
  if (mGroupSealCount > 0) {
    throw std::runtime_error(std::string("failed to ") + operation +
                             ": this is a setup-only light property sealed by a render system "
                             "group; destroy the group before changing it");
  }
}

void SapienRenderLightComponent::internalNotePoseUpdate(Pose const &globalPose) {
  if (mGroupSealCount == 0) {
    return;
  }
  bool poseChanged =
      globalPose.p.x != mLastCpuStatePose.p.x || globalPose.p.y != mLastCpuStatePose.p.y ||
      globalPose.p.z != mLastCpuStatePose.p.z || globalPose.q.w != mLastCpuStatePose.q.w ||
      globalPose.q.x != mLastCpuStatePose.q.x || globalPose.q.y != mLastCpuStatePose.q.y ||
      globalPose.q.z != mLastCpuStatePose.q.z;
  if (!poseChanged) {
    return;
  }
  if (mPoseMode == LightPoseMode::eStatic) {
    throw std::runtime_error(
        "failed to update light: its pose changed after RenderSystemGroup.gpu_init() but the "
        "light pose mode is 'static' (default); set pose mode 'cpu' before gpu_init() to move "
        "it");
  }
  // CPU pose stays authoritative: propagate and mark the scene light state dirty so the
  // owning group re-uploads light state at its next update_render().
  mLastCpuStatePose = globalPose;
  markLightStateDirty();
}

void SapienRenderLightComponent::internalSealGroupState() {
  if (mGroupSealCount++ == 0) {
    mSealedGlobalPose = getGlobalPose();
    mLastCpuStatePose = mSealedGlobalPose;
  }
}

void SapienRenderLightComponent::internalReleaseGroupStateSeal() {
  if (mGroupSealCount == 0) {
    throw std::runtime_error("light group state seal count is already zero");
  }
  --mGroupSealCount;
}

void SapienRenderLightComponent::markLightStateDirty() {
  if (auto scene = getScene()) {
    scene->getSapienRendererSystem()->internalNotifyLightStateChanged();
  }
}

void SapienRenderLightComponent::setColor(Vec3 color) {
  mColor = color;
  markLightStateDirty();
}
void SapienRenderPointLightComponent::setColor(Vec3 color) {
  mColor = color;
  if (mPointLight) {
    mPointLight->setColor({color.x, color.y, color.z});
  }
  markLightStateDirty();
}
void SapienRenderDirectionalLightComponent::setColor(Vec3 color) {
  mColor = color;
  if (mDirectionalLight) {
    mDirectionalLight->setColor({color.x, color.y, color.z});
  }
  markLightStateDirty();
}
void SapienRenderSpotLightComponent::setColor(Vec3 color) {
  mColor = color;
  if (mSpotLight) {
    mSpotLight->setColor({color.x, color.y, color.z});
  }
  markLightStateDirty();
}
void SapienRenderParallelogramLightComponent::setColor(Vec3 color) {
  mColor = color;
  if (mParallelogramLight) {
    mParallelogramLight->setColor({color.x, color.y, color.z});
  }
  markLightStateDirty();
}

void SapienRenderPointLightComponent::internalApplyShadowParameters() {
  if (mPointLight) {
    mPointLight->setShadowParameters(mShadowNear, mShadowFar, mShadowMapSize);
  }
}
void SapienRenderDirectionalLightComponent::internalApplyShadowParameters() {
  if (mDirectionalLight) {
    mDirectionalLight->setShadowParameters(mShadowNear, mShadowFar, mShadowHalfSize,
                                           mShadowMapSize);
  }
}
void SapienRenderSpotLightComponent::internalApplyShadowParameters() {
  if (mSpotLight) {
    mSpotLight->setShadowParameters(mShadowNear, mShadowFar, mShadowMapSize);
  }
}

} // namespace sapien_renderer
} // namespace sapien
