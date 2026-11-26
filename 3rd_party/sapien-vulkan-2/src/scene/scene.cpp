#include "svulkan2/scene/scene.h"
#include "../common/logger.h"
#include "svulkan2/core/context.h"
#include "svulkan2/resource/render_target.h"
#include <algorithm>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace svulkan2 {
namespace scene {

namespace {
std::mutex gSceneDependenciesMutex;
std::unordered_map<Scene *, std::vector<Scene *>> gSceneDependencies;
} // namespace

namespace detail {
void registerSceneDependency(Scene *source, Scene *dependent) {
  if (!source || !dependent) {
    return;
  }

  std::lock_guard lock(gSceneDependenciesMutex);
  auto &dependents = gSceneDependencies[source];
  if (std::find(dependents.begin(), dependents.end(), dependent) == dependents.end()) {
    dependents.push_back(dependent);
  }
}

void unregisterSceneDependency(Scene *source, Scene *dependent) {
  std::lock_guard lock(gSceneDependenciesMutex);
  auto it = gSceneDependencies.find(source);
  if (it == gSceneDependencies.end()) {
    return;
  }

  auto &dependents = it->second;
  dependents.erase(std::remove(dependents.begin(), dependents.end(), dependent), dependents.end());
  if (dependents.empty()) {
    gSceneDependencies.erase(it);
  }
}

void notifySceneVersionChanged(Scene *source, bool renderOnly) {
  std::vector<Scene *> dependents;
  {
    std::lock_guard lock(gSceneDependenciesMutex);
    auto it = gSceneDependencies.find(source);
    if (it != gSceneDependencies.end()) {
      dependents = it->second;
    }
  }

  for (auto dependent : dependents) {
    if (renderOnly) {
      dependent->updateRenderVersion();
    } else {
      dependent->updateVersion();
    }
  }
}
} // namespace detail

// Matches GLSL std140 array stride (16-byte aligned struct = 272 bytes) for ShadowBuffer arrays
// as well as the standalone LightBuffer UBO of shadow passes (first 264 bytes).
struct alignas(16) LightBufferData {
  glm::mat4 viewMatrix;
  glm::mat4 viewMatrixInverse;
  glm::mat4 projectionMatrix;
  glm::mat4 projectionMatrixInverse;
  int width;
  int height;
  int padding0{0};
  int padding1{0};
};
static_assert(sizeof(LightBufferData) == 272);

struct ScopedBufferMapping {
  explicit ScopedBufferMapping(core::Buffer &buffer) : buffer(buffer) { buffer.map(); }
  ~ScopedBufferMapping() { buffer.unmap(); }

  core::Buffer &buffer;
};

Scene::Scene(std::shared_ptr<core::Context> context) : mContext(std::move(context)) {
  if (!mContext) {
    throw std::runtime_error("failed to create scene: context must not be null");
  }
  mNodes.push_back(std::make_unique<Node>());
  mRootNode = mNodes.back().get();
  mRootNode->setScene(this);
}

Node &Scene::addNode(Node &parent, Transform const &transform) {
  updateVersion();
  forceRemove();
  mNodes.push_back(std::make_unique<Node>());
  mNodes.back()->setScene(this);
  mNodes.back()->setParent(parent);
  parent.addChild(*mNodes.back());

  // update global matrix now, so the next call to update the scene
  // will update the current matrix into the previous matrix
  mNodes.back()->setTransform(transform);
  mNodes.back()->updateGlobalModelMatrixRecursive();
  return *mNodes.back();
}

Node &Scene::addNode(Transform const &transform) { return addNode(getRootNode(), transform); }

Object &Scene::addObject(std::shared_ptr<resource::SVModel> model, Transform const &transform) {
  return addObject(getRootNode(), model, transform);
}

Object &Scene::addObject(Node &parent, std::shared_ptr<resource::SVModel> model,
                         Transform const &transform) {
  updateVersion();
  forceRemove();
  auto obj = std::make_unique<Object>(model);
  auto &result = *obj;
  mObjects.push_back(std::move(obj));
  mObjects.back()->setScene(this);
  mObjects.back()->setParent(parent);
  parent.addChild(*mObjects.back());

  mObjects.back()->setTransform(transform);
  mObjects.back()->updateGlobalModelMatrixRecursive();
  return result;
}

Object &Scene::addDeformableObject(std::shared_ptr<resource::SVModel> model) {
  updateVersion();
  forceRemove();
  auto obj = std::make_unique<Object>(model);
  auto &result = *obj;
  mDeformableObjects.push_back(std::move(obj));
  mDeformableObjects.back()->setScene(this);
  mDeformableObjects.back()->setParent(getRootNode());
  getRootNode().addChild(*mDeformableObjects.back());

  mDeformableObjects.back()->setTransform({});
  mDeformableObjects.back()->updateGlobalModelMatrixRecursive();
  return result;
}

LineObject &Scene::addLineObject(std::shared_ptr<resource::SVLineSet> lineSet,
                                 Transform const &transform) {
  return addLineObject(getRootNode(), lineSet, transform);
}

LineObject &Scene::addLineObject(Node &parent, std::shared_ptr<resource::SVLineSet> lineSet,
                                 Transform const &transform) {
  updateVersion();
  forceRemove();
  auto obj = std::make_unique<LineObject>(lineSet);
  auto &result = *obj;
  mLineObjects.push_back(std::move(obj));
  mLineObjects.back()->setScene(this);
  mLineObjects.back()->setParent(parent);
  parent.addChild(*mLineObjects.back());

  mLineObjects.back()->setTransform(transform);
  mLineObjects.back()->updateGlobalModelMatrixRecursive();
  return result;
}

PointObject &Scene::addPointObject(std::shared_ptr<resource::SVPointSet> pointSet,
                                   Transform const &transform) {
  return addPointObject(getRootNode(), pointSet, transform);
}

PointObject &Scene::addPointObject(Node &parent, std::shared_ptr<resource::SVPointSet> pointSet,
                                   Transform const &transform) {
  updateVersion();
  forceRemove();
  auto obj = std::make_unique<PointObject>(pointSet);
  auto &result = *obj;
  mPointObjects.push_back(std::move(obj));
  mPointObjects.back()->setScene(this);
  mPointObjects.back()->setParent(parent);
  parent.addChild(*mPointObjects.back());

  mPointObjects.back()->setTransform(transform);
  mPointObjects.back()->updateGlobalModelMatrixRecursive();
  return result;
}

Camera &Scene::addCamera(Transform const &transform) {
  return addCamera(getRootNode(), transform);
}
Camera &Scene::addCamera(Node &parent, Transform const &transform) {
  updateVersion();
  forceRemove();
  auto cam = std::make_unique<Camera>();
  auto &result = *cam;
  mCameras.push_back(std::move(cam));
  mCameras.back()->setScene(this);
  mCameras.back()->setParent(parent);

  mCameras.back()->setTransform(transform);
  mCameras.back()->updateGlobalModelMatrixRecursive();
  parent.addChild(*mCameras.back());
  return result;
}

PointLight &Scene::addPointLight() { return addPointLight(getRootNode()); }
PointLight &Scene::addPointLight(Node &parent) {
  updateVersion();
  forceRemove();
  auto pointLight = std::make_unique<PointLight>();
  auto &result = *pointLight;
  mPointLights.push_back(std::move(pointLight));
  mPointLights.back()->setScene(this);
  mPointLights.back()->setParent(parent);
  parent.addChild(*mPointLights.back());
  return result;
}

DirectionalLight &Scene::addDirectionalLight() { return addDirectionalLight(getRootNode()); }
DirectionalLight &Scene::addDirectionalLight(Node &parent) {
  updateVersion();
  forceRemove();
  auto directionalLight = std::make_unique<DirectionalLight>();
  auto &result = *directionalLight;
  mDirectionalLights.push_back(std::move(directionalLight));
  mDirectionalLights.back()->setScene(this);
  mDirectionalLights.back()->setParent(parent);
  parent.addChild(*mDirectionalLights.back());
  return result;
}

SpotLight &Scene::addSpotLight() { return addSpotLight(getRootNode()); }
SpotLight &Scene::addSpotLight(Node &parent) {
  updateVersion();
  forceRemove();
  auto spotLight = std::make_unique<SpotLight>();
  auto &result = *spotLight;
  mSpotLights.push_back(std::move(spotLight));
  mSpotLights.back()->setScene(this);
  mSpotLights.back()->setParent(parent);
  parent.addChild(*mSpotLights.back());
  return result;
}

TexturedLight &Scene::addTexturedLight() { return addTexturedLight(getRootNode()); }
TexturedLight &Scene::addTexturedLight(Node &parent) {
  updateVersion();
  forceRemove();
  auto texturedLight = std::make_unique<TexturedLight>();
  auto &result = *texturedLight;
  mTexturedLights.push_back(std::move(texturedLight));
  mTexturedLights.back()->setScene(this);
  mTexturedLights.back()->setParent(parent);
  parent.addChild(*mTexturedLights.back());
  return result;
}

ParallelogramLight &Scene::addParallelogramLight() { return addParallelogramLight(getRootNode()); }
ParallelogramLight &Scene::addParallelogramLight(Node &parent) {
  updateVersion();
  forceRemove();
  auto parallelogramLight = std::make_unique<ParallelogramLight>();
  auto &result = *parallelogramLight;
  mParallelogramLights.push_back(std::move(parallelogramLight));
  mParallelogramLights.back()->setScene(this);
  mParallelogramLights.back()->setParent(parent);
  parent.addChild(*mParallelogramLights.back());
  return result;
}

void Scene::removeNode(Node &node) {
  updateVersion();
  mRequireForceRemove = true;
  node.markRemovedRecursive();
  node.getParent().removeChild(node);
}

void Scene::clearNodes() {
  updateVersion();
  mNodes.resize(1);
  mObjects.clear();
  mDeformableObjects.clear();
  mLineObjects.clear();
  mPointObjects.clear();
  mCameras.clear();
  mPointLights.clear();
  mDirectionalLights.clear();
  mSpotLights.clear();
  mTexturedLights.clear();
  mParallelogramLights.clear();
}

void Scene::forceRemove() {
  if (!mRequireForceRemove) {
    return;
  }
  mNodes.erase(std::remove_if(mNodes.begin(), mNodes.end(),
                              [](std::unique_ptr<Node> &node) { return node->isMarkedRemoved(); }),
               mNodes.end());
  mObjects.erase(
      std::remove_if(mObjects.begin(), mObjects.end(),
                     [](std::unique_ptr<Object> &node) { return node->isMarkedRemoved(); }),
      mObjects.end());
  mDeformableObjects.erase(
      std::remove_if(mDeformableObjects.begin(), mDeformableObjects.end(),
                     [](std::unique_ptr<Object> &node) { return node->isMarkedRemoved(); }),
      mDeformableObjects.end());
  mLineObjects.erase(
      std::remove_if(mLineObjects.begin(), mLineObjects.end(),
                     [](std::unique_ptr<LineObject> &node) { return node->isMarkedRemoved(); }),
      mLineObjects.end());
  mPointObjects.erase(
      std::remove_if(mPointObjects.begin(), mPointObjects.end(),
                     [](std::unique_ptr<PointObject> &node) { return node->isMarkedRemoved(); }),
      mPointObjects.end());
  mCameras.erase(
      std::remove_if(mCameras.begin(), mCameras.end(),
                     [](std::unique_ptr<Camera> &node) { return node->isMarkedRemoved(); }),
      mCameras.end());
  mPointLights.erase(
      std::remove_if(mPointLights.begin(), mPointLights.end(),
                     [](std::unique_ptr<PointLight> &node) { return node->isMarkedRemoved(); }),
      mPointLights.end());
  mDirectionalLights.erase(std::remove_if(mDirectionalLights.begin(), mDirectionalLights.end(),
                                          [](std::unique_ptr<DirectionalLight> &node) {
                                            return node->isMarkedRemoved();
                                          }),
                           mDirectionalLights.end());
  mSpotLights.erase(
      std::remove_if(mSpotLights.begin(), mSpotLights.end(),
                     [](std::unique_ptr<SpotLight> &node) { return node->isMarkedRemoved(); }),
      mSpotLights.end());

  mTexturedLights.erase(
      std::remove_if(mTexturedLights.begin(), mTexturedLights.end(),
                     [](std::unique_ptr<TexturedLight> &node) { return node->isMarkedRemoved(); }),
      mTexturedLights.end());

  mParallelogramLights.erase(std::remove_if(mParallelogramLights.begin(),
                                            mParallelogramLights.end(),
                                            [](std::unique_ptr<ParallelogramLight> &node) {
                                              return node->isMarkedRemoved();
                                            }),
                             mParallelogramLights.end());

  mRequireForceRemove = false;
}

std::vector<Object *> Scene::getObjects() {
  forceRemove();
  std::vector<Object *> result;
  for (auto &obj : mObjects) {
    result.push_back(obj.get());
  }
  for (auto &obj : mDeformableObjects) {
    result.push_back(obj.get());
  }
  return result;
}

std::vector<LineObject *> Scene::getLineObjects() {
  forceRemove();
  std::vector<LineObject *> result;
  for (auto &obj : mLineObjects) {
    result.push_back(obj.get());
  }
  return result;
}

std::vector<PointObject *> Scene::getPointObjects() {
  forceRemove();
  std::vector<PointObject *> result;
  for (auto &obj : mPointObjects) {
    result.push_back(obj.get());
  }
  return result;
}

std::vector<Camera *> Scene::getCameras() {
  forceRemove();
  std::vector<Camera *> result;
  for (auto &obj : mCameras) {
    result.push_back(obj.get());
  }
  return result;
}

std::vector<PointLight *> Scene::getPointLights() {
  forceRemove();
  std::vector<PointLight *> result;
  for (auto &light : mPointLights) {
    result.push_back(light.get());
  }
  return result;
}

std::vector<DirectionalLight *> Scene::getDirectionalLights() {
  forceRemove();
  std::vector<DirectionalLight *> result;
  for (auto &light : mDirectionalLights) {
    result.push_back(light.get());
  }
  return result;
}

std::vector<SpotLight *> Scene::getSpotLights() {
  forceRemove();
  std::vector<SpotLight *> result;
  for (auto &light : mSpotLights) {
    result.push_back(light.get());
  }
  return result;
}

std::vector<TexturedLight *> Scene::getTexturedLights() {
  forceRemove();
  std::vector<TexturedLight *> result;
  for (auto &light : mTexturedLights) {
    result.push_back(light.get());
  }
  return result;
}

std::vector<ParallelogramLight *> Scene::getParallelogramLights() {
  forceRemove();
  std::vector<ParallelogramLight *> result;
  for (auto &light : mParallelogramLights) {
    result.push_back(light.get());
  }
  return result;
}

void Scene::updateModelMatrices() {
  updateRenderVersion();
  // Recomputing every node costs more than everything else in a frame of a large static
  // scene, so skip it while no local transform and no topology changed. One more pass after
  // the last change lets prevWorldModelMatrix catch up with worldModelMatrix.
  bool changed = mRootNode->isSubtreeDirty() || mModelMatricesVersion != mVersion;
  if (changed || mModelMatricesSettling) {
    mRootNode->updateGlobalModelMatrixRecursive();
  }
  mModelMatricesSettling = changed;
  mModelMatricesVersion = mVersion;
}

std::shared_ptr<RasterShadowMaps>
Scene::getOrCreateRasterShadowMaps(vk::Format depthFormat, size_t lightBufferSize) {
  std::vector<uint32_t> dirSizes;
  std::vector<uint32_t> ptSizes;
  std::vector<uint32_t> spSizes;
  std::vector<uint32_t> texSizes;

  for (auto *l : getDirectionalLights()) {
    if (l->isShadowEnabled()) {
      dirSizes.push_back(l->getShadowMapSize());
    }
  }
  for (auto *l : getPointLights()) {
    if (l->isShadowEnabled()) {
      ptSizes.push_back(l->getShadowMapSize());
    }
  }
  for (auto *l : getSpotLights()) {
    if (l->isShadowEnabled()) {
      spSizes.push_back(l->getShadowMapSize());
    }
  }
  for (auto *l : getTexturedLights()) {
    texSizes.push_back(l->getShadowMapSize());
  }

  auto it = mRasterShadowMaps.find(depthFormat);
  if (it != mRasterShadowMaps.end() && it->second &&
      it->second->matches(depthFormat, lightBufferSize, dirSizes, ptSizes, spSizes, texSizes)) {
    return it->second;
  }

  auto maps = std::make_shared<RasterShadowMaps>();
  maps->depthFormat = depthFormat;
  maps->sceneVersion = mVersion;
  maps->lightBufferSize = lightBufferSize;
  maps->directionalShadowSizes = std::move(dirSizes);
  maps->pointShadowSizes = std::move(ptSizes);
  maps->spotShadowSizes = std::move(spSizes);
  maps->texturedShadowSizes = std::move(texSizes);

  bool hasAnyShadow = !maps->directionalShadowSizes.empty() || !maps->pointShadowSizes.empty() ||
                      !maps->spotShadowSizes.empty() || !maps->texturedShadowSizes.empty();

  if (hasAnyShadow && mContext->isVulkanAvailable()) {
    auto pool = mContext->createCommandPool();
    auto commandBuffer = pool->allocateCommandBuffer();
    commandBuffer->begin(
        vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    vk::ComponentMapping componentMapping(vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
                                          vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA);

    auto makeShadowSampler = [&]() {
      return mContext->createSampler(vk::SamplerCreateInfo(
          {}, vk::Filter::eNearest, vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest,
          vk::SamplerAddressMode::eClampToBorder, vk::SamplerAddressMode::eClampToBorder,
          vk::SamplerAddressMode::eClampToBorder, 0.f, false, 0.f, false, vk::CompareOp::eNever,
          0.f, 0.f, vk::BorderColor::eFloatOpaqueWhite));
    };

    // 1. Directional lights: kShadowCascadeCount-layer 2D array per shadow-casting light
    for (uint32_t size : maps->directionalShadowSizes) {
      constexpr uint32_t kCascades = DirectionalLight::kShadowCascadeCount;
      auto image = std::make_shared<core::Image>(
          *mContext, vk::ImageType::e2D, vk::Extent3D{size, size, 1}, depthFormat,
          vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
          VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, vk::SampleCountFlagBits::e1, 1, kCascades);
      image->transitionLayout(commandBuffer.get(), vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eShaderReadOnlyOptimal, {},
                              vk::AccessFlagBits::eShaderRead,
                              vk::PipelineStageFlagBits::eBottomOfPipe,
                              vk::PipelineStageFlagBits::eFragmentShader);

      vk::ImageViewCreateInfo readViewInfo(
          {}, image->getVulkanImage(), vk::ImageViewType::e2DArray, depthFormat, componentMapping,
          vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, kCascades));
      auto readView = mContext->getDevice().createImageViewUnique(readViewInfo);
      maps->directionalReadTargets.push_back(std::make_shared<resource::SVRenderTarget>(
          mContext, "DirectionalShadow", size, size, image, std::move(readView),
          makeShadowSampler()));

      for (uint32_t cascade = 0; cascade < kCascades; ++cascade) {
        vk::ImageViewCreateInfo writeViewInfo(
            {}, image->getVulkanImage(), vk::ImageViewType::e2D, depthFormat, componentMapping,
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, cascade, 1));
        auto writeView = mContext->getDevice().createImageViewUnique(writeViewInfo);
        maps->allWriteTargets.push_back(std::make_shared<resource::SVRenderTarget>(
            mContext, "DirectionalShadow", size, size, image, std::move(writeView), vk::Sampler{}));
        maps->allShadowSizes.push_back(size);
      }
    }

    // 2. Point lights: 6-layer cubemap per shadow-casting light
    for (uint32_t size : maps->pointShadowSizes) {
      auto image = std::make_shared<core::Image>(
          *mContext, vk::ImageType::e2D, vk::Extent3D{size, size, 1}, depthFormat,
          vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
          VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, vk::SampleCountFlagBits::e1, 1, 6,
          vk::ImageTiling::eOptimal, vk::ImageCreateFlagBits::eCubeCompatible);
      image->transitionLayout(commandBuffer.get(), vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eShaderReadOnlyOptimal, {},
                              vk::AccessFlagBits::eShaderRead,
                              vk::PipelineStageFlagBits::eBottomOfPipe,
                              vk::PipelineStageFlagBits::eFragmentShader);

      vk::ImageViewCreateInfo readViewInfo(
          {}, image->getVulkanImage(), vk::ImageViewType::eCube, depthFormat, componentMapping,
          vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 6));
      auto readView = mContext->getDevice().createImageViewUnique(readViewInfo);
      maps->pointReadTargets.push_back(std::make_shared<resource::SVRenderTarget>(
          mContext, "PointShadow", size, size, image, std::move(readView), makeShadowSampler()));

      for (uint32_t face = 0; face < 6; ++face) {
        vk::ImageViewCreateInfo writeViewInfo(
            {}, image->getVulkanImage(), vk::ImageViewType::e2D, depthFormat, componentMapping,
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, face, 1));
        auto writeView = mContext->getDevice().createImageViewUnique(writeViewInfo);
        maps->allWriteTargets.push_back(std::make_shared<resource::SVRenderTarget>(
            mContext, "PointShadow", size, size, image, std::move(writeView), vk::Sampler{}));
        maps->allShadowSizes.push_back(size);
      }
    }

    // 3. Spot lights: single 2D depth map per shadow-casting light
    for (uint32_t size : maps->spotShadowSizes) {
      auto image = std::make_shared<core::Image>(
          *mContext, vk::ImageType::e2D, vk::Extent3D{size, size, 1}, depthFormat,
          vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
          VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, vk::SampleCountFlagBits::e1, 1, 1);
      image->transitionLayout(commandBuffer.get(), vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eShaderReadOnlyOptimal, {},
                              vk::AccessFlagBits::eShaderRead,
                              vk::PipelineStageFlagBits::eBottomOfPipe,
                              vk::PipelineStageFlagBits::eFragmentShader);

      vk::ImageViewCreateInfo readViewInfo(
          {}, image->getVulkanImage(), vk::ImageViewType::e2D, depthFormat, componentMapping,
          vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1));
      auto readView = mContext->getDevice().createImageViewUnique(readViewInfo);
      maps->spotReadTargets.push_back(std::make_shared<resource::SVRenderTarget>(
          mContext, "SpotShadow", size, size, image, std::move(readView), makeShadowSampler()));

      vk::ImageViewCreateInfo writeViewInfo(
          {}, image->getVulkanImage(), vk::ImageViewType::e2D, depthFormat, componentMapping,
          vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1));
      auto writeView = mContext->getDevice().createImageViewUnique(writeViewInfo);
      maps->allWriteTargets.push_back(std::make_shared<resource::SVRenderTarget>(
          mContext, "SpotShadow", size, size, image, std::move(writeView), vk::Sampler{}));
      maps->allShadowSizes.push_back(size);
    }

    // 4. Textured lights: single 2D depth map per textured light
    for (uint32_t size : maps->texturedShadowSizes) {
      auto image = std::make_shared<core::Image>(
          *mContext, vk::ImageType::e2D, vk::Extent3D{size, size, 1}, depthFormat,
          vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
          VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, vk::SampleCountFlagBits::e1, 1, 1);
      image->transitionLayout(commandBuffer.get(), vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eShaderReadOnlyOptimal, {},
                              vk::AccessFlagBits::eShaderRead,
                              vk::PipelineStageFlagBits::eBottomOfPipe,
                              vk::PipelineStageFlagBits::eFragmentShader);

      vk::ImageViewCreateInfo readViewInfo(
          {}, image->getVulkanImage(), vk::ImageViewType::e2D, depthFormat, componentMapping,
          vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1));
      auto readView = mContext->getDevice().createImageViewUnique(readViewInfo);
      maps->texturedReadTargets.push_back(std::make_shared<resource::SVRenderTarget>(
          mContext, "TexturedLightShadow", size, size, image, std::move(readView),
          makeShadowSampler()));

      vk::ImageViewCreateInfo writeViewInfo(
          {}, image->getVulkanImage(), vk::ImageViewType::e2D, depthFormat, componentMapping,
          vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1));
      auto writeView = mContext->getDevice().createImageViewUnique(writeViewInfo);
      maps->allWriteTargets.push_back(std::make_shared<resource::SVRenderTarget>(
          mContext, "TexturedLightShadow", size, size, image, std::move(writeView),
          vk::Sampler{}));
      maps->allShadowSizes.push_back(size);
    }

    commandBuffer->end();
    mContext->getQueue().submitAndWait(commandBuffer.get());

    size_t const allocSize = std::max(lightBufferSize, sizeof(LightBufferData));
    maps->lightBuffers.reserve(maps->allWriteTargets.size());
    for (size_t i = 0; i < maps->allWriteTargets.size(); ++i) {
      maps->lightBuffers.push_back(core::Buffer::CreateUniform(*mContext, allocSize));
    }
  }

  mRasterShadowMaps[depthFormat] = maps;
  return maps;
}

void Scene::uploadShadowToDevice(core::Buffer &shadowBuffer, RasterShadowMaps &shadowMaps,
                                 StructDataLayout const &shadowLayout) {
  constexpr uint32_t kCascades = DirectionalLight::kShadowCascadeCount;

  uint32_t maxNumDirectionalLightShadows =
      shadowLayout.elements.at("directionalLightBuffers").array.at(0) / kCascades;
  uint32_t maxNumPointLightShadows =
      shadowLayout.elements.at("pointLightBuffers").array.at(0) / 6;
  uint32_t maxNumSpotLightShadows =
      shadowLayout.elements.at("spotLightBuffers").array.at(0);
  uint32_t maxNumTexturedLightShadows =
      shadowLayout.elements.at("texturedLightBuffers").array.at(0);

  ScopedBufferMapping shadowMapping(shadowBuffer);

  std::vector<LightBufferData> allLightData;
  allLightData.reserve(shadowMaps.lightBuffers.size());

  {
    std::vector<LightBufferData> directionalLightShadowData;
    uint32_t numDirectionalLightShadows = 0;
    for (auto &l : getDirectionalLights()) {
      if (!l->isShadowEnabled()) {
        continue;
      }
      if (numDirectionalLightShadows >= maxNumDirectionalLightShadows) {
        throw std::runtime_error("The scene contains too many directional "
                                 "lights that cast shadows.");
      }
      numDirectionalLightShadows++;

      auto modelMat = l->getTransform().worldModelMatrix;
      auto viewMat = glm::affineInverse(modelMat);
      auto projMats = l->getShadowProjectionMatrices();
      for (uint32_t c = 0; c < kCascades; ++c) {
        directionalLightShadowData.push_back({.viewMatrix = viewMat,
                                              .viewMatrixInverse = modelMat,
                                              .projectionMatrix = projMats[c],
                                              .projectionMatrixInverse = glm::inverse(projMats[c]),
                                              .width = static_cast<int>(l->getShadowMapSize()),
                                              .height = static_cast<int>(l->getShadowMapSize())});
        allLightData.push_back(directionalLightShadowData.back());
      }
    }
    if (!directionalLightShadowData.empty()) {
      shadowBuffer.upload(directionalLightShadowData.data(),
                          directionalLightShadowData.size() * sizeof(LightBufferData),
                          shadowLayout.elements.at("directionalLightBuffers").offset);
    }
  }

  {
    std::vector<LightBufferData> pointLightShadowData;
    uint32_t numPointLightShadows = 0;
    for (auto &l : getPointLights()) {
      if (!l->isShadowEnabled()) {
        continue;
      }
      if (numPointLightShadows >= maxNumPointLightShadows) {
        throw std::runtime_error("The scene contains too many point "
                                 "lights that cast shadows.");
      }
      numPointLightShadows++;
      auto modelMats =
          PointLight::getModelMatrices(glm::vec3(l->getTransform().worldModelMatrix[3]));
      auto projMat = l->getShadowProjectionMatrix();
      for (uint32_t i = 0; i < 6; ++i) {
        pointLightShadowData.push_back({.viewMatrix = glm::affineInverse(modelMats[i]),
                                        .viewMatrixInverse = modelMats[i],
                                        .projectionMatrix = projMat,
                                        .projectionMatrixInverse = glm::inverse(projMat),
                                        .width = static_cast<int>(l->getShadowMapSize()),
                                        .height = static_cast<int>(l->getShadowMapSize())});
        allLightData.push_back(pointLightShadowData.back());
      }
    }
    if (!pointLightShadowData.empty()) {
      shadowBuffer.upload(pointLightShadowData.data(),
                          pointLightShadowData.size() * sizeof(LightBufferData),
                          shadowLayout.elements.at("pointLightBuffers").offset);
    }
  }

  {
    std::vector<LightBufferData> spotLightShadowData;
    uint32_t numSpotLightShadows = 0;
    for (auto &l : getSpotLights()) {
      if (!l->isShadowEnabled()) {
        continue;
      }
      if (numSpotLightShadows >= maxNumSpotLightShadows) {
        throw std::runtime_error("The scene contains too many spot "
                                 "lights that cast shadows.");
      }
      numSpotLightShadows++;

      auto modelMat = l->getTransform().worldModelMatrix;
      auto projMat = l->getShadowProjectionMatrix();
      spotLightShadowData.push_back({.viewMatrix = glm::affineInverse(modelMat),
                                     .viewMatrixInverse = modelMat,
                                     .projectionMatrix = projMat,
                                     .projectionMatrixInverse = glm::inverse(projMat),
                                     .width = static_cast<int>(l->getShadowMapSize()),
                                     .height = static_cast<int>(l->getShadowMapSize())});
      allLightData.push_back(spotLightShadowData.back());
    }
    if (!spotLightShadowData.empty()) {
      shadowBuffer.upload(spotLightShadowData.data(),
                          spotLightShadowData.size() * sizeof(LightBufferData),
                          shadowLayout.elements.at("spotLightBuffers").offset);
    }
  }

  {
    std::vector<LightBufferData> texturedLightShadowData;
    uint32_t numTexturedLightShadows = 0;
    for (auto l : getTexturedLights()) {
      if (numTexturedLightShadows >= maxNumTexturedLightShadows) {
        throw std::runtime_error("The scene contains too many textured lights.");
      }
      numTexturedLightShadows++;
      auto modelMat = l->getTransform().worldModelMatrix;
      auto projMat = l->getShadowProjectionMatrix();

      texturedLightShadowData.push_back({.viewMatrix = glm::affineInverse(modelMat),
                                         .viewMatrixInverse = modelMat,
                                         .projectionMatrix = projMat,
                                         .projectionMatrixInverse = glm::inverse(projMat),
                                         .width = static_cast<int>(l->getShadowMapSize()),
                                         .height = static_cast<int>(l->getShadowMapSize())});
      allLightData.push_back(texturedLightShadowData.back());
    }
    if (!texturedLightShadowData.empty()) {
      shadowBuffer.upload(texturedLightShadowData.data(),
                          texturedLightShadowData.size() * sizeof(LightBufferData),
                          shadowLayout.elements.at("texturedLightBuffers").offset);
    }
  }

  // Upload shared per-view LightBuffers only when the packed shadow matrices actually changed,
  // so multiple cameras uploading on the same frame do not invalidate already-rendered shadows.
  size_t const totalBytes = allLightData.size() * sizeof(LightBufferData);
  bool changed = shadowMaps.lastLightBufferBytes.size() != totalBytes ||
                 (totalBytes > 0 && std::memcmp(shadowMaps.lastLightBufferBytes.data(),
                                                allLightData.data(), totalBytes) != 0);
  if (changed) {
    for (size_t i = 0; i < allLightData.size() && i < shadowMaps.lightBuffers.size(); ++i) {
      shadowMaps.lightBuffers[i]->upload(&allLightData[i], sizeof(LightBufferData));
    }
    shadowMaps.lastLightBufferBytes.resize(totalBytes);
    if (totalBytes > 0) {
      std::memcpy(shadowMaps.lastLightBufferBytes.data(), allLightData.data(), totalBytes);
    }
    markShadowsDirty();
  }
}

void Scene::markShadowsDirty() {
  mShadowEpoch++;
  detail::notifySceneVersionChanged(this, true);
}

void Scene::reorderLights() {
  updateVersion();
  std::sort(mPointLights.begin(), mPointLights.end(),
            [](auto &l1, auto &l2) { return l1->isShadowEnabled() && !l2->isShadowEnabled(); });
  std::sort(mDirectionalLights.begin(), mDirectionalLights.end(),
            [](auto &l1, auto &l2) { return l1->isShadowEnabled() && !l2->isShadowEnabled(); });
  std::sort(mSpotLights.begin(), mSpotLights.end(),
            [](auto &l1, auto &l2) { return l1->isShadowEnabled() && !l2->isShadowEnabled(); });
}

void Scene::updateVersion() {
  mVersion++;
  mRenderVersion++;
  mShadowEpoch++;
  detail::notifySceneVersionChanged(this, false);
}

void Scene::updateRenderVersion() {
  mRenderVersion++;
  mShadowEpoch++;
  detail::notifySceneVersionChanged(this, true);
}

void Scene::prepareObjectTransformBuffer() {
  size_t gpuTransformBufferSize = getGpuTransformBufferSize();

  // transform buffer is up-to-date
  if (mTransformBufferVersion == mVersion) {
    return;
  }

  // handle back-to-back render, update, render
  mContext->waitIdle();

  uint32_t count = 0;

  auto objects = getObjects();
  for (auto obj : objects) {
    obj->setInternalGpuIndex(count++);
  }
  auto lineObjects = getLineObjects();
  for (auto obj : lineObjects) {
    obj->setInternalGpuIndex(count++);
  }
  auto pointObjects = getPointObjects();
  for (auto obj : pointObjects) {
    obj->setInternalGpuIndex(count++);
  }

  // no need to create buffer if it is large enough
  if (mTransformBuffer && mTransformBuffer->getSize() >= gpuTransformBufferSize * count) {
    mTransformBufferVersion = mVersion;
    return;
  }

  logger::info("recreating object transform buffer.");

  count = std::max(count, 1u);

  mTransformBufferCpu =
      core::Buffer::Create(*mContext, gpuTransformBufferSize * count,
                           vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_ONLY);
  mTransformBuffer = core::Buffer::Create(
      *mContext, gpuTransformBufferSize * count,
      vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
      VMA_MEMORY_USAGE_GPU_ONLY, VmaAllocationCreateFlags{}, true);
  mTransformBufferVersion = mVersion;
}

size_t Scene::getGpuTransformBufferSize() {
  if (!mGpuTransformBufferSize) {
    // The transform table is read as a std430 mat4[] SSBO, whose stride is exactly one mat4.
    // There is no UBO dynamic-offset alignment requirement anymore.
    mGpuTransformBufferSize = sizeof(glm::mat4);
  }
  return mGpuTransformBufferSize;
}

void Scene::acquireExternalTransformOwnership(void const *owner) {
  if (!owner) {
    throw std::runtime_error("external transform owner must not be null");
  }
  mExternalTransformOwners.insert(owner);
}

void Scene::releaseExternalTransformOwnership(void const *owner) {
  mExternalTransformOwners.erase(owner);
}

void Scene::uploadObjectTransforms() {
  if (mTransformBufferRenderVersion == mRenderVersion) {
    return;
  }

  prepareObjectTransformBuffer();

  // collect data
  auto objects = getObjects();
  auto lineObjects = getLineObjects();
  auto pointObjects = getPointObjects();

  size_t totalSize{0};
  uint8_t *buffer = reinterpret_cast<uint8_t *>(mTransformBufferCpu->map());
  size_t step = getGpuTransformBufferSize();
  for (auto obj : objects) {
    auto mat = obj->getTransform().worldModelMatrix;
    std::memcpy(buffer + totalSize, &mat, sizeof(glm::mat4));
    totalSize += step;
  }
  for (auto obj : lineObjects) {
    auto mat = obj->getTransform().worldModelMatrix;
    std::memcpy(buffer + totalSize, &mat, sizeof(glm::mat4));
    totalSize += step;
  }
  for (auto obj : pointObjects) {
    auto mat = obj->getTransform().worldModelMatrix;
    std::memcpy(buffer + totalSize, &mat, sizeof(glm::mat4));
    totalSize += step;
  }
  mTransformBufferCpu->unmap();

  // rendering empty scene
  if (totalSize == 0) {
    return;
  }

  if (!mTransformUpdateCommandBuffer) {
    mTransformUpdateCommandBuffer = getCommandPool().allocateCommandBuffer();
  }

  mTransformUpdateCommandBuffer->reset();
  mTransformUpdateCommandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  {
    // make sure transfer happens after previous reads
    mTransformUpdateCommandBuffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eFragmentShader,
        vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, {});
  }

  vk::BufferCopy region(0, 0, totalSize);
  mTransformUpdateCommandBuffer->copyBuffer(mTransformBufferCpu->getVulkanBuffer(),
                                            mTransformBuffer->getVulkanBuffer(), region);
  {
    // wait for transfer write to be available, make visible to shader read
    vk::MemoryBarrier barrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead);

    // TODO: more shader stages required?
    mTransformUpdateCommandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                                   vk::PipelineStageFlagBits::eVertexShader |
                                                       vk::PipelineStageFlagBits::eFragmentShader,
                                                   {}, barrier, {}, {});
  }

  mTransformUpdateCommandBuffer->end();

  mContext->getQueue().submit(mTransformUpdateCommandBuffer.get(), {});

  mTransformBufferRenderVersion = mRenderVersion;
}

core::CommandPool &Scene::getCommandPool() {
  if (!mCommandPool) {
    mCommandPool = mContext->createCommandPool();
  }
  return *mCommandPool;
}

std::shared_ptr<core::Buffer> Scene::getObjectTransformBuffer() {
  prepareObjectTransformBuffer();
  return mTransformBuffer;
}

void Scene::createMissingBLAS(std::vector<core::BLAS *> &blases,
                              std::unordered_set<resource::SVModel *> &seen) {
  // Use the virtual getters so SceneGroup also initializes resources owned by
  // its child scenes. Child objects are built update-capable because their
  // rigid/deformable storage is private to the source scene.
  for (auto obj : getObjects()) {
    auto model = obj->getModel().get();
    if (model->getBLAS(*mContext) || !seen.insert(model).second) {
      continue;
    }
    bool isLocalRigid = std::any_of(mObjects.begin(), mObjects.end(),
                                    [obj](auto const &localObject) {
                                      return localObject.get() == obj;
                                    });
    blases.push_back(model->createBLAS(*mContext, !isLocalRigid));
  }
  for (auto obj : getPointObjects()) {
    if (!obj->getPointSet()->getBLAS(*mContext)) {
      obj->getPointSet()->buildBLAS(*mContext, true);
    }
  }
}

core::TLAS *Scene::createTLAS(bool externalInstanceBuffer) {
  std::vector<vk::AccelerationStructureInstanceKHR> instances;
  uint32_t instanceIndex{0};

  for (auto obj : getObjects()) {
    glm::mat4 modelTranspose =
        glm::transpose(obj->getTransform().worldModelMatrix); // column major matrix
    vk::TransformMatrixKHR mat;
    static_assert(sizeof(mat) == sizeof(float) * 12);
    std::memcpy(&mat.matrix[0][0], &modelTranspose, sizeof(mat));

    // Instance mask bit 0: primary/path visibility; bit 1: shadow casting.
    uint32_t mask = obj->getCastShadow() ? 0x03u : 0x01u;
    vk::AccelerationStructureInstanceKHR inst(
        mat, instanceIndex, mask, 0, vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable,
        obj->getModel()->getBLAS(*mContext)->getAddress());

    instanceIndex += obj->getModel()->getShapes().size();

    instances.push_back(inst);
  }

  instanceIndex = 0;
  for (auto obj : getPointObjects()) {
    glm::mat4 modelTranspose = glm::transpose(obj->getTransform().worldModelMatrix);
    vk::TransformMatrixKHR mat;
    std::memcpy(&mat.matrix[0][0], &modelTranspose, sizeof(mat));

    vk::AccelerationStructureInstanceKHR inst(
        mat, instanceIndex, 0x03u, 1, {}, obj->getPointSet()->getBLAS(*mContext)->getAddress());
    instanceIndex++;
    instances.push_back(inst);
  }

  mTLAS = std::make_unique<core::TLAS>(*mContext, instances, externalInstanceBuffer);
  return mTLAS.get();
}

void Scene::prepareRTBuildLocked() {
  // make sure scene is not in use
  if (mAccessFences.size()) {
    if (mContext->getDevice().waitForFences(mAccessFences, VK_TRUE, UINT64_MAX) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to wait for scene access fence");
    }
  }
  forceRemove();
}

void Scene::BuildAccelerationStructures(std::vector<AccelerationStructureBuild> const &builds) {
  std::vector<std::unique_lock<std::mutex>> locks;
  locks.reserve(builds.size());
  for (auto const &build : builds) {
    locks.emplace_back(build.scene->mRTResourcesLock);
    build.scene->prepareRTBuildLocked();
  }
  buildAccelerationStructuresLocked(builds);
}

void Scene::buildAccelerationStructuresLocked(
    std::vector<AccelerationStructureBuild> const &builds) {
  std::vector<AccelerationStructureBuild> stale;
  for (auto const &build : builds) {
    if (build.scene->mRTAccelerationStructureVersion != build.scene->mVersion) {
      stale.push_back(build);
    }
  }
  if (stale.empty()) {
    return;
  }
  auto &context = *stale.front().scene->mContext;

  std::vector<core::BLAS *> blases;
  std::unordered_set<resource::SVModel *> seen;
  for (auto const &build : stale) {
    build.scene->createMissingBLAS(blases, seen);
  }
  core::BLAS::BuildAll(context, blases);

  std::vector<core::TLAS *> tlases;
  for (auto const &build : stale) {
    tlases.push_back(build.scene->createTLAS(build.externalInstanceBuffer));
  }
  core::TLAS::BuildAll(context, tlases);
  for (auto const &build : stale) {
    build.scene->mRTAccelerationStructureVersion = build.scene->mVersion;
  }
}

static std::vector<vk::AccelerationStructureInstanceKHR>
packPrevTLASInstances(std::vector<Object *> const &objects,
                      std::vector<PointObject *> const &pointObjects) {
  std::vector<vk::AccelerationStructureInstanceKHR> rows;
  rows.reserve(objects.size() + pointObjects.size());
  auto append = [&](TransformWithCache const &t) {
    glm::mat4 const &m =
        t.prevWorldModelMatrix[3][3] != 0.f ? t.prevWorldModelMatrix : t.worldModelMatrix;
    glm::mat4 transposed = glm::transpose(m);
    vk::AccelerationStructureInstanceKHR inst{};
    std::memcpy(&inst.transform.matrix[0][0], &transposed, sizeof(inst.transform));
    rows.push_back(inst);
  };
  for (auto obj : objects) {
    append(obj->getTransform());
  }
  for (auto obj : pointObjects) {
    append(obj->getTransform());
  }
  return rows;
}

void Scene::updateTLAS(bool useDeviceTransforms) {
  if (!mASUpdateCommandBuffer) {
    mASUpdateCommandBuffer = getCommandPool().allocateCommandBuffer();
  }
  mASUpdateCommandBuffer->reset();
  mASUpdateCommandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  recordDynamicBLASUpdates(mASUpdateCommandBuffer.get());
  if (useDeviceTransforms) {
    mTLAS->recordUpdate(mASUpdateCommandBuffer.get());
  } else {
    std::vector<vk::TransformMatrixKHR> transforms;
    auto objects = getObjects();
    auto pointObjects = getPointObjects();

    transforms.reserve(objects.size() + pointObjects.size());

    for (auto obj : objects) {
      glm::mat4 modelTranspose = glm::transpose(obj->getTransform().worldModelMatrix);
      vk::TransformMatrixKHR mat;
      std::memcpy(&mat.matrix[0][0], &modelTranspose, sizeof(mat));
      transforms.push_back(mat);
    }

    for (auto obj : pointObjects) {
      glm::mat4 modelTranspose = glm::transpose(obj->getTransform().worldModelMatrix);
      vk::TransformMatrixKHR mat;
      std::memcpy(&mat.matrix[0][0], &modelTranspose, sizeof(mat));
      transforms.push_back(mat);
    }

    mTLAS->recordUpdate(mASUpdateCommandBuffer.get(), transforms);
    if (mPrevTLASInstanceBuffer) {
      auto prevRows = packPrevTLASInstances(objects, pointObjects);
      if (!prevRows.empty()) {
        mPrevTLASInstanceBuffer->upload(prevRows);
      }
    }
  }
  mASUpdateCommandBuffer->end();
  mContext->getQueue().submit(mASUpdateCommandBuffer.get(), {});
}

bool Scene::recordBLASRefits(vk::CommandBuffer commandBuffer) {
  std::lock_guard lock(mRTResourcesLock);
  if (mRTAccelerationStructureVersion != mVersion) {
    throw std::runtime_error("recordBLASRefits failed: scene has changed, call "
                             "BuildAccelerationStructures first");
  }
  bool recorded = recordDynamicBLASUpdates(commandBuffer);
  mRTResourcesRenderVersion = mRenderVersion;
  return recorded;
}

bool Scene::recordDynamicBLASUpdates(vk::CommandBuffer commandBuffer) {
  for (auto &obj : mDeformableObjects) {
    obj->getModel()->recordUpdateBLAS(*mContext, commandBuffer);
  }

  // FIXME: update per pointset, not per point cloud
  auto pointObjects = getPointObjects();
  for (auto &obj : pointObjects) {
    obj->getPointSet()->recordUpdateBLAS(*mContext, commandBuffer);
  }
  return !mDeformableObjects.empty() || !pointObjects.empty();
}

void Scene::createRTStorageBuffers(StructDataLayout const &geometryInstanceBufferLayout) {

  uint32_t geometryInstanceSize = geometryInstanceBufferLayout.size;
  uint32_t materialIndexOffset = geometryInstanceBufferLayout.elements.at("materialIndex").offset;
  uint32_t geometryIndexOffset = geometryInstanceBufferLayout.elements.at("geometryIndex").offset;
  auto objects = getObjects();

  uint32_t instanceCount{0};
  uint32_t meshCount{0};
  uint32_t materialCount{0};

  std::unordered_map<std::shared_ptr<resource::SVMesh>, uint32_t> mesh2Id;
  std::unordered_map<std::shared_ptr<resource::SVMaterial>, uint32_t> material2Id;
  std::unordered_map<std::shared_ptr<resource::SVTexture>, uint32_t> texture2Id;

  std::vector<std::shared_ptr<resource::SVMesh>> meshes;
  std::vector<std::shared_ptr<resource::SVMetallicMaterial>> materials;
  std::vector<std::shared_ptr<resource::SVTexture>> textures;

  for (auto &obj : objects) {
    for (auto &shape : obj->getModel()->getShapes()) {
      shape->mesh->uploadToDevice(*mContext);
      ++instanceCount;

      if (!mesh2Id.contains(shape->mesh)) {
        mesh2Id[shape->mesh] = meshCount++;
        meshes.push_back(shape->mesh);
      }
      if (!material2Id.contains(shape->material)) {
        material2Id[shape->material] = materialCount++;
        materials.push_back(std::static_pointer_cast<resource::SVMetallicMaterial>(shape->material));
      }
    }
  }

  std::vector<uint8_t> geometryInstanceBuffer(instanceCount * geometryInstanceSize);
  uint32_t instanceIndex = 0;
  for (auto &obj : objects) {
    for (auto &shape : obj->getModel()->getShapes()) {
      uint32_t meshId = mesh2Id.at(shape->mesh);
      uint32_t materialId = material2Id.at(shape->material);
      std::memcpy(geometryInstanceBuffer.data() + geometryInstanceSize * instanceIndex +
                      geometryIndexOffset,
                  &meshId, sizeof(uint32_t));
      std::memcpy(geometryInstanceBuffer.data() + geometryInstanceSize * instanceIndex +
                      materialIndexOffset,
                  &materialId, sizeof(uint32_t));
      ++instanceIndex;
    }
  }

  // textures[]: the materials' textures, then those of textured lights
  mRTTextureSlots.clear();
  auto textureSlot = [&](std::shared_ptr<resource::SVTexture> const &texture) {
    if (!texture2Id.contains(texture)) {
      texture->loadAsync().get();
      texture->uploadToDevice(*mContext);
      texture2Id[texture] = static_cast<uint32_t>(textures.size());
      textures.push_back(texture);
    }
    return mRTTextureSlots[texture.get()] = static_cast<int>(texture2Id.at(texture));
  };
  mRTMaterials = std::move(materials);
  mRTMaterialRecords.clear();
  for (auto const &material : mRTMaterials) {
    mRTMaterialRecords.push_back(packMaterial(*material, textureSlot));
  }
  mRTMaterialBuffer = core::Buffer::Create(
      *mContext, std::max<size_t>(mRTMaterialRecords.size(), 1) * sizeof(GpuMaterial),
      vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_CPU_TO_GPU);
  if (!mRTMaterialRecords.empty()) {
    mRTMaterialBuffer->upload(mRTMaterialRecords);
  }
  mRTMaterialVersion = resource::SVMetallicMaterial::getGlobalVersion();
  ++mRTMaterialVersions.content;
  ++mRTMaterialVersions.emission;
  ++mRTTablesVersion;

  mVertexBuffers.clear();
  mIndexBuffers.clear();
  for (auto mesh : meshes) {
    mVertexBuffers.push_back(mesh->getVertexBuffer(*mContext).getVulkanBuffer());
    mIndexBuffers.push_back(mesh->getIndexBuffer(*mContext).getVulkanBuffer());
  }

  mGeometryInstanceBuffer = core::Buffer::Create(
      *mContext, std::max(size_t(1), geometryInstanceBuffer.size()),
      vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_GPU_ONLY);
  mGeometryInstanceBuffer->upload(geometryInstanceBuffer);

  // point objects
  uint32_t pointsetCount{0};
  auto pointObjects = getPointObjects();
  std::unordered_map<std::shared_ptr<resource::SVPointSet>, uint32_t> pointset2Id;
  std::vector<std::shared_ptr<resource::SVPointSet>> pointsets;
  std::vector<int> pointInstanceBuffer;
  for (auto &obj : pointObjects) {
    if (!pointset2Id.contains(obj->getPointSet())) {
      pointset2Id[obj->getPointSet()] = pointsetCount++;
      pointsets.push_back(obj->getPointSet());
    }
    pointInstanceBuffer.push_back(pointset2Id[obj->getPointSet()]);
  }
  mPointInstanceBuffer = core::Buffer::Create(
      *mContext, std::max(size_t(1), pointInstanceBuffer.size() * sizeof(int)),
      vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_GPU_ONLY);
  mPointInstanceBuffer->upload(pointInstanceBuffer);

  // point sets
  mPointSetBuffers.clear();
  for (auto s : pointsets) {
    mPointSetBuffers.push_back(s->getVertexBuffer(*mContext).getVulkanBuffer());
  }

  // lights; a textured light's texture joins textures[]
  mRTLights.clear();
  mRTLights.append(*this, textureSlot);
  auto lightBuffer = [&](auto const &lights) {
    // a shader array of zero length still needs a buffer to bind
    return core::Buffer::Create(
        *mContext, std::max<size_t>(lights.size(), 1) * sizeof(lights[0]),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_CPU_TO_GPU);
  };
  mRTPointLightBuffer = lightBuffer(mRTLights.point);
  mRTDirectionalLightBuffer = lightBuffer(mRTLights.directional);
  mRTSpotLightBuffer = lightBuffer(mRTLights.spot);
  mRTParallelogramLightBuffer = lightBuffer(mRTLights.parallelogram);
  writeRTLightBuffers();

  // previous-frame TLAS instance transforms for ray-traced motion vectors
  auto prevRows = packPrevTLASInstances(objects, pointObjects);
  mPrevTLASInstanceBuffer = core::Buffer::Create(
      *mContext,
      std::max<size_t>(1, prevRows.size()) * sizeof(vk::AccelerationStructureInstanceKHR),
      vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_CPU_TO_GPU);
  if (!prevRows.empty()) {
    mPrevTLASInstanceBuffer->upload(prevRows);
  }

  // build textures
  mTextures.clear();
  for (auto texture : textures) {
    mTextures.push_back({texture->getImageView(*mContext), texture->getSampler(*mContext)});
  }
}

void Scene::writeRTLightBuffers() {
  auto upload = [](core::Buffer &buffer, auto const &lights) {
    if (!lights.empty()) {
      buffer.upload(lights);
    }
  };
  upload(*mRTPointLightBuffer, mRTLights.point);
  upload(*mRTDirectionalLightBuffer, mRTLights.directional);
  upload(*mRTSpotLightBuffer, mRTLights.spot);
  upload(*mRTParallelogramLightBuffer, mRTLights.parallelogram);
}

void Scene::updateRTStorageBuffers() {
  // the scene version did not change, so the same lights and textures are repacked
  mRTLights.clear();
  mRTLights.append(*this, [&](std::shared_ptr<resource::SVTexture> const &texture) {
    return mRTTextureSlots.at(texture.get());
  });
  writeRTLightBuffers();
}

uint64_t Scene::getRTTablesVersion() {
  std::lock_guard lock(mRTResourcesLock);
  return mRTTablesVersion;
}

std::optional<Scene::RTMaterialVersions> Scene::uploadRTMaterials() {
  std::lock_guard lock(mRTResourcesLock);
  if (mRTResourcesVersion != mVersion) {
    return std::nullopt;
  }
  uint64_t version = resource::SVMetallicMaterial::getGlobalVersion();
  if (version == mRTMaterialVersion) {
    return mRTMaterialVersions;
  }

  bool newTexture = false;
  std::vector<GpuMaterial> records;
  records.reserve(mRTMaterials.size());
  for (auto const &material : mRTMaterials) {
    records.push_back(
        packMaterial(*material, [&](std::shared_ptr<resource::SVTexture> const &texture) {
          auto slot = mRTTextureSlots.find(texture.get());
          newTexture |= slot == mRTTextureSlots.end();
          return slot == mRTTextureSlots.end() ? -1 : slot->second;
        }));
  }
  if (newTexture) {
    // textures[] is sized at build time: drop the tables so buildRTResources repacks them
    mRTResourcesVersion = 0;
    return std::nullopt;
  }
  mRTMaterialVersion = version;
  // edits of other scenes' materials bump the global version too; they leave this table as is
  if (std::memcmp(records.data(), mRTMaterialRecords.data(),
                  records.size() * sizeof(GpuMaterial)) == 0) {
    return mRTMaterialVersions;
  }
  bool emissionChanged = false;
  for (size_t i = 0; i < records.size(); ++i) {
    auto const &now = records[i];
    auto const &before = mRTMaterialRecords[i];
    emissionChanged |= now.parameters.emissionColor != before.parameters.emissionColor ||
                       now.parameters.emissionStrength != before.parameters.emissionStrength ||
                       now.textures[resource::SVMetallicMaterial::eEmission] !=
                           before.textures[resource::SVMetallicMaterial::eEmission];
  }

  // make sure scene is not in use
  if (mAccessFences.size()) {
    if (mContext->getDevice().waitForFences(mAccessFences, VK_TRUE, UINT64_MAX) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to wait for scene access fence");
    }
  }
  mRTMaterialBuffer->upload(records);
  mRTMaterialRecords = std::move(records);
  ++mRTMaterialVersions.content;
  mRTMaterialVersions.emission += emissionChanged;
  return mRTMaterialVersions;
}

void Scene::buildRTResources(StructDataLayout const &geometryInstanceBufferLayout,
                             bool externalInstanceBuffer) {

  std::lock_guard lock(mRTResourcesLock);
  prepareRTBuildLocked();

  if (mRTAccelerationStructureVersion != mVersion) {
    buildAccelerationStructuresLocked({{this, externalInstanceBuffer}});
  } else if (externalInstanceBuffer && !mTLAS->isInstanceBufferExternal()) {
    mTLAS->enableExternalInstanceBuffer();
  }
  if (mRTResourcesVersion != mVersion) {
    createRTStorageBuffers(geometryInstanceBufferLayout);
    mRTResourcesVersion = mVersion;
  }
}

void Scene::updateRTResources(bool useDeviceTransforms) {
  updateRTTransforms(useDeviceTransforms);
  // the batched renderer builds only the acceleration structures and keeps its own light tables
  bool hasStorageBuffers;
  {
    std::lock_guard lock(mRTResourcesLock);
    hasStorageBuffers = mRTResourcesVersion == mVersion;
  }
  if (hasStorageBuffers) {
    uploadRTLights();
  }
}

void Scene::updateRTTransforms(bool useDeviceTransforms) {
  std::lock_guard lock(mRTResourcesLock);
  if (mRTAccelerationStructureVersion != mVersion) {
    throw std::runtime_error("updateRTTransforms failed: scene has changed, "
                             "call buildRTResources first");
  }

  // make sure scene is not in use
  if (mAccessFences.size()) {
    if (mContext->getDevice().waitForFences(mAccessFences, VK_TRUE, UINT64_MAX) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to wait for scene access fence");
    }
  }

  updateTLAS(useDeviceTransforms);
  mRTResourcesRenderVersion = mRenderVersion;
}

void Scene::uploadRTLights() {
  std::lock_guard lock(mRTResourcesLock);
  if (mRTResourcesVersion != mVersion) {
    throw std::runtime_error("uploadRTLights failed: scene has changed, "
                             "call buildRTResources first");
  }

  // make sure scene is not in use
  if (mAccessFences.size()) {
    if (mContext->getDevice().waitForFences(mAccessFences, VK_TRUE, UINT64_MAX) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to wait for scene access fence");
    }
  }

  updateRTStorageBuffers();
}

void Scene::registerAccessFence(vk::Fence fence) { mAccessFences.push_back(fence); }
void Scene::unregisterAccessFence(vk::Fence fence) { std::erase(mAccessFences, fence); }

} // namespace scene
} // namespace svulkan2
