#include "svulkan2/scene/scene.h"
#include "../common/logger.h"
#include "svulkan2/core/context.h"
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

struct LightBufferData {
  glm::mat4 viewMatrix;
  glm::mat4 viewMatrixInverse;
  glm::mat4 projectionMatrix;
  glm::mat4 projectionMatrixInverse;
  int width;
  int height;
};

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

void Scene::uploadShadowToDevice(core::Buffer &shadowBuffer,
                                 std::vector<std::unique_ptr<core::Buffer>> &lightBuffers,
                                 StructDataLayout const &shadowLayout) {

  uint32_t maxNumDirectionalLightShadows =
      shadowLayout.elements.at("directionalLightBuffers").size /
      shadowLayout.elements.at("directionalLightBuffers").member->size;
  uint32_t maxNumPointLightShadows = shadowLayout.elements.at("pointLightBuffers").size /
                                     shadowLayout.elements.at("pointLightBuffers").member->size /
                                     6;
  uint32_t maxNumSpotLightShadows = shadowLayout.elements.at("spotLightBuffers").size /
                                    shadowLayout.elements.at("spotLightBuffers").member->size;
  uint32_t maxNumTexturedLightShadows =
      shadowLayout.elements.at("texturedLightBuffers").size /
      shadowLayout.elements.at("texturedLightBuffers").member->size;

  ScopedBufferMapping shadowMapping(shadowBuffer);

  uint32_t lightBufferIndex = 0;
  {
    std::vector<LightBufferData> directionalLightShadowData;
    uint32_t numDirectionalLightShadows = 0;
    for (auto &l : getDirectionalLights()) {
      if (l->isShadowEnabled()) {
        if (numDirectionalLightShadows >= maxNumDirectionalLightShadows) {
          throw std::runtime_error("The scene contains too many directional "
                                   "lights that cast shadows.");
          break;
        }
        numDirectionalLightShadows++;

        auto modelMat = l->getTransform().worldModelMatrix;
        auto projMat = l->getShadowProjectionMatrix();
        directionalLightShadowData.push_back({.viewMatrix = glm::affineInverse(modelMat),
                                              .viewMatrixInverse = modelMat,
                                              .projectionMatrix = projMat,
                                              .projectionMatrixInverse = glm::inverse(projMat),
                                              .width = static_cast<int>(l->getShadowMapSize()),
                                              .height = static_cast<int>(l->getShadowMapSize())});
        lightBuffers[lightBufferIndex++]->upload(&directionalLightShadowData.back(),
                                                 sizeof(LightBufferData));
      } else {
        break;
      }
    }
    shadowBuffer.upload(directionalLightShadowData.data(),
                        directionalLightShadowData.size() * sizeof(LightBufferData),
                        shadowLayout.elements.at("directionalLightBuffers").offset);
  }

  {
    std::vector<LightBufferData> pointLightShadowData;
    uint32_t numPointLightShadows = 0;
    for (auto &l : getPointLights()) {
      if (l->isShadowEnabled()) {
        if (numPointLightShadows >= maxNumPointLightShadows) {
          throw std::runtime_error("The scene contains too many point "
                                   "lights that cast shadows.");
          // log::warn(
          //     "The scene contains too many point lights that cast shadows. "
          //     "Truncated.");
          break;
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
          lightBuffers[lightBufferIndex++]->upload(&pointLightShadowData.back(),
                                                   sizeof(LightBufferData));
        }
      } else {
        break;
      }
    }
    shadowBuffer.upload(pointLightShadowData.data(),
                        pointLightShadowData.size() * sizeof(LightBufferData),
                        shadowLayout.elements.at("pointLightBuffers").offset);
  }

  {
    std::vector<LightBufferData> spotLightShadowData;
    uint32_t numSpotLightShadows = 0;
    for (auto &l : getSpotLights()) {
      if (l->isShadowEnabled()) {
        if (numSpotLightShadows >= maxNumSpotLightShadows) {
          throw std::runtime_error("The scene contains too many spot "
                                   "lights that cast shadows.");
          break;
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
        lightBuffers[lightBufferIndex++]->upload(&spotLightShadowData.back(),
                                                 sizeof(LightBufferData));
      } else {
        break;
      }
    }
    shadowBuffer.upload(spotLightShadowData.data(),
                        spotLightShadowData.size() * sizeof(LightBufferData),
                        shadowLayout.elements.at("spotLightBuffers").offset);
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
      lightBuffers[lightBufferIndex++]->upload(&texturedLightShadowData.back(),
                                               sizeof(LightBufferData));
    }
    shadowBuffer.upload(texturedLightShadowData.data(),
                        texturedLightShadowData.size() * sizeof(LightBufferData),
                        shadowLayout.elements.at("texturedLightBuffers").offset);
  }
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
  detail::notifySceneVersionChanged(this, false);
}

void Scene::updateRenderVersion() {
  mRenderVersion++;
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

    vk::AccelerationStructureInstanceKHR inst(
        mat, instanceIndex, 0xff, 0, vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable,
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
        mat, instanceIndex, 0xff, 1, {}, obj->getPointSet()->getBLAS(*mContext)->getAddress());
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
    emissionChanged |= records[i].emission != mRTMaterialRecords[i].emission ||
                       records[i].emissionTexture != mRTMaterialRecords[i].emissionTexture;
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
