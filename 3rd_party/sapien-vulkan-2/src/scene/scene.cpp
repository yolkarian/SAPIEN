#include "svulkan2/scene/scene.h"
#include "../common/logger.h"
#include "svulkan2/core/context.h"
#include <algorithm>
#include <mutex>
#include <unordered_map>
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

struct PointLightData {
  glm::vec4 position;
  glm::vec4 color;
};

struct DirectionalLightData {
  glm::vec3 direction;
  float softness;
  glm::vec4 color;
};

struct SpotLightData {
  glm::vec4 position;
  glm::vec4 direction;
  glm::vec4 color;
};

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
  mRootNode->updateGlobalModelMatrixRecursive();
}

void Scene::uploadToDevice(core::Buffer &sceneBuffer, StructDataLayout const &sceneLayout) {
  auto pointLights = getPointLights();
  auto directionalLights = getDirectionalLights();
  auto spotLights = getSpotLights();
  auto texturedLights = getTexturedLights();

  // Raster shaders bind shadow maps before unshadowed lights. Keep the light
  // buffers in the same stable order so mixed shadow settings address the
  // matching light and shadow map.
  auto shadowsFirst = [](auto const *light) { return light->isShadowEnabled(); };
  std::stable_partition(pointLights.begin(), pointLights.end(), shadowsFirst);
  std::stable_partition(directionalLights.begin(), directionalLights.end(), shadowsFirst);
  std::stable_partition(spotLights.begin(), spotLights.end(), shadowsFirst);
  std::stable_partition(texturedLights.begin(), texturedLights.end(), shadowsFirst);

  std::vector<PointLightData> pointLightData;
  std::vector<DirectionalLightData> directionalLightData;
  std::vector<SpotLightData> spotLightData;
  std::vector<SpotLightData> texturedLightData;
  pointLightData.reserve(pointLights.size());
  directionalLightData.reserve(directionalLights.size());
  spotLightData.reserve(spotLights.size());
  texturedLightData.reserve(texturedLights.size());
  for (auto light : pointLights) {
    pointLightData.push_back(
        {.position = light->getTransform().worldModelMatrix * glm::vec4(0, 0, 0, 1),
         .color = glm::vec4(light->getColor(), 0)});
  }

  for (auto light : directionalLights) {
    directionalLightData.push_back(
        {.direction = glm::vec3(light->getTransform().worldModelMatrix * glm::vec4(0, 0, -1, 0)),
         .softness = light->getSoftness(),
         .color = glm::vec4(light->getColor(), 0)});
  }

  for (auto light : spotLights) {
    spotLightData.push_back({.position = glm::vec4(light->getPosition(), 1),
                             .direction = glm::vec4(light->getDirection(), light->getFov()),
                             .color = glm::vec4(light->getColor(), light->getFovSmall())});
  }

  for (auto light : texturedLights) {
    texturedLightData.push_back({.position = glm::vec4(light->getPosition(), 1),
                                 .direction = glm::vec4(light->getDirection(), light->getFov()),
                                 .color = glm::vec4(light->getColor(), light->getFovSmall())});
  }

  ScopedBufferMapping sceneMapping(sceneBuffer);
  sceneBuffer.upload(&mAmbientLight, 16, sceneLayout.elements.at("ambientLight").offset);
  // Use the aggregated (virtual getter) counts so a SceneGroup uploads member-scene
  // lights; the base member vectors are empty for groups.
  uint32_t numPointLights = pointLights.size();
  uint32_t numDirectionalLights = directionalLights.size();
  uint32_t numSpotLights = spotLights.size();
  uint32_t numTexturedLights = texturedLights.size();
  uint32_t maxNumPointLights = sceneLayout.elements.at("pointLights").size /
                               sceneLayout.elements.at("pointLights").member->size;
  uint32_t maxNumDirectionalLights = sceneLayout.elements.at("directionalLights").size /
                                     sceneLayout.elements.at("directionalLights").member->size;
  uint32_t maxNumSpotLights = sceneLayout.elements.at("spotLights").size /
                              sceneLayout.elements.at("spotLights").member->size;
  uint32_t maxNumTexturedLights = sceneLayout.elements.at("texturedLights").size /
                                  sceneLayout.elements.at("texturedLights").member->size;

  if (maxNumPointLights < numPointLights) {
    logger::warn("The scene contains more point lights than the maximum number of "
                 "point lights in the shader. Truncated.");
    numPointLights = maxNumPointLights;
  }
  if (maxNumDirectionalLights < numDirectionalLights) {
    logger::warn("The scene contains more directional lights than the maximum number of "
                 "directional lights in the shader. Truncated.");
    numDirectionalLights = maxNumDirectionalLights;
  }
  if (maxNumSpotLights < numSpotLights) {
    logger::warn("The scene contains more spot lights than the maximum number of "
                 "spot lights in the shader. Truncated.");
    numSpotLights = maxNumSpotLights;
  }
  if (maxNumTexturedLights < numTexturedLights) {
    logger::warn("The scene contains more textured lights than the maximum number of "
                 "textured lights in the shader. Truncated.");
    numTexturedLights = maxNumTexturedLights;
  }

  sceneBuffer.upload(pointLightData.data(), numPointLights * sizeof(PointLightData),
                     sceneLayout.elements.at("pointLights").offset);
  sceneBuffer.upload(directionalLightData.data(),
                     numDirectionalLights * sizeof(DirectionalLightData),
                     sceneLayout.elements.at("directionalLights").offset);
  sceneBuffer.upload(spotLightData.data(), numSpotLights * sizeof(SpotLightData),
                     sceneLayout.elements.at("spotLights").offset);
  sceneBuffer.upload(texturedLightData.data(), numTexturedLights * sizeof(SpotLightData),
                     sceneLayout.elements.at("texturedLights").offset);
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
      vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst |
          vk::BufferUsageFlagBits::eStorageBuffer,
      VMA_MEMORY_USAGE_GPU_ONLY, VmaAllocationCreateFlags{}, true);
  mTransformBufferVersion = mVersion;
}

size_t Scene::getGpuTransformBufferSize() {
  if (!mGpuTransformBufferSize) {
    mGpuTransformBufferSize =
        std::max((uint64_t)sizeof(glm::mat4),
                 (uint64_t)mContext->getPhysicalDeviceLimits().minUniformBufferOffsetAlignment);
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

void Scene::ensureBLAS() {
  // Use the virtual getters so SceneGroup also initializes resources owned by
  // its child scenes. Child objects are built update-capable because their
  // rigid/deformable storage is private to the source scene.
  for (auto obj : getObjects()) {
    if (!obj->getModel()->getBLAS(*mContext)) {
      bool isLocalRigid = std::any_of(mObjects.begin(), mObjects.end(),
                                      [obj](auto const &localObject) {
                                        return localObject.get() == obj;
                                      });
      obj->getModel()->buildBLAS(*mContext, !isLocalRigid);
    }
  }
  for (auto obj : getPointObjects()) {
    if (!obj->getPointSet()->getBLAS(*mContext)) {
      obj->getPointSet()->buildBLAS(*mContext, true);
    }
  }
}

void Scene::buildTLAS(bool externalInstanceBuffer) {
  logger::info("building TLAS");
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

  mTLAS = std::make_unique<core::TLAS>(*mContext, instances);
  mTLAS->build(externalInstanceBuffer);
}

void Scene::updateTLAS(bool useDeviceTransforms) {
  if (!mASUpdateCommandBuffer) {
    mASUpdateCommandBuffer = getCommandPool().allocateCommandBuffer();
  }
  mASUpdateCommandBuffer->reset();
  mASUpdateCommandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  // update BLAS
  for (auto &obj : mDeformableObjects) {
    obj->getModel()->recordUpdateBLAS(*mContext, mASUpdateCommandBuffer.get());
  }

  // FIXME: update per pointset, not per point cloud
  for (auto &obj : getPointObjects()) {
    obj->getPointSet()->recordUpdateBLAS(*mContext, mASUpdateCommandBuffer.get());
  }

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

void Scene::createRTStorageBuffers(StructDataLayout const &materialBufferLayout,
                                   StructDataLayout const &textureIndexBufferLayout,
                                   StructDataLayout const &geometryInstanceBufferLayout) {

  uint32_t geometryInstanceSize = geometryInstanceBufferLayout.size;
  uint32_t materialIndexOffset = geometryInstanceBufferLayout.elements.at("materialIndex").offset;
  uint32_t geometryIndexOffset = geometryInstanceBufferLayout.elements.at("geometryIndex").offset;

  uint32_t textureIndexSize = textureIndexBufferLayout.size;
  uint32_t emissionOffset = textureIndexBufferLayout.elements.at("emission").offset;
  uint32_t diffuseOffset = textureIndexBufferLayout.elements.at("diffuse").offset;
  uint32_t metallicOffset = textureIndexBufferLayout.elements.at("metallic").offset;
  uint32_t roughnessOffset = textureIndexBufferLayout.elements.at("roughness").offset;
  uint32_t normalOffset = textureIndexBufferLayout.elements.at("normal").offset;
  uint32_t transmissionOffset = textureIndexBufferLayout.elements.at("transmission").offset;

  auto objects = getObjects();

  uint32_t instanceCount{0};
  uint32_t meshCount{0};
  uint32_t materialCount{0};
  uint32_t textureCount{0};

  std::unordered_map<std::shared_ptr<resource::SVMesh>, uint32_t> mesh2Id;
  std::unordered_map<std::shared_ptr<resource::SVMaterial>, uint32_t> material2Id;
  std::unordered_map<std::shared_ptr<resource::SVTexture>, uint32_t> texture2Id;

  std::vector<std::shared_ptr<resource::SVMesh>> meshes;
  std::vector<std::shared_ptr<resource::SVMetallicMaterial>> materials;
  std::vector<std::shared_ptr<resource::SVTexture>> textures;

  for (auto &obj : objects) {
    for (auto &shape : obj->getModel()->getShapes()) {
      shape->mesh->uploadToDevice(*mContext);
      shape->material->uploadToDevice(*mContext);
      ++instanceCount;

      if (!mesh2Id.contains(shape->mesh)) {
        mesh2Id[shape->mesh] = meshCount++;
        meshes.push_back(shape->mesh);
      }
      if (!material2Id.contains(shape->material)) {
        material2Id[shape->material] = materialCount++;
        auto mat = std::static_pointer_cast<resource::SVMetallicMaterial>(shape->material);
        materials.push_back(mat);

        std::array matTextures{mat->getEmissionTexture(),  mat->getBaseColorTexture(),
                               mat->getNormalTexture(),    mat->getMetallicTexture(),
                               mat->getRoughnessTexture(), mat->getTransmissionTexture()};
        for (auto tex : matTextures) {
          if (tex && !texture2Id.contains(tex)) {
            texture2Id[tex] = textureCount++;
            textures.push_back(tex);
          }
        }
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

  std::vector<uint8_t> textureIndexBuffer(materialCount * textureIndexSize);
  uint32_t materialIndex{0};
  for (auto mat : materials) {
    auto emissionTex = mat->getEmissionTexture();
    auto diffuseTex = mat->getBaseColorTexture();
    auto normalTex = mat->getNormalTexture();
    auto metallicTex = mat->getMetallicTexture();
    auto roughnessTex = mat->getRoughnessTexture();
    auto transmissionTex = mat->getTransmissionTexture();

    int emissionId = emissionTex ? texture2Id.at(emissionTex) : -1;
    int diffuseId = diffuseTex ? texture2Id.at(diffuseTex) : -1;
    int normalId = normalTex ? texture2Id.at(normalTex) : -1;
    int metallicId = metallicTex ? texture2Id.at(metallicTex) : -1;
    int roughnessId = roughnessTex ? texture2Id.at(roughnessTex) : -1;
    int transmissionId = transmissionTex ? texture2Id.at(transmissionTex) : -1;

    std::memcpy(textureIndexBuffer.data() + materialIndex * textureIndexSize + emissionOffset,
                &emissionId, sizeof(int));
    std::memcpy(textureIndexBuffer.data() + materialIndex * textureIndexSize + diffuseOffset,
                &diffuseId, sizeof(int));
    std::memcpy(textureIndexBuffer.data() + materialIndex * textureIndexSize + normalOffset,
                &normalId, sizeof(int));
    std::memcpy(textureIndexBuffer.data() + materialIndex * textureIndexSize + metallicOffset,
                &metallicId, sizeof(int));
    std::memcpy(textureIndexBuffer.data() + materialIndex * textureIndexSize + roughnessOffset,
                &roughnessId, sizeof(int));
    std::memcpy(textureIndexBuffer.data() + materialIndex * textureIndexSize + transmissionOffset,
                &transmissionId, sizeof(int));

    ++materialIndex;
  }

  mVertexBuffers.clear();
  mIndexBuffers.clear();
  for (auto mesh : meshes) {
    mVertexBuffers.push_back(mesh->getVertexBuffer(*mContext).getVulkanBuffer());
    mIndexBuffers.push_back(mesh->getIndexBuffer(*mContext).getVulkanBuffer());
  }

  // build materials
  mMaterialBuffers.clear();
  for (auto mat : materials) {
    mMaterialBuffers.push_back(mat->getDeviceBuffer(*mContext).getVulkanBuffer());
  }

  // build texture index
  mTextureIndexBuffer = core::Buffer::Create(
      *mContext, std::max(size_t(1), textureIndexBuffer.size()),
      vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_GPU_ONLY);
  mTextureIndexBuffer->upload(textureIndexBuffer);

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

  // begin lights
  auto pointLights = getPointLights();
  auto directionalLights = getDirectionalLights();
  auto spotLights = getSpotLights();
  auto texturedLights = getTexturedLights();
  auto parallelogramLights = getParallelogramLights();

  mRTPointLightBufferHost.clear();
  for (auto l : pointLights) {
    mRTPointLightBufferHost.push_back(
        RTPointLight{.position = l->getPosition(), .radius = 0, .rgb = l->getColor()});
  }
  mRTPointLightBuffer = core::Buffer::Create(
      *mContext, std::max(pointLights.size(), size_t(1)) * sizeof(RTPointLight),
      vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_CPU_TO_GPU);
  mRTPointLightBuffer->upload(mRTPointLightBufferHost);

  mRTDirectionalLightBufferHost.clear();
  for (auto l : directionalLights) {
    mRTDirectionalLightBufferHost.push_back(RTDirectionalLight{
        .direction = l->getDirection(), .softness = l->getSoftness(), .rgb = l->getColor()});
  }
  mRTDirectionalLightBuffer = core::Buffer::Create(
      *mContext, std::max(directionalLights.size(), size_t(1)) * sizeof(RTDirectionalLight),
      vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_CPU_TO_GPU);
  mRTDirectionalLightBuffer->upload(mRTDirectionalLightBufferHost);

  mRTSpotLightBufferHost.clear();
  for (auto l : spotLights) {
    mRTSpotLightBufferHost.push_back(
        RTSpotLight{.viewMat = glm::affineInverse(l->getTransform().worldModelMatrix),
                    .projMat = l->getShadowProjectionMatrix(),
                    .rgb = l->getColor(),
                    .position = l->getPosition(),
                    .fovInner = l->getFovSmall(),
                    .fovOuter = l->getFov(),
                    .textureId = -1});
  }
  for (auto l : texturedLights) {
    auto tex = l->getTexture();
    tex->loadAsync().get(); // TODO: move to top
    tex->uploadToDevice(*mContext);
    if (tex && !texture2Id.contains(tex)) {
      textures.push_back(tex);
      texture2Id[tex] = textures.size() - 1;
    }
    int textureId = texture2Id.at(tex);

    mRTSpotLightBufferHost.push_back(
        RTSpotLight{.viewMat = glm::affineInverse(l->getTransform().worldModelMatrix),
                    .projMat = l->getShadowProjectionMatrix(),
                    .rgb = l->getColor(),
                    .position = l->getPosition(),
                    .fovInner = l->getFovSmall(),
                    .fovOuter = l->getFov(),
                    .textureId = textureId});
  }
  mRTSpotLightBuffer = core::Buffer::Create(
      *mContext,
      std::max(texturedLights.size() + spotLights.size(), size_t(1)) * sizeof(RTSpotLight),
      vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_CPU_TO_GPU);
  mRTSpotLightBuffer->upload(mRTSpotLightBufferHost);

  mRTParallelogramLightBufferHost.clear();
  for (auto l : parallelogramLights) {
    mRTParallelogramLightBufferHost.push_back(RTParallelogramLight{
        .color = l->getColor(),
        .position = l->getOrigin(),
        .edge0 = l->getEdgeX(),
        .edge1 = l->getEdgeY(),
    });
  }
  mRTParallelogramLightBuffer = core::Buffer::Create(
      *mContext, std::max(parallelogramLights.size(), size_t(1)) * sizeof(RTParallelogramLight),
      vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_CPU_TO_GPU);
  mRTParallelogramLightBuffer->upload(mRTParallelogramLightBufferHost);
  // end lights

  // build textures
  mTextures.clear();
  for (auto texture : textures) {
    mTextures.push_back({texture->getImageView(*mContext), texture->getSampler(*mContext)});
  }
}

void Scene::updateRTStorageBuffers() {
  uint32_t i = 0;
  for (auto l : getPointLights()) {
    auto &b = mRTPointLightBufferHost.at(i);
    b.position = l->getPosition();
    b.radius = 0;
    b.rgb = l->getColor();
    ++i;
  }
  mRTPointLightBuffer->upload(mRTPointLightBufferHost);

  i = 0;
  for (auto l : getDirectionalLights()) {
    auto &b = mRTDirectionalLightBufferHost.at(i);
    b.direction = l->getDirection();
    b.softness = l->getSoftness();
    b.rgb = l->getColor();
    ++i;
  }
  mRTDirectionalLightBuffer->upload(mRTDirectionalLightBufferHost);

  i = 0;
  for (auto l : getSpotLights()) {
    auto &b = mRTSpotLightBufferHost.at(i);
    b.viewMat = glm::affineInverse(l->getTransform().worldModelMatrix);
    b.projMat = l->getShadowProjectionMatrix();
    b.rgb = l->getColor();
    b.position = l->getPosition();
    b.fovInner = l->getFovSmall();
    b.fovOuter = l->getFov();
    ++i;
  }
  for (auto l : getTexturedLights()) {
    auto &b = mRTSpotLightBufferHost.at(i);
    b.viewMat = glm::affineInverse(l->getTransform().worldModelMatrix);
    b.projMat = l->getShadowProjectionMatrix();
    b.rgb = l->getColor();
    b.position = l->getPosition();
    b.fovInner = l->getFovSmall();
    b.fovOuter = l->getFov();
    ++i;
  }
  mRTSpotLightBuffer->upload(mRTSpotLightBufferHost);

  i = 0;
  for (auto l : getParallelogramLights()) {
    auto &b = mRTParallelogramLightBufferHost.at(i);
    b.color = l->getColor();
    b.position = l->getOrigin();
    b.edge0 = l->getEdgeX();
    b.edge1 = l->getEdgeY();
    ++i;
  }
  mRTParallelogramLightBuffer->upload(mRTParallelogramLightBufferHost);
}

void Scene::buildRTResources(StructDataLayout const &materialBufferLayout,
                             StructDataLayout const &textureIndexBufferLayout,
                             StructDataLayout const &geometryInstanceBufferLayout,
                             bool externalInstanceBuffer) {

  std::lock_guard lock(mRTResourcesLock);

  // make sure scene is not in use
  if (mAccessFences.size()) {
    if (mContext->getDevice().waitForFences(mAccessFences, VK_TRUE, UINT64_MAX) !=
        vk::Result::eSuccess) {
      throw std::runtime_error("failed to wait for scene access fence");
    }
  }

  forceRemove();

  if (mRTResourcesVersion != mVersion) {
    ensureBLAS();
    buildTLAS(externalInstanceBuffer);
    createRTStorageBuffers(materialBufferLayout, textureIndexBufferLayout,
                           geometryInstanceBufferLayout);
    mRTResourcesVersion = mVersion;
  } else if (externalInstanceBuffer && !mTLAS->isInstanceBufferExternal()) {
    mTLAS->enableExternalInstanceBuffer();
  }
}

void Scene::updateRTResources(bool useDeviceTransforms) {
  updateRTTransforms(useDeviceTransforms);
  uploadRTLights();
}

void Scene::updateRTTransforms(bool useDeviceTransforms) {
  std::lock_guard lock(mRTResourcesLock);
  if (mRTResourcesVersion != mVersion) {
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
