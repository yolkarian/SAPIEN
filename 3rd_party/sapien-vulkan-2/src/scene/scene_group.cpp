#include "svulkan2/scene/scene_group.h"
#include "../common/logger.h"
#include "svulkan2/core/context.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>

namespace svulkan2 {
namespace scene {

namespace detail {
void registerSceneDependency(Scene *source, Scene *dependent);
void unregisterSceneDependency(Scene *source, Scene *dependent);
} // namespace detail

namespace {
std::shared_ptr<Scene> makeDependencyScene(std::shared_ptr<Scene> scene, Scene *dependent) {
  auto source = scene.get();
  auto registered = std::shared_ptr<Scene>(
      source, [scene = std::move(scene), dependent](Scene *source) {
        detail::unregisterSceneDependency(source, dependent);
      });
  detail::registerSceneDependency(source, dependent);
  return registered;
}
} // namespace

SceneGroup::SceneGroup(std::vector<std::shared_ptr<Scene>> const &scenes,
                       std::vector<Transform> const &transforms)
    : mTransforms(transforms) {
  if (scenes.size() != transforms.size()) {
    throw std::runtime_error(
        "failed to create scene group: scenes and transforms should have the same length");
  }

  mScenes.reserve(scenes.size());
  for (auto const &scene : scenes) {
    mScenes.push_back(makeDependencyScene(scene, this));
  }
}

std::vector<Object *> SceneGroup::getObjects() {
  std::vector<Object *> allObjs;
  for (auto s : mScenes) {
    auto objs = s->getObjects();
    allObjs.insert(allObjs.end(), objs.begin(), objs.end());
  }

  auto objs = Scene::getObjects();
  allObjs.insert(allObjs.end(), objs.begin(), objs.end());

  return allObjs;
}

std::vector<LineObject *> SceneGroup::getLineObjects() {
  std::vector<LineObject *> allObjs;
  for (auto s : mScenes) {
    auto objs = s->getLineObjects();
    allObjs.insert(allObjs.end(), objs.begin(), objs.end());
  }

  auto objs = Scene::getLineObjects();
  allObjs.insert(allObjs.end(), objs.begin(), objs.end());

  return allObjs;
}

std::vector<PointObject *> SceneGroup::getPointObjects() {
  std::vector<PointObject *> allObjs;
  for (auto s : mScenes) {
    auto objs = s->getPointObjects();
    allObjs.insert(allObjs.end(), objs.begin(), objs.end());
  }

  auto objs = Scene::getPointObjects();
  allObjs.insert(allObjs.end(), objs.begin(), objs.end());

  return allObjs;
}

std::vector<Camera *> SceneGroup::getCameras() {
  std::vector<Camera *> allCameras;
  for (auto s : mScenes) {
    auto cameras = s->getCameras();
    allCameras.insert(allCameras.end(), cameras.begin(), cameras.end());
  }

  auto cameras = Scene::getCameras();
  allCameras.insert(allCameras.end(), cameras.begin(), cameras.end());

  return allCameras;
}

std::vector<PointLight *> SceneGroup::getPointLights() {
  std::vector<PointLight *> allLights;
  for (auto s : mScenes) {
    auto lights = s->getPointLights();
    allLights.insert(allLights.end(), lights.begin(), lights.end());
  }

  auto lights = Scene::getPointLights();
  allLights.insert(allLights.end(), lights.begin(), lights.end());

  return allLights;
}

std::vector<DirectionalLight *> SceneGroup::getDirectionalLights() {
  std::vector<DirectionalLight *> allLights;
  for (auto s : mScenes) {
    auto lights = s->getDirectionalLights();
    allLights.insert(allLights.end(), lights.begin(), lights.end());
  }

  auto lights = Scene::getDirectionalLights();
  allLights.insert(allLights.end(), lights.begin(), lights.end());

  return allLights;
}

std::vector<SpotLight *> SceneGroup::getSpotLights() {
  std::vector<SpotLight *> allLights;
  for (auto s : mScenes) {
    auto lights = s->getSpotLights();
    allLights.insert(allLights.end(), lights.begin(), lights.end());
  }

  auto lights = Scene::getSpotLights();
  allLights.insert(allLights.end(), lights.begin(), lights.end());

  return allLights;
}

std::vector<TexturedLight *> SceneGroup::getTexturedLights() {
  std::vector<TexturedLight *> allLights;
  for (auto s : mScenes) {
    auto lights = s->getTexturedLights();
    allLights.insert(allLights.end(), lights.begin(), lights.end());
  }

  auto lights = Scene::getTexturedLights();
  allLights.insert(allLights.end(), lights.begin(), lights.end());

  return allLights;
}

std::vector<ParallelogramLight *> SceneGroup::getParallelogramLights() {
  std::vector<ParallelogramLight *> allLights;
  for (auto s : mScenes) {
    auto lights = s->getParallelogramLights();
    allLights.insert(allLights.end(), lights.begin(), lights.end());
  }

  auto lights = Scene::getParallelogramLights();
  allLights.insert(allLights.end(), lights.begin(), lights.end());

  return allLights;
}

void SceneGroup::uploadObjectTransforms() {
  if (mTransformBufferRenderVersion == mRenderVersion) {
    return;
  }

  prepareObjectTransformBuffer();

  size_t totalSize{0};
  size_t step = getGpuTransformBufferSize();
  uint8_t *buffer = reinterpret_cast<uint8_t *>(mTransformBufferCpu->map());

  auto writeMatrix = [&](glm::mat4 const &mat) {
    std::memcpy(buffer + totalSize, &mat, sizeof(glm::mat4));
    totalSize += step;
  };

  // collect data
  for (uint32_t i = 0; i < mScenes.size(); ++i) {
    auto sceneMat = mTransforms[i].matrix();
    auto objs = mScenes[i]->getObjects();
    for (auto obj : objs) {
      writeMatrix(sceneMat * obj->getTransform().worldModelMatrix);
    }
  }
  auto objs = Scene::getObjects();
  for (auto obj : objs) {
    writeMatrix(obj->getTransform().worldModelMatrix);
  }

  for (uint32_t i = 0; i < mScenes.size(); ++i) {
    auto sceneMat = mTransforms[i].matrix();
    auto lineObjects = mScenes[i]->getLineObjects();
    for (auto obj : lineObjects) {
      writeMatrix(sceneMat * obj->getTransform().worldModelMatrix);
    }
  }
  auto lineObjects = Scene::getLineObjects();
  for (auto obj : lineObjects) {
    writeMatrix(obj->getTransform().worldModelMatrix);
  }

  for (uint32_t i = 0; i < mScenes.size(); ++i) {
    auto sceneMat = mTransforms[i].matrix();
    auto pointObjects = mScenes[i]->getPointObjects();
    for (auto obj : pointObjects) {
      writeMatrix(sceneMat * obj->getTransform().worldModelMatrix);
    }
  }
  auto pointObjects = Scene::getPointObjects();
  for (auto obj : pointObjects) {
    writeMatrix(obj->getTransform().worldModelMatrix);
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

  vk::BufferCopy region(0, 0, totalSize);
  mTransformUpdateCommandBuffer->copyBuffer(mTransformBufferCpu->getVulkanBuffer(),
                                            mTransformBuffer->getVulkanBuffer(), region);

  vk::MemoryBarrier barrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead);
  // vk::BufferMemoryBarrier barrier(
  //     vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
  //     VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, mTransformBuffer->getVulkanBuffer(), 0,
  //     VK_WHOLE_SIZE);
  mTransformUpdateCommandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                                 vk::PipelineStageFlagBits::eVertexShader |
                                                     vk::PipelineStageFlagBits::eFragmentShader,
                                                 {}, barrier, {}, {});

  mTransformUpdateCommandBuffer->end();

  core::Context::Get()->getQueue().submit(mTransformUpdateCommandBuffer.get(), {});

  mTransformBufferRenderVersion = mRenderVersion;
}

} // namespace scene
} // namespace svulkan2
