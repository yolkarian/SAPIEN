#include "render_scene_resolver.h"
#include "sapien/sapien_renderer/sapien_renderer_system.h"
#include <stdexcept>
#include <svulkan2/scene/scene_group.h>
#include <unordered_set>

namespace sapien::sapien_renderer {

std::vector<std::shared_ptr<SapienRendererSystem>> RenderSceneResolver::resolve(
    std::shared_ptr<svulkan2::core::Context> const &context,
    std::vector<std::shared_ptr<SapienRendererSystem>> const &baseSystems,
    std::vector<std::shared_ptr<SapienRendererSystem>> const &contextSystems) {
  std::vector<std::shared_ptr<SapienRendererSystem>> result;
  std::unordered_set<SapienRendererSystem *> added;

  // Every selected scene must render on `context`: the resolved render scene (a SceneGroup
  // when there are several) is built for that one device.
  for (auto const &system : baseSystems) {
    if (!system) {
      continue;
    }
    if (system->getContext() != context) {
      throw std::runtime_error(
          "failed to select render scenes: a scene renders on device \"" +
          system->getDevice()->getAlias() +
          "\", which differs from the device of the render scene it would join");
    }
  }

  for (auto const &system : baseSystems) {
    if (system && added.insert(system.get()).second) {
      result.push_back(system);
    }
  }
  for (auto const &system : contextSystems) {
    // Shared scenes of another device cannot join this render scene.
    if (system && system->getContext() == context && system->isBatchedRenderShared() &&
        added.insert(system.get()).second) {
      result.push_back(system);
    }
  }
  return result;
}

std::shared_ptr<svulkan2::scene::Scene>
RenderSceneResolver::build(std::vector<std::shared_ptr<SapienRendererSystem>> const &systems) {
  if (systems.empty()) {
    return nullptr;
  }
  if (systems.size() == 1) {
    return systems.front()->getScene();
  }

  std::vector<std::shared_ptr<svulkan2::scene::Scene>> scenes;
  scenes.reserve(systems.size());
  for (auto const &system : systems) {
    scenes.push_back(system->getScene());
  }
  std::vector<svulkan2::scene::Transform> identityTransforms(scenes.size());
  auto group = std::make_shared<svulkan2::scene::SceneGroup>(scenes, identityTransforms);

  // The first base scene owns aggregate ambient light. Prefer its environment map, then use the
  // first shared map so shared worlds can still supply image-based lighting.
  glm::vec4 ambient = systems.front()->getScene()->getAmbientLight();
  auto environmentMap = systems.front()->getScene()->getEnvironmentMap();
  if (!environmentMap) {
    for (auto const &system : systems) {
      environmentMap = system->getScene()->getEnvironmentMap();
      if (environmentMap) {
        break;
      }
    }
  }
  ambient.a = environmentMap ? 0.f : 1.f;
  group->setAmbientLight(ambient);
  group->setEnvironmentMap(environmentMap);
  return group;
}

} // namespace sapien::sapien_renderer
