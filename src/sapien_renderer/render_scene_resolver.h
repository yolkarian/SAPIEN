#pragma once

#include <memory>
#include <vector>

namespace svulkan2::scene {
class Scene;
}

namespace sapien::sapien_renderer {

class SapienRendererSystem;

class RenderSceneResolver {
public:
  static std::vector<std::shared_ptr<SapienRendererSystem>>
  resolve(std::vector<std::shared_ptr<SapienRendererSystem>> const &baseSystems,
          std::vector<std::shared_ptr<SapienRendererSystem>> const &contextSystems);

  static std::shared_ptr<svulkan2::scene::Scene>
  build(std::vector<std::shared_ptr<SapienRendererSystem>> const &systems);
};

} // namespace sapien::sapien_renderer
