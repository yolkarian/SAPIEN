#pragma once

#include <memory>
#include <vector>

namespace svulkan2::core {
class Context;
} // namespace svulkan2::core

namespace svulkan2::scene {
class Scene;
}

namespace sapien::sapien_renderer {

class SapienRendererSystem;

/** Resolve the render systems whose scenes make up one output render scene of `context`. */
class RenderSceneResolver {
public:
  /** `baseSystems` are the explicitly selected scenes; `contextSystems` contribute the
   *  batched-render-shared scenes of the same device. Throws when a base scene lives on
   *  another device. */
  static std::vector<std::shared_ptr<SapienRendererSystem>>
  resolve(std::shared_ptr<svulkan2::core::Context> const &context,
          std::vector<std::shared_ptr<SapienRendererSystem>> const &baseSystems,
          std::vector<std::shared_ptr<SapienRendererSystem>> const &contextSystems);

  static std::shared_ptr<svulkan2::scene::Scene>
  build(std::vector<std::shared_ptr<SapienRendererSystem>> const &systems);
};

} // namespace sapien::sapien_renderer
