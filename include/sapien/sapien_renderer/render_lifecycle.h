#pragma once
#include <cstdint>
#include <string>

namespace sapien {
namespace sapien_renderer {

struct RenderLiveResources {
  bool engineExists{false};
  uint64_t rendererSystems{0};
  /** Strong SapienRenderEngine owners excluding the temporary local snapshot. */
  uint64_t externalEngineOwners{0};

  bool canShutdown() const { return externalEngineOwners == 0; }
  std::string describe() const;
};

RenderLiveResources renderLiveResources();
bool renderCanShutdown();
/** Wait for Vulkan work and destroy the SAPIEN render engine/context. Idempotent;
 *  raises without mutating state while render systems/groups/materials/textures or
 *  viewers still own the engine. */
void renderShutdown();

} // namespace sapien_renderer
} // namespace sapien
