#include "sapien/sapien_renderer/render_lifecycle.h"
#include "sapien/sapien_renderer/sapien_renderer_system.h"

#include <sstream>
#include <stdexcept>

namespace sapien {
namespace sapien_renderer {

std::string RenderLiveResources::describe() const {
  std::ostringstream ss;
  ss << "renderer_systems=" << rendererSystems
     << ", external_engine_owners=" << externalEngineOwners
     << ", engine=" << (engineExists ? "exists" : "none");
  return ss.str();
}

RenderLiveResources renderLiveResources() {
  RenderLiveResources out;
  auto engine = SapienRenderEngine::GetIfExists();
  if (!engine) {
    return out;
  }
  out.engineExists = true;
  out.rendererSystems = engine->getRenderSystems().size();
  // gRenderEngine is weak. This local shared_ptr is the only library-owned strong
  // reference outside the objects being diagnosed.
  out.externalEngineOwners = engine.use_count() > 0 ? engine.use_count() - 1 : 0;
  return out;
}

bool renderCanShutdown() { return renderLiveResources().canShutdown(); }

void renderShutdown() {
  RenderLiveResources snapshot = renderLiveResources();
  if (!snapshot.canShutdown()) {
    throw std::runtime_error(
        "failed to shut down SAPIEN renderer: caller-owned resources are still alive (" +
        snapshot.describe() +
        "). Close viewers, render camera/system groups and renderer systems, and drop "
        "render material/texture/shape references first.");
  }
  auto engine = SapienRenderEngine::GetIfExists();
  if (engine && !engine->isShutdown()) {
    engine->shutdown();
  }
}

} // namespace sapien_renderer
} // namespace sapien
