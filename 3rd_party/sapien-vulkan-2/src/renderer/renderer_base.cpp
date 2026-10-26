#include "svulkan2/renderer/renderer_base.h"
#include "svulkan2/renderer/renderer.h"
#include "svulkan2/renderer/rt_renderer.h"
#include "svulkan2/shader/rt.h"
#include <filesystem>
#include <utility>

namespace fs = std::filesystem;

namespace svulkan2 {
namespace renderer {
std::unique_ptr<RendererBase> RendererBase::Create(std::shared_ptr<core::Context> context,
                                                   std::shared_ptr<RendererConfig> config) {
  if (fs::exists((fs::path(config->shaderDir) / "gbuffer.vert"))) {
    return std::make_unique<Renderer>(std::move(context), config);
  }
  if (fs::exists((fs::path(config->shaderDir) / "camera.rgen"))) {
    return std::make_unique<RTRenderer>(std::move(context), config->shaderDir);
  }
  throw std::runtime_error("Shader directory must contain gbuffer.vert for "
                           "rasterization or camera.rgen for ray tracing");
}

} // namespace renderer
} // namespace svulkan2
