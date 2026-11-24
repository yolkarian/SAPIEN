#include "sapien/sapien_renderer/sapien_renderer_default.h"
#include <filesystem>
#include <svulkan2/core/context.h>
#include <svulkan2/renderer/vr.h>

namespace fs = std::filesystem;

namespace sapien {
namespace sapien_renderer {

void SapienRendererDefault::setGlobalConfig(int32_t maxNumMaterials, uint32_t maxNumTextures,
                                            uint32_t defaultMipMaps, bool doNotLoadTexture) {
  auto &d = Get();

  d.defaultMipMaps = defaultMipMaps;
  d.doNotLoadTexture = doNotLoadTexture;
  d.maxNumMaterials = maxNumMaterials;
  d.maxNumTextures = maxNumTextures;
}

uint32_t SapienRendererDefault::getDefaultMipMaps() { return Get().defaultMipMaps; }

bool SapienRendererDefault::getDoNotLoadTexture() { return Get().doNotLoadTexture; }
uint32_t SapienRendererDefault::getMaxNumMaterials() { return Get().maxNumMaterials; }
uint32_t SapienRendererDefault::getMaxNumTextures() { return Get().maxNumTextures; }

void SapienRendererDefault::setDefaultRenderConfig(std::string const &target,
                                                   RenderConfig const &config) {
  auto &stored = Get().getOrCreateDefault(target);
  stored = config;
}

RenderConfig SapienRendererDefault::getDefaultRenderConfig(std::string const &target) {
  return Get().getOrCreateDefault(target);
}

RenderConfig &SapienRendererDefault::getOrCreateDefault(std::string const &target) {
  if (target == "camera") {
    if (!cameraRenderConfig) {
      cameraRenderConfig.emplace("default");
    }
    return *cameraRenderConfig;
  }
  if (target == "viewer") {
    if (!viewerRenderConfig) {
      viewerRenderConfig.emplace("default");
    }
    return *viewerRenderConfig;
  }
  throw std::runtime_error("render config target must be \"camera\" or \"viewer\", got \"" +
                           target + "\"");
}

SapienRendererDefault &SapienRendererDefault::Get() {
  static SapienRendererDefault gSapienRendererDefault;
  return gSapienRendererDefault;
}

void SapienRendererDefault::internalSetShaderSearchPath(std::string const &dir) {
  Get().shaderSearchPath = dir;
  if (!fs::is_directory(fs::path(dir))) {
    throw std::runtime_error("failed to set shader search path: invalid directory");
  }
}

std::string SapienRendererDefault::internalGetShaderSearchPath() {
  return Get().shaderSearchPath;
}

void SapienRendererDefault::setImguiIniFilename(std::string const &filename) {
  svulkan2::renderer::GuiWindow::setImguiIniFileLocation(filename);
}

std::string SapienRendererDefault::getImguiIniFilename() {
  return svulkan2::renderer::GuiWindow::getImguiIniFileLocation();
}

void SapienRendererDefault::setVRActionManifestFilename(std::string const &filename) {
  svulkan2::renderer::VRDisplay::setActionManifestPath(filename);
}
std::string SapienRendererDefault::getVRActionManifestFilename() {
  return svulkan2::renderer::VRDisplay::getActionManifestPath();
}

void SapienRendererDefault::enableVR() { Get().vrEnabled = true; }
bool SapienRendererDefault::getVREnabled() { return Get().vrEnabled; }

} // namespace sapien_renderer
} // namespace sapien
