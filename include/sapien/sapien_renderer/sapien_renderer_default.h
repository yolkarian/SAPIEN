#pragma once
#include "render_config.h"
#include <memory>
#include <optional>
#include <string>
#include <svulkan2/common/vk.h>
#include <unordered_map>

namespace sapien {
namespace sapien_renderer {

class SapienRendererDefault {
public:
  static SapienRendererDefault &Get();

  static void setGlobalConfig(int32_t maxNumMaterials, uint32_t maxNumTextures,
                              uint32_t defaultMipMaps, bool doNotLoadTexture);

  static uint32_t getDefaultMipMaps();
  static bool getDoNotLoadTexture();
  static uint32_t getMaxNumMaterials();
  static uint32_t getMaxNumTextures();

  static void setImguiIniFilename(std::string const &filename);
  static std::string getImguiIniFilename();

  // vr
  static void setVRActionManifestFilename(std::string const &filename);
  static std::string getVRActionManifestFilename();
  static void enableVR();
  static bool getVREnabled();

  /** process-wide default render configs: "camera" for cameras and "viewer" for windows and VR.
   *  Consumers copy the default when they are created without an explicit config. */
  static void setDefaultRenderConfig(std::string const &target, RenderConfig const &config);
  static RenderConfig getDefaultRenderConfig(std::string const &target);

  static void internalSetShaderSearchPath(std::string const &dir);
  static std::string internalGetShaderSearchPath();

public:
  uint32_t defaultMipMaps = 1;
  bool doNotLoadTexture = false;
  uint32_t maxNumMaterials = 128;
  uint32_t maxNumTextures = 512;

  bool vrEnabled = false;

private:
  std::string shaderSearchPath;
  /** resolved on first use: a config resolves its shader pack, which needs the search path */
  std::optional<RenderConfig> cameraRenderConfig;
  std::optional<RenderConfig> viewerRenderConfig;

  RenderConfig &getOrCreateDefault(std::string const &target);
};

} // namespace sapien_renderer
} // namespace sapien
