#pragma once
#include <map>
#include <string>
#include <svulkan2/common/config.h>
#include <svulkan2/common/vk.h>
#include <svulkan2/renderer/renderer_base.h>
#include <svulkan2/renderer/rt_renderer.h>
#include <variant>

namespace svulkan2::renderer {
class BatchRTRenderer;
}

namespace sapien {
namespace sapien_renderer {

/** Per-consumer renderer configuration, replacing the process-global renderer setters.
 *
 *  A config names a shader pack (a built-in pack under the shader search path or a directory),
 *  carries shader properties applied as renderer custom properties, shader preprocessor
 *  definitions, a denoiser, MSAA, and render target formats.
 *
 *  Consumers (camera, window, VR) copy the config they are given: later edits to the Python
 *  object do not affect renderers that already exist. Property names and types are validated
 *  when a renderer is created from the config, so an unknown key raises instead of being
 *  silently dropped. */
class RenderConfig {
public:
  /** `shaderPack` is a pack name resolved under the shader search path, or a directory path.
   *  Resolved and validated immediately; throws std::runtime_error when it is not a directory
   *  holding `gbuffer.vert` (rasterization) or `camera.rgen` (ray tracing). */
  explicit RenderConfig(std::string const &shaderPack = "default");

  /** canonical shader pack directory */
  std::string getShaderPack() const { return mShaderPack; }
  /** resolve and validate a new pack; keeps the current settings */
  void setShaderPack(std::string const &shaderPack);

  /** what the pack renders: "raster" (entry point gbuffer.vert), "ray_tracing" (entry point
   *  camera.rgen, one renderer per camera) or "batch_ray_tracing" (entry point batch.comp, one
   *  renderer per camera group; such a pack has no per-camera renderer) */
  enum class Kind { eRaster, eRayTracing, eBatchRayTracing };
  Kind getKind() const { return mKind; }
  /** human-readable kind, the value of the Python `kind` property */
  std::string getKindName() const;

  /** true for both ray tracing kinds: a batch pack also traces rays */
  bool isRayTracing() const {
    return mKind == Kind::eRayTracing || mKind == Kind::eBatchRayTracing;
  }
  /** true when the pack renders only through a RenderCameraGroup */
  bool isBatchRayTracing() const { return mKind == Kind::eBatchRayTracing; }

  /** shader properties become renderer custom properties: ray tracing push constants or
   *  rasterization specialization constants */
  void setProperty(std::string const &name, int value);
  void setProperty(std::string const &name, float value);
  bool hasProperty(std::string const &name) const { return mProperties.contains(name); }
  std::variant<int, float> getProperty(std::string const &name) const;
  void removeProperty(std::string const &name) { mProperties.erase(name); }
  std::map<std::string, std::variant<int, float>> getProperties() const { return mProperties; }

  std::map<std::string, std::string> getDefines() const { return mDefines; }
  void setDefines(std::map<std::string, std::string> defines);

  svulkan2::renderer::RTRenderer::DenoiserType getDenoiser() const { return mDenoiser; }
  void setDenoiser(svulkan2::renderer::RTRenderer::DenoiserType denoiser);

  /** 1, 2, 4 or 8 */
  int getMSAA() const { return mMSAA; }
  void setMSAA(int msaa);

  /** render target name -> image format, in the same string form as the Python format caster */
  std::map<std::string, vk::Format> getPictureFormats() const { return mPictureFormats; }
  void setPictureFormats(std::map<std::string, vk::Format> formats);

  bool operator==(RenderConfig const &other) const = default;

  /** human-readable summary, also the Python `__repr__` */
  std::string toString() const;

private:
  std::string mShaderPack;
  Kind mKind{Kind::eRaster};
  std::map<std::string, std::variant<int, float>> mProperties;
  std::map<std::string, std::string> mDefines;
  svulkan2::renderer::RTRenderer::DenoiserType mDenoiser{
      svulkan2::renderer::RTRenderer::DenoiserType::eNONE};
  int mMSAA{1};
  std::map<std::string, vk::Format> mPictureFormats;
};

/** svulkan2 renderer options of `config`: shader pack, defines, MSAA, picture formats */
std::shared_ptr<svulkan2::RendererConfig> buildRendererConfig(RenderConfig const &config);

/** Create the renderer of `config` on `context` and apply every setting of the config.
 *  Raises std::runtime_error naming the key and the pack when a property is not declared by the
 *  pack or has another type, when a picture format is set for a ray tracing pack, or when a
 *  denoiser is set for a rasterization pack. */
std::unique_ptr<svulkan2::renderer::RendererBase>
createRenderer(std::shared_ptr<svulkan2::core::Context> context, RenderConfig const &config);

/** Validate and apply the properties and denoiser of `config` on an existing renderer. */
void applyRenderConfig(RenderConfig const &config, svulkan2::renderer::RendererBase &renderer);

/** Validate and apply the properties of a batched ray tracing `config` on its camera group's
 *  renderer. Raises like createRenderer; a denoiser cannot be set for a batched pack. */
void applyRenderConfig(RenderConfig const &config, svulkan2::renderer::BatchRTRenderer &renderer);

} // namespace sapien_renderer
} // namespace sapien
