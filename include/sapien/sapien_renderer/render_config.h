#pragma once
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <svulkan2/common/config.h>
#include <svulkan2/common/vk.h>
#include <svulkan2/renderer/renderer_base.h>
#include <svulkan2/renderer/rt_renderer.h>
#include <variant>
#include <vector>

namespace svulkan2::renderer {
class BatchRTRenderer;
}

namespace sapien {
namespace sapien_renderer {

namespace detail {
class PackManifest;
}

/** One parameter value: a bool, int or float uniform, or a string enum variant. */
using PackValue = std::variant<bool, int, float, std::string>;

/** One parameter a pack.json declares, as reported to Python. */
struct RenderConfigParameter {
  std::string name;
  /** "uniform" (applied to a renderer) or "variant" (selects source through a define) */
  std::string kind;
  /** "bool", "int", "float" or "enum" */
  std::string type;
  /** the value the parameter takes when the config does not assign it */
  PackValue defaultValue{false};
  /** inclusive bounds; only int and float parameters declare them */
  std::optional<double> minimum;
  std::optional<double> maximum;
  /** enum parameters: the allowed values */
  std::vector<std::string> values;
  std::string description;
};

/** Per-consumer renderer configuration, replacing the process-global renderer setters.
 *
 *  A config names a shader pack (a built-in pack under the shader search path or a directory),
 *  assigns values to the parameters the pack declares in its `pack.json`, and carries a
 *  denoiser, MSAA, and render target formats. Uniform parameters become renderer properties
 *  (ray tracing push constants or rasterization specialization constants); variant parameters
 *  become shader preprocessor definitions.
 *
 *  Every pack must have a `pack.json`; a pack without one, or with a malformed one, raises when
 *  the config is created or its pack is changed, and every assignment is checked against it
 *  (name, type and range) as it is made.
 *
 *  Consumers (camera, window, VR) copy the config they are given: later edits do not affect
 *  renderers that already exist. */
class RenderConfig {
public:
  /** `shaderPack` is a pack name resolved under the shader search path, or a directory path.
   *  `preset`, when given, must be a preset the pack declares; it expands into assignments.
   *  Resolved and validated immediately; throws std::runtime_error when the pack is not a
   *  directory holding `pack.json` next to `gbuffer.vert` (rasterization), `camera.rgen` (ray
   *  tracing) or `batch.comp` (batched ray tracing), or when its pack.json is malformed. */
  explicit RenderConfig(std::string const &shaderPack = "default",
                        std::optional<std::string> const &preset = std::nullopt);

  /** canonical shader pack directory */
  std::string getShaderPack() const { return mShaderPack; }
  /** resolve a new pack and re-validate every explicit assignment against its pack.json;
   *  raises without changing anything when one is invalid */
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

  /** true when the pack's pack.json declares `name` */
  bool declaresParameter(std::string const &name) const;
  /** true when `name` is a variant parameter, i.e. selects shader source and so cannot change
   *  after a renderer exists; raises when the pack does not declare it */
  bool isVariantParameter(std::string const &name) const;

  /** assign `name`, checked against the pack's pack.json: an unknown name, a wrong type or an
   *  out-of-range value raises std::runtime_error naming the key and the pack. An int is
   *  accepted for a float parameter and stored as a float. */
  void setParameter(std::string const &name, PackValue value);
  /** check `value` for `name` without assigning it, converting an int to a float for a float
   *  parameter; raises like setParameter */
  PackValue checkParameter(std::string const &name, PackValue value) const;
  /** effective value of `name`: the assignment when there is one, else the pack's default.
   *  Raises when the pack does not declare `name`. */
  PackValue getParameter(std::string const &name) const;
  /** revert `name` to the pack's default; raises when the pack does not declare it */
  void removeParameter(std::string const &name);
  /** true when `name` is explicitly assigned (the Python `name in cfg`) */
  bool hasParameter(std::string const &name) const { return mParameters.contains(name); }
  /** copy of the explicit assignments, in name order */
  std::map<std::string, PackValue> getProperties() const { return mParameters; }
  /** the pack's parameters in file order, as reported to Python */
  std::vector<RenderConfigParameter> describeParameters() const;
  /** names of the presets the pack declares, in file order */
  std::vector<std::string> getPresets() const;

  /** preprocessor definitions of the effective variant values: a bool defines its name when
   *  true, an int or float defines it to the value, an enum defines its mapped name (an enum
   *  value mapping to "" defines nothing). Feed to svulkan2::RendererConfig::shaderDefines. */
  std::map<std::string, std::string> getShaderDefinitions() const;

  svulkan2::renderer::RTRenderer::DenoiserType getDenoiser() const { return mDenoiser; }
  void setDenoiser(svulkan2::renderer::RTRenderer::DenoiserType denoiser);

  /** 1, 2, 4 or 8 */
  int getMSAA() const { return mMSAA; }
  void setMSAA(int msaa);

  /** render target name -> image format, in the same string form as the Python format caster */
  std::map<std::string, vk::Format> getPictureFormats() const { return mPictureFormats; }
  void setPictureFormats(std::map<std::string, vk::Format> formats);

  /** same pack and kind, same assignments, denoiser, MSAA and picture formats */
  bool operator==(RenderConfig const &other) const;

  /** human-readable summary, also the Python `__repr__` */
  std::string toString() const;

private:
  /** resolve `shaderPack` and load its manifest without modifying the config */
  std::pair<std::string, Kind> resolvePack(std::string const &shaderPack) const;
  detail::PackManifest const &manifest() const;

  std::string mShaderPack;
  Kind mKind{Kind::eRaster};
  std::shared_ptr<detail::PackManifest const> mManifest;
  std::map<std::string, PackValue> mParameters;
  svulkan2::renderer::RTRenderer::DenoiserType mDenoiser{
      svulkan2::renderer::RTRenderer::DenoiserType::eNONE};
  int mMSAA{1};
  std::map<std::string, vk::Format> mPictureFormats;
};

/** svulkan2 renderer options of `config`: shader pack, variant definitions, MSAA, picture
 *  formats */
std::shared_ptr<svulkan2::RendererConfig> buildRendererConfig(RenderConfig const &config);

/** Create the renderer of `config` on `context` and apply every uniform parameter with its
 *  effective value. Raises std::runtime_error naming the parameter and the pack when the shader
 *  does not declare a uniform the pack.json declares (or declares it with another type), when a
 *  picture format is set for a ray tracing pack, or when a denoiser is set for a rasterization
 *  pack. */
std::unique_ptr<svulkan2::renderer::RendererBase>
createRenderer(std::shared_ptr<svulkan2::core::Context> context, RenderConfig const &config);

/** Validate and apply the uniform parameters and the denoiser of `config` on an existing
 *  renderer. */
void applyRenderConfig(RenderConfig const &config, svulkan2::renderer::RendererBase &renderer);

/** Validate and apply the uniform parameters of a batched ray tracing `config` on its camera
 *  group's renderer. Raises like createRenderer; a denoiser cannot be set for a batched pack. */
void applyRenderConfig(RenderConfig const &config, svulkan2::renderer::BatchRTRenderer &renderer);

/** Drop every cached pack.json; configs created earlier keep theirs alive. Called by
 *  sapien.render.clear_cache(shaders=True). */
void clearPackManifestCache();

} // namespace sapien_renderer
} // namespace sapien
