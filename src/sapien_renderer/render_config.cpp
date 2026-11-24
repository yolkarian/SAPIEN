#include "sapien/sapien_renderer/render_config.h"
#include "sapien/sapien_renderer/sapien_renderer_default.h"
#include <svulkan2/renderer/batch_rt_renderer.h>
#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

namespace sapien {
namespace sapien_renderer {

namespace {

std::string denoiserToString(svulkan2::renderer::RTRenderer::DenoiserType type) {
  switch (type) {
  case svulkan2::renderer::RTRenderer::DenoiserType::eOPTIX:
    return "optix";
  case svulkan2::renderer::RTRenderer::DenoiserType::eOIDN:
    return "oidn";
  case svulkan2::renderer::RTRenderer::DenoiserType::eNONE:
  default:
    return "none";
  }
}

/** a pack name is looked up under the shader search path; an existing directory is used as is */
fs::path resolveShaderPack(std::string const &shaderPack) {
  fs::path path(shaderPack);
  if (!fs::is_directory(path)) {
    auto const searchPath = SapienRendererDefault::internalGetShaderSearchPath();
    if (searchPath.empty()) {
      throw std::runtime_error("invalid shader pack \"" + shaderPack +
                               "\": it is not a directory and no shader search path is set");
    }
    path = fs::path(searchPath) / shaderPack;
  }
  if (!fs::is_directory(path)) {
    throw std::runtime_error("invalid shader pack \"" + shaderPack + "\": " + path.string() +
                             " is not a directory");
  }
  path = fs::canonical(path);
  if (!fs::exists(path / "gbuffer.vert") && !fs::exists(path / "camera.rgen") &&
      !fs::exists(path / "batch.comp")) {
    throw std::runtime_error("invalid shader pack \"" + shaderPack + "\": " + path.string() +
                             " contains neither gbuffer.vert (rasterization), camera.rgen "
                             "(ray tracing) nor batch.comp (batched ray tracing)");
  }
  return path;
}

RenderConfig::Kind packKind(fs::path const &pack) {
  // a batch pack renders only through a camera group and takes precedence over camera.rgen, so a
  // pack cannot silently fall back to one renderer per camera
  if (fs::exists(pack / "batch.comp")) {
    return RenderConfig::Kind::eBatchRayTracing;
  }
  // the order of RendererBase::Create: a pack with both entry points renders with rasterization
  if (fs::exists(pack / "gbuffer.vert")) {
    return RenderConfig::Kind::eRaster;
  }
  return RenderConfig::Kind::eRayTracing;
}

vk::SampleCountFlagBits msaaSampleCount(int msaa) {
  switch (msaa) {
  case 1:
    return vk::SampleCountFlagBits::e1;
  case 2:
    return vk::SampleCountFlagBits::e2;
  case 4:
    return vk::SampleCountFlagBits::e4;
  case 8:
    return vk::SampleCountFlagBits::e8;
  default:
    throw std::runtime_error("MSAA count must be one of [1, 2, 4, 8]");
  }
}

bool propertyTypeMatches(svulkan2::renderer::CustomPropertyType type, bool isInt) {
  using T = svulkan2::renderer::CustomPropertyType;
  return isInt ? type == T::eInt : type == T::eFloat;
}

/** validate one property of `config` against the pack of `renderer`, then apply it */
template <typename Renderer>
void applyProperty(RenderConfig const &config, Renderer &renderer, std::string const &name,
                   std::variant<int, float> const &value) {
  using T = svulkan2::renderer::CustomPropertyType;
  auto type = renderer.getCustomPropertyType(name);
  auto pack = config.getShaderPack();
  if (type == T::eNone) {
    throw std::runtime_error("render config property \"" + name +
                             "\" is not declared by shader pack \"" + pack + "\"");
  }
  bool isInt = std::holds_alternative<int>(value);
  if (!propertyTypeMatches(type, isInt)) {
    std::string declared = type == T::eInt ? "int" : (type == T::eFloat ? "float" : "a vector type");
    throw std::runtime_error("render config property \"" + name + "\" is declared as " +
                             declared + " by shader pack \"" + pack + "\" but set to a " +
                             (isInt ? "int" : "float"));
  }
  if (isInt) {
    renderer.setCustomProperty(name, std::get<int>(value));
  } else {
    renderer.setCustomProperty(name, std::get<float>(value));
  }
}

/** apply `value` only when the pack declares it with the expected type */
template <typename T>
void applyDefaultIfDeclared(svulkan2::renderer::RendererBase &renderer, std::string const &name,
                            T value) {
  using P = svulkan2::renderer::CustomPropertyType;
  auto type = renderer.getCustomPropertyType(name);
  if (propertyTypeMatches(type, std::is_same_v<T, int>) && type != P::eNone) {
    renderer.setCustomProperty(name, value);
  }
}

} // namespace

RenderConfig::RenderConfig(std::string const &shaderPack) { setShaderPack(shaderPack); }

void RenderConfig::setShaderPack(std::string const &shaderPack) {
  auto pack = resolveShaderPack(shaderPack);
  mShaderPack = pack.string();
  mKind = packKind(pack);
}

void RenderConfig::setProperty(std::string const &name, int value) { mProperties[name] = value; }

void RenderConfig::setProperty(std::string const &name, float value) { mProperties[name] = value; }

std::variant<int, float> RenderConfig::getProperty(std::string const &name) const {
  auto it = mProperties.find(name);
  if (it == mProperties.end()) {
    throw std::runtime_error("render config has no property \"" + name + "\"");
  }
  return it->second;
}

void RenderConfig::setDefines(std::map<std::string, std::string> defines) {
  mDefines = std::move(defines);
}

void RenderConfig::setDenoiser(svulkan2::renderer::RTRenderer::DenoiserType denoiser) {
  mDenoiser = denoiser;
}

void RenderConfig::setMSAA(int msaa) {
  msaaSampleCount(msaa);
  mMSAA = msaa;
}

void RenderConfig::setPictureFormats(std::map<std::string, vk::Format> formats) {
  mPictureFormats = std::move(formats);
}

std::string RenderConfig::getKindName() const {
  switch (mKind) {
  case Kind::eRaster:
    return "raster";
  case Kind::eRayTracing:
    return "ray_tracing";
  case Kind::eBatchRayTracing:
    return "batch_ray_tracing";
  }
  throw std::runtime_error("corrupted render config kind");
}

std::string RenderConfig::toString() const {
  std::ostringstream stream;
  stream << "RenderConfig(shader_pack=\"" << mShaderPack << "\", kind=\""
         << getKindName() << "\", properties={";
  bool first = true;
  for (auto const &[name, value] : mProperties) {
    stream << (first ? "" : ", ") << "\"" << name << "\": ";
    if (std::holds_alternative<int>(value)) {
      stream << std::get<int>(value);
    } else {
      stream << std::get<float>(value);
    }
    first = false;
  }
  stream << "}, defines={";
  first = true;
  for (auto const &[name, value] : mDefines) {
    stream << (first ? "" : ", ") << "\"" << name << "\": \"" << value << "\"";
    first = false;
  }
  stream << "}, denoiser=\"" << denoiserToString(mDenoiser) << "\", msaa=" << mMSAA
         << ", picture_formats={";
  first = true;
  for (auto const &[name, format] : mPictureFormats) {
    stream << (first ? "" : ", ") << "\"" << name << "\": \""
           << vk::to_string(format) << "\"";
    first = false;
  }
  stream << "})";
  return stream.str();
}

std::shared_ptr<svulkan2::RendererConfig> buildRendererConfig(RenderConfig const &config) {
  auto result = std::make_shared<svulkan2::RendererConfig>();
  result->shaderDir = config.getShaderPack();
  result->shaderDefines = config.getDefines();
  result->msaa = msaaSampleCount(config.getMSAA());
  for (auto const &[name, format] : config.getPictureFormats()) {
    result->textureFormat[name] = format;
  }
  return result;
}

void applyRenderConfig(RenderConfig const &config, svulkan2::renderer::RendererBase &renderer) {
  auto pack = config.getShaderPack();
  if (config.isRayTracing()) {
    if (!config.getPictureFormats().empty()) {
      throw std::runtime_error("render config property \"picture_formats\" is not supported by "
                               "ray tracing shader pack \"" +
                               pack + "\"");
    }
    // SAPIEN defaults for keys the config leaves out; the renderer's own defaults differ
    applyDefaultIfDeclared(renderer, "spp", 32);
    applyDefaultIfDeclared(renderer, "maxDepth", 8);
    applyDefaultIfDeclared(renderer, "russianRoulette", 0);
    applyDefaultIfDeclared(renderer, "aperture", 0.f);
    applyDefaultIfDeclared(renderer, "focusPlane", 1.f);
  } else if (config.getDenoiser() != svulkan2::renderer::RTRenderer::DenoiserType::eNONE) {
    throw std::runtime_error("render config property \"denoiser\" can only be set with a ray "
                             "tracing shader pack, but the config uses rasterization pack \"" +
                             pack + "\"");
  }

  for (auto const &[name, value] : config.getProperties()) {
    applyProperty(config, renderer, name, value);
  }

  if (config.isRayTracing()) {
    auto rtRenderer = dynamic_cast<svulkan2::renderer::RTRenderer *>(&renderer);
    if (rtRenderer) {
      rtRenderer->enableDenoiser(config.getDenoiser(), "HdrColor", "Albedo", "Normal");
    }
  }
}

std::unique_ptr<svulkan2::renderer::RendererBase>
createRenderer(std::shared_ptr<svulkan2::core::Context> context, RenderConfig const &config) {
  if (config.isBatchRayTracing()) {
    throw std::runtime_error("batched ray tracing shader pack \"" + config.getShaderPack() +
                             "\" has no per-camera renderer: add its cameras to one "
                             "RenderCameraGroup, which builds a single group renderer");
  }
  auto renderer = svulkan2::renderer::RendererBase::Create(context, buildRendererConfig(config));
  applyRenderConfig(config, *renderer);
  return renderer;
}

void applyRenderConfig(RenderConfig const &config,
                       svulkan2::renderer::BatchRTRenderer &renderer) {
  if (config.getDenoiser() != svulkan2::renderer::RTRenderer::DenoiserType::eNONE) {
    throw std::runtime_error("render config property \"denoiser\" is not supported by batched "
                             "ray tracing shader pack \"" +
                             config.getShaderPack() + "\"");
  }
  for (auto const &[name, value] : config.getProperties()) {
    applyProperty(config, renderer, name, value);
  }
}

} // namespace sapien_renderer
} // namespace sapien
