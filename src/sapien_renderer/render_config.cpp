#include "sapien/sapien_renderer/render_config.h"
#include "pack_manifest.hpp"
#include "sapien/sapien_renderer/sapien_renderer_default.h"
#include <charconv>
#include <filesystem>
#include <svulkan2/renderer/batch_rt_renderer.h>
#include <sstream>

namespace fs = std::filesystem;

namespace sapien {
namespace sapien_renderer {

namespace {

using svulkan2::renderer::CustomPropertyType;
using detail::PackManifest;
using detail::PackParameterKind;
using detail::PackParameterType;

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

/** shortest decimal text that reads back as the same value; a define may be pasted anywhere */
std::string numberText(float value) {
  char buffer[64];
  auto [end, error] =
      std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::general);
  if (error != std::errc()) {
    throw std::runtime_error("failed to format the shader definition value " +
                             std::to_string(value));
  }
  return std::string(buffer, end);
}

std::string valueText(PackValue const &value) {
  switch (value.index()) {
  case 0:
    return std::get<bool>(value) ? "1" : "0";
  case 1:
    return std::to_string(std::get<int>(value));
  case 2:
    return numberText(std::get<float>(value));
  default:
    return std::get<std::string>(value);
  }
}

/** check `value` for parameter `name` of `manifest`; an int becomes a float for a float
 *  parameter. Unknown names, wrong types and out-of-range values raise naming both. */
PackValue checkAgainst(PackManifest const &manifest, std::string const &pack,
                       std::string const &name, PackValue value) {
  int const index = manifest.findParameter(name);
  if (index < 0) {
    std::string available;
    for (auto const &parameter : manifest.getParameters()) {
      available += (available.empty() ? "" : ", ") + parameter.name;
    }
    throw std::runtime_error("shader pack \"" + pack + "\" declares no parameter \"" + name +
                             "\"" +
                             (available.empty() ? ": its pack.json declares no parameters"
                                                : ": its pack.json declares " + available));
  }
  try {
    manifest.check(static_cast<std::size_t>(index), value);
  } catch (std::runtime_error const &error) {
    throw std::runtime_error("shader pack \"" + pack + "\": " + error.what());
  }
  return value;
}

/** the property type a uniform parameter is applied with: bools and ints are int properties,
 *  floats are float properties */
CustomPropertyType uniformPropertyType(PackParameterType type) {
  return type == PackParameterType::eInt || type == PackParameterType::eBool
             ? CustomPropertyType::eInt
             : CustomPropertyType::eFloat;
}

std::string declaredTypeName(CustomPropertyType type) {
  switch (type) {
  case CustomPropertyType::eInt:
    return "int";
  case CustomPropertyType::eFloat:
    return "float";
  case CustomPropertyType::eVec3:
    return "vec3";
  case CustomPropertyType::eVec4:
    return "vec4";
  default:
    return "";
  }
}

/** apply every uniform parameter of `config` with its effective value; `renderer` declares the
 *  properties as specialization constants or push constant members */
template <typename Renderer>
void applyUniformParameters(RenderConfig const &config, Renderer &renderer) {
  auto const &manifest = *detail::getPackManifest(fs::path(config.getShaderPack()));
  std::string const manifestPath = (fs::path(config.getShaderPack()) / "pack.json").string();
  for (auto const &parameter : manifest.getParameters()) {
    if (parameter.kind != PackParameterKind::eUniform) {
      continue;
    }
    std::string const type = PackManifest::typeName(parameter.type);
    CustomPropertyType const expected = uniformPropertyType(parameter.type);
    CustomPropertyType const declared = renderer.getCustomPropertyType(parameter.name);
    if (declared == CustomPropertyType::eNone) {
      throw std::runtime_error(manifestPath + " declares uniform \"" + parameter.name + "\" as " +
                               type + " but the shader declares no such property");
    }
    if (declared != expected) {
      throw std::runtime_error(manifestPath + " declares uniform \"" + parameter.name + "\" as " +
                               type + " but the shader declares it as " +
                               declaredTypeName(declared));
    }
    PackValue const value = checkAgainst(manifest, config.getShaderPack(), parameter.name,
                                         config.getParameter(parameter.name));
    if (expected == CustomPropertyType::eInt) {
      int const integer = std::holds_alternative<bool>(value) ? (std::get<bool>(value) ? 1 : 0)
                                                              : std::get<int>(value);
      renderer.setCustomProperty(parameter.name, integer);
    } else {
      renderer.setCustomProperty(parameter.name, std::get<float>(value));
    }
  }
}

} // namespace

RenderConfig::RenderConfig(std::string const &shaderPack,
                           std::optional<std::string> const &preset) {
  auto const [pack, kind] = resolvePack(shaderPack);
  mShaderPack = pack;
  mKind = kind;
  mManifest = detail::getPackManifest(pack);
  if (preset) {
    // an unknown preset raises listing the pack's presets
    for (auto const &[name, value] : mManifest->getPreset(*preset)) {
      mParameters[name] = value;
    }
  }
}

std::pair<std::string, RenderConfig::Kind>
RenderConfig::resolvePack(std::string const &shaderPack) const {
  fs::path const path = resolveShaderPack(shaderPack);
  // a pack without pack.json raises here, naming the directory and pointing to the docs
  detail::getPackManifest(path);
  return {path.string(), packKind(path)};
}

detail::PackManifest const &RenderConfig::manifest() const {
  if (!mManifest) {
    throw std::runtime_error("shader pack \"" + mShaderPack + "\" has no pack.json");
  }
  return *mManifest;
}

void RenderConfig::setShaderPack(std::string const &shaderPack) {
  auto const [pack, kind] = resolvePack(shaderPack);
  auto manifest = detail::getPackManifest(pack);
  // re-validate every explicit assignment against the new pack.json before keeping it
  for (auto const &[name, assigned] : mParameters) {
    checkAgainst(*manifest, pack, name, assigned);
  }
  mShaderPack = pack;
  mKind = kind;
  mManifest = std::move(manifest);
}

bool RenderConfig::declaresParameter(std::string const &name) const {
  return manifest().findParameter(name) >= 0;
}

bool RenderConfig::isVariantParameter(std::string const &name) const {
  int const index = manifest().findParameter(name);
  if (index < 0) {
    throw std::runtime_error("shader pack \"" + mShaderPack + "\" declares no parameter \"" +
                             name + "\"");
  }
  return manifest().getParameter(static_cast<std::size_t>(index)).kind ==
         PackParameterKind::eVariant;
}

PackValue RenderConfig::checkParameter(std::string const &name, PackValue value) const {
  return checkAgainst(manifest(), mShaderPack, name, std::move(value));
}

void RenderConfig::setParameter(std::string const &name, PackValue value) {
  mParameters[name] = checkParameter(name, std::move(value));
}

PackValue RenderConfig::getParameter(std::string const &name) const {
  auto it = mParameters.find(name);
  if (it != mParameters.end()) {
    return it->second;
  }
  int const index = manifest().findParameter(name);
  if (index < 0) {
    throw std::runtime_error("shader pack \"" + mShaderPack + "\" declares no parameter \"" +
                             name + "\"");
  }
  return manifest().getParameter(static_cast<std::size_t>(index)).defaultValue;
}

void RenderConfig::removeParameter(std::string const &name) {
  if (!declaresParameter(name)) {
    throw std::runtime_error("shader pack \"" + mShaderPack + "\" declares no parameter \"" +
                             name + "\"");
  }
  mParameters.erase(name);
}

std::vector<RenderConfigParameter> RenderConfig::describeParameters() const {
  std::vector<RenderConfigParameter> result;
  for (auto const &parameter : manifest().getParameters()) {
    RenderConfigParameter described;
    described.name = parameter.name;
    described.kind = PackManifest::kindName(parameter.kind);
    described.type = PackManifest::typeName(parameter.type);
    described.defaultValue = parameter.defaultValue;
    described.minimum = parameter.minimum;
    described.maximum = parameter.maximum;
    described.values = parameter.values;
    described.description = parameter.description;
    result.push_back(std::move(described));
  }
  return result;
}

std::vector<std::string> RenderConfig::getPresets() const { return manifest().getPresets(); }

std::map<std::string, std::string> RenderConfig::getShaderDefinitions() const {
  std::map<std::string, std::string> definitions;
  auto const &parameters = manifest().getParameters();
  for (std::size_t index = 0; index < parameters.size(); ++index) {
    auto const &parameter = parameters[index];
    if (parameter.kind != PackParameterKind::eVariant) {
      continue;
    }
    PackValue const value = getParameter(parameter.name);
    std::string definition = manifest().getDefinition(index, value);
    if (definition.empty()) {
      continue;
    }
    // an int or float variant defines its name to the value; bool and enum variants define a name
    bool const numeric = parameter.type == PackParameterType::eInt ||
                         parameter.type == PackParameterType::eFloat;
    definitions[definition] = numeric ? valueText(value) : "";
  }
  return definitions;
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

bool RenderConfig::operator==(RenderConfig const &other) const {
  return mShaderPack == other.mShaderPack && mKind == other.mKind &&
         mParameters == other.mParameters && mDenoiser == other.mDenoiser &&
         mMSAA == other.mMSAA && mPictureFormats == other.mPictureFormats;
}

std::string RenderConfig::toString() const {
  std::ostringstream stream;
  stream << "RenderConfig(shader_pack=\"" << mShaderPack << "\", kind=\"" << getKindName()
         << "\", properties={";
  bool first = true;
  for (auto const &[name, value] : mParameters) {
    stream << (first ? "" : ", ") << "\"" << name << "\": ";
    if (std::holds_alternative<std::string>(value)) {
      stream << "\"" << std::get<std::string>(value) << "\"";
    } else if (std::holds_alternative<bool>(value)) {
      stream << (std::get<bool>(value) ? "True" : "False");
    } else {
      stream << valueText(value);
    }
    first = false;
  }
  stream << "}, denoiser=\"" << denoiserToString(mDenoiser) << "\", msaa=" << mMSAA
         << ", picture_formats={";
  first = true;
  for (auto const &[name, format] : mPictureFormats) {
    stream << (first ? "" : ", ") << "\"" << name << "\": \"" << vk::to_string(format) << "\"";
    first = false;
  }
  stream << "})";
  return stream.str();
}

std::shared_ptr<svulkan2::RendererConfig> buildRendererConfig(RenderConfig const &config) {
  auto result = std::make_shared<svulkan2::RendererConfig>();
  result->shaderDir = config.getShaderPack();
  result->shaderDefines = config.getShaderDefinitions();
  result->msaa = msaaSampleCount(config.getMSAA());
  for (auto const &[name, format] : config.getPictureFormats()) {
    result->textureFormat[name] = format;
  }
  return result;
}

void applyRenderConfig(RenderConfig const &config, svulkan2::renderer::RendererBase &renderer) {
  auto const &pack = config.getShaderPack();
  if (config.isRayTracing()) {
    if (!config.getPictureFormats().empty()) {
      throw std::runtime_error("render config property \"picture_formats\" is not supported by "
                               "ray tracing shader pack \"" +
                               pack + "\"");
    }
  } else if (config.getDenoiser() != svulkan2::renderer::RTRenderer::DenoiserType::eNONE) {
    throw std::runtime_error("render config property \"denoiser\" can only be set with a ray "
                             "tracing shader pack, but the config uses rasterization pack \"" +
                             pack + "\"");
  }

  applyUniformParameters(config, renderer);

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

void clearPackManifestCache() { detail::clearPackManifestCache(); }

void applyRenderConfig(RenderConfig const &config,
                       svulkan2::renderer::BatchRTRenderer &renderer) {
  if (config.getDenoiser() != svulkan2::renderer::RTRenderer::DenoiserType::eNONE) {
    throw std::runtime_error("render config property \"denoiser\" is not supported by batched "
                             "ray tracing shader pack \"" +
                             config.getShaderPack() + "\"");
  }
  applyUniformParameters(config, renderer);
}

} // namespace sapien_renderer
} // namespace sapien
