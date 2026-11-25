#include "pack_manifest.hpp"
#include <algorithm>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace fs = std::filesystem;

namespace sapien {
namespace sapien_renderer {

namespace detail {

namespace {

constexpr std::size_t ERROR_BUFFER_SIZE = 1024;

std::string toString(char const *text, std::size_t length) {
  return text ? std::string(text, length) : std::string();
}

/** the parser's error buffer as a message; the parser always NUL-terminates */
std::string messageOf(char const *buffer) { return std::string(buffer); }

} // namespace

PackManifest::PackManifest(fs::path directory, SapienPackManifest *handle)
    : mDirectory(std::move(directory)), mHandle(handle) {
  if (!mHandle) {
    throw std::runtime_error("pack manifest handle must not be null");
  }
  std::size_t const count = sapien_pack_manifest_parameter_count(mHandle);
  mParameters.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    SapienPackParameter declared{};
    sapien_pack_manifest_parameter(mHandle, index, &declared);
    PackParameter parameter;
    parameter.name = toString(declared.name, declared.name_len);
    parameter.kind = declared.kind == SAPIEN_PACK_VARIANT ? PackParameterKind::eVariant
                                                         : PackParameterKind::eUniform;
    switch (declared.type) {
    case SAPIEN_PACK_BOOL:
      parameter.type = PackParameterType::eBool;
      parameter.defaultValue = declared.default_value.integer != 0;
      break;
    case SAPIEN_PACK_INT:
      parameter.type = PackParameterType::eInt;
      parameter.defaultValue = static_cast<int>(declared.default_value.integer);
      break;
    case SAPIEN_PACK_FLOAT:
      parameter.type = PackParameterType::eFloat;
      parameter.defaultValue = static_cast<float>(declared.default_value.real);
      break;
    case SAPIEN_PACK_ENUM:
      parameter.type = PackParameterType::eEnum;
      parameter.defaultValue =
          toString(declared.default_value.string, declared.default_value.string_len);
      break;
    default:
      throw std::runtime_error("pack.json declares an unknown type for parameter \"" +
                               parameter.name + "\"");
    }
    if (declared.has_minimum) {
      parameter.minimum = declared.minimum;
    }
    if (declared.has_maximum) {
      parameter.maximum = declared.maximum;
    }
    parameter.description = toString(declared.description, declared.description_len);
    parameter.values.reserve(declared.enum_count);
    for (std::size_t value = 0; value < declared.enum_count; ++value) {
      char const *text{};
      std::size_t length{};
      sapien_pack_manifest_enum_value(mHandle, index, value, &text, &length);
      parameter.values.push_back(toString(text, length));
    }
    mParameters.push_back(std::move(parameter));
  }

  std::size_t const presets = sapien_pack_manifest_preset_count(mHandle);
  for (std::size_t preset = 0; preset < presets; ++preset) {
    char const *text{};
    std::size_t length{};
    sapien_pack_manifest_preset_name(mHandle, preset, &text, &length);
    std::string name = toString(text, length);
    std::map<std::string, PackValue> &entries = mPresets[name];
    std::size_t const count = sapien_pack_manifest_preset_entry_count(mHandle, preset);
    for (std::size_t index = 0; index < count; ++index) {
      std::size_t parameter{};
      SapienPackValue value{};
      sapien_pack_manifest_preset_entry(mHandle, preset, index, &parameter, &value);
      if (parameter >= mParameters.size()) {
        throw std::runtime_error("pack.json preset \"" + name + "\" refers to an unknown parameter");
      }
      PackValue converted;
      switch (value.type) {
      case SAPIEN_PACK_BOOL:
        converted = value.integer != 0;
        break;
      case SAPIEN_PACK_INT:
        converted = static_cast<int>(value.integer);
        break;
      case SAPIEN_PACK_FLOAT:
        converted = static_cast<float>(value.real);
        break;
      case SAPIEN_PACK_ENUM:
        converted = toString(value.string, value.string_len);
        break;
      default:
        throw std::runtime_error("pack.json preset \"" + name + "\" sets parameter \"" +
                                 mParameters[parameter].name + "\" to an unknown type");
      }
      entries[mParameters[parameter].name] = std::move(converted);
    }
    mPresetNames.push_back(std::move(name));
  }

  std::size_t const passes = sapien_pack_manifest_post_count(mHandle);
  for (std::size_t pass = 0; pass < passes; ++pass) {
    char const *text{};
    std::size_t length{};
    sapien_pack_manifest_post_name(mHandle, pass, &text, &length);
    mPostPasses.push_back(toString(text, length));
  }
}

PackManifest::~PackManifest() { sapien_pack_manifest_destroy(mHandle); }

std::shared_ptr<PackManifest const> PackManifest::load(fs::path const &directory) {
  fs::path const file = directory / "pack.json";
  if (!fs::is_regular_file(file)) {
    throw std::runtime_error("shader pack directory \"" + directory.string() +
                             "\" has no pack.json: every shader pack must declare its parameters "
                             "and presets in pack.json (see "
                             "docs/source/tutorial/rendering/customize_shaders.md)");
  }
  std::ifstream stream(file, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("failed to read " + file.string());
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  std::string const text = buffer.str();

  char error[ERROR_BUFFER_SIZE] = {};
  SapienPackManifest *handle{};
  if (sapien_pack_manifest_parse(text.data(), text.size(), &handle, error, sizeof(error)) != 0) {
    throw std::runtime_error(file.string() + ": " + messageOf(error));
  }
  return std::shared_ptr<PackManifest const>(new PackManifest(directory, handle));
}

std::map<std::string, PackValue> const &PackManifest::getPreset(std::string const &name) const {
  auto it = mPresets.find(name);
  if (it == mPresets.end()) {
    std::string available;
    for (auto const &preset : mPresetNames) {
      available += (available.empty() ? "" : ", ") + preset;
    }
    throw std::runtime_error("preset \"" + name + "\" is not declared by shader pack \"" +
                             mDirectory.string() + "\"" +
                             (available.empty() ? std::string(": the pack declares no presets")
                                                : ": available presets are " + available));
  }
  return it->second;
}

int PackManifest::findParameter(std::string const &name) const {
  for (std::size_t index = 0; index < mParameters.size(); ++index) {
    if (mParameters[index].name == name) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

PackParameter const &PackManifest::getParameter(std::size_t index) const {
  return mParameters.at(index);
}

void PackManifest::check(std::size_t index, PackValue &value) const {
  SapienPackValue converted{};
  std::string const &name = mParameters.at(index).name;
  switch (value.index()) {
  case 0:
    converted.type = SAPIEN_PACK_BOOL;
    converted.integer = std::get<bool>(value) ? 1 : 0;
    break;
  case 1:
    converted.type = SAPIEN_PACK_INT;
    converted.integer = std::get<int>(value);
    break;
  case 2:
    converted.type = SAPIEN_PACK_FLOAT;
    converted.real = std::get<float>(value);
    break;
  case 3: {
    std::string const &text = std::get<std::string>(value);
    converted.type = SAPIEN_PACK_ENUM;
    converted.string = text.data();
    converted.string_len = text.size();
    break;
  }
  default:
    throw std::runtime_error("parameter \"" + name + "\": unknown value type");
  }

  char error[ERROR_BUFFER_SIZE] = {};
  if (sapien_pack_manifest_check(mHandle, index, &converted, error, sizeof(error)) != 0) {
    throw std::runtime_error(messageOf(error));
  }
  switch (converted.type) {
  case SAPIEN_PACK_BOOL:
    value = converted.integer != 0;
    break;
  case SAPIEN_PACK_INT:
    value = static_cast<int>(converted.integer);
    break;
  case SAPIEN_PACK_FLOAT:
    value = static_cast<float>(converted.real);
    break;
  case SAPIEN_PACK_ENUM:
    value = toString(converted.string, converted.string_len);
    break;
  default:
    throw std::runtime_error("parameter \"" + name + "\": unknown checked type");
  }
}

std::string PackManifest::getDefinition(std::size_t index, PackValue const &value) const {
  SapienPackValue converted{};
  switch (value.index()) {
  case 0:
    converted.type = SAPIEN_PACK_BOOL;
    converted.integer = std::get<bool>(value) ? 1 : 0;
    break;
  case 1:
    converted.type = SAPIEN_PACK_INT;
    converted.integer = std::get<int>(value);
    break;
  case 2:
    converted.type = SAPIEN_PACK_FLOAT;
    converted.real = std::get<float>(value);
    break;
  case 3: {
    std::string const &text = std::get<std::string>(value);
    converted.type = SAPIEN_PACK_ENUM;
    converted.string = text.data();
    converted.string_len = text.size();
    break;
  }
  default:
    throw std::runtime_error("parameter \"" + mParameters.at(index).name +
                             "\": unknown value type");
  }

  char error[ERROR_BUFFER_SIZE] = {};
  char const *definition{};
  std::size_t length{};
  if (sapien_pack_manifest_define(mHandle, index, &converted, &definition, &length, error,
                                  sizeof(error)) != 0) {
    throw std::runtime_error(messageOf(error));
  }
  return toString(definition, length);
}

char const *PackManifest::kindName(PackParameterKind kind) {
  return kind == PackParameterKind::eVariant ? "variant" : "uniform";
}

char const *PackManifest::typeName(PackParameterType type) {
  switch (type) {
  case PackParameterType::eBool:
    return "bool";
  case PackParameterType::eInt:
    return "int";
  case PackParameterType::eFloat:
    return "float";
  case PackParameterType::eEnum:
    return "enum";
  }
  return "unknown";
}

namespace {

struct PackManifestCache {
  std::mutex mutex;
  std::unordered_map<std::string, std::shared_ptr<PackManifest const>> manifests;
};

PackManifestCache &cache() {
  static PackManifestCache instance;
  return instance;
}

} // namespace

std::shared_ptr<PackManifest const> getPackManifest(fs::path const &directory) {
  std::string const key = fs::canonical(directory).string();
  {
    PackManifestCache &store = cache();
    std::lock_guard<std::mutex> lock(store.mutex);
    auto it = store.manifests.find(key);
    if (it != store.manifests.end()) {
      return it->second;
    }
  }

  // parse outside the lock: a slow disk must not block other packs, and a duplicate parse of the
  // same pack yields an equal manifest
  auto manifest = PackManifest::load(key);

  PackManifestCache &store = cache();
  std::lock_guard<std::mutex> lock(store.mutex);
  auto [it, inserted] = store.manifests.try_emplace(key, std::move(manifest));
  return it->second;
}

void clearPackManifestCache() {
  PackManifestCache &store = cache();
  std::lock_guard<std::mutex> lock(store.mutex);
  store.manifests.clear();
}

} // namespace detail
} // namespace sapien_renderer
} // namespace sapien
