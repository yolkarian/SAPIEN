#pragma once
#include "pack_manifest.h"
#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace sapien {
namespace sapien_renderer {

/** one parameter value: a bool, int or float uniform, or a string enum variant */
using PackValue = std::variant<bool, int, float, std::string>;

namespace detail {

enum class PackParameterKind { eUniform, eVariant };
enum class PackParameterType { eBool, eInt, eFloat, eEnum };

/** one parameter of a pack.json, copied out of the parsed manifest */
struct PackParameter {
  std::string name;
  PackParameterKind kind{PackParameterKind::eUniform};
  PackParameterType type{PackParameterType::eBool};
  PackValue defaultValue{false};
  std::optional<double> minimum;
  std::optional<double> maximum;
  /** enum parameters: the allowed values, in file order */
  std::vector<std::string> values;
  std::string description;
};

/** RAII wrapper of a pack.json parsed by src/sapien_renderer/pack_manifest.zig.
 *
 *  The manifest is immutable once parsed and owned by a pack cache, so callers share it through
 *  a shared_ptr and keep it alive after the cache drops it. All errors raise
 *  std::runtime_error; `load` prefixes its message with `<directory>/pack.json: `. */
class PackManifest {
public:
  /** parse `<directory>/pack.json`; raises when the file is missing or malformed */
  static std::shared_ptr<PackManifest const> load(std::filesystem::path const &directory);

  ~PackManifest();
  PackManifest(PackManifest const &) = delete;
  PackManifest &operator=(PackManifest const &) = delete;

  std::filesystem::path const &getDirectory() const { return mDirectory; }
  /** parameters in file order */
  std::vector<PackParameter> const &getParameters() const { return mParameters; }
  /** preset names in file order */
  std::vector<std::string> const &getPresets() const { return mPresetNames; }
  /** the parameter assignments of preset `name`; raises when the preset is unknown */
  std::map<std::string, PackValue> const &getPreset(std::string const &name) const;

  /** index of the parameter called `name`, or -1 */
  int findParameter(std::string const &name) const;
  /** parameter `index`; indices come from findParameter, so out of range is a programming error */
  PackParameter const &getParameter(std::size_t index) const;

  /** check the type, range and enum membership of `value` for parameter `index`; an int is
   *  converted to a float for a float parameter. Raises with the parser's message. */
  void check(std::size_t index, PackValue &value) const;
  /** the definition variant parameter `index` selects with `value`; empty when none */
  std::string getDefinition(std::size_t index, PackValue const &value) const;

  static char const *kindName(PackParameterKind kind);
  static char const *typeName(PackParameterType type);

private:
  PackManifest(std::filesystem::path directory, SapienPackManifest *handle);

  std::filesystem::path mDirectory;
  SapienPackManifest *mHandle{};
  std::vector<PackParameter> mParameters;
  std::vector<std::string> mPresetNames;
  std::map<std::string, std::map<std::string, PackValue>> mPresets;
};

/** cached parse of `<directory>/pack.json`; `directory` must be canonical. The cache is
 *  mutex-protected and dropped by clearPackManifestCache(). */
std::shared_ptr<PackManifest const> getPackManifest(std::filesystem::path const &directory);

/** drop every cached manifest; configs created earlier keep theirs alive */
void clearPackManifestCache();

} // namespace detail
} // namespace sapien_renderer
} // namespace sapien
