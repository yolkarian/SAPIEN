#pragma once
#include "svulkan2/build_options.h"
#include <filesystem>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace svulkan2 {
namespace fs = std::filesystem;

/** Preprocessor definitions that select a variant of a shader: name -> value, where an empty
 *  value defines the bare name. Ordered, so equal sets compare and hash equal. */
using ShaderDefines = std::map<std::string, std::string>;

class GLSLCompiler {
public:
  /** Compiles a GLSL file with its includes expanded and `defines` inserted after `#version`.
   *  The result is cached by stage and expanded source text, so an edited file, an edited
   *  include or another set of defines compiles again while repeated requests do not. */
  static std::vector<std::uint32_t> const &
  compileGlslFileCached(vk::ShaderStageFlagBits stage, fs::path const &filepath,
                        ShaderDefines const &defines = {});

  static std::tuple<std::string, std::vector<std::tuple<std::string, int>>>
  loadGlslCodeWithDebugInfo(fs::path const &filepath);
  // static std::string loadGlslCode(fs::path const &filepath);
  static std::vector<std::uint32_t>
  compileToSpirv(vk::ShaderStageFlagBits shaderStage,
                 std::string const &glslCode,
                 std::vector<std::tuple<std::string, int>> const &debugInfo = {});
  static void InitializeProcess();
  static void FinalizeProcess();

private:
};

} // namespace svulkan2
