#pragma once
#include "layout.h"
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace svulkan2 {

/** Renderer options configured by API */
struct RendererConfig {
  std::string shaderDir;

  // preprocessor definitions that select a variant of the shader pack: name -> value
  std::map<std::string, std::string> shaderDefines;

  // compute shader files of the post-processing chain (renderer::PostChain), in order
  std::vector<std::string> postPasses;

  // default 1-channel texture format
  vk::Format colorFormat1{vk::Format::eR32Sfloat}; // R8Unorm, R32Sfloat

  // default 4-channel texture format
  vk::Format colorFormat4{vk::Format::eR32G32B32A32Sfloat}; // R8G8B8A8Unorm, R32G32B32A32Sfloat

  vk::SampleCountFlagBits msaa{vk::SampleCountFlagBits::e1}; // msaa

  // texture foramt for specific textures
  std::unordered_map<std::string, vk::Format> textureFormat;

  vk::Format depthFormat{vk::Format::eD32Sfloat}; // D32Sfloat
  // vk::CullModeFlags culling{vk::CullModeFlagBits::eBack};

  // when true, a single depth texture is used for all gbuffer passes (incremental rendering)
  bool shareGbufferDepths{true};

  bool operator==(RendererConfig const &other) const = default;
};

} // namespace svulkan2
