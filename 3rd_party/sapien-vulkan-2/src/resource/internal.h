#pragma once
#include "svulkan2/core/context.h"
#include "svulkan2/resource/image.h"
#include <ktx.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace svulkan2 {
namespace resource {
namespace internal {

/** the srgb variant of `format`, or `format` when it has none */
inline vk::Format toSrgbFormat(vk::Format format) {
  switch (format) {
  case vk::Format::eR8G8B8A8Unorm:
    return vk::Format::eR8G8B8A8Srgb;
  case vk::Format::eR8G8B8Unorm:
    return vk::Format::eR8G8B8Srgb;
  case vk::Format::eR8G8Unorm:
    return vk::Format::eR8G8Srgb;
  case vk::Format::eR8Unorm:
    return vk::Format::eR8Srgb;
  default:
    return format;
  }
}

/** stages that sample images */
inline vk::PipelineStageFlags shaderReadStages(core::Context const &context) {
  vk::PipelineStageFlags stages = vk::PipelineStageFlagBits::eFragmentShader;
  if (context.isRayTracingAvailable()) {
    stages |= vk::PipelineStageFlagBits::eRayTracingShaderKHR;
  }
  return stages;
}

/** host ktx texture with its destroy function; the SVImage takes it over */
struct KtxDeleter {
  void operator()(ktxTexture *texture) const { ktxTexture_Destroy(texture); }
};
using KtxTexture = std::shared_ptr<ktxTexture>;

/** take over a ktx texture returned by the open functions below */
inline KtxTexture wrap(ktxTexture *texture) { return KtxTexture(texture, KtxDeleter{}); }

/** open a ktx texture from memory, transcoding Basis data when needed; nullptr-free, throws */
inline ktxTexture *openFromMemory(void const *data, size_t size) {
  ktxTexture *texture{};
  auto result = ktxTexture_CreateFromMemory(reinterpret_cast<ktx_uint8_t const *>(data), size,
                                            KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture);
  if (result != KTX_SUCCESS || !texture) {
    throw std::runtime_error("failed to load ktx texture from memory: " +
                             std::string(ktxErrorString(result)));
  }
  if (ktxTexture_NeedsTranscoding(texture) && texture->classId == ktxTexture2_c) {
    result =
        ktxTexture2_TranscodeBasis(reinterpret_cast<ktxTexture2 *>(texture), KTX_TTF_BC7_RGBA, {});
    if (result != KTX_SUCCESS) {
      ktxTexture_Destroy(texture);
      throw std::runtime_error("failed to transcode ktx texture: " +
                               std::string(ktxErrorString(result)));
    }
  }
  return texture;
}

/** open a ktx texture from a file, transcoding Basis data when needed */
inline ktxTexture *openFromFile(std::string const &filename) {
  ktxTexture *texture{};
  auto result = ktxTexture_CreateFromNamedFile(filename.c_str(),
                                               KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture);
  if (result != KTX_SUCCESS || !texture) {
    throw std::runtime_error("failed to load ktx texture " + filename + ": " +
                             std::string(ktxErrorString(result)));
  }
  if (ktxTexture_NeedsTranscoding(texture) && texture->classId == ktxTexture2_c) {
    result =
        ktxTexture2_TranscodeBasis(reinterpret_cast<ktxTexture2 *>(texture), KTX_TTF_BC7_RGBA, {});
    if (result != KTX_SUCCESS) {
      ktxTexture_Destroy(texture);
      throw std::runtime_error("failed to transcode ktx texture " + filename + ": " +
                               std::string(ktxErrorString(result)));
    }
  }
  return texture;
}

/** a host image of an embedded ktx texture */
inline std::shared_ptr<SVImage> imageFromMemory(void const *data, size_t size) {
  return SVImage::FromKTX(openFromMemory(data, size));
}

/** a host image of a ktx file */
inline std::shared_ptr<SVImage> imageFromFile(std::string const &filename) {
  return SVImage::FromKTX(openFromFile(filename));
}

} // namespace internal
} // namespace resource
} // namespace svulkan2
