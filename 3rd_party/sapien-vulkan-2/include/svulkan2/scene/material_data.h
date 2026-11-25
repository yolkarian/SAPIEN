#pragma once
#include "svulkan2/resource/material.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace svulkan2 {
namespace scene {

/** The material record of vulkan_shader/common/material.glsl, shared by the rasterization, ray
 *  tracing and batched ray tracing renderers: the first 256 bytes are
 *  SVMetallicMaterial::Buffer, then the slots of its seven textures in the renderer's textures[]
 *  (-1 for none); the tail is reserved for the OpenPBR parameters still to come. 512 bytes, so a
 *  rasterizer can bind every record at a dynamic offset under any
 *  minUniformBufferOffsetAlignment Vulkan allows. */
struct GpuMaterial {
  resource::SVMetallicMaterial::Buffer parameters;
  int32_t textures[resource::SVMetallicMaterial::eTextureSlotCount];
  int32_t padding;
  glm::vec4 reserved[14];
};
static_assert(sizeof(GpuMaterial) == 512, "GpuMaterial must be 512 bytes");
static_assert(offsetof(GpuMaterial, textures) == 256, "GpuMaterial layout");

/** maps a texture of a material to the slot of the renderer's textures[] holding it */
using TextureSlot = std::function<int32_t(std::shared_ptr<resource::SVTexture> const &)>;

/** the record of `material`; `textureSlot` is called for each texture the material has */
GpuMaterial packMaterial(resource::SVMetallicMaterial const &material,
                         TextureSlot const &textureSlot);

} // namespace scene
} // namespace svulkan2
