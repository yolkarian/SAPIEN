#pragma once
#include "svulkan2/common/glm.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace svulkan2 {
namespace resource {
class SVMetallicMaterial;
class SVTexture;
} // namespace resource
namespace scene {

/** The material record of vulkan_shader/common/material.glsl, shared by the rasterization, ray
 *  tracing and batched ray tracing renderers: the first 160 bytes are
 *  SVMetallicMaterial::Buffer, then the slots of its six textures in the renderer's textures[]
 *  (-1 for none); the tail is reserved. 256 bytes, so a rasterizer can bind every record at a
 *  dynamic offset under any minUniformBufferOffsetAlignment Vulkan allows. */
struct GpuMaterial {
  glm::vec4 emission;
  glm::vec4 baseColor;
  float fresnel;
  float roughness;
  float metallic;
  float transmission;
  float ior;
  float transmissionRoughness;
  int32_t textureMask;
  int32_t padding1;
  glm::vec4 textureTransforms[6];
  int32_t baseColorTexture;
  int32_t roughnessTexture;
  int32_t normalTexture;
  int32_t metallicTexture;
  int32_t emissionTexture;
  int32_t transmissionTexture;
  int32_t padding2[2];
  glm::vec4 reserved[4];
};
static_assert(sizeof(GpuMaterial) == 256, "GpuMaterial must be 256 bytes");
static_assert(offsetof(GpuMaterial, baseColorTexture) == 160, "GpuMaterial layout");

/** maps a texture of a material to the slot of the renderer's textures[] holding it */
using TextureSlot = std::function<int32_t(std::shared_ptr<resource::SVTexture> const &)>;

/** the record of `material`; `textureSlot` is called for each texture the material has */
GpuMaterial packMaterial(resource::SVMetallicMaterial const &material,
                         TextureSlot const &textureSlot);

} // namespace scene
} // namespace svulkan2
