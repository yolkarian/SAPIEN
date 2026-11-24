#pragma once
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
namespace svulkan2 {
namespace renderer {

/** A draw's `instance` uniform (std140), mirroring vulkan_shader/common/scene.glsl. Instances
 *  are stored minUniformBufferOffsetAlignment apart so each is bindable at a dynamic offset. */
struct InstanceInfo {
  glm::uvec4 segmentation;
  int32_t transformIndex;
  int32_t materialIndex;
  float transparency;
  int32_t shadeFlat;
  int32_t customDataIndex;
  int32_t padding[3];
};
static_assert(sizeof(InstanceInfo) == 48, "InstanceInfo must be 48 bytes");
static_assert(offsetof(InstanceInfo, customDataIndex) == 32, "InstanceInfo layout");

/** One element of materials[] and a draw's `material` uniform, mirroring
 *  vulkan_shader/common/scene.glsl: the first 160 bytes are SVMetallicMaterial::Buffer, then six
 *  textures[] slots; the tail is reserved. 256 bytes, so every element is bindable at a dynamic
 *  offset under any minUniformBufferOffsetAlignment Vulkan allows. */
struct GPUMaterial {
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
static_assert(sizeof(GPUMaterial) == 256, "GPUMaterial must be 256 bytes");
static_assert(offsetof(GPUMaterial, baseColorTexture) == 160, "GPUMaterial layout");

} // namespace renderer
} // namespace svulkan2
