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

} // namespace renderer
} // namespace svulkan2
