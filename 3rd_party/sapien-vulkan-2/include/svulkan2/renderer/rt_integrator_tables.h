#pragma once
#include <memory>

namespace svulkan2 {
namespace core {
class Context;
class Buffer;
} // namespace core
namespace scene {
class Scene;
}
namespace resource {
class SVCubemap;
}

namespace renderer {

/** Light-sampling tables of the rt path tracer (vulkan_shader/rt/scene_set.glsl), rebuilt with
 *  its scene descriptor set: the environment map over an equal-solid-angle grid, and the
 *  triangles of emissive meshes. Their alias tables come from src/sapien_renderer/alias_table.zig.
 */
struct IntegratorTables {
  // EnvAlias: one uvec4 per grid cell, (alias, threshold, pdf bits, 0)
  std::unique_ptr<core::Buffer> envAliasBuffer;
  // EmissiveTriangles (object space, with their TLAS rows), EmissiveAlias, and EmissiveBase:
  // per geometry instance the index of its first emissive triangle, or -1
  std::unique_ptr<core::Buffer> emissiveBuffer;
  std::unique_ptr<core::Buffer> emissiveAliasBuffer;
  std::unique_ptr<core::Buffer> emissiveBaseBuffer;
  int envWidth{0};      // grid columns (rows are half as many); 0 without an environment map
  int emissiveCount{0}; // 0 without emissive meshes

  void build(core::Context &context, scene::Scene &scene, resource::SVCubemap *environment);
};

} // namespace renderer
} // namespace svulkan2
