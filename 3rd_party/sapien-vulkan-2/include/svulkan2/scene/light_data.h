#pragma once
#include <functional>
#include "svulkan2/common/glm.h"
#include <memory>
#include <vector>

namespace svulkan2 {
namespace resource {
class SVTexture;
}
namespace scene {
class Scene;

// The storage-buffer layouts of vulkan_shader/common/light_data.glsl, shared by the
// rasterization, ray tracing and batched ray tracing renderers.

struct GpuPointLight {
  glm::vec3 position;
  float radius{0.f};
  glm::vec3 rgb;
  float padding{0.f};
};
static_assert(sizeof(GpuPointLight) == 32);

struct GpuDirectionalLight {
  glm::vec3 direction; // the direction the light travels
  float softness{0.f};
  glm::vec3 rgb;
  float padding{0.f};
};
static_assert(sizeof(GpuDirectionalLight) == 32);

struct GpuSpotLight {
  glm::mat4 viewMat{1.f}; // world to light space; the light shines along its -z
  glm::mat4 projMat{1.f}; // the light's shadow projection, also used for its texture
  glm::vec3 rgb;
  int padding0{0};
  glm::vec3 position;
  int padding1{0};
  float fovInner{0.f}; // full angles, radians
  float fovOuter{0.f};
  int textureId{-1}; // renderer-specific texture slot of a textured light, -1 for none
  int padding2{0};
};
static_assert(sizeof(GpuSpotLight) == 176);

struct GpuParallelogramLight {
  glm::vec3 rgb;
  float padding0{0.f};
  glm::vec3 position;
  float padding1{0.f};
  glm::vec3 edge0;
  float padding2{0.f};
  glm::vec3 edge1;
  float padding3{0.f};
};
static_assert(sizeof(GpuParallelogramLight) == 64);

/** The lights of one or more scenes in the shader layout.
 *
 *  Within each kind the lights that cast shadows come first, in scene order, so a
 *  rasterization renderer pairs light i with shadow map i; textured lights always cast shadows
 *  and follow the plain spot lights in `spot`. */
struct LightTable {
  std::vector<GpuPointLight> point;
  std::vector<GpuDirectionalLight> directional;
  std::vector<GpuSpotLight> spot; // plain spot lights, then textured lights
  std::vector<GpuParallelogramLight> parallelogram;

  /** maps the texture of a textured light to the textureId its renderer's shaders read */
  using TextureIndex = std::function<int(std::shared_ptr<resource::SVTexture> const &)>;

  /** append the lights of `scene` (a SceneGroup appends those of its member scenes) */
  void append(Scene &scene, TextureIndex const &textureIndex);
  void clear();
};

} // namespace scene
} // namespace svulkan2
