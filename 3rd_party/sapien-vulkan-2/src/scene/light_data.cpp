#include "svulkan2/scene/light_data.h"
#include "svulkan2/scene/scene.h"
#include <algorithm>

namespace svulkan2 {
namespace scene {

namespace {
template <typename Light> std::vector<Light *> shadowsFirst(std::vector<Light *> lights) {
  std::stable_partition(lights.begin(), lights.end(),
                        [](Light const *light) { return light->isShadowEnabled(); });
  return lights;
}

glm::mat4 worldMatrix(Node const &node) { return node.computeWorldModelMatrix(); }

GpuSpotLight spotLightData(SpotLight const &light, int textureId) {
  glm::mat4 world = worldMatrix(light);
  return {.viewMat = glm::affineInverse(world),
          .projMat = light.getShadowProjectionMatrix(),
          .rgb = light.getColor(),
          .shadowEnabled = light.isShadowEnabled() ? 1 : 0,
          .position = glm::vec3(world[3]),
          .fovInner = light.getFovSmall(),
          .fovOuter = light.getFov(),
          .textureId = textureId,
          .radius = light.getRadius()};
}
} // namespace

void LightTable::append(Scene &scene, TextureIndex const &textureIndex) {
  for (auto *light : shadowsFirst(scene.getPointLights())) {
    glm::mat4 world = worldMatrix(*light);
    point.push_back({.position = glm::vec3(world[3]),
                     .radius = light->getRadius(),
                     .rgb = light->getColor()});
  }
  for (auto *light : shadowsFirst(scene.getDirectionalLights())) {
    glm::mat4 world = worldMatrix(*light);
    glm::vec3 dir = glm::vec3(world * glm::vec4(0.f, 0.f, -1.f, 0.f));
    float len = glm::length(dir);
    directional.push_back({.direction = len > 1e-6f ? dir / len : glm::vec3(0.f, 0.f, -1.f),
                           .softness = light->getSoftness(),
                           .rgb = light->getColor()});
  }
  for (auto *light : shadowsFirst(scene.getSpotLights())) {
    spot.push_back(spotLightData(*light, -1));
  }
  for (auto *light : scene.getTexturedLights()) {
    // Every textured light asks for a slot, texture or not: the rasterizer binds the default
    // texture into the slot of a light without one, and the shaders index the array by that slot
    // (the ray tracers return -1 for a missing texture and skip it).
    spot.push_back(spotLightData(*light, textureIndex(light->getTexture())));
  }
  for (auto *light : scene.getParallelogramLights()) {
    glm::mat4 world = worldMatrix(*light);
    glm::vec2 half = light->getHalfSize();
    float angle = light->getAngle();
    parallelogram.push_back(
        {.rgb = light->getColor(),
         .position = glm::vec3(world * glm::vec4(-half.x, -half.y, 0.f, 1.f)),
         .edge0 = glm::vec3(world * glm::vec4(half.x * 2.f, 0.f, 0.f, 0.f)),
         .edge1 = glm::vec3(
             world * glm::vec4(glm::cos(angle) * half.y * 2.f,
                               glm::sin(angle) * half.y * 2.f, 0.f, 0.f))});
  }
}

void LightTable::clear() {
  point.clear();
  directional.clear();
  spot.clear();
  parallelogram.clear();
}

} // namespace scene
} // namespace svulkan2
