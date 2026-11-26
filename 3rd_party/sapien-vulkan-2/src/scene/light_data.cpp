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

GpuSpotLight spotLightData(SpotLight const &light, int textureId) {
  return {.viewMat = glm::affineInverse(light.getTransform().worldModelMatrix),
          .projMat = light.getShadowProjectionMatrix(),
          .rgb = light.getColor(),
          .position = light.getPosition(),
          .fovInner = light.getFovSmall(),
          .fovOuter = light.getFov(),
          .textureId = textureId,
          .radius = light.getRadius()};
}
} // namespace

void LightTable::append(Scene &scene, TextureIndex const &textureIndex) {
  for (auto *light : shadowsFirst(scene.getPointLights())) {
    point.push_back({.position = light->getPosition(),
                     .radius = light->getRadius(),
                     .rgb = light->getColor()});
  }
  for (auto *light : shadowsFirst(scene.getDirectionalLights())) {
    directional.push_back({.direction = light->getDirection(),
                           .softness = light->getSoftness(),
                           .rgb = light->getColor()});
  }
  for (auto *light : shadowsFirst(scene.getSpotLights())) {
    spot.push_back(spotLightData(*light, -1));
  }
  for (auto *light : scene.getTexturedLights()) {
    spot.push_back(spotLightData(*light, textureIndex(light->getTexture())));
  }
  for (auto *light : scene.getParallelogramLights()) {
    parallelogram.push_back({.rgb = light->getColor(),
                             .position = light->getOrigin(),
                             .edge0 = light->getEdgeX(),
                             .edge1 = light->getEdgeY()});
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
