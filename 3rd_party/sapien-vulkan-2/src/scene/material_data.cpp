#include "svulkan2/scene/material_data.h"
#include "svulkan2/resource/material.h"
#include <cstring>

namespace svulkan2 {
namespace scene {

GpuMaterial packMaterial(resource::SVMetallicMaterial const &material,
                         TextureSlot const &textureSlot) {
  static_assert(sizeof(resource::SVMetallicMaterial::Buffer) ==
                offsetof(GpuMaterial, baseColorTexture));
  GpuMaterial record{};
  std::memcpy(&record, &material.getBuffer(), sizeof(resource::SVMetallicMaterial::Buffer));
  auto slot = [&](std::shared_ptr<resource::SVTexture> const &texture) {
    return texture ? textureSlot(texture) : -1;
  };
  record.baseColorTexture = slot(material.getBaseColorTexture());
  record.roughnessTexture = slot(material.getRoughnessTexture());
  record.normalTexture = slot(material.getNormalTexture());
  record.metallicTexture = slot(material.getMetallicTexture());
  record.emissionTexture = slot(material.getEmissionTexture());
  record.transmissionTexture = slot(material.getTransmissionTexture());
  return record;
}

} // namespace scene
} // namespace svulkan2
