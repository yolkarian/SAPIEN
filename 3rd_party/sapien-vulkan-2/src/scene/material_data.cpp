#include "svulkan2/scene/material_data.h"

namespace svulkan2 {
namespace scene {

GpuMaterial packMaterial(resource::SVMetallicMaterial const &material,
                         TextureSlot const &textureSlot) {
  GpuMaterial record{};
  record.parameters = material.getBuffer();
  for (int slot = 0; slot < resource::SVMetallicMaterial::eTextureSlotCount; ++slot) {
    auto texture = material.getTexture(static_cast<resource::SVMetallicMaterial::TextureSlot>(slot));
    record.textures[slot] = texture ? textureSlot(texture) : -1;
  }
  return record;
}

} // namespace scene
} // namespace svulkan2
