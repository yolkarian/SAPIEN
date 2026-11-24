#include "svulkan2/resource/material.h"
#include <atomic>
#include "svulkan2/core/buffer.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace resource {

namespace {
std::atomic<uint64_t> gMaterialVersion{1};
}

uint64_t SVMetallicMaterial::getGlobalVersion() { return gMaterialVersion.load(); }

SVMetallicMaterial::SVMetallicMaterial(glm::vec4 emission, glm::vec4 baseColor, float fresnel,
                                       float roughness, float metallic, float transparency,
                                       float ior, float transmissionRoughness) {
  mBuffer.emission = emission;
  mBuffer.baseColor = baseColor;
  mBuffer.fresnel = fresnel;
  mBuffer.roughness = roughness;
  mBuffer.metallic = metallic;
  mBuffer.transmission = transparency;
  mBuffer.ior = ior;
  mBuffer.transmissionRoughness = transmissionRoughness;
  mBuffer.textureMask = 0;
}

SVMetallicMaterial::~SVMetallicMaterial() { removeFromDevice(); }

core::Buffer &SVMetallicMaterial::getDeviceBuffer(core::Context const &context) const {
  auto state = mDeviceStates.find(context);
  if (!state || !state->buffer) {
    throw std::runtime_error("failed to get device buffer: the material is not on this device");
  }
  return *state->buffer;
}

std::vector<std::shared_ptr<core::Context>> SVMetallicMaterial::deviceContexts() {
  std::vector<std::shared_ptr<core::Context>> contexts;
  mDeviceStates.forEach([&](DeviceState const &state) {
    if (state.context) {
      contexts.push_back(state.context);
    }
  });
  return contexts;
}

// Edits reach every device that already holds the material (the ray tracers bind its buffer)
// and bump the global version that rasterizers compare to refresh their materials[] table.
void SVMetallicMaterial::bufferChanged() {
  ++mBufferVersion;
  ++gMaterialVersion;
  for (auto const &context : deviceContexts()) {
    uploadToDevice(*context);
  }
}

void SVMetallicMaterial::texturesChanged() {
  ++mTextureVersion;
  bufferChanged(); // the texture mask lives in the uniform buffer
}

void SVMetallicMaterial::setEmission(glm::vec4 emission) {
  mBuffer.emission = emission;
  bufferChanged();
}

glm::vec4 SVMetallicMaterial::getEmission() const { return mBuffer.emission; }

void SVMetallicMaterial::setBaseColor(glm::vec4 baseColor) {
  mBuffer.baseColor = baseColor;
  bufferChanged();
}

glm::vec4 SVMetallicMaterial::getBaseColor() const { return mBuffer.baseColor; }

void SVMetallicMaterial::setRoughness(float roughness) {
  mBuffer.roughness = roughness;
  bufferChanged();
}

float SVMetallicMaterial::getRoughness() const { return mBuffer.roughness; }

void SVMetallicMaterial::setFresnel(float fresnel) {
  mBuffer.fresnel = fresnel;
  bufferChanged();
}

float SVMetallicMaterial::getFresnel() const { return mBuffer.fresnel; }

void SVMetallicMaterial::setMetallic(float metallic) {
  mBuffer.metallic = metallic;
  bufferChanged();
}

float SVMetallicMaterial::getMetallic() const { return mBuffer.metallic; }

void SVMetallicMaterial::setTransmission(float transmission) {
  mBuffer.transmission = transmission;
  bufferChanged();
}
float SVMetallicMaterial::getTransmission() const { return mBuffer.transmission; }

void SVMetallicMaterial::setIor(float ior) {
  mBuffer.ior = ior;
  bufferChanged();
}
float SVMetallicMaterial::getIor() const { return mBuffer.ior; }

void SVMetallicMaterial::setTransmissionRoughness(float roughness) {
  mBuffer.transmissionRoughness = roughness;
  bufferChanged();
}
float SVMetallicMaterial::getTransmissionRoughness() const {
  return mBuffer.transmissionRoughness;
}

std::shared_ptr<SVTexture> SVMetallicMaterial::getBaseColorTexture() const {
  if (getBit(mBuffer.textureMask, TextureBit::eBaseColor) == 0) {
    return nullptr;
  }
  return mBaseColorTexture;
}

std::shared_ptr<SVTexture> SVMetallicMaterial::getRoughnessTexture() const {
  if (getBit(mBuffer.textureMask, TextureBit::eRoughness) == 0) {
    return nullptr;
  }
  return mRoughnessTexture;
}

std::shared_ptr<SVTexture> SVMetallicMaterial::getNormalTexture() const {
  if (getBit(mBuffer.textureMask, TextureBit::eNormal) == 0) {
    return nullptr;
  }
  return mNormalTexture;
}

std::shared_ptr<SVTexture> SVMetallicMaterial::getMetallicTexture() const {
  if (getBit(mBuffer.textureMask, TextureBit::eMetallic) == 0) {
    return nullptr;
  }
  return mMetallicTexture;
}

std::shared_ptr<SVTexture> SVMetallicMaterial::getEmissionTexture() const {
  if (getBit(mBuffer.textureMask, TextureBit::eEmission) == 0) {
    return nullptr;
  }
  return mEmissionTexture;
}

std::shared_ptr<SVTexture> SVMetallicMaterial::getTransmissionTexture() const {
  if (getBit(mBuffer.textureMask, TextureBit::eTransmission) == 0) {
    return nullptr;
  }
  return mTransmissionTexture;
}

void SVMetallicMaterial::setBaseColorTexture(std::shared_ptr<SVTexture> texture) {
  mBaseColorTexture = texture;
  if (mBaseColorTexture) {
    setBit(mBuffer.textureMask, TextureBit::eBaseColor);
  } else {
    unsetBit(mBuffer.textureMask, TextureBit::eBaseColor);
  }
  texturesChanged();
}
void SVMetallicMaterial::setRoughnessTexture(std::shared_ptr<SVTexture> texture) {
  mRoughnessTexture = texture;
  if (mRoughnessTexture) {
    setBit(mBuffer.textureMask, TextureBit::eRoughness);
  } else {
    unsetBit(mBuffer.textureMask, TextureBit::eRoughness);
  }
  texturesChanged();
}

void SVMetallicMaterial::setNormalTexture(std::shared_ptr<SVTexture> texture) {
  mNormalTexture = texture;
  if (mNormalTexture) {
    setBit(mBuffer.textureMask, TextureBit::eNormal);
  } else {
    unsetBit(mBuffer.textureMask, TextureBit::eNormal);
  }
  texturesChanged();
}

void SVMetallicMaterial::setMetallicTexture(std::shared_ptr<SVTexture> texture) {
  mMetallicTexture = texture;
  if (mMetallicTexture) {
    setBit(mBuffer.textureMask, TextureBit::eMetallic);
  } else {
    unsetBit(mBuffer.textureMask, TextureBit::eMetallic);
  }
  texturesChanged();
}

void SVMetallicMaterial::setEmissionTexture(std::shared_ptr<SVTexture> texture) {
  mEmissionTexture = texture;
  if (mEmissionTexture) {
    setBit(mBuffer.textureMask, TextureBit::eEmission);
  } else {
    unsetBit(mBuffer.textureMask, TextureBit::eEmission);
  }
  texturesChanged();
}

void SVMetallicMaterial::setTransmissionTexture(std::shared_ptr<SVTexture> texture) {
  mTransmissionTexture = texture;
  if (mTransmissionTexture) {
    setBit(mBuffer.textureMask, TextureBit::eTransmission);
  } else {
    unsetBit(mBuffer.textureMask, TextureBit::eTransmission);
  }
  texturesChanged();
}

void SVMetallicMaterial::setBaseColorTextureTransform(glm::vec4 const &transform) {
  mBuffer.textureTransform[TextureBit::eBaseColor] = transform;
  bufferChanged();
}
void SVMetallicMaterial::setRoughnessTextureTransform(glm::vec4 const &transform) {
  mBuffer.textureTransform[TextureBit::eRoughness] = transform;
  bufferChanged();
}
void SVMetallicMaterial::setNormalTextureTransform(glm::vec4 const &transform) {
  mBuffer.textureTransform[TextureBit::eNormal] = transform;
  bufferChanged();
}
void SVMetallicMaterial::setMetallicTextureTransform(glm::vec4 const &transform) {
  mBuffer.textureTransform[TextureBit::eMetallic] = transform;
  bufferChanged();
}
void SVMetallicMaterial::setEmissionTextureTransform(glm::vec4 const &transform) {
  mBuffer.textureTransform[TextureBit::eEmission] = transform;
  bufferChanged();
}
void SVMetallicMaterial::setTransmissionTextureTransform(glm::vec4 const &transform) {
  mBuffer.textureTransform[TextureBit::eTransmission] = transform;
  bufferChanged();
}

glm::vec4 SVMetallicMaterial::getBaseColorTextureTransform() const {
  return mBuffer.textureTransform[TextureBit::eBaseColor];
}
glm::vec4 SVMetallicMaterial::getRoughnessTextureTransform() const {
  return mBuffer.textureTransform[TextureBit::eRoughness];
}
glm::vec4 SVMetallicMaterial::getNormalTextureTransform() const {
  return mBuffer.textureTransform[TextureBit::eNormal];
}
glm::vec4 SVMetallicMaterial::getMetallicTextureTransform() const {
  return mBuffer.textureTransform[TextureBit::eMetallic];
}
glm::vec4 SVMetallicMaterial::getEmissionTextureTransform() const {
  return mBuffer.textureTransform[TextureBit::eEmission];
}
glm::vec4 SVMetallicMaterial::getTransmissionTextureTransform() const {
  return mBuffer.textureTransform[TextureBit::eTransmission];
}

void SVMetallicMaterial::setTextures(std::shared_ptr<SVTexture> baseColorTexture,
                                     std::shared_ptr<SVTexture> roughnessTexture,
                                     std::shared_ptr<SVTexture> normalTexture,
                                     std::shared_ptr<SVTexture> metallicTexture,
                                     std::shared_ptr<SVTexture> emissionTexture,
                                     std::shared_ptr<SVTexture> transmissionTexture) {
  mBaseColorTexture = baseColorTexture;
  mRoughnessTexture = roughnessTexture;
  mNormalTexture = normalTexture;
  mMetallicTexture = metallicTexture;
  mEmissionTexture = emissionTexture;
  mTransmissionTexture = transmissionTexture;

  if (mBaseColorTexture) {
    setBit(mBuffer.textureMask, TextureBit::eBaseColor);
  } else {
    unsetBit(mBuffer.textureMask, TextureBit::eBaseColor);
  }

  if (mRoughnessTexture) {
    setBit(mBuffer.textureMask, TextureBit::eRoughness);
  } else {
    unsetBit(mBuffer.textureMask, TextureBit::eRoughness);
  }

  if (mNormalTexture) {
    setBit(mBuffer.textureMask, TextureBit::eNormal);
  } else {
    unsetBit(mBuffer.textureMask, TextureBit::eNormal);
  }

  if (mMetallicTexture) {
    setBit(mBuffer.textureMask, TextureBit::eMetallic);
  } else {
    unsetBit(mBuffer.textureMask, TextureBit::eMetallic);
  }

  if (mEmissionTexture) {
    setBit(mBuffer.textureMask, TextureBit::eEmission);
  } else {
    unsetBit(mBuffer.textureMask, TextureBit::eEmission);
  }

  if (mTransmissionTexture) {
    setBit(mBuffer.textureMask, TextureBit::eTransmission);
  } else {
    unsetBit(mBuffer.textureMask, TextureBit::eTransmission);
  }
  texturesChanged();
}

void SVMetallicMaterial::uploadToDevice(core::Context &context) {
  std::scoped_lock lock(context.getGlobalLock());

  auto &state = mDeviceStates.get(context);

  if (!state.buffer) {
    vk::BufferUsageFlags flags = vk::BufferUsageFlagBits::eUniformBuffer;
    if (context.isRayTracingAvailable()) {
      flags |= vk::BufferUsageFlagBits::eStorageBuffer;
    }
    state.buffer = core::Buffer::Create(context, sizeof(SVMetallicMaterial::Buffer), flags,
                                        VMA_MEMORY_USAGE_CPU_TO_GPU);
    state.context = context.shared_from_this();
  }

  if (state.bufferVersion != mBufferVersion) {
    state.buffer->upload(mBuffer);
    state.bufferVersion = mBufferVersion;
  }

  if (state.textureVersion != mTextureVersion) {
    // the textures live on the device too; renderers index them from their own tables
    for (auto const &texture : {mBaseColorTexture, mRoughnessTexture, mNormalTexture,
                                mMetallicTexture, mEmissionTexture, mTransmissionTexture}) {
      if (texture) {
        texture->uploadToDevice(context);
      }
    }
    state.textureVersion = mTextureVersion;
  }
}

void SVMetallicMaterial::removeFromDevice() {
  // collect first: erase cannot run inside forEach (it would deadlock on the PerContext lock)
  for (auto const &context : deviceContexts()) {
    std::scoped_lock lock(context->getGlobalLock());
    mDeviceStates.erase(*context);
  }
}

} // namespace resource
} // namespace svulkan2
