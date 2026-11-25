#include "svulkan2/resource/material.h"
#include <atomic>

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

void SVMetallicMaterial::changed() { ++gMaterialVersion; }

void SVMetallicMaterial::setEmission(glm::vec4 emission) {
  mBuffer.emission = emission;
  changed();
}

glm::vec4 SVMetallicMaterial::getEmission() const { return mBuffer.emission; }

void SVMetallicMaterial::setBaseColor(glm::vec4 baseColor) {
  mBuffer.baseColor = baseColor;
  changed();
}

glm::vec4 SVMetallicMaterial::getBaseColor() const { return mBuffer.baseColor; }

void SVMetallicMaterial::setRoughness(float roughness) {
  mBuffer.roughness = roughness;
  changed();
}

float SVMetallicMaterial::getRoughness() const { return mBuffer.roughness; }

void SVMetallicMaterial::setFresnel(float fresnel) {
  mBuffer.fresnel = fresnel;
  changed();
}

float SVMetallicMaterial::getFresnel() const { return mBuffer.fresnel; }

void SVMetallicMaterial::setMetallic(float metallic) {
  mBuffer.metallic = metallic;
  changed();
}

float SVMetallicMaterial::getMetallic() const { return mBuffer.metallic; }

void SVMetallicMaterial::setTransmission(float transmission) {
  mBuffer.transmission = transmission;
  changed();
}
float SVMetallicMaterial::getTransmission() const { return mBuffer.transmission; }

void SVMetallicMaterial::setIor(float ior) {
  mBuffer.ior = ior;
  changed();
}
float SVMetallicMaterial::getIor() const { return mBuffer.ior; }

void SVMetallicMaterial::setTransmissionRoughness(float roughness) {
  mBuffer.transmissionRoughness = roughness;
  changed();
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
  changed();
}
void SVMetallicMaterial::setRoughnessTexture(std::shared_ptr<SVTexture> texture) {
  mRoughnessTexture = texture;
  if (mRoughnessTexture) {
    setBit(mBuffer.textureMask, TextureBit::eRoughness);
  } else {
    unsetBit(mBuffer.textureMask, TextureBit::eRoughness);
  }
  changed();
}

void SVMetallicMaterial::setNormalTexture(std::shared_ptr<SVTexture> texture) {
  mNormalTexture = texture;
  if (mNormalTexture) {
    setBit(mBuffer.textureMask, TextureBit::eNormal);
  } else {
    unsetBit(mBuffer.textureMask, TextureBit::eNormal);
  }
  changed();
}

void SVMetallicMaterial::setMetallicTexture(std::shared_ptr<SVTexture> texture) {
  mMetallicTexture = texture;
  if (mMetallicTexture) {
    setBit(mBuffer.textureMask, TextureBit::eMetallic);
  } else {
    unsetBit(mBuffer.textureMask, TextureBit::eMetallic);
  }
  changed();
}

void SVMetallicMaterial::setEmissionTexture(std::shared_ptr<SVTexture> texture) {
  mEmissionTexture = texture;
  if (mEmissionTexture) {
    setBit(mBuffer.textureMask, TextureBit::eEmission);
  } else {
    unsetBit(mBuffer.textureMask, TextureBit::eEmission);
  }
  changed();
}

void SVMetallicMaterial::setTransmissionTexture(std::shared_ptr<SVTexture> texture) {
  mTransmissionTexture = texture;
  if (mTransmissionTexture) {
    setBit(mBuffer.textureMask, TextureBit::eTransmission);
  } else {
    unsetBit(mBuffer.textureMask, TextureBit::eTransmission);
  }
  changed();
}

void SVMetallicMaterial::setBaseColorTextureTransform(glm::vec4 const &transform) {
  mBuffer.textureTransform[TextureBit::eBaseColor] = transform;
  changed();
}
void SVMetallicMaterial::setRoughnessTextureTransform(glm::vec4 const &transform) {
  mBuffer.textureTransform[TextureBit::eRoughness] = transform;
  changed();
}
void SVMetallicMaterial::setNormalTextureTransform(glm::vec4 const &transform) {
  mBuffer.textureTransform[TextureBit::eNormal] = transform;
  changed();
}
void SVMetallicMaterial::setMetallicTextureTransform(glm::vec4 const &transform) {
  mBuffer.textureTransform[TextureBit::eMetallic] = transform;
  changed();
}
void SVMetallicMaterial::setEmissionTextureTransform(glm::vec4 const &transform) {
  mBuffer.textureTransform[TextureBit::eEmission] = transform;
  changed();
}
void SVMetallicMaterial::setTransmissionTextureTransform(glm::vec4 const &transform) {
  mBuffer.textureTransform[TextureBit::eTransmission] = transform;
  changed();
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
  changed();
}

} // namespace resource
} // namespace svulkan2
