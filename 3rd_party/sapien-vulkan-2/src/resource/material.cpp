#include "svulkan2/resource/material.h"
#include <atomic>

namespace svulkan2 {
namespace resource {

namespace {
std::atomic<uint64_t> gMaterialVersion{1};
}

uint64_t SVMetallicMaterial::getGlobalVersion() { return gMaterialVersion.load(); }

void SVMetallicMaterial::changed() { ++gMaterialVersion; }

void SVMetallicMaterial::setCullMode(vk::CullModeFlagBits cullMode) {
  if (mCullMode == cullMode) {
    return;
  }
  mCullMode = cullMode;
  changed();
}

std::shared_ptr<SVMetallicMaterial> SVMetallicMaterial::clone() const {
  auto copy = std::make_shared<SVMetallicMaterial>();
  copy->mBuffer = mBuffer;
  copy->mTextures = mTextures;
  copy->mCullMode = mCullMode;
  copy->mAlphaMode = mAlphaMode;
  copy->mAlphaCutoff = mAlphaCutoff;
  return copy;
}

void SVMetallicMaterial::setEmission(glm::vec4 emission) {
  mBuffer.emissionColor = glm::vec3(emission);
  mBuffer.emissionStrength = emission.a;
  changed();
}
glm::vec4 SVMetallicMaterial::getEmission() const {
  return glm::vec4(mBuffer.emissionColor, mBuffer.emissionStrength);
}

void SVMetallicMaterial::setBaseColor(glm::vec4 baseColor) {
  mBuffer.baseColor = baseColor;
  changed();
}
void SVMetallicMaterial::setBaseWeight(float weight) {
  mBuffer.baseWeight = weight;
  changed();
}
void SVMetallicMaterial::setMetallic(float metallic) {
  mBuffer.metallic = metallic;
  changed();
}

void SVMetallicMaterial::setSpecularWeight(float weight) {
  mBuffer.specularWeight = weight;
  changed();
}
void SVMetallicMaterial::setSpecularColor(glm::vec3 color) {
  mBuffer.specularColor = color;
  changed();
}
void SVMetallicMaterial::setRoughness(float roughness) {
  mBuffer.roughness = roughness;
  changed();
}
void SVMetallicMaterial::setSpecularAnisotropy(float anisotropy) {
  mBuffer.specularAnisotropy = anisotropy;
  changed();
}
void SVMetallicMaterial::setIor(float ior) {
  mBuffer.ior = ior;
  changed();
}

void SVMetallicMaterial::setTransmission(float weight) {
  mBuffer.transmissionWeight = weight;
  changed();
}
void SVMetallicMaterial::setTransmissionColor(glm::vec3 color) {
  mBuffer.transmissionColor = color;
  changed();
}
void SVMetallicMaterial::setTransmissionDepth(float depth) {
  mBuffer.transmissionDepth = depth;
  changed();
}

void SVMetallicMaterial::setCoatWeight(float weight) {
  mBuffer.coatWeight = weight;
  changed();
}
void SVMetallicMaterial::setCoatColor(glm::vec3 color) {
  mBuffer.coatColor = color;
  changed();
}
void SVMetallicMaterial::setCoatRoughness(float roughness) {
  mBuffer.coatRoughness = roughness;
  changed();
}
void SVMetallicMaterial::setCoatIor(float ior) {
  mBuffer.coatIor = ior;
  changed();
}

void SVMetallicMaterial::setFuzzWeight(float weight) {
  mBuffer.fuzzWeight = weight;
  changed();
}
void SVMetallicMaterial::setFuzzColor(glm::vec3 color) {
  mBuffer.fuzzColor = color;
  changed();
}
void SVMetallicMaterial::setFuzzRoughness(float roughness) {
  mBuffer.fuzzRoughness = roughness;
  changed();
}

void SVMetallicMaterial::setThinWalled(bool thinWalled) {
  mBuffer.thinWalled = thinWalled ? 1 : 0;
  changed();
}
void SVMetallicMaterial::setOcclusionStrength(float strength) {
  mBuffer.occlusionStrength = strength;
  changed();
}

void SVMetallicMaterial::setTexture(TextureSlot slot, std::shared_ptr<SVTexture> texture) {
  mTextures.at(slot) = std::move(texture);
  changed();
}

void SVMetallicMaterial::setTextureTransform(TextureSlot slot, glm::vec4 const &transform) {
  mBuffer.textureTransforms.at(slot) = transform;
  changed();
}

} // namespace resource
} // namespace svulkan2
