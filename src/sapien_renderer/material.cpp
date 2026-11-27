#include "sapien/sapien_renderer/material.h"
#include "sapien/sapien_renderer/sapien_renderer_system.h"

namespace sapien {
namespace sapien_renderer {

SapienRenderMaterial::SapienRenderMaterial()
    : SapienRenderMaterial({0.f, 0.f, 0.f, 0.f}, {1.f, 1.f, 1.f, 1.f}, 1.f, 0.45f, 0.f, 0.f,
                           1.5f) {}

SapienRenderMaterial::SapienRenderMaterial(std::array<float, 4> emission,
                                           std::array<float, 4> baseColor, float specularWeight,
                                           float roughness, float metallic, float transmission,
                                           float ior) {
  mEngine = SapienRenderEngine::Get();
  mMaterial = std::make_shared<svulkan2::resource::SVMetallicMaterial>();
  setEmission(emission);
  setBaseColor(baseColor);
  setSpecularWeight(specularWeight);
  setRoughness(roughness);
  setMetallic(metallic);
  setTransmission(transmission);
  setIOR(ior);
}

SapienRenderMaterial::SapienRenderMaterial(
    std::shared_ptr<svulkan2::resource::SVMetallicMaterial> material) {
  if (!material) {
    throw std::runtime_error("invalid material");
  }
  mEngine = SapienRenderEngine::Get();
  mMaterial = material;
}

void SapienRenderMaterial::setEmission(std::array<float, 4> color) {
  mMaterial->setEmission({color[0], color[1], color[2], color[3]});
}
std::array<float, 4> SapienRenderMaterial::getEmission() const {
  auto color = mMaterial->getEmission();
  return {color.r, color.g, color.b, color.a};
}

void SapienRenderMaterial::setBaseColor(std::array<float, 4> color) {
  mMaterial->setBaseColor({color[0], color[1], color[2], color[3]});
}
std::array<float, 4> SapienRenderMaterial::getBaseColor() const {
  auto color = mMaterial->getBaseColor();
  return {color.r, color.g, color.b, color.a};
}

void SapienRenderMaterial::setBaseWeight(float weight) { mMaterial->setBaseWeight(weight); }
float SapienRenderMaterial::getBaseWeight() const { return mMaterial->getBaseWeight(); }

void SapienRenderMaterial::setRoughness(float roughness) { mMaterial->setRoughness(roughness); }
float SapienRenderMaterial::getRoughness() const { return mMaterial->getRoughness(); }
void SapienRenderMaterial::setMetallic(float metallic) { mMaterial->setMetallic(metallic); }
float SapienRenderMaterial::getMetallic() const { return mMaterial->getMetallic(); }
void SapienRenderMaterial::setSpecularWeight(float weight) { mMaterial->setSpecularWeight(weight); }
float SapienRenderMaterial::getSpecularWeight() const { return mMaterial->getSpecularWeight(); }

void SapienRenderMaterial::setSpecularColor(std::array<float, 3> color) {
  mMaterial->setSpecularColor({color[0], color[1], color[2]});
}
std::array<float, 3> SapienRenderMaterial::getSpecularColor() const {
  auto color = mMaterial->getSpecularColor();
  return {color.r, color.g, color.b};
}

void SapienRenderMaterial::setSpecularAnisotropy(float anisotropy) {
  mMaterial->setSpecularAnisotropy(anisotropy);
}
float SapienRenderMaterial::getSpecularAnisotropy() const {
  return mMaterial->getSpecularAnisotropy();
}

void SapienRenderMaterial::setIOR(float ior) { mMaterial->setIor(ior); }
float SapienRenderMaterial::getIOR() const { return mMaterial->getIor(); }

void SapienRenderMaterial::setTransmission(float transmission) {
  mMaterial->setTransmission(transmission);
}
float SapienRenderMaterial::getTransmission() const { return mMaterial->getTransmission(); }

void SapienRenderMaterial::setTransmissionColor(std::array<float, 3> color) {
  mMaterial->setTransmissionColor({color[0], color[1], color[2]});
}
std::array<float, 3> SapienRenderMaterial::getTransmissionColor() const {
  auto color = mMaterial->getTransmissionColor();
  return {color.r, color.g, color.b};
}

void SapienRenderMaterial::setTransmissionDepth(float depth) {
  mMaterial->setTransmissionDepth(depth);
}
float SapienRenderMaterial::getTransmissionDepth() const {
  return mMaterial->getTransmissionDepth();
}

void SapienRenderMaterial::setCoatWeight(float weight) { mMaterial->setCoatWeight(weight); }
float SapienRenderMaterial::getCoatWeight() const { return mMaterial->getCoatWeight(); }

void SapienRenderMaterial::setCoatColor(std::array<float, 3> color) {
  mMaterial->setCoatColor({color[0], color[1], color[2]});
}
std::array<float, 3> SapienRenderMaterial::getCoatColor() const {
  auto color = mMaterial->getCoatColor();
  return {color.r, color.g, color.b};
}

void SapienRenderMaterial::setCoatRoughness(float roughness) {
  mMaterial->setCoatRoughness(roughness);
}
float SapienRenderMaterial::getCoatRoughness() const { return mMaterial->getCoatRoughness(); }

void SapienRenderMaterial::setCoatIor(float ior) { mMaterial->setCoatIor(ior); }
float SapienRenderMaterial::getCoatIor() const { return mMaterial->getCoatIor(); }

void SapienRenderMaterial::setFuzzWeight(float weight) { mMaterial->setFuzzWeight(weight); }
float SapienRenderMaterial::getFuzzWeight() const { return mMaterial->getFuzzWeight(); }

void SapienRenderMaterial::setFuzzColor(std::array<float, 3> color) {
  mMaterial->setFuzzColor({color[0], color[1], color[2]});
}
std::array<float, 3> SapienRenderMaterial::getFuzzColor() const {
  auto color = mMaterial->getFuzzColor();
  return {color.r, color.g, color.b};
}

void SapienRenderMaterial::setFuzzRoughness(float roughness) {
  mMaterial->setFuzzRoughness(roughness);
}
float SapienRenderMaterial::getFuzzRoughness() const { return mMaterial->getFuzzRoughness(); }

void SapienRenderMaterial::setThinWalled(bool thinWalled) {
  mMaterial->setThinWalled(thinWalled);
}
bool SapienRenderMaterial::getThinWalled() const { return mMaterial->getThinWalled(); }

namespace {
using Slot = svulkan2::resource::SVMetallicMaterial::TextureSlot;

void setTexture(svulkan2::resource::SVMetallicMaterial &material, Slot slot,
                std::shared_ptr<SapienRenderTexture2D> const &texture) {
  material.setTexture(slot, texture ? texture->getTexture() : nullptr);
}

std::shared_ptr<SapienRenderTexture2D>
getTexture(svulkan2::resource::SVMetallicMaterial const &material, Slot slot) {
  auto tex = material.getTexture(slot);
  return tex ? std::make_shared<SapienRenderTexture2D>(tex) : nullptr;
}
} // namespace

void SapienRenderMaterial::setEmissionTexture(std::shared_ptr<SapienRenderTexture2D> texture) {
  setTexture(*mMaterial, Slot::eEmission, texture);
}
std::shared_ptr<SapienRenderTexture2D> SapienRenderMaterial::getEmissionTexture() const {
  return getTexture(*mMaterial, Slot::eEmission);
}

void SapienRenderMaterial::setBaseColorTexture(std::shared_ptr<SapienRenderTexture2D> texture) {
  setTexture(*mMaterial, Slot::eBaseColor, texture);
}
std::shared_ptr<SapienRenderTexture2D> SapienRenderMaterial::getBaseColorTexture() const {
  return getTexture(*mMaterial, Slot::eBaseColor);
}

void SapienRenderMaterial::setRoughnessTexture(std::shared_ptr<SapienRenderTexture2D> texture) {
  setTexture(*mMaterial, Slot::eRoughness, texture);
}
std::shared_ptr<SapienRenderTexture2D> SapienRenderMaterial::getRoughnessTexture() const {
  return getTexture(*mMaterial, Slot::eRoughness);
}

void SapienRenderMaterial::setMetallicTexture(std::shared_ptr<SapienRenderTexture2D> texture) {
  setTexture(*mMaterial, Slot::eMetallic, texture);
}
std::shared_ptr<SapienRenderTexture2D> SapienRenderMaterial::getMetallicTexture() const {
  return getTexture(*mMaterial, Slot::eMetallic);
}

void SapienRenderMaterial::setNormalTexture(std::shared_ptr<SapienRenderTexture2D> texture) {
  setTexture(*mMaterial, Slot::eNormal, texture);
}
std::shared_ptr<SapienRenderTexture2D> SapienRenderMaterial::getNormalTexture() const {
  return getTexture(*mMaterial, Slot::eNormal);
}

void SapienRenderMaterial::setTransmissionTexture(std::shared_ptr<SapienRenderTexture2D> texture) {
  setTexture(*mMaterial, Slot::eTransmission, texture);
}
std::shared_ptr<SapienRenderTexture2D> SapienRenderMaterial::getTransmissionTexture() const {
  return getTexture(*mMaterial, Slot::eTransmission);
}

void SapienRenderMaterial::setOcclusionTexture(std::shared_ptr<SapienRenderTexture2D> texture) {
  setTexture(*mMaterial, Slot::eOcclusion, texture);
}
std::shared_ptr<SapienRenderTexture2D> SapienRenderMaterial::getOcclusionTexture() const {
  return getTexture(*mMaterial, Slot::eOcclusion);
}
void SapienRenderMaterial::setOcclusionStrength(float strength) {
  mMaterial->setOcclusionStrength(strength);
}
float SapienRenderMaterial::getOcclusionStrength() const {
  return mMaterial->getOcclusionStrength();
}

} // namespace sapien_renderer
} // namespace sapien
