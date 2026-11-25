#pragma once
#include "texture.h"
#include <array>
#include <svulkan2/resource/material.h>

namespace sapien {
namespace sapien_renderer {
class SapienRenderEngine;

class SapienRenderMaterial {
public:
  SapienRenderMaterial();
  SapienRenderMaterial(std::array<float, 4> emission, std::array<float, 4> baseColor,
                       float specularWeight, float roughness, float metallic, float transmission,
                       float ior);
  explicit SapienRenderMaterial(std::shared_ptr<svulkan2::resource::SVMetallicMaterial> material);

  void setEmission(std::array<float, 4> color);
  [[nodiscard]] std::array<float, 4> getEmission() const;
  void setBaseColor(std::array<float, 4> color);
  [[nodiscard]] std::array<float, 4> getBaseColor() const;
  void setBaseWeight(float weight);
  [[nodiscard]] float getBaseWeight() const;
  void setMetallic(float metallic);
  [[nodiscard]] float getMetallic() const;
  void setRoughness(float roughness);
  [[nodiscard]] float getRoughness() const;
  void setSpecularWeight(float weight);
  [[nodiscard]] float getSpecularWeight() const;
  void setSpecularColor(std::array<float, 3> color);
  [[nodiscard]] std::array<float, 3> getSpecularColor() const;
  void setSpecularAnisotropy(float anisotropy);
  [[nodiscard]] float getSpecularAnisotropy() const;
  void setIOR(float ior);
  [[nodiscard]] float getIOR() const;
  void setTransmission(float transmission);
  [[nodiscard]] float getTransmission() const;
  void setTransmissionColor(std::array<float, 3> color);
  [[nodiscard]] std::array<float, 3> getTransmissionColor() const;
  void setTransmissionDepth(float depth);
  [[nodiscard]] float getTransmissionDepth() const;
  void setCoatWeight(float weight);
  [[nodiscard]] float getCoatWeight() const;
  void setCoatColor(std::array<float, 3> color);
  [[nodiscard]] std::array<float, 3> getCoatColor() const;
  void setCoatRoughness(float roughness);
  [[nodiscard]] float getCoatRoughness() const;
  void setCoatIor(float ior);
  [[nodiscard]] float getCoatIor() const;
  void setFuzzWeight(float weight);
  [[nodiscard]] float getFuzzWeight() const;
  void setFuzzColor(std::array<float, 3> color);
  [[nodiscard]] std::array<float, 3> getFuzzColor() const;
  void setFuzzRoughness(float roughness);
  [[nodiscard]] float getFuzzRoughness() const;
  void setThinWalled(bool thinWalled);
  [[nodiscard]] bool getThinWalled() const;

  void setEmissionTexture(std::shared_ptr<SapienRenderTexture2D> texture);
  [[nodiscard]] std::shared_ptr<SapienRenderTexture2D> getEmissionTexture() const;
  void setBaseColorTexture(std::shared_ptr<SapienRenderTexture2D> texture);
  [[nodiscard]] std::shared_ptr<SapienRenderTexture2D> getBaseColorTexture() const;
  void setRoughnessTexture(std::shared_ptr<SapienRenderTexture2D> texture);
  [[nodiscard]] std::shared_ptr<SapienRenderTexture2D> getRoughnessTexture() const;
  void setMetallicTexture(std::shared_ptr<SapienRenderTexture2D> texture);
  [[nodiscard]] std::shared_ptr<SapienRenderTexture2D> getMetallicTexture() const;
  void setNormalTexture(std::shared_ptr<SapienRenderTexture2D> texture);
  [[nodiscard]] std::shared_ptr<SapienRenderTexture2D> getNormalTexture() const;
  void setTransmissionTexture(std::shared_ptr<SapienRenderTexture2D> texture);
  [[nodiscard]] std::shared_ptr<SapienRenderTexture2D> getTransmissionTexture() const;
  void setOcclusionTexture(std::shared_ptr<SapienRenderTexture2D> texture);
  [[nodiscard]] std::shared_ptr<SapienRenderTexture2D> getOcclusionTexture() const;

  [[nodiscard]] std::shared_ptr<svulkan2::resource::SVMetallicMaterial> getMaterial() const {
    return mMaterial;
  }

  std::shared_ptr<SapienRenderEngine> mEngine;
  std::shared_ptr<svulkan2::resource::SVMetallicMaterial> mMaterial;
};

} // namespace sapien_renderer
} // namespace sapien
