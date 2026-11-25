#pragma once
#include "svulkan2/common/glm.h"
#include "texture.h"
#include <array>
#include <cstddef>

namespace svulkan2 {
namespace resource {

/** Device-independent material. Renderers pack materials into their own scene tables
 *  (svulkan2/scene/material_data.h) and upload the textures to their devices. */
class SVMaterial {
public:
  virtual float getOpacity() const = 0;

  virtual ~SVMaterial() = default;

  void setCullMode(vk::CullModeFlagBits cullMode) { mCullMode = cullMode; }
  vk::CullModeFlagBits getCullMode() const { return mCullMode; }

protected:
  vk::CullModeFlagBits mCullMode{vk::CullModeFlagBits::eBack};
};

/** The first batch of the OpenPBR Surface 1.1 parameters (base, specular, transmission, coat,
 *  fuzz, emission, geometry); the names follow OpenPBR except where SAPIEN kept its own:
 *  metallic = base_metalness, roughness = specular_roughness, ior = specular_ior,
 *  emission strength = emission_luminance (a plain radiance multiplier, not nits). */
class SVMetallicMaterial : public SVMaterial {
public:
  /** texture slot, also the index of its UV transform */
  enum TextureSlot {
    eBaseColor = 0,
    eRoughness = 1,
    eNormal = 2,
    eMetallic = 3,
    eEmission = 4,
    eTransmission = 5,
    eOcclusion = 6,
    eTextureSlotCount = 7
  };

  /** The device-independent parameters: the first 256 bytes of scene::GpuMaterial and of
   *  `Material` in vulkan_shader/common/material.glsl. Defaults are OpenPBR's except
   *  base_color (white) and the specular roughness (0.45), SAPIEN's previous defaults. */
  struct Buffer {
    glm::vec4 baseColor{1, 1, 1, 1}; // rgb base_color, a geometry_opacity
    glm::vec3 emissionColor{0, 0, 0};
    float emissionStrength{0};
    glm::vec3 specularColor{1, 1, 1};
    float specularWeight{1};
    glm::vec3 transmissionColor{1, 1, 1};
    float transmissionWeight{0};
    glm::vec3 coatColor{1, 1, 1};
    float coatWeight{0};
    glm::vec3 fuzzColor{1, 1, 1};
    float fuzzWeight{0};
    float baseWeight{1};
    float metallic{0};
    float roughness{0.45f};
    float ior{1.5f};
    float specularAnisotropy{0};
    float transmissionDepth{0};
    float coatRoughness{0};
    float coatIor{1.6f};
    float fuzzRoughness{0.5f};
    int thinWalled{0};
    int padding[2]{};
    // offset xy, scale zw of the UVs of each texture slot
    std::array<glm::vec4, eTextureSlotCount> textureTransforms{
        glm::vec4{0, 0, 1, 1}, glm::vec4{0, 0, 1, 1}, glm::vec4{0, 0, 1, 1},
        glm::vec4{0, 0, 1, 1}, glm::vec4{0, 0, 1, 1}, glm::vec4{0, 0, 1, 1},
        glm::vec4{0, 0, 1, 1}};
  };
  static_assert(sizeof(Buffer) == 256, "Material Buffer layout");
  static_assert(offsetof(Buffer, baseWeight) == 96, "Material Buffer layout");
  static_assert(offsetof(Buffer, textureTransforms) == 144, "Material Buffer layout");

  SVMetallicMaterial() = default;

  void setEmission(glm::vec4 emission); // rgb colour, a strength
  glm::vec4 getEmission() const;
  void setBaseColor(glm::vec4 baseColor); // rgb colour, a opacity
  glm::vec4 getBaseColor() const { return mBuffer.baseColor; }
  void setBaseWeight(float weight);
  float getBaseWeight() const { return mBuffer.baseWeight; }
  void setMetallic(float metallic);
  float getMetallic() const { return mBuffer.metallic; }

  void setSpecularWeight(float weight);
  float getSpecularWeight() const { return mBuffer.specularWeight; }
  void setSpecularColor(glm::vec3 color);
  glm::vec3 getSpecularColor() const { return mBuffer.specularColor; }
  void setRoughness(float roughness);
  float getRoughness() const { return mBuffer.roughness; }
  void setSpecularAnisotropy(float anisotropy);
  float getSpecularAnisotropy() const { return mBuffer.specularAnisotropy; }
  void setIor(float ior);
  float getIor() const { return mBuffer.ior; }

  void setTransmission(float weight);
  float getTransmission() const { return mBuffer.transmissionWeight; }
  void setTransmissionColor(glm::vec3 color);
  glm::vec3 getTransmissionColor() const { return mBuffer.transmissionColor; }
  void setTransmissionDepth(float depth);
  float getTransmissionDepth() const { return mBuffer.transmissionDepth; }

  void setCoatWeight(float weight);
  float getCoatWeight() const { return mBuffer.coatWeight; }
  void setCoatColor(glm::vec3 color);
  glm::vec3 getCoatColor() const { return mBuffer.coatColor; }
  void setCoatRoughness(float roughness);
  float getCoatRoughness() const { return mBuffer.coatRoughness; }
  void setCoatIor(float ior);
  float getCoatIor() const { return mBuffer.coatIor; }

  void setFuzzWeight(float weight);
  float getFuzzWeight() const { return mBuffer.fuzzWeight; }
  void setFuzzColor(glm::vec3 color);
  glm::vec3 getFuzzColor() const { return mBuffer.fuzzColor; }
  void setFuzzRoughness(float roughness);
  float getFuzzRoughness() const { return mBuffer.fuzzRoughness; }

  void setThinWalled(bool thinWalled);
  bool getThinWalled() const { return mBuffer.thinWalled != 0; }

  void setTexture(TextureSlot slot, std::shared_ptr<SVTexture> texture);
  std::shared_ptr<SVTexture> getTexture(TextureSlot slot) const { return mTextures.at(slot); }
  void setTextureTransform(TextureSlot slot, glm::vec4 const &transform);
  glm::vec4 getTextureTransform(TextureSlot slot) const {
    return mBuffer.textureTransforms.at(slot);
  }

  inline float getOpacity() const override { return mBuffer.baseColor.a; }

  inline Buffer const &getBuffer() const { return mBuffer; }
  /** Process-wide stamp of the most recent edit to any material; renderers use it to decide
   *  whether their material tables need repacking. */
  static uint64_t getGlobalVersion();
  /** a new material with the same parameters, textures and cull mode */
  std::shared_ptr<SVMetallicMaterial> clone() const;

  SVMetallicMaterial(SVMetallicMaterial const &) = delete;
  SVMetallicMaterial &operator=(SVMetallicMaterial const &) = delete;
  SVMetallicMaterial(SVMetallicMaterial const &&) = delete;
  SVMetallicMaterial &operator=(SVMetallicMaterial const &&) = delete;

  ~SVMetallicMaterial() override = default;

private:
  Buffer mBuffer{};
  std::array<std::shared_ptr<SVTexture>, eTextureSlotCount> mTextures;

  /** every edit bumps the process-wide version renderers compare to repack their tables */
  void changed();
};

} // namespace resource
} // namespace svulkan2
