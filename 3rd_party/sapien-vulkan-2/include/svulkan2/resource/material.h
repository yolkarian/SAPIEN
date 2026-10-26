#pragma once
#include "svulkan2/common/glm.h"
#include "svulkan2/core/allocator.h"
#include "texture.h"

namespace svulkan2 {

namespace core {
class Context;
class Buffer;
} // namespace core

namespace resource {

inline void setBit(int &number, int bit) { number |= 1 << bit; }
inline void unsetBit(int &number, int bit) { number &= ~(1 << bit); }
inline int getBit(int number, int bit) { return (number >> bit) & 1; }

/** Device-independent material; its uniform buffer and descriptor set live per device
 *  (Context) and are refreshed on upload when the material changed. */
class SVMaterial {
public:
  /** this device's descriptor set; throws when the material is not uploaded there */
  virtual vk::DescriptorSet getDescriptorSet(core::Context const &context) const = 0;
  virtual void uploadToDevice(core::Context &context) = 0;
  /** release the device state of every device */
  virtual void removeFromDevice() = 0;
  virtual float getOpacity() const = 0;

  virtual ~SVMaterial() = default;

  void setCullMode(vk::CullModeFlagBits cullMode) { mCullMode = cullMode; }
  vk::CullModeFlagBits getCullMode() const { return mCullMode; }

protected:
  vk::CullModeFlagBits mCullMode{vk::CullModeFlagBits::eBack};
};

class SVMetallicMaterial : public SVMaterial {

  enum TextureBit {
    eBaseColor = 0,
    eRoughness = 1,
    eNormal = 2,
    eMetallic = 3,
    eEmission = 4,
    eTransmission = 5
  };

  struct Buffer {
    glm::vec4 emission{};
    glm::vec4 baseColor{};
    float fresnel{};
    float roughness{};
    float metallic{};
    float transmission{};
    float ior{};
    float transmissionRoughness{};
    int textureMask{};
    int padding1;
    glm::vec4 textureTransform[6] = {
        {0, 0, 1, 1}, {0, 0, 1, 1}, {0, 0, 1, 1}, {0, 0, 1, 1}, {0, 0, 1, 1}, {0, 0, 1, 1},
    };
  } mBuffer{};
  std::shared_ptr<SVTexture> mBaseColorTexture;
  std::shared_ptr<SVTexture> mRoughnessTexture;
  std::shared_ptr<SVTexture> mNormalTexture;
  std::shared_ptr<SVTexture> mMetallicTexture;
  std::shared_ptr<SVTexture> mEmissionTexture;
  std::shared_ptr<SVTexture> mTransmissionTexture;

  struct DeviceState {
    std::shared_ptr<core::Context> context; // keeps the descriptor pool alive
    std::unique_ptr<core::Buffer> buffer;
    vk::UniqueDescriptorSet descriptorSet;
    uint64_t bufferVersion{};  // mBufferVersion last written to `buffer`
    uint64_t textureVersion{}; // mTextureVersion last bound to `descriptorSet`
  };
  core::PerContext<DeviceState> mDeviceStates;
  /** bumped by every setter of the matching kind; devices upload when behind */
  uint64_t mBufferVersion{1};
  uint64_t mTextureVersion{1};

  DeviceState const &getDeviceState(core::Context const &context) const;
  std::vector<std::shared_ptr<core::Context>> deviceContexts();
  /** bump the version and re-upload to every device holding the material */
  void bufferChanged();
  void texturesChanged();

public:
  SVMetallicMaterial(glm::vec4 emission = {0, 0, 0, 1}, glm::vec4 baseColor = {0, 0, 0, 1},
                     float fresnel = 0, float roughness = 1, float metallic = 0,
                     float transparency = 0, float ior = 1.f, float transmissionRoughness = 0.f);

  void setEmission(glm::vec4 emission);
  glm::vec4 getEmission() const;

  void setBaseColor(glm::vec4 baseColor);
  glm::vec4 getBaseColor() const;

  void setRoughness(float roughness);
  float getRoughness() const;

  void setFresnel(float fresnel);
  float getFresnel() const;

  void setMetallic(float metallic);
  float getMetallic() const;

  void setTransmission(float transmission);
  float getTransmission() const;

  void setIor(float ior);
  float getIor() const;

  void setTransmissionRoughness(float roughness);
  float getTransmissionRoughness() const;

  void setBaseColorTexture(std::shared_ptr<SVTexture> texture);
  void setRoughnessTexture(std::shared_ptr<SVTexture> texture);
  void setNormalTexture(std::shared_ptr<SVTexture> texture);
  void setMetallicTexture(std::shared_ptr<SVTexture> texture);
  void setEmissionTexture(std::shared_ptr<SVTexture> texture);
  void setTransmissionTexture(std::shared_ptr<SVTexture> texture);

  void setBaseColorTextureTransform(glm::vec4 const &transform);
  void setRoughnessTextureTransform(glm::vec4 const &transform);
  void setNormalTextureTransform(glm::vec4 const &transform);
  void setMetallicTextureTransform(glm::vec4 const &transform);
  void setEmissionTextureTransform(glm::vec4 const &transform);
  void setTransmissionTextureTransform(glm::vec4 const &transform);

  glm::vec4 getBaseColorTextureTransform() const;
  glm::vec4 getRoughnessTextureTransform() const;
  glm::vec4 getNormalTextureTransform() const;
  glm::vec4 getMetallicTextureTransform() const;
  glm::vec4 getEmissionTextureTransform() const;
  glm::vec4 getTransmissionTextureTransform() const;

  // TODO: getters
  std::shared_ptr<SVTexture> getBaseColorTexture() const;
  std::shared_ptr<SVTexture> getRoughnessTexture() const;
  std::shared_ptr<SVTexture> getNormalTexture() const;
  std::shared_ptr<SVTexture> getMetallicTexture() const;
  std::shared_ptr<SVTexture> getEmissionTexture() const;
  std::shared_ptr<SVTexture> getTransmissionTexture() const;

  void setTextures(std::shared_ptr<SVTexture> baseColorTexture,
                   std::shared_ptr<SVTexture> roughnessTexture,
                   std::shared_ptr<SVTexture> normalTexture,
                   std::shared_ptr<SVTexture> metallicTexture,
                   std::shared_ptr<SVTexture> emissionTexture,
                   std::shared_ptr<SVTexture> transmissionTexture);

  void uploadToDevice(core::Context &context) override;
  void removeFromDevice() override;
  vk::DescriptorSet getDescriptorSet(core::Context const &context) const override;
  inline float getOpacity() const override { return mBuffer.baseColor.a; }

  /** this device's uniform buffer; throws when the material is not uploaded there */
  core::Buffer &getDeviceBuffer(core::Context const &context) const;

  SVMetallicMaterial(SVMetallicMaterial const &) = delete;
  SVMetallicMaterial &operator=(SVMetallicMaterial const &) = delete;
  SVMetallicMaterial(SVMetallicMaterial const &&) = delete;
  SVMetallicMaterial &operator=(SVMetallicMaterial const &&) = delete;

  virtual ~SVMetallicMaterial();
};

} // namespace resource
} // namespace svulkan2
