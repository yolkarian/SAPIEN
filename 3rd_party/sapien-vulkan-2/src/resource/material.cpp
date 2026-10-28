#include "svulkan2/resource/material.h"
#include "svulkan2/core/buffer.h"
#include "svulkan2/core/context.h"
#include "svulkan2/resource/manager.h"

namespace svulkan2 {
namespace resource {

static void updateDescriptorSets(
    vk::Device device, vk::DescriptorSet descriptorSet,
    std::vector<std::tuple<vk::DescriptorType, vk::Buffer, vk::BufferView>> const &bufferData,
    std::vector<std::tuple<vk::ImageView, vk::Sampler>> textureData, uint32_t bindingOffset) {
  std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
  writeDescriptorSets.reserve(bufferData.size() + textureData.size());

  std::vector<vk::DescriptorBufferInfo> bufferInfos;
  bufferInfos.reserve(bufferData.size());
  for (auto const &bd : bufferData) {
    bufferInfos.push_back(vk::DescriptorBufferInfo(std::get<1>(bd), 0, VK_WHOLE_SIZE));
    writeDescriptorSets.push_back(
        vk::WriteDescriptorSet(descriptorSet, bindingOffset++, 0, 1, std::get<0>(bd), nullptr,
                               &bufferInfos.back(), std::get<2>(bd) ? &std::get<2>(bd) : nullptr));
  }
  std::vector<vk::DescriptorImageInfo> imageInfos;
  imageInfos.reserve(textureData.size());
  for (auto const &tex : textureData) {
    imageInfos.push_back(vk::DescriptorImageInfo(std::get<1>(tex), std::get<0>(tex),
                                                 vk::ImageLayout::eShaderReadOnlyOptimal));
    writeDescriptorSets.push_back(
        vk::WriteDescriptorSet(descriptorSet, bindingOffset++, 0, 1,
                               vk::DescriptorType::eCombinedImageSampler, &imageInfos.back()));
  }
  device.updateDescriptorSets(writeDescriptorSets, nullptr);
}

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

SVMetallicMaterial::DeviceState const &
SVMetallicMaterial::getDeviceState(core::Context const &context) const {
  auto state = mDeviceStates.find(context);
  if (!state || !state->descriptorSet) {
    throw std::runtime_error("failed to get material: it is not uploaded to this device");
  }
  return *state;
}

vk::DescriptorSet SVMetallicMaterial::getDescriptorSet(core::Context const &context) const {
  return getDeviceState(context).descriptorSet.get();
}

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

// Edits reach every device that already holds the material, so renderers whose recorded
// commands bind its buffer and descriptor set see them without re-recording.
void SVMetallicMaterial::bufferChanged() {
  ++mBufferVersion;
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

  if (!state.descriptorSet) {
    state.descriptorSet =
        context.getDescriptorPool().allocateSet(context.getMetallicDescriptorSetLayout());
  }

  if (state.bufferVersion != mBufferVersion) {
    state.buffer->upload(mBuffer);
    state.bufferVersion = mBufferVersion;
  }

  if (state.textureVersion != mTextureVersion) {
    auto manager = SVResourceManager::Get();
    auto defaultTexture = manager->getDefaultTexture();

    auto bindTexture = [&](std::shared_ptr<SVTexture> const &texture) {
      auto t = texture ? texture : defaultTexture;
      t->uploadToDevice(context);
      return std::tuple<vk::ImageView, vk::Sampler>{t->getImageView(context),
                                                    t->getSampler(context)};
    };

    std::vector<std::tuple<vk::ImageView, vk::Sampler>> textures;
    textures.reserve(6);
    textures.push_back(bindTexture(mBaseColorTexture));
    textures.push_back(bindTexture(mRoughnessTexture));
    textures.push_back(bindTexture(mNormalTexture));
    textures.push_back(bindTexture(mMetallicTexture));
    textures.push_back(bindTexture(mEmissionTexture));
    textures.push_back(bindTexture(mTransmissionTexture));

    updateDescriptorSets(
        context.getDevice(), state.descriptorSet.get(),
        {{vk::DescriptorType::eUniformBuffer, state.buffer->getVulkanBuffer(), nullptr}}, textures,
        0);
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
