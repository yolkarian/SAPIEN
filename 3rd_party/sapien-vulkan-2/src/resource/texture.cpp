#include "svulkan2/resource/texture.h"
#include "../common/logger.h"
#include "internal.h"
#include "svulkan2/common/launch_policy.h"
#include "svulkan2/core/context.h"
#include "svulkan2/resource/manager.h"
#include <algorithm>

namespace svulkan2 {
namespace resource {

std::shared_ptr<SVTexture> SVTexture::FromFile(std::string const &filename, uint32_t mipLevels,
                                               vk::Filter magFilter, vk::Filter minFilter,
                                               vk::SamplerAddressMode addressModeU,
                                               vk::SamplerAddressMode addressModeV, bool srgb,
                                               uint32_t desiredChannels) {
  auto texture = std::shared_ptr<SVTexture>(new SVTexture);
  texture->mDescription = SVTextureDescription{.source = SVTextureDescription::SourceType::eFILE,
                                               .filename = filename,
                                               .desiredChannels = desiredChannels,
                                               .mipLevels = mipLevels,
                                               .magFilter = magFilter,
                                               .minFilter = minFilter,
                                               .addressModeU = addressModeU,
                                               .addressModeV = addressModeV,
                                               .srgb = srgb,
                                               .dim = 2};
  return texture;
}

std::shared_ptr<SVTexture> SVTexture::FromRawData(uint32_t width, uint32_t height, uint32_t depth,
                                                  vk::Format format, std::vector<char> const &data,
                                                  int dim, uint32_t mipLevels,
                                                  vk::Filter magFilter, vk::Filter minFilter,
                                                  vk::SamplerAddressMode addressModeU,
                                                  vk::SamplerAddressMode addressModeV,
                                                  vk::SamplerAddressMode addressModeW, bool srgb) {
  if (srgb && !getFormatSupportSrgb(format)) {
    throw std::runtime_error("format does not support srgb");
  }

  if (dim <= 0 || dim > 3) {
    throw std::runtime_error("texture dimension must be 1, 2 or 3");
  }
  if (dim == 1 && (height != 1 || depth != 1)) {
    throw std::runtime_error("1D texture must have height = 1 and depth = 1");
  }
  if (dim == 2 && (depth != 1)) {
    throw std::runtime_error("2D texture must have depth = 1");
  }

  vk::ImageType type =
      dim == 1 ? vk::ImageType::e1D : (dim == 2 ? vk::ImageType::e2D : vk::ImageType::e3D);

  auto texture = std::shared_ptr<SVTexture>(new SVTexture);
  texture->mDescription = {.source = SVTextureDescription::SourceType::eCUSTOM,
                           .format = format,
                           .filename = {},
                           .mipLevels = mipLevels,
                           .magFilter = magFilter,
                           .minFilter = minFilter,
                           .addressModeU = addressModeU,
                           .addressModeV = addressModeV,
                           .addressModeW = addressModeW,
                           .srgb = srgb,
                           .dim = dim};
  texture->mImage = SVImage::FromRawData(type, width, height, depth, format, {data}, mipLevels);
  texture->mLoaded = true;
  return texture;
}

std::shared_ptr<SVTexture> SVTexture::FromImage(std::shared_ptr<SVImage> image,
                                                SVTextureDescription const &description) {
  if (!image) {
    throw std::runtime_error("failed to create texture: invalid image");
  }
  auto texture = std::shared_ptr<SVTexture>(new SVTexture);
  texture->mDescription = description;
  texture->mImage = std::move(image);
  texture->mLoaded = texture->mImage->isLoaded();
  return texture;
}

SVTexture::DeviceState const &SVTexture::getDeviceState(core::Context const &context) const {
  auto state = mDeviceStates.find(context);
  if (!state || !state->imageView) {
    throw std::runtime_error("failed to get texture: it is not uploaded to this device");
  }
  return *state;
}

vk::ImageView SVTexture::getImageView(core::Context const &context) const {
  return getDeviceState(context).imageView.get();
}

vk::Sampler SVTexture::getSampler(core::Context const &context) const {
  return getDeviceState(context).sampler;
}

bool SVTexture::isOnDevice(core::Context const &context) const {
  auto state = mDeviceStates.find(context);
  return state && state->imageView;
}

void SVTexture::uploadToDevice(core::Context &context) {
  std::scoped_lock lock(mUploadingMutex);
  auto existing = mDeviceStates.find(context);
  if (existing && existing->imageView) {
    return;
  }

  if (!mLoaded) {
    loadAsync().get();
  }
  if (!mImage) {
    throw std::runtime_error("failed to upload texture: no image data");
  }

  mImage->uploadToDevice(context);
  auto image = mImage->getDeviceImage(context);
  if (!image) {
    throw std::runtime_error("failed to upload texture: the image is not on this device");
  }

  auto format = image->getFormat();
  if (mDescription.srgb) {
    format = internal::toSrgbFormat(format);
  }

  vk::ImageViewType viewType = vk::ImageViewType::e2D;
  if (mDescription.dim == 1) {
    viewType = vk::ImageViewType::e1D;
  } else if (mDescription.dim == 3) {
    viewType = vk::ImageViewType::e3D;
  }

  auto &state = mDeviceStates.get(context);
  uint32_t mipLevels = std::min(mDescription.mipLevels, image->getMipLevels());
  if (mipLevels == 0) {
    mipLevels = 1;
  }
  state.context = context.shared_from_this();
  state.imageView = context.getDevice().createImageViewUnique(vk::ImageViewCreateInfo(
      {}, image->getVulkanImage(), viewType, format, mDescription.components,
      vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 1)));
  state.sampler = context.createSampler(vk::SamplerCreateInfo(
      {}, mDescription.magFilter, mDescription.minFilter, vk::SamplerMipmapMode::eLinear,
      mDescription.addressModeU, mDescription.addressModeV, mDescription.addressModeW, 0.f, false,
      0.f, false, vk::CompareOp::eNever, 0.f, static_cast<float>(mipLevels),
      vk::BorderColor::eFloatOpaqueBlack));
}

void SVTexture::removeFromDevice() {
  // the image may be shared with other textures (e.g. a channel split), so only this
  // texture's views go; the image's owner releases the image
  mDeviceStates.clear();
}

std::future<void> SVTexture::loadAsync() {
  if (mLoaded) {
    return std::async(std::launch::deferred, []() {});
  }

  if (!mImage && mDescription.source != SVTextureDescription::SourceType::eFILE) {
    throw std::runtime_error("failed to load texture: the texture is not specified by a file");
  }

  if (mDescription.source == SVTextureDescription::SourceType::eFILE) {
    logger::info("Loading: {}", mDescription.filename);
  }
  return std::async(LAUNCH_ASYNC, [this]() {
    std::scoped_lock lock(mLoadingMutex);
    if (mLoaded) {
      return;
    }
    if (!mImage) {
      mImage = SVResourceManager::Get()->CreateImageFromFile(
          mDescription.filename, mDescription.mipLevels, mDescription.desiredChannels);
    }
    mImage->loadAsync().get();
    mLoaded = true;
    if (mDescription.source == SVTextureDescription::SourceType::eFILE) {
      logger::info("Loaded: {}", mDescription.filename);
    }
  });
}

} // namespace resource
} // namespace svulkan2
