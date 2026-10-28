#include "svulkan2/resource/image.h"
#include "../common/logger.h"
#include "internal.h"
#include "svulkan2/common/image.h"
#include "svulkan2/common/launch_policy.h"
#include "svulkan2/core/context.h"
#include <cstring>
#include <ktx.h>
#include <ktxvulkan.h>

namespace svulkan2 {
namespace resource {

namespace {

/** upload the host ktx texture to `context` with `usage`; the caller owns the returned struct */
std::unique_ptr<ktxVulkanTexture> uploadKtxToDevice(core::Context &context, ktxTexture *texture,
                                                    vk::ImageUsageFlags usage) {
  auto pool = context.createCommandPool();

  ktxVulkanFunctions funcs{};
  funcs.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
  funcs.vkGetDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

  ktxVulkanDeviceInfo vdi{};
  auto result = ktxVulkanDeviceInfo_ConstructEx(
      &vdi, context.getInstance(), context.getPhysicalDevice(), context.getDevice(),
      context.getQueue().getVulkanQueue(), pool->getVulkanCommandPool(), nullptr, &funcs);
  if (result != KTX_SUCCESS) {
    throw std::runtime_error("failed to construct ktx vulkan device: " +
                             std::string(ktxErrorString(result)));
  }

  ktxVulkanTexture vkTexture{};
  {
    auto queueLock = context.getQueue().lock(); // libktx submits to the raw queue and waits on it
    result = ktxTexture_VkUploadEx(texture, &vdi, &vkTexture, VK_IMAGE_TILING_OPTIMAL,
                                   static_cast<VkImageUsageFlags>(usage),
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
  ktxVulkanDeviceInfo_Destruct(&vdi);
  if (result != KTX_SUCCESS) {
    throw std::runtime_error("failed to upload ktx texture: " +
                             std::string(ktxErrorString(result)));
  }
  return std::make_unique<ktxVulkanTexture>(vkTexture);
}

} // namespace

std::shared_ptr<SVImage> SVImage::FromRawData(vk::ImageType type, uint32_t width, uint32_t height,
                                              uint32_t depth, vk::Format format,
                                              std::vector<std::vector<char>> const &data,
                                              uint32_t mipLevels) {
  size_t imageSize = width * height * depth;
  for (auto &d : data) {
    if (imageSize * getFormatSize(format) != d.size()) {
      throw std::runtime_error(
          "failed to create image: image data size does not match image dimensions");
    }
  }
  auto image = std::shared_ptr<SVImage>(new SVImage);
  image->mDescription = {.source = SVImageDescription::SourceType::eCUSTOM,
                         .format = format,
                         .filenames = {},
                         .mipLevels = mipLevels};

  image->mType = type;
  image->mFormat = format;
  image->mWidth = width;
  image->mHeight = height;
  image->mDepth = depth;
  image->mRawData = data;
  image->mLoaded = true;
  return image;
}

std::shared_ptr<SVImage> SVImage::FromFile(std::vector<std::string> const &filenames,
                                           uint32_t mipLevels, uint32_t desiredChannels) {
  auto image = std::shared_ptr<SVImage>(new SVImage);
  image->mDescription = SVImageDescription{.source = SVImageDescription::SourceType::eFILE,
                                           .filenames = filenames,
                                           .desiredChannels = desiredChannels,
                                           .mipLevels = mipLevels};
  return image;
}

std::shared_ptr<SVImage> SVImage::FromKTX(ktxTexture *texture) {
  if (!texture) {
    throw std::runtime_error("failed to create image: invalid ktx texture");
  }
  // the vulkan format of a ktx1 texture is only derived on request
  auto format = static_cast<vk::Format>(ktxTexture_GetVkFormat(texture));
  if (format == vk::Format::eUndefined) {
    ktxTexture_Destroy(texture);
    throw std::runtime_error("failed to create image: the ktx texture has no vulkan format");
  }

  auto image = std::shared_ptr<SVImage>(new SVImage);
  image->mKtxTexture = internal::wrap(texture);
  image->mDescription = {.source = SVImageDescription::SourceType::eKTX,
                         .format = format,
                         .filenames = {},
                         .mipLevels = texture->numLevels};

  image->mType = texture->numDimensions == 1
                     ? vk::ImageType::e1D
                     : (texture->baseDepth > 1 ? vk::ImageType::e3D : vk::ImageType::e2D);
  image->mFormat = format;
  image->mWidth = texture->baseWidth;
  image->mHeight = texture->baseHeight;
  image->mDepth = texture->baseDepth;
  image->mLoaded = true;
  image->mMipLoaded = true;
  return image;
}

std::shared_ptr<SVImage> SVImage::FromDeviceImage(std::unique_ptr<core::Image> image) {
  if (!image) {
    throw std::runtime_error("failed to create image: invalid device image");
  }
  auto context = image->getContext();
  if (!context) {
    throw std::runtime_error("failed to create image: the device image has no context");
  }

  auto result = std::shared_ptr<SVImage>(new SVImage);
  result->mDescription = SVImageDescription{.source = SVImageDescription::SourceType::eDEVICE,
                                            .format = image->getFormat(),
                                            .mipLevels = image->getMipLevels()};
  result->mType = image->getType();
  result->mFormat = image->getFormat();
  result->mWidth = image->getExtent().width;
  result->mHeight = image->getExtent().height;
  result->mDepth = image->getExtent().depth;
  result->mUsage = image->getUsage();
  result->mLoaded = true;
  result->mMipLoaded = true;
  result->mDeviceStates.get(*context).image = std::move(image);
  return result;
}

void SVImage::setUsage(vk::ImageUsageFlags usage) {
  bool onDevice = false;
  mDeviceStates.forEach([&](DeviceState &state) {
    if (state.image) {
      onDevice = true;
    }
  });
  if (onDevice) {
    throw std::runtime_error("failed to set usage: device image already created");
  }
  mUsage = usage;
}

void SVImage::setCreateFlags(vk::ImageCreateFlags flags) {
  bool onDevice = false;
  mDeviceStates.forEach([&](DeviceState &state) {
    if (state.image) {
      onDevice = true;
    }
  });
  if (onDevice) {
    throw std::runtime_error("failed to set create flags: device image already created");
  }
  mCreateFlags = flags;
}

void SVImage::uploadToDevice(core::Context &context, bool generateMipmaps) {
  std::scoped_lock lock(mUploadingMutex);

  auto state = mDeviceStates.find(context);
  if (state && state->image) {
    return;
  }

  if (mDescription.source == SVImageDescription::SourceType::eDEVICE) {
    throw std::runtime_error("failed to upload to device: the image is bound to another device");
  }

  if (!mLoaded) {
    throw std::runtime_error("failed to upload to device: image does not exist in memory");
  }

  if (mFormat == vk::Format::eUndefined) {
    throw std::runtime_error("failed to upload to device: image format is not determined");
  }

  // the host ktx texture is uploaded by libktx on this device
  if (mKtxTexture) {
    mDeviceStates.get(context).image = std::make_unique<core::Image>(
        context, uploadKtxToDevice(context, mKtxTexture.get(), mUsage));
    return;
  }

  if (mRawData.empty()) {
    throw std::runtime_error("failed to upload to device: image has no host data");
  }

  auto image = std::make_unique<core::Image>(
      context, mType, vk::Extent3D{mWidth, mHeight, mDepth}, mFormat, mUsage,
      VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, vk::SampleCountFlagBits::e1,
      mDescription.mipLevels, mRawData.size(), vk::ImageTiling::eOptimal,
      mCreateFlags | vk::ImageCreateFlagBits::eMutableFormat);

  if (mMipLoaded) {
    // upload all levels if mipmap is provided
    auto pool = context.createCommandPool();
    auto cb = pool->allocateCommandBuffer();
    cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    image->transitionLayout(
        cb.get(), vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, {},
        vk::AccessFlagBits::eTransferWrite, vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTransfer);
    cb->end();
    context.getQueue().submitAndWait(cb.get());

    for (uint32_t layer = 0; layer < mRawData.size(); ++layer) {
      uint32_t idx = 0;
      for (uint32_t l = 0; l < mDescription.mipLevels; ++l) {
        auto size = computeMipLevelSize({mWidth, mHeight, mDepth}, l) * getFormatSize(mFormat);
        image->uploadLevel(mRawData[layer].data() + idx, size, layer, l);
        idx += size;
      }
    }

    cb = pool->allocateCommandBuffer();
    cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    image->transitionLayout(
        cb.get(), vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
        vk::PipelineStageFlagBits::eTransfer, internal::shaderReadStages(context));
    cb->end();
    context.getQueue().submitAndWait(cb.get());
  } else {
    for (uint32_t layer = 0; layer < mRawData.size(); ++layer) {
      image->upload(mRawData[layer].data(), mRawData[layer].size(), layer, generateMipmaps);
    }
  }

  mDeviceStates.get(context).image = std::move(image);
}

void SVImage::removeFromDevice() {
  // a device-only image has no host copy: it cannot be re-created, so it is kept
  if (mDescription.source == SVImageDescription::SourceType::eDEVICE) {
    return;
  }
  mDeviceStates.clear();
}

bool SVImage::isOnDevice(core::Context const &context) const {
  auto state = mDeviceStates.find(context);
  return state && state->image;
}

core::Image *SVImage::getDeviceImage(core::Context const &context) const {
  auto state = mDeviceStates.find(context);
  return state && state->image ? state->image.get() : nullptr;
}

void SVImage::updateBaseLevel(void const *data, size_t size) {
  size_t baseSize = static_cast<size_t>(mWidth) * mHeight * mDepth * getFormatSize(mFormat);
  if (size != baseSize) {
    throw std::runtime_error("failed to update image: expecting size " + std::to_string(baseSize) +
                             ", got " + std::to_string(size));
  }

  std::scoped_lock lock(mUploadingMutex);
  if (mKtxTexture || mRawData.empty() || mRawData[0].size() < size) {
    throw std::runtime_error("failed to update image: the image has no host pixel data");
  }

  std::memcpy(mRawData[0].data(), data, size);

  mDeviceStates.forEach([&](DeviceState &state) {
    if (!state.image) {
      return;
    }
    auto &image = *state.image;
    auto context = image.getContext();
    auto pool = context->createCommandPool();

    auto cb = pool->allocateCommandBuffer();
    cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    image.transitionLayout(
        cb.get(), vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferDstOptimal,
        vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eTransferWrite,
        internal::shaderReadStages(*context), vk::PipelineStageFlagBits::eTransfer, 0);
    cb->end();
    context->getQueue().submitAndWait(cb.get());

    image.uploadLevel(data, size, 0, 0);

    cb = pool->allocateCommandBuffer();
    cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    image.transitionLayout(
        cb.get(), vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
        vk::PipelineStageFlagBits::eTransfer, internal::shaderReadStages(*context), 0);
    cb->end();
    context->getQueue().submitAndWait(cb.get());
  });
}

std::future<void> SVImage::loadAsync() {
  if (mLoaded) {
    return std::async(std::launch::deferred, []() {});
  }
  return std::async(LAUNCH_ASYNC, [this]() {
    std::scoped_lock lock(mLoadingMutex);
    if (mLoaded) {
      return;
    }
    if (mDescription.source != SVImageDescription::SourceType::eFILE) {
      throw std::runtime_error("failed to load image: the image is not specified with a file");
    }
    if (mDescription.filenames.empty()) {
      throw std::runtime_error("failed to load image: no file is specified");
    }

    // load ktx file: keep the host ktx texture, upload it to each device with libktx
    if (mDescription.filenames[0].ends_with(".ktx") ||
        mDescription.filenames[0].ends_with(".KTX")) {
      auto image = internal::imageFromFile(mDescription.filenames[0]);
      if (mDescription.mipLevels != image->getMipLevels()) {
        logger::warn("ktx texture has {} levels but {} levels requested", image->getMipLevels(),
                     mDescription.mipLevels);
      }
      mKtxTexture = image->mKtxTexture;
      mType = image->mType;
      mFormat = image->mFormat;
      mWidth = image->mWidth;
      mHeight = image->mHeight;
      mDepth = image->mDepth;
      mMipLoaded = true;
      mLoaded = true;
      return;
    }

    // every file must match the first one's size and format
    mWidth = 0;
    mHeight = 0;
    for (uint32_t i = 0; i < mDescription.filenames.size(); ++i) {
      int width, height, channels;

      vk::Format format{vk::Format::eUndefined};
      if (mDescription.filenames[i].ends_with(".exr")) {
        // exr
        auto rawData = loadExrImage(mDescription.filenames[i], width, height);
        format = vk::Format::eR16G16B16A16Sfloat;
        channels = 4;

        mRawData.push_back(toRawBytes(rawData));
      } else {
        // png, jpg
        std::vector<uint8_t> rawData = loadImage(mDescription.filenames[i], width, height,
                                                 channels, mDescription.desiredChannels);

        switch (channels) {
        case 1:
          format = vk::Format::eR8Unorm;
          break;
        case 2:
          format = vk::Format::eR8G8Unorm;
          break;
        case 3:
          format = vk::Format::eR8G8B8Unorm;
          break;
        case 4:
          format = vk::Format::eR8G8B8A8Unorm;
          break;
        default:
          throw std::runtime_error("invalid image channels");
        }

        mRawData.push_back(toRawBytes(rawData));
      }

      if ((mWidth != 0 && mWidth != static_cast<uint32_t>(width)) ||
          (mHeight != 0 && mHeight != static_cast<uint32_t>(height)) ||
          (mFormat != vk::Format::eUndefined && mFormat != format)) {
        throw std::runtime_error(
            "image load failed: provided files have different sizes or formats");
      }

      mFormat = format;
      mWidth = width;
      mHeight = height;
    }
    mLoaded = true;
  });
}

} // namespace resource
} // namespace svulkan2
