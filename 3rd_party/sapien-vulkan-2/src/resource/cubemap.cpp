#include "svulkan2/resource/cubemap.h"
#include "../common/logger.h"
#include "internal.h"
#include "svulkan2/common/launch_policy.h"
#include "svulkan2/core/context.h"
#include "svulkan2/shader/compute.h"
#include <algorithm>
#include <ktx.h>

namespace svulkan2 {
namespace resource {

SVCubemap::DeviceState const &SVCubemap::getDeviceState(core::Context const &context) const {
  auto state = mDeviceStates.find(context);
  if (!state || !state->imageView) {
    throw std::runtime_error("failed to get cubemap: it is not uploaded to this device");
  }
  return *state;
}

vk::ImageView SVCubemap::getImageView(core::Context const &context) const {
  return getDeviceState(context).imageView.get();
}

vk::Sampler SVCubemap::getSampler(core::Context const &context) const {
  return getDeviceState(context).sampler;
}

bool SVCubemap::isOnDevice(core::Context const &context) const {
  auto state = mDeviceStates.find(context);
  return state && state->imageView;
}

std::shared_ptr<SVCubemap> SVCubemap::FromFile(std::string const &filename, uint32_t mipLevels,
                                               vk::Filter magFilter, vk::Filter minFilter,
                                               bool srgb) {
  auto texture = std::shared_ptr<SVCubemap>(new SVCubemap);
  auto source = (filename.ends_with(".ktx") || filename.ends_with(".KTX"))
                    ? SVCubemapDescription::SourceType::eKTX
                    : SVCubemapDescription::SourceType::eLATLONG;
  texture->mDescription = {.source = source,
                           .filenames = {filename, "", "", "", "", ""},
                           .mipLevels = mipLevels,
                           .magFilter = magFilter,
                           .minFilter = minFilter,
                           .srgb = srgb};
  return texture;
}

std::shared_ptr<SVCubemap> SVCubemap::FromFile(std::array<std::string, 6> const &filenames,
                                               uint32_t mipLevels, vk::Filter magFilter,
                                               vk::Filter minFilter, bool srgb) {
  auto texture = std::shared_ptr<SVCubemap>(new SVCubemap);
  texture->mDescription = {.source = SVCubemapDescription::SourceType::eFACES,
                           .filenames = filenames,
                           .mipLevels = mipLevels,
                           .magFilter = magFilter,
                           .minFilter = minFilter,
                           .srgb = srgb};
  return texture;
}

std::shared_ptr<SVCubemap> SVCubemap::FromData(uint32_t size, vk::Format format,
                                               std::array<std::vector<char>, 6> const &data,
                                               uint32_t mipLevels, vk::Filter magFilter,
                                               vk::Filter minFilter, bool srgb) {
  auto texture = std::shared_ptr<SVCubemap>(new SVCubemap);
  texture->mDescription = {.source = SVCubemapDescription::SourceType::eCUSTOM,
                           .filenames = {},
                           .mipLevels = mipLevels,
                           .magFilter = magFilter,
                           .minFilter = minFilter,
                           .srgb = srgb};
  std::vector<std::vector<char>> vdata(data.begin(), data.end());
  texture->mImage =
      SVImage::FromRawData(vk::ImageType::e2D, size, size, 1, format, vdata, mipLevels);
  texture->mImage->setCreateFlags(vk::ImageCreateFlagBits::eCubeCompatible);
  texture->mLoaded = true;
  return texture;
}

void SVCubemap::uploadToDevice(core::Context &context) {
  std::scoped_lock lock(mUploadingMutex);

  auto existing = mDeviceStates.find(context);
  if (existing && existing->imageView) {
    return;
  }

  loadAsync().get();
  if (!mImage && !mLatLongImage) {
    throw std::runtime_error("failed to upload cubemap: not loaded");
  }

  // the cube image of this device: converted from lat-long, or uploaded from the host faces
  core::Image *cube = nullptr;
  bool mipsProvided = mImage && mImage->mipmapIsLoaded();
  if (!mImage) {
    mLatLongImage->uploadToDevice(context);
    auto latLong = mLatLongImage->getDeviceImage(context);
    if (!latLong) {
      throw std::runtime_error("failed to upload cubemap: the lat-long image was not uploaded");
    }
    auto converted = shader::latlongToCube(context, *latLong, mDescription.mipLevels);
    cube = converted.get();
    mDeviceStates.get(context).latLongCube = std::move(converted);
  } else {
    // the host image is shared by every device: set usage before the first upload
    if (!mImageUsageSet) {
      mImage->setUsage(vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst |
                       vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eStorage);
      mImageUsageSet = true;
    }
    mImage->uploadToDevice(context, false);
    cube = mImage->getDeviceImage(context);
    if (!cube) {
      throw std::runtime_error("failed to upload cubemap: the image was not uploaded");
    }
  }

  auto viewFormat = cube->getFormat();
  if (mDescription.srgb) {
    viewFormat = internal::toSrgbFormat(viewFormat);
  }

  uint32_t mipLevels = std::min(mDescription.mipLevels, cube->getMipLevels());
  if (mipLevels == 0) {
    mipLevels = 1;
  }

  vk::ImageViewUsageCreateInfo usageInfo(vk::ImageUsageFlagBits::eSampled);
  vk::ImageViewCreateInfo viewInfo(
      {}, cube->getVulkanImage(), vk::ImageViewType::eCube, viewFormat,
      vk::ComponentSwizzle::eIdentity,
      vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 6));
  viewInfo.setPNext(&usageInfo);

  auto &state = mDeviceStates.get(context);
  state.context = context.shared_from_this();
  state.imageView = context.getDevice().createImageViewUnique(viewInfo);
  state.sampler = context.createSampler(vk::SamplerCreateInfo(
      {}, mDescription.magFilter, mDescription.minFilter, vk::SamplerMipmapMode::eLinear,
      vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode::eRepeat, 0.f, false, 0.f, false, vk::CompareOp::eNever, 0.f,
      static_cast<float>(mipLevels), vk::BorderColor::eFloatOpaqueBlack));

  // levels beyond the base one are written by the prefilter pass; loaded mips are kept
  if (mipLevels > 1 && !mipsProvided) {
    logger::info("Prefiltering cube map...");
    shader::prefilterCubemap(context, *cube);
    logger::info("Prefiltering cube map completed");
  }
  cube->setCurrentLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
}

void SVCubemap::removeFromDevice() { mDeviceStates.clear(); }

std::future<void> SVCubemap::loadAsync() {
  if (mLoaded) {
    return std::async(std::launch::deferred, []() {});
  }
  switch (mDescription.source) {
  case SVCubemapDescription::SourceType::eFACES:
    for (auto f : mDescription.filenames) {
      logger::info("Loading: {}", f);
    }
    break;
  case (SVCubemapDescription::SourceType::eKTX):
  case (SVCubemapDescription::SourceType::eLATLONG):
    logger::info("Loading: {}", mDescription.filenames[0]);
    break;
  default:
    throw std::runtime_error("failed to load cubemap: not specified by a file");
  }

  return std::async(LAUNCH_ASYNC, [this]() {
    std::scoped_lock lock(mLoadingMutex);
    if (mLoaded) {
      return;
    }
    switch (mDescription.source) {
    case SVCubemapDescription::SourceType::eFACES: {
      auto vfiles =
          std::vector<std::string>(mDescription.filenames.begin(), mDescription.filenames.end());
      mImage = SVImage::FromFile(vfiles, mDescription.mipLevels, 4);
      mImage->setCreateFlags(vk::ImageCreateFlagBits::eCubeCompatible);
      mImage->loadAsync().get();
      break;
    }
    case SVCubemapDescription::SourceType::eKTX: {
      // libktx creates a cube-compatible image for a cube map
      mImage = internal::imageFromFile(mDescription.filenames[0]);
      if (mImage->getMipLevels() != mDescription.mipLevels) {
        logger::warn("cube map ktx has {} levels but {} levels requested", mImage->getMipLevels(),
                     mDescription.mipLevels);
      }
      break;
    }
    case SVCubemapDescription::SourceType::eLATLONG: {
      mLatLongImage = SVImage::FromFile({mDescription.filenames[0]}, 1, 4);
      mLatLongImage->loadAsync().get();
      break;
    }
    default:
      throw std::runtime_error("failed to load cubemap: not specified by a file.");
    };

    mLoaded = true;
  });
}

void SVCubemap::load() { loadAsync().get(); }

void SVCubemap::exportKTX(core::Context &context, std::string const &filename) {
  auto state = mDeviceStates.find(context);
  if (!state || !state->imageView) {
    throw std::runtime_error("failed to export KTX: the cubemap is not uploaded to this device");
  }

  core::Image *img = state->latLongCube ? state->latLongCube.get()
                                        : (mImage ? mImage->getDeviceImage(context) : nullptr);
  if (!img) {
    throw std::runtime_error("failed to export KTX: no cube image on this device");
  }
  auto extent = img->getExtent();

  if (img->getFormat() != vk::Format::eR8G8B8A8Unorm) {
    throw std::runtime_error("exporting to ktx only supports R8G8B8A8Unorm texture for now");
  }

  ktxTexture2 *texture{};
  ktxTextureCreateInfo createInfo;

  createInfo.vkFormat = VK_FORMAT_R8G8B8A8_UNORM;
  createInfo.baseWidth = extent.width;
  createInfo.baseHeight = extent.height;
  createInfo.baseDepth = 1;
  createInfo.numDimensions = 2;
  createInfo.numLevels = img->getMipLevels();
  createInfo.numLayers = 1;
  createInfo.numFaces = 6;
  createInfo.isArray = KTX_FALSE;
  createInfo.generateMipmaps = KTX_FALSE;

  auto result = ktxTexture2_Create(&createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &texture);
  if (result != KTX_SUCCESS) {
    throw std::runtime_error("failed to export KTX: cannot create texture: " +
                             std::string(ktxErrorString(result)));
  }
  std::vector<uint8_t> data(static_cast<size_t>(extent.width) * extent.height * 4);

  for (uint32_t face = 0; face < 6; ++face) {
    uint32_t width = extent.width;
    uint32_t height = extent.height;
    for (uint32_t level = 0; level < img->getMipLevels(); ++level) {
      uint32_t size = width * height * 4;
      img->download(data.data(), size, {0, 0, 0}, {width, height, 1}, face, level);
      ktxTexture_SetImageFromMemory(ktxTexture(texture), level, 0, face, data.data(), size);
      width /= 2;
      height /= 2;
      if (width == 0) {
        width = 1;
      }
      if (height == 0) {
        height = 1;
      }
    }
  }
  ktxTexture_WriteToNamedFile(ktxTexture(texture), filename.c_str());
  ktxTexture_Destroy(ktxTexture(texture));

  auto pool = context.createCommandPool();
  auto buffer = pool->allocateCommandBuffer();
  buffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  img->transitionLayout(buffer.get(), vk::ImageLayout::eTransferSrcOptimal,
                        vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eTransferRead,
                        vk::AccessFlagBits::eShaderRead, vk::PipelineStageFlagBits::eTransfer,
                        internal::shaderReadStages(context));
  buffer->end();
  context.getQueue().submitAndWait(buffer.get());
}

} // namespace resource
} // namespace svulkan2
