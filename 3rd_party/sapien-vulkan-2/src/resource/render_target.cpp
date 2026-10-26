#include "svulkan2/resource/render_target.h"
#include "svulkan2/core/context.h"
#include <stdexcept>

namespace svulkan2 {
namespace resource {

SVRenderTarget::SVRenderTarget(std::shared_ptr<core::Context> context, std::string const &name,
                               uint32_t width, uint32_t height, vk::Format format,
                               vk::SampleCountFlagBits sampleCount)
    : mContext(std::move(context)), mName(name), mFormat(format), mWidth(width), mHeight(height),
      mSampleCount(sampleCount) {
  if (!mContext) {
    throw std::runtime_error("failed to create render target: invalid context");
  }
}

SVRenderTarget::SVRenderTarget(std::shared_ptr<core::Context> context, std::string const &name,
                               uint32_t width, uint32_t height, std::shared_ptr<core::Image> image,
                               vk::UniqueImageView imageView, vk::Sampler sampler)
    : mContext(std::move(context)), mName(name),
      mFormat(image ? image->getFormat() : vk::Format::eUndefined), mWidth(width), mHeight(height),
      mImage(std::move(image)), mImageView(std::move(imageView)), mSampler(sampler) {
  if (!mContext) {
    throw std::runtime_error("failed to create render target: invalid context");
  }
  if (!mImage) {
    throw std::runtime_error("failed to create render target: invalid image");
  }
  if (mImage->getContext() != mContext) {
    throw std::runtime_error(
        "failed to create render target: the image belongs to another device");
  }
}

void SVRenderTarget::createDeviceResources() {
  if (mImage) {
    return;
  }

  bool isDepth = false;
  vk::ImageUsageFlags usage;
  if (mFormat == vk::Format::eR8G8B8A8Unorm ||

      mFormat == vk::Format::eR32G32B32A32Sfloat || mFormat == vk::Format::eR16G16B16A16Sfloat ||

      mFormat == vk::Format::eR32G32B32A32Uint || mFormat == vk::Format::eR32G32B32A32Sint ||

      mFormat == vk::Format::eR16G16B16A16Uint || mFormat == vk::Format::eR16G16B16A16Sint ||

      mFormat == vk::Format::eR32Sfloat || mFormat == vk::Format::eR16Sfloat) {
    usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment |
            vk::ImageUsageFlagBits::eTransferSrc;
  } else if (mFormat == vk::Format::eD32Sfloat) {
    usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eDepthStencilAttachment |
            vk::ImageUsageFlagBits::eTransferSrc;
    isDepth = true;
  } else {
    throw std::runtime_error("failed to create image resources: unsupported image format.");
  }

  mImage = std::make_shared<core::Image>(
      *mContext, vk::ImageType::e2D, vk::Extent3D{mWidth, mHeight, 1}, mFormat, usage,
      VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, mSampleCount, 1);
  vk::ComponentMapping componentMapping(vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
                                        vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA);

  vk::ImageViewCreateInfo info({}, mImage->getVulkanImage(), vk::ImageViewType::e2D, mFormat,
                               componentMapping,
                               vk::ImageSubresourceRange(isDepth ? vk::ImageAspectFlagBits::eDepth
                                                                 : vk::ImageAspectFlagBits::eColor,
                                                         0, 1, 0, 1));
  mImageView = mContext->getDevice().createImageViewUnique(info);
  mSampler = mContext->createSampler(vk::SamplerCreateInfo(
      {}, vk::Filter::eNearest, vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest,
      vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
      vk::SamplerAddressMode::eClampToEdge, 0.f, false, 0.f, false, vk::CompareOp::eNever, 0.f,
      0.f, vk::BorderColor::eFloatOpaqueBlack));
}

} // namespace resource
} // namespace svulkan2
