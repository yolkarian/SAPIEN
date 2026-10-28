#pragma once
#include "image.h"
#include <atomic>
#include <memory>
#include <vector>

namespace svulkan2 {
namespace resource {

struct SVTextureDescription {
  enum class SourceType { eFILE, eCUSTOM } source{SourceType::eCUSTOM};

  vk::Format format{vk::Format::eUndefined};

  std::string filename{};
  uint32_t desiredChannels{0}; // affect loaded file, 0 for auto
  uint32_t mipLevels{1};
  vk::Filter magFilter{vk::Filter::eNearest};
  vk::Filter minFilter{vk::Filter::eNearest};
  vk::SamplerAddressMode addressModeU{vk::SamplerAddressMode::eRepeat};
  vk::SamplerAddressMode addressModeV{vk::SamplerAddressMode::eRepeat};
  vk::SamplerAddressMode addressModeW{vk::SamplerAddressMode::eRepeat};
  bool srgb{false};
  int dim{2};
  /** swizzle of the image view, e.g. to sample a single channel of a shared image */
  vk::ComponentMapping components{};

  inline bool operator==(SVTextureDescription const &other) const {
    return source == other.source && filename == other.filename && format == other.format &&
           mipLevels == other.mipLevels && magFilter == other.magFilter &&
           minFilter == other.minFilter && addressModeU == other.addressModeU &&
           addressModeV == other.addressModeV && addressModeW == other.addressModeW &&
           srgb == other.srgb && dim == other.dim && components == other.components;
  }
};

/** A device-independent sampled texture; its image view and sampler are created lazily on
 *  every device (Context) that samples it. */
class SVTexture {
public:
  static std::shared_ptr<SVTexture>
  FromFile(std::string const &filename, uint32_t mipLevels,
           vk::Filter magFilter = vk::Filter::eLinear, vk::Filter minFilter = vk::Filter::eLinear,
           vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
           vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat,
           bool srgb = false, uint32_t desiredChannels = 0);

  static std::shared_ptr<SVTexture> FromRawData(
      uint32_t width, uint32_t height, uint32_t depth, vk::Format format,
      std::vector<char> const &data, int dim, uint32_t mipLevels = 1,
      vk::Filter magFilter = vk::Filter::eLinear, vk::Filter minFilter = vk::Filter::eLinear,
      vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode addressModeW = vk::SamplerAddressMode::eRepeat, bool srgb = false);

  /** a texture sampling an existing (possibly shared) image with the sampler, mip levels and
   *  view swizzle of `description`; a loaded image counts as loaded */
  static std::shared_ptr<SVTexture> FromImage(std::shared_ptr<SVImage> image,
                                              SVTextureDescription const &description);

  /** upload the image and create this device's view and sampler; no-op when present */
  void uploadToDevice(core::Context &context);
  /** release the views and images of every device */
  void removeFromDevice();

  bool isOnDevice(core::Context const &context) const;
  inline bool isLoaded() const { return mLoaded; }
  std::future<void> loadAsync();

  inline SVTextureDescription const &getDescription() const { return mDescription; }

  inline std::shared_ptr<SVImage> getImage() const { return mImage; }
  /** this device's view and sampler; throw when the texture is not uploaded there */
  vk::ImageView getImageView(core::Context const &context) const;
  vk::Sampler getSampler(core::Context const &context) const;

  SVTexture(SVTexture const &other) = delete;
  SVTexture &operator=(SVTexture const &other) = delete;
  SVTexture(SVTexture &&other) = delete;
  SVTexture &operator=(SVTexture &&other) = delete;
  inline SVTexture() {}

private:
  struct DeviceState {
    std::shared_ptr<core::Context> context; // keeps the device alive for the view
    vk::UniqueImageView imageView;
    vk::Sampler sampler;
  };
  DeviceState const &getDeviceState(core::Context const &context) const;

  SVTextureDescription mDescription{};
  std::shared_ptr<SVImage> mImage{};
  core::PerContext<DeviceState> mDeviceStates;

  std::atomic<bool> mLoaded{false};

  std::mutex mUploadingMutex;
  std::mutex mLoadingMutex;
};

} // namespace resource
} // namespace svulkan2
