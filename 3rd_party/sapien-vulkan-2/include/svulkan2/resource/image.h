#pragma once
#include "svulkan2/core/image.h"
#include "svulkan2/core/per_context.h"
#include <atomic>
#include <future>

struct ktxTexture;

namespace svulkan2 {
namespace resource {

struct SVImageDescription {
  enum class SourceType { eFILE, eCUSTOM, eKTX, eDEVICE } source{SourceType::eCUSTOM};
  vk::Format format{vk::Format::eUndefined};
  std::vector<std::string> filenames{};
  uint32_t desiredChannels{0};
  uint32_t mipLevels{1};

  inline bool operator==(SVImageDescription const &other) const {
    return source == other.source && filenames == other.filenames &&
           mipLevels == other.mipLevels && format == other.format;
  }
};

/** A device-independent image: host pixels (or a host KTX texture) uploaded lazily to every
 *  device (Context) that samples it. eDEVICE images have no host data and live on exactly
 *  the device that created them. */
class SVImage {
  SVImageDescription mDescription{};

  struct DeviceState {
    std::unique_ptr<core::Image> image;
  };
  core::PerContext<DeviceState> mDeviceStates;

  vk::ImageType mType{vk::ImageType::e2D};
  vk::Format mFormat{vk::Format::eUndefined};
  uint32_t mWidth{1};
  uint32_t mHeight{1};
  uint32_t mDepth{1};
  vk::ImageCreateFlags mCreateFlags{};
  vk::ImageUsageFlags mUsage{vk::ImageUsageFlagBits::eSampled |
                             vk::ImageUsageFlagBits::eTransferDst |
                             vk::ImageUsageFlagBits::eTransferSrc};

  /** host pixels per array layer */
  std::vector<std::vector<char>> mRawData;
  /** host KTX texture of an eKTX image, uploaded to each device through libktx */
  std::shared_ptr<ktxTexture> mKtxTexture;

  /** the image is on the host */
  std::atomic<bool> mLoaded{false};

  /** mipmap levels are provided on host */
  bool mMipLoaded{false};

  std::mutex mLoadingMutex;
  std::mutex mUploadingMutex;

public:
  static std::shared_ptr<SVImage> FromRawData(vk::ImageType type, uint32_t width, uint32_t height,
                                              uint32_t depth, vk::Format format,
                                              std::vector<std::vector<char>> const &data,
                                              uint32_t mipLevels = 1);

  static std::shared_ptr<SVImage> FromFile(std::vector<std::string> const &filenames,
                                           uint32_t mipLevels, uint32_t desiredChannels);

  /** take ownership of a loaded (and, if needed, transcoded) host KTX texture */
  static std::shared_ptr<SVImage> FromKTX(ktxTexture *texture);

  /** device-only image owned by image->getContext(); it cannot be uploaded elsewhere */
  static std::shared_ptr<SVImage> FromDeviceImage(std::unique_ptr<core::Image> image);

  /** must be called before the image is uploaded to any device */
  void setUsage(vk::ImageUsageFlags usage);
  void setCreateFlags(vk::ImageCreateFlags flags);

  /** create this device's image from the host data; no-op when already on the device */
  void uploadToDevice(core::Context &context, bool generateMipmaps = true);
  /** release the images of every device (host data is kept) */
  void removeFromDevice();
  inline bool isLoaded() const { return mLoaded; }
  bool isOnDevice(core::Context const &context) const;

  /** load the image to host memory */
  std::future<void> loadAsync();

  /** replace mip level 0 of array layer 0 on the host and on every device holding the image */
  void updateBaseLevel(void const *data, size_t size);

  /** this device's image, or nullptr when it is not uploaded there */
  core::Image *getDeviceImage(core::Context const &context) const;

  inline SVImageDescription const &getDescription() const { return mDescription; }

  std::vector<std::vector<char>> const &getRawData() const { return mRawData; }

  inline uint32_t getWidth() const { return mWidth; }
  inline uint32_t getHeight() const { return mHeight; }
  inline uint32_t getDepth() const { return mDepth; }
  inline uint32_t getChannels() const { return getFormatChannels(mFormat); }
  inline vk::Format getFormat() const { return mFormat; }
  inline uint32_t getMipLevels() const { return mDescription.mipLevels; }

  /** Indicate that the image loads mipmap into the data and does not require
   * generation */
  bool mipmapIsLoaded() const { return mMipLoaded; }

private:
  SVImage() = default;
};

} // namespace resource
} // namespace svulkan2
