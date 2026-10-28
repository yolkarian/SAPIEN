#pragma once
#include "image.h"
#include <atomic>
#include <memory>
#include <vector>

namespace svulkan2 {
namespace resource {

struct SVCubemapDescription {
  enum class SourceType { eFACES, eKTX, eLATLONG, eCUSTOM } source{SourceType::eCUSTOM};
  std::array<std::string, 6> filenames{};
  uint32_t mipLevels{1};
  vk::Filter magFilter{vk::Filter::eLinear};
  vk::Filter minFilter{vk::Filter::eLinear};
  bool srgb{false};

  inline bool operator==(SVCubemapDescription const &other) const {
    return source == other.source && filenames == other.filenames &&
           mipLevels == other.mipLevels && magFilter == other.magFilter &&
           minFilter == other.minFilter && srgb == other.srgb;
  }
};

/** A device-independent cubemap. Each device (Context) that samples it gets its own cube
 *  image (converted from lat-long and prefiltered on that device when needed), view and
 *  sampler. */
class SVCubemap {
  struct DeviceState {
    std::shared_ptr<core::Context> context;   // keeps the device alive for the view
    std::unique_ptr<core::Image> latLongCube; // this device's cube converted from lat-long
    vk::UniqueImageView imageView;
    vk::Sampler sampler;
  };

  SVCubemapDescription mDescription;
  /** host cube faces (eFACES, eKTX, eCUSTOM) */
  std::shared_ptr<SVImage> mImage;
  /** host lat-long image (eLATLONG) */
  std::shared_ptr<SVImage> mLatLongImage;
  core::PerContext<DeviceState> mDeviceStates;

  std::atomic<bool> mLoaded{false};

  /** the cube usage (including storage, needed by the prefilter pass) is set on mImage */
  bool mImageUsageSet{false};

  std::mutex mUploadingMutex;
  std::mutex mLoadingMutex;

  DeviceState const &getDeviceState(core::Context const &context) const;

public:
  static std::shared_ptr<SVCubemap> FromFile(std::string const &filename, uint32_t mipLevels,
                                             vk::Filter magFilter = vk::Filter::eLinear,
                                             vk::Filter minFilter = vk::Filter::eLinear,
                                             bool srgb = false);

  static std::shared_ptr<SVCubemap> FromFile(std::array<std::string, 6> const &filenames,
                                             uint32_t mipLevels,
                                             vk::Filter magFilter = vk::Filter::eLinear,
                                             vk::Filter minFilter = vk::Filter::eLinear,
                                             bool srgb = false);

  static std::shared_ptr<SVCubemap>
  FromData(uint32_t size, vk::Format format, std::array<std::vector<char>, 6> const &data,
           uint32_t mipLevels = 1, vk::Filter magFilter = vk::Filter::eLinear,
           vk::Filter minFilter = vk::Filter::eLinear, bool srgb = false);

  /** upload (convert, prefilter) and create this device's view and sampler; no-op when
   *  present */
  void uploadToDevice(core::Context &context);
  /** release the device state of every device */
  void removeFromDevice();

  bool isOnDevice(core::Context const &context) const;
  inline bool isLoaded() const { return mLoaded; }
  std::future<void> loadAsync();
  void load();

  inline SVCubemapDescription const &getDescription() const { return mDescription; }

  /** this device's view and sampler; throw when the cubemap is not uploaded there */
  vk::ImageView getImageView(core::Context const &context) const;
  vk::Sampler getSampler(core::Context const &context) const;

  /** write this device's (prefiltered) cube image to a KTX file */
  void exportKTX(core::Context &context, std::string const &filename);

private:
  SVCubemap() = default;
};

} // namespace resource
} // namespace svulkan2
