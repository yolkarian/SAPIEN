#pragma once
#include "svulkan2/core/image.h"
#include <memory>
#include <string>
#include <vector>

namespace svulkan2 {
namespace resource {

/** Storage image of one device (Context). It downloads from the device that created it. */
class SVStorageImage {
  std::shared_ptr<core::Context> mContext;
  std::string mName;

  vk::Format mFormat;
  uint32_t mWidth{};
  uint32_t mHeight{};

  std::shared_ptr<core::Image> mImage{};
  vk::UniqueImageView mImageView;

public:
  SVStorageImage(std::shared_ptr<core::Context> context, std::string const &name, uint32_t width,
                 uint32_t height, vk::Format format);

  /** create the image and view of `context`; no-op when they already exist */
  void createDeviceResources();
  template <typename T> std::vector<T> download() {
    return mImage->download<T>();
  }

  inline uint32_t getWidth() const { return mWidth; }
  inline uint32_t getHeight() const { return mHeight; }

  inline core::Image &getImage() const { return *mImage; }
  inline vk::ImageView getImageView() const { return mImageView.get(); };
  inline vk::Format getFormat() { return mFormat; };
};

} // namespace resource
} // namespace svulkan2
