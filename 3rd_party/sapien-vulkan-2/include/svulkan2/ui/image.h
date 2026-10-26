#pragma once
#include "svulkan2/core/image.h"
#include "widget.h"

namespace svulkan2 {
namespace core {
class CommandPool;
class Context;
}

namespace ui {

UI_CLASS(DisplayImage) {
  UI_DECLARE_LABEL(DisplayImage);

  // TODO: own core::Image
  std::shared_ptr<DisplayImage> Image(core::Image &);
  std::shared_ptr<DisplayImage> Clear();
  UI_ATTRIBUTE(DisplayImage, glm::vec2, Size);

  void build() override;

  ~DisplayImage();

protected:
  std::shared_ptr<core::Context> mContext;
  std::unique_ptr<core::CommandPool> mCommandPool;
  vk::UniqueCommandBuffer mCommandBuffer;

  core::Image *mImage{};

  vk::Sampler mSampler;
  vk::UniqueImageView mImageView;

  VkDescriptorSet mDS{};
};

} // namespace ui
} // namespace svulkan2
