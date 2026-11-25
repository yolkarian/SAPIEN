#pragma once
#include "renderer_base.h"
#include "svulkan2/common/layout.h"
#include "svulkan2/core/descriptor_pool.h"
#include "svulkan2/resource/storage_image.h"
#include "svulkan2/shader/glsl_compiler.h"
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace svulkan2 {
namespace core {
class Context;
}
namespace renderer {

/** A linear chain of post-processing compute passes shared by the renderers.
 *
 *  Each pass is one compute shader. Its storage images are bound by name:
 *    - `inColor` (rgba32f, read): the colour the previous pass wrote; for the first pass, the
 *      renderer's source image;
 *    - `outColor` (rgba32f, written): read by the next pass; for the last pass, the renderer's
 *      destination image;
 *    - any other name: the renderer's image of that name (for example `Albedo`).
 *  The members of a pass's push constant block (int or float) are the chain's properties; a
 *  name declared by several passes is one property. Passes run in the order given; the chain
 *  owns the images between two passes. */
class PostChain {
public:
  PostChain(std::shared_ptr<core::Context> context, std::vector<std::string> const &files,
            ShaderDefines const &defines);
  ~PostChain();
  PostChain(PostChain const &) = delete;
  PostChain &operator=(PostChain const &) = delete;

  bool empty() const { return mPasses.empty(); }

  /** eNone when no pass declares `name` */
  CustomPropertyType getPropertyType(std::string const &name) const;
  void setProperty(std::string const &name, int value);
  void setProperty(std::string const &name, float value);
  int getPropertyInt(std::string const &name) const;
  float getPropertyFloat(std::string const &name) const;

  /** Bind the images of a `width` x `height` frame, all in layout eGeneral. `lookup` returns
   *  the renderer's image of a name other than inColor/outColor and throws when it has none. */
  void prepare(uint32_t width, uint32_t height, vk::ImageView source, vk::ImageView destination,
               std::function<vk::ImageView(std::string const &)> const &lookup);

  /** Record every pass with the current property values. The caller makes earlier writes to
   *  the bound images visible to compute shaders. */
  void record(vk::CommandBuffer commandBuffer) const;

private:
  struct Pass {
    std::string file;
    // storage image names by binding of set 0
    std::map<uint32_t, std::string> images;
    std::shared_ptr<StructDataLayout> parameters; // null without a push constant block
    uint32_t localSize[2]{1, 1};
    vk::UniqueDescriptorSetLayout setLayout;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline pipeline;
    core::DescriptorSet set;
  };
  struct Property {
    DataType dtype;
    char value[4]{};
  };

  template <typename T> void store(std::string const &name, DataType const &dtype, T value);

  std::shared_ptr<core::Context> mContext;
  std::vector<Pass> mPasses;
  std::map<std::string, Property> mProperties;

  uint32_t mWidth{};
  uint32_t mHeight{};
  std::unique_ptr<core::DynamicDescriptorPool> mPool;
  // the colour between pass i and pass i + 1 lives in mScratch[i % 2]
  std::vector<std::shared_ptr<resource::SVStorageImage>> mScratch;
};

} // namespace renderer
} // namespace svulkan2
