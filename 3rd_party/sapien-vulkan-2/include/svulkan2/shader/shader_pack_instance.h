#pragma once
#include "svulkan2/common/config.h"
#include "svulkan2/common/layout.h"
#include "svulkan2/shader/shader_pack.h"
#include <atomic>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>

namespace svulkan2 {
namespace core {
class Context;
}
namespace shader {

struct ShaderPackInstanceDesc {
  std::shared_ptr<RendererConfig> config;
  std::map<std::string, SpecializationConstantValue> specializationConstants;

  bool operator==(ShaderPackInstanceDesc const &other) const {
    return *config == *other.config && specializationConstants == other.specializationConstants;
  }
};

enum class RenderTargetOperation { eNoOp, eRead, eColorWrite, eDepthWrite };
using optable_t = std::unordered_map<std::string, std::vector<RenderTargetOperation>>;

class ShaderPackInstance {
public:
  /** pipelines of `desc` on `context` */
  ShaderPackInstance(std::shared_ptr<core::Context> context, ShaderPackInstanceDesc const &desc);

  std::future<void> loadAsync();

  inline ShaderPackInstanceDesc const &getDesc() const { return mDesc; }

  struct PipelineResources {
    vk::UniquePipelineLayout layout{};
    vk::UniqueRenderPass renderPass{};
    vk::UniquePipeline pipeline{};
  };

  inline std::shared_ptr<ShaderPack> getShaderPack() const { return mShaderPack; }

  inline std::vector<PipelineResources> const &getNonShadowPassResources() const {
    return mNonShadowPassResources;
  }

  inline PipelineResources const &getShadowPassResources() const { return mShadowPassResources; }

  inline PipelineResources const &getPointShadowPassResources() const {
    return mPointShadowPassResources;
  }

  inline std::unordered_map<std::string, vk::Format> getRenderTargetFormats() const {
    return mRenderTargetFormats;
  }

  inline std::unordered_map<std::string, float> getRenderTargetScales() const {
    return mRenderTargetScales;
  }

  inline std::unordered_map<std::string, vk::ImageLayout> getRenderTargetFinalLayouts() const {
    return mRenderTargetFinalLayouts;
  }

  inline bool isPassActive(uint32_t passIndex) const {
    return passIndex < mActivePasses.size() && mActivePasses[passIndex];
  }
  inline bool isColorOutputActive(uint32_t passIndex, uint32_t outputIndex) const {
    return passIndex < mActiveColorOutputs.size() &&
           outputIndex < mActiveColorOutputs[passIndex].size() &&
           mActiveColorOutputs[passIndex][outputIndex];
  }
  inline uint32_t getActiveColorOutputCount(uint32_t passIndex) const {
    if (passIndex >= mActiveColorOutputs.size()) {
      return 0;
    }
    uint32_t count = 0;
    for (bool active : mActiveColorOutputs[passIndex]) {
      if (active) {
        ++count;
      }
    }
    return count;
  }
  inline bool needsSceneSet() const { return mNeedsSceneSet; }
  inline bool needsShadowPass() const { return mNeedsShadowPass; }
  inline bool needsPostChain() const { return mNeedsPostChain; }
  inline bool hasRenderTarget(std::string const &name) const {
    return mRenderTargetFormats.contains(name);
  }

  inline vk::DescriptorSetLayout getSceneDescriptorSetLayout() const {
    return mSceneSetLayout.get();
  }
  inline vk::DescriptorSetLayout getGPUSceneDescriptorSetLayout() const {
    return mGPUSceneSetLayout.get();
  }
  inline vk::DescriptorSetLayout getCameraDescriptorSetLayout() const {
    return mCameraSetLayout.get();
  }
  inline vk::DescriptorSetLayout getLightDescriptorSetLayout() const {
    return mLightSetLayout.get();
  }

  inline std::vector<vk::DescriptorSetLayout> getInputTextureLayouts() const {
    std::vector<vk::DescriptorSetLayout> result;
    for (auto &layout : mTextureSetLayouts) {
      result.push_back(layout.get());
    }
    return result;
  }

  std::optional<std::string> getDepthRenderTargetName(BaseParser const &pass) const;

private:
  ShaderPackInstanceDesc mDesc;
  std::shared_ptr<ShaderPack> mShaderPack;
  std::shared_ptr<core::Context> mContext;

  vk::UniqueDescriptorSetLayout mSceneSetLayout;
  vk::UniqueDescriptorSetLayout mGPUSceneSetLayout;
  vk::UniqueDescriptorSetLayout mCameraSetLayout;
  vk::UniqueDescriptorSetLayout mLightSetLayout;
  std::vector<vk::UniqueDescriptorSetLayout> mTextureSetLayouts;

  optable_t mTextureOperationTable;
  optable_t generateTextureOperationTable() const;

  std::vector<PipelineResources> mNonShadowPassResources;
  PipelineResources mShadowPassResources;
  PipelineResources mPointShadowPassResources;

  std::unordered_map<std::string, vk::Format> mRenderTargetFormats;
  std::unordered_map<std::string, float> mRenderTargetScales;
  std::unordered_map<std::string, vk::ImageLayout> mRenderTargetFinalLayouts;
  std::vector<bool> mActivePasses;
  std::vector<std::vector<bool>> mActiveColorOutputs;
  bool mNeedsSceneSet{false};
  bool mNeedsShadowPass{false};
  bool mNeedsPostChain{false};

  std::mutex mLoadingMutex;
  std::atomic<bool> mLoaded{false};
};

} // namespace shader
} // namespace svulkan2
