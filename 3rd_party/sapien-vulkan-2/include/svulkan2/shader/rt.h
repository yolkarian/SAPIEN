#pragma once
#include "base_parser.h"

namespace svulkan2 {
namespace core {
class Buffer;
class Context;
}

namespace shader {

/** Mesh vertex layout implied by the `Vertices` storage buffer array of a scene descriptor set
 *  (members x y z, then optionally nx.. u v, tx.., bx.., r g b a, in that order without padding);
 *  empty when the set declares no `Vertices`. Throws when the struct breaks that order. */
std::shared_ptr<InputDataLayout> computeVertexLayout(DescriptorSetDescription const &set);
/** point layout implied by the `Points` storage buffer array of a scene descriptor set; empty
 *  when the set declares no `Points` */
std::shared_ptr<InputDataLayout> computePointLayout(DescriptorSetDescription const &set);

class RayTracingStageParser {
public:
  std::future<void> loadFileAsync(std::string const &filepath, vk::ShaderStageFlagBits stage,
                                  ShaderDefines const &defines = {});
  void reflectSPV();
  inline std::unordered_map<uint32_t, DescriptorSetDescription> const &getResources() const {
    return mResources;
  }
  inline std::shared_ptr<StructDataLayout> getPushConstantLayout() const {
    return mPushConstantLayout;
  }
  inline std::vector<uint32_t> const &getCode() const { return mSPVCode; }
  std::string summary() const;

  void setNmae(std::string const &name) { mName = name; }
  std::string getName() const { return mName; }

private:
  std::vector<uint32_t> mSPVCode;
  std::unordered_map<uint32_t, DescriptorSetDescription> mResources;
  std::shared_ptr<StructDataLayout> mPushConstantLayout;
  std::string mName;
};

class RayTracingShaderPack {
public:
  struct HitGroupParser {
    std::string name;
    std::unique_ptr<RayTracingStageParser> any;
    std::unique_ptr<RayTracingStageParser> closest;
    std::unique_ptr<RayTracingStageParser> intersect;
  };

  struct MissGroupParser {
    std::string name;
    std::unique_ptr<RayTracingStageParser> miss;
  };

  RayTracingShaderPack(std::string const &shaderDir, ShaderDefines const &defines = {});

  std::shared_ptr<InputDataLayout> computeCompatibleInputVertexLayout() const;
  std::shared_ptr<InputDataLayout> computePrimitiveLayout() const;

  inline std::unordered_map<uint32_t, DescriptorSetDescription> const &getResources() const {
    return mResources;
  }
  inline std::shared_ptr<StructDataLayout> getPushConstantLayout() const {
    return mPushConstantLayout;
  }

  std::string summary() const;

  RayTracingStageParser *getRaygenStageParser() const { return mRaygenStageParser.get(); };
  inline std::vector<MissGroupParser> const &getMissGroupParsers() const {
    return mMissGroupParsers;
  }
  inline std::vector<HitGroupParser> const &getHitGroupParsers() const { return mHitGroupParsers; }

  // inline std::vector<std::unique_ptr<RayTracingStageParser>> const &getAnyHitStageParsers()
  // const {
  //   return mAnyHitStageParsers;
  // }
  // inline std::vector<std::unique_ptr<RayTracingStageParser>> const &
  // getClosestHitStageParsers() const {
  //   return mClosestHitStageParsers;
  // }
  // inline std::vector<std::unique_ptr<RayTracingStageParser>> const &
  // getIntersectStageParsers() const {
  //   return mIntersectStageParsers;
  // }
  StructDataLayout const &getObjectBufferLayout() const;
  StructDataLayout const &getGeometryInstanceBufferLayout() const;
  StructDataLayout const &getCameraBufferLayout() const;
  DescriptorSetDescription const &getOutputDescription() const;
  DescriptorSetDescription const &getSceneDescription() const;
  DescriptorSetDescription const &getCameraDescription() const;

private:
  std::unique_ptr<RayTracingStageParser> mRaygenStageParser;
  std::vector<MissGroupParser> mMissGroupParsers;
  std::vector<HitGroupParser> mHitGroupParsers;

  std::unordered_map<uint32_t, DescriptorSetDescription> mResources;
  std::shared_ptr<StructDataLayout> mPushConstantLayout;
};

struct RayTracingShaderPackInstanceDesc {
  std::string shaderDir{};
  ShaderDefines defines{};
  uint32_t maxMeshes{};
  uint32_t maxTextures{};
  uint32_t maxPointSets{};
};

class RayTracingShaderPackInstance {

public:
  /** pipelines of `desc` on `context`; the binding table is allocated there */
  RayTracingShaderPackInstance(std::shared_ptr<core::Context> context,
                               RayTracingShaderPackInstanceDesc desc);
  vk::PipelineLayout getPipelineLayout();
  vk::Pipeline getPipeline();
  core::Buffer &getShaderBindingTable();
  vk::DescriptorSetLayout getOutputSetLayout();
  vk::DescriptorSetLayout getSceneSetLayout();
  vk::DescriptorSetLayout getCameraSetLayout();

  inline std::shared_ptr<RayTracingShaderPack> getShaderPack() const { return mShaderPack; }

  vk::StridedDeviceAddressRegionKHR const &getRgenRegion();
  vk::StridedDeviceAddressRegionKHR const &getMissRegion();
  vk::StridedDeviceAddressRegionKHR const &getHitRegion();
  vk::StridedDeviceAddressRegionKHR const &getCallRegion();

  RayTracingShaderPackInstance(RayTracingShaderPackInstance const &) = delete;
  RayTracingShaderPackInstance &operator=(RayTracingShaderPackInstance const &) = delete;

private:
  void initPipeline();

  void initSBT();

  std::shared_ptr<core::Context> mContext;
  RayTracingShaderPackInstanceDesc mDesc;

  // vk::UniqueShaderModule mRgenModule;
  // std::vector<vk::UniqueShaderModule> mMissModules;
  // std::vector<vk::UniqueShaderModule> mRcHitModules;
  // std::vector<vk::UniqueShaderModule> mRaHitModules;
  // std::vector<vk::UniqueShaderModule> mIntersectModules;

  vk::UniqueDescriptorSetLayout mSceneSetLayout;
  vk::UniqueDescriptorSetLayout mCameraSetLayout;
  vk::UniqueDescriptorSetLayout mOutputSetLayout;
  vk::UniquePipelineLayout mPipelineLayout;
  vk::UniquePipeline mPipeline;
  std::unique_ptr<core::Buffer> mSBTBuffer;
  vk::StridedDeviceAddressRegionKHR mRgenRegion{};
  vk::StridedDeviceAddressRegionKHR mMissRegion{};
  vk::StridedDeviceAddressRegionKHR mHitRegion{};
  vk::StridedDeviceAddressRegionKHR mCallRegion{};

  std::shared_ptr<RayTracingShaderPack> mShaderPack;
};

} // namespace shader
} // namespace svulkan2
