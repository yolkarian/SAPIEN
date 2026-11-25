#pragma once
#include "base_parser.h"

namespace svulkan2 {
namespace core {
class Buffer;
class Context;
} // namespace core

namespace shader {

/** A batched ray-tracing shader pack: a directory holding `batch.comp`, one compute shader
 *  that renders every camera of a camera group in a single dispatch.
 *
 *  The output set (the set holding the `Out<Name>` storage buffers) declares the pictures the
 *  pack produces: the element type of each block's array selects the image format
 *  (vec4 -> rgba32f, uint -> rgba8, uvec4 -> rgba32ui, float -> r32f). The scene set declares the
 *  group- and environment-level tables; bindings are resolved by name. */
class BatchShaderPack {
public:
  /** one declared picture */
  struct Output {
    std::string name;      // picture name, the block name without the "Out" prefix
    std::string blockName; // e.g. "OutColor"
    vk::Format format;
    uint32_t binding;
  };

  /** Loads and reflects `batch.comp` compiled with `defines`. Throws when the directory has no
   *  batch.comp, when the shader does not compile, or when the reflected descriptor sets are
   *  not the two the renderer expects. */
  BatchShaderPack(std::string const &shaderDir, ShaderDefines const &defines = {});

  inline std::string const &getShaderDir() const { return mShaderDir; }

  inline std::vector<uint32_t> const &getCode() const { return mSPVCode; }

  std::shared_ptr<StructDataLayout> getPushConstantLayout() const { return mPushConstantLayout; }

  inline std::unordered_map<uint32_t, DescriptorSetDescription> const &getResources() const {
    return mResources;
  }

  /** descriptor set holding the `Out<Name>` storage buffers */
  inline uint32_t getOutputSetId() const { return mOutputSetId; }
  /** descriptor set holding the acceleration structure and the scene tables */
  inline uint32_t getSceneSetId() const { return mSceneSetId; }

  DescriptorSetDescription const &getOutputDescription() const;
  DescriptorSetDescription const &getSceneDescription() const;

  inline std::vector<Output> const &getOutputs() const { return mOutputs; }
  bool hasOutput(std::string const &name) const { return getOutputFormat(name) != vk::Format::eUndefined; }
  /** format of picture `name`, or eUndefined when the pack does not declare it */
  vk::Format getOutputFormat(std::string const &name) const;

  /** element structs of the scene tables as the pack declares them (`Object`,
   *  `GeometryInstance`); throws when the block is missing or malformed */
  StructDataLayout const &getObjectBufferLayout() const;
  StructDataLayout const &getGeometryInstanceBufferLayout() const;

  std::string summary() const;

private:
  void reflectSPV();

  std::string mShaderDir;
  std::vector<uint32_t> mSPVCode;
  std::unordered_map<uint32_t, DescriptorSetDescription> mResources;
  std::shared_ptr<StructDataLayout> mPushConstantLayout;
  std::vector<Output> mOutputs;
  uint32_t mOutputSetId{0};
  uint32_t mSceneSetId{1};
};

/** map the element type of an `Out<Name>` block to the picture format it produces */
vk::Format getBatchOutputFormat(DataType const &dtype);

/** upper-case define suffix of a picture format, e.g. r8g8b8a8Unorm -> "R8G8B8A8UNORM" */
std::string getBatchFormatDefineName(vk::Format format);

} // namespace shader
} // namespace svulkan2
