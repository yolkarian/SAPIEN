#pragma once
#include "cubemap.h"
#include "model.h"
#include "svulkan2/common/config.h"
#include "svulkan2/core/descriptor_pool.h"
#include "svulkan2/shader/shader_pack.h"
#include "svulkan2/shader/shader_pack_instance.h"
#include <memory>
#include <unordered_map>

namespace svulkan2 {
namespace shader {
class RayTracingShaderPack;
class DescriptorSetDescription;
}; // namespace shader

namespace resource {

/** Process-wide cache of device-independent resources (models, images, textures,
 *  cubemaps, shader packs, vertex layouts) shared by every device (Context). Device objects
 *  it hands out (shader pack instances, BRDF lookup tables) are cached per context. */
class SVResourceManager {
public:
  /** the live manager, or a new one when none is alive */
  static std::shared_ptr<SVResourceManager> Create();
  /** the live manager; throws when none is alive */
  static std::shared_ptr<SVResourceManager> Get();

  std::shared_ptr<shader::ShaderPack> CreateShaderPack(std::string const &dirname);

  /** pipelines of `desc` on `context`, cached per context */
  std::shared_ptr<shader::ShaderPackInstance>
  CreateShaderPackInstance(std::shared_ptr<core::Context> context,
                           shader::ShaderPackInstanceDesc const &desc);

  std::shared_ptr<shader::RayTracingShaderPack> CreateRTShaderPack(std::string const &dirname);

  std::shared_ptr<SVImage> CreateImageFromFile(std::string const &filename, uint32_t mipLevels,
                                               uint32_t desiredChannels);

  std::shared_ptr<SVTexture>
  CreateTextureFromFile(std::string const &filename, uint32_t mipLevels,
                        vk::Filter magFilter = vk::Filter::eLinear,
                        vk::Filter minFilter = vk::Filter::eLinear,
                        vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
                        vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat,
                        bool srgb = false, uint32_t desiredChannels = 0);

  std::shared_ptr<SVTexture> CreateTextureFromRawData(
      uint32_t width, uint32_t height, uint32_t depth, vk::Format format,
      std::vector<char> const &data, int dim, uint32_t mipLevels = 1,
      vk::Filter magFilter = vk::Filter::eLinear, vk::Filter minFilter = vk::Filter::eLinear,
      vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode addressModeW = vk::SamplerAddressMode::eRepeat, bool srgb = false);

  std::shared_ptr<SVCubemap> CreateCubemapFromFile(std::string const &filename,
                                                   uint32_t mipLevels = 1,
                                                   vk::Filter magFilter = vk::Filter::eLinear,
                                                   vk::Filter minFilter = vk::Filter::eLinear,
                                                   bool srgb = true);

  std::shared_ptr<SVCubemap> CreateCubemapFromFiles(std::array<std::string, 6> const &filenames,
                                                    uint32_t mipLevels = 1,
                                                    vk::Filter magFilter = vk::Filter::eLinear,
                                                    vk::Filter minFilter = vk::Filter::eLinear,
                                                    bool srgb = true);

  /** a BRDF lookup table computed on `context` */
  std::shared_ptr<SVTexture> generateBRDFLUT(core::Context &context, uint32_t size);

  /** the default BRDF lookup table of `context`, computed once per context */
  std::shared_ptr<SVTexture> getDefaultBRDFLUT(core::Context &context);

  std::shared_ptr<SVTexture> CreateRandomTexture(std::string const &name);

  inline std::shared_ptr<SVTexture> getDefaultTexture() const { return mDefaultTexture2D; };

  inline std::shared_ptr<SVTexture> getDefaultTexture1D() const { return mDefaultTexture1D; };

  inline std::shared_ptr<SVTexture> getDefaultTexture3D() const { return mDefaultTexture3D; };

  inline std::shared_ptr<SVCubemap> getDefaultCubemap() const { return mDefaultCubemap; };

  std::shared_ptr<SVModel> CreateModelFromFile(std::string const &filename);

  std::shared_ptr<SVMetallicMaterial> createMetallicMaterial(glm::vec4 emission,
                                                             glm::vec4 baseColor, float fresnel,
                                                             float roughness, float metallic,
                                                             float transparency, float ior);

  std::shared_ptr<resource::SVModel>
  createModel(std::vector<std::shared_ptr<resource::SVMesh>> const &meshes,
              std::vector<std::shared_ptr<resource::SVMaterial>> const &materials);

  /** release all cached resources */
  void clearCachedResources(bool models = true, bool images = true, bool shaders = true);

  /** release gpu resources of every device.
   * NOTE: This MUST be called when no rendering is running!
   * NOTE: All renders become invalid after calling this function!*/
  void releaseGPUResourcesUnsafe();

  /** drop the device objects cached for `context` (shader pack instances, BRDF LUT) */
  void releaseContext(core::Context const &context);

  void setVertexLayout(std::shared_ptr<InputDataLayout> layout);
  inline std::shared_ptr<InputDataLayout> getVertexLayout() const {
    if (!mVertexLayout) {
      throw std::runtime_error("failed to infer vertex layout: you need to load a shader first "
                               "(e.g. by creating a camera)");
    }
    return mVertexLayout;
  }

  void setPrimitiveVertexLayout(std::shared_ptr<InputDataLayout> layout);
  inline std::shared_ptr<InputDataLayout> getLineVertexLayout() const {
    if (!mLineVertexLayout) {
      throw std::runtime_error(
          "failed to infer vertex layout: you need to load a shader first (e.g. by "
          "creating a camera) ");
    }
    return mLineVertexLayout;
  }

  inline std::unordered_map<std::string, std::vector<std::shared_ptr<SVModel>>> const &
  getModels() const {
    return mModelRegistry;
  }

  inline std::unordered_map<std::string, std::vector<std::shared_ptr<SVTexture>>> const &
  getTextures() const {
    return mTextureRegistry;
  }

  inline std::unordered_map<std::string, std::vector<std::shared_ptr<SVCubemap>>> const &
  getCubemaps() const {
    return mCubemapRegistry;
  }

  inline std::unordered_map<std::string, std::vector<std::shared_ptr<SVImage>>> const &
  getImages() const {
    return mImageRegistry;
  }

  void setDefaultMipLevels(uint32_t level) { mDefaultMipLevels = level; }
  uint32_t getDefaultMipLevels() const { return mDefaultMipLevels; }

  /** when set, models load without their textures */
  void setDoNotLoadTexture(bool doNotLoad) { mDoNotLoadTexture = doNotLoad; }
  bool shouldNotLoadTexture() const { return mDoNotLoadTexture; }

  ~SVResourceManager() = default;

private:
  SVResourceManager();

  std::unordered_map<std::string, std::vector<std::shared_ptr<SVModel>>> mModelRegistry;
  std::unordered_map<std::string, std::vector<std::shared_ptr<SVTexture>>> mTextureRegistry;
  std::unordered_map<std::string, std::vector<std::shared_ptr<SVCubemap>>> mCubemapRegistry;
  std::unordered_map<std::string, std::vector<std::shared_ptr<SVImage>>> mImageRegistry;
  std::unordered_map<std::string, std::shared_ptr<SVTexture>> mRandomTextureRegistry;

  std::mutex mShaderPackLock{};
  std::unordered_map<std::string, std::shared_ptr<shader::ShaderPack>> mShaderPackRegistry;

  std::mutex mShaderPackInstanceLock{};
  // Context::getId() -> shader directory -> instances
  std::unordered_map<
      uint64_t,
      std::unordered_map<std::string, std::vector<std::shared_ptr<shader::ShaderPackInstance>>>>
      mShaderPackInstanceRegistry;

  std::mutex mRTShaderPackLock{};
  std::unordered_map<std::string, std::shared_ptr<shader::RayTracingShaderPack>>
      mRTShaderPackRegistry;

  std::shared_ptr<InputDataLayout> mVertexLayout{};
  std::shared_ptr<InputDataLayout> mLineVertexLayout{};

  std::shared_ptr<SVTexture> mDefaultTexture1D;
  std::shared_ptr<SVTexture> mDefaultTexture2D;
  std::shared_ptr<SVTexture> mDefaultTexture3D;

  std::shared_ptr<SVCubemap> mDefaultCubemap;

  std::mutex mBRDFLUTLock{};
  // Context::getId() -> default BRDF lookup table
  std::unordered_map<uint64_t, std::shared_ptr<SVTexture>> mDefaultBRDFLUTs;

  uint32_t mDefaultMipLevels{1};
  bool mDoNotLoadTexture{false};

  std::mutex mCreateLock{};
};

} // namespace resource
} // namespace svulkan2
