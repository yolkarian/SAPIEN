#pragma once
#include "renderer_base.h"
#include "svulkan2/core/context.h"
#include "svulkan2/core/descriptor_pool.h"
#include "svulkan2/resource/cubemap.h"
#include "svulkan2/resource/render_target.h"
#include "svulkan2/resource/storage_image.h"
#include "svulkan2/scene/scene.h"
#include "svulkan2/shader/glsl_compiler.h"
#include "svulkan2/renderer/post_chain.h"
#include "svulkan2/renderer/rt_integrator_tables.h"
#include <utility>
#include <variant>

namespace svulkan2 {
namespace core {
class Context;
class CommandPool;
} // namespace core

namespace shader {
class RayTracingShaderPack;
class RayTracingShaderPackInstance;
}; // namespace shader

namespace renderer {

class RTRenderer : public RendererBase {
public:
  enum class DenoiserType { eNONE, eOPTIX, eOIDN };

  /** `postPasses`: the compute shader files of the post-processing chain (PostChain), in
   *  order. It reads `HdrColor` and writes `Color`; without passes there is no `Color`. */
  RTRenderer(std::shared_ptr<core::Context> context, std::string const &shaderDir,
             ShaderDefines const &shaderDefines = {},
             std::vector<std::string> const &postPasses = {});

  void resize(int width, int height) override;
  void setScene(std::shared_ptr<scene::Scene> scene) override;

  void render(scene::Camera &camera, std::vector<vk::Semaphore> const &waitSemaphores,
              std::vector<vk::PipelineStageFlags> const &waitStages,
              std::vector<vk::Semaphore> const &signalSemaphores, vk::Fence fence) override;
  void render(scene::Camera &camera,
              vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &waitSemaphores,
              vk::ArrayProxyNoTemporaries<vk::PipelineStageFlags const> const &waitStageMasks,
              vk::ArrayProxyNoTemporaries<uint64_t const> const &waitValues,
              vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &signalSemaphores,
              vk::ArrayProxyNoTemporaries<uint64_t const> const &signalValues) override;
  void display(std::string const &imageName, vk::Image backBuffer, vk::Format format,
               uint32_t width, uint32_t height, std::vector<vk::Semaphore> const &waitSemaphores,
               std::vector<vk::PipelineStageFlags> const &waitStages,
               std::vector<vk::Semaphore> const &signalSemaphores, vk::Fence fence) override;

  void enableDenoiser(DenoiserType type, std::string const &colorName,
                      std::string const &albedoName, std::string const &normalName);
  DenoiserType getDenoiserType() const;

  bool denoiserEnabled() const {
#ifdef SVULKAN2_CUDA_INTEROP
    return mDenoiser != nullptr;
#else
    return false;
#endif
  }
  void disableDenoiser();

  CustomPropertyType getCustomPropertyType(std::string const &name) const override;

  int getCustomPropertyInt(std::string const &name) const override;
  float getCustomPropertyFloat(std::string const &name) const override;

  glm::vec3 getCustomPropertyVec3(std::string const &name) const override {
    return mCustomPropertiesVec3.at(name);
  }

  glm::vec4 getCustomPropertyVec4(std::string const &name) const override {
    return mCustomPropertiesVec4.at(name);
  }

  void setCustomProperty(std::string const &name, int p) override;
  void setCustomProperty(std::string const &name, float p) override;
  void setCustomProperty(std::string const &name, glm::vec3 p) override {
    setProperty(mCustomPropertiesVec3, name, p);
  }
  void setCustomProperty(std::string const &name, glm::vec4 p) override {
    setProperty(mCustomPropertiesVec4, name, p);
  }

  std::vector<std::string> getDisplayTargetNames() const override;
  std::vector<std::string> getRenderTargetNames() const override;

  core::Buffer &getCameraBuffer();
  /** float offsets in the camera buffer of prevViewMatrix and prevViewMatrixInverse, or
   *  {-1, -1} when the pack's camera layout has none (a GPU pose writer must then leave the
   *  buffer's prev slots alone) */
  std::pair<int, int> getCameraPrevViewOffsets() const;
  void prepareResources(scene::Camera &camera) override;
  void uploadCpuFrameState(scene::Camera &camera) override;
  void uploadCpuCameraState(scene::Camera &camera) override;
  void uploadCpuSceneLightState() override;
  void setExternalCameraUpdatesEnabled(bool enable);
  bool getExternalCameraUpdatesEnabled() const { return mExternalCameraUpdates; }
  void setExternalTransformUpdatesEnabled(bool enable, bool cudaInterop = true);
  void initializeExternalTransformResources(scene::Camera &camera);

  core::Image &getRenderImage(std::string const &name) override {
    return mRenderImages.at(name)->getImage();
  }

  vk::ImageLayout getRenderTargetImageLayout(std::string const &name) override {
    return vk::ImageLayout::eGeneral;
  }

  inline void setCustomTexture(std::string const &name,
                               std::shared_ptr<resource::SVTexture> texture) override {
    mCustomTextures[name] = texture;
  };
  inline void setCustomCubemap(std::string const &name,
                               std::shared_ptr<resource::SVCubemap> cubemap) override {
    mCustomCubemaps[name] = cubemap;
  };

  RTRenderer(RTRenderer const &other) = delete;
  RTRenderer &operator=(RTRenderer const &other) = delete;
  RTRenderer(RTRenderer &&other) = default;
  RTRenderer &operator=(RTRenderer &&other) = default;

  ~RTRenderer();

private:
  std::unordered_map<std::string, int> mCustomPropertiesInt;
  std::unordered_map<std::string, float> mCustomPropertiesFloat;
  std::unordered_map<std::string, glm::vec3> mCustomPropertiesVec3;
  std::unordered_map<std::string, glm::vec4> mCustomPropertiesVec4;

  /** Stores a property of the ray tracing shaders; a changed value restarts accumulation. */
  template <typename T>
  void setProperty(std::unordered_map<std::string, T> &properties, std::string const &name,
                   T const &value) {
    auto it = properties.find(name);
    if (it != properties.end() && it->second == value) {
      return;
    }
    properties[name] = value;
    mAccumulationDirty = true;
  }
  /** whether `name` is a property of the post-processing chain, which reads the accumulated
   *  image: changing it keeps the accumulation */
  bool isPostProperty(std::string const &name) const {
    return mPostChain && mPostChain->getPropertyType(name) != CustomPropertyType::eNone;
  }

  void prepareObjects();
  void updateObjects();

  void prepareOutput();
  void prepareScene();
  void prepareCamera();
  void preparePostChain();

  void recordRender();
  void recordPostprocess();

  void updatePushConstant();

  void prepareRender(scene::Camera &camera);

  std::shared_ptr<core::Context> mContext;
  std::string mShaderDir;
  ShaderDefines mShaderDefines;
  std::shared_ptr<shader::RayTracingShaderPack> mShaderPack;
  std::shared_ptr<shader::RayTracingShaderPackInstance> mShaderPackInstance;

  StructDataLayout mGeometryInstanceBufferLayout;
  StructDataLayout mCameraBufferLayout;
  StructDataLayout mObjectBufferLayout;

  int mWidth{};
  int mHeight{};

  std::shared_ptr<scene::Scene> mScene;
  uint64_t mSceneVersion{0l}; // check for rebuild
  uint64_t mRTTablesVersion{};      // the scene's ray tracing tables the descriptors name
  scene::Scene::RTMaterialVersions mMaterialVersions; // the scene's materials[] last rendered

  uint64_t mSceneRenderVersion{0l}; // check for updating matrices
  int mFrameCount = 0;
  uint32_t mRenderCount = 0;
  bool mAccumulationDirty = false;

  std::unordered_map<std::string, std::shared_ptr<resource::SVStorageImage>> mRenderImages;
  // subset of render images used in RT
  std::vector<std::shared_ptr<resource::SVStorageImage>> mRTImages;

  std::unique_ptr<core::DynamicDescriptorPool> mOutputPool;
  std::unique_ptr<core::DynamicDescriptorPool> mCameraPool;
  std::unique_ptr<core::DynamicDescriptorPool> mScenePool;

  core::DescriptorSet mSceneSet;
  core::DescriptorSet mOutputSet;
  core::DescriptorSet mCameraSet;

  std::unique_ptr<PostChain> mPostChain;

  std::unique_ptr<core::Buffer> mCameraBuffer;
  std::unique_ptr<core::Buffer> mObjectBuffer;

  // Importance-sampling tables for the environment map and emissive triangles, rebuilt with
  // the scene descriptor set. Owns their GPU storage buffers.
  IntegratorTables mIntegratorTables;

  std::map<std::string, std::shared_ptr<resource::SVTexture>> mCustomTextures;
  std::map<std::string, std::shared_ptr<resource::SVCubemap>> mCustomCubemaps;
  std::shared_ptr<resource::SVCubemap> mEnvironmentMap{};

  std::unique_ptr<core::CommandPool> mRenderCommandPool;
  vk::UniqueCommandBuffer mRenderCommandBuffer;
  vk::UniqueCommandBuffer mPostprocessCommandBuffer;

  std::unique_ptr<core::CommandPool> mDisplayCommandPool;
  vk::UniqueCommandBuffer mDisplayCommandBuffer;

  std::vector<uint8_t> mPushConstantBuffer;

  bool mRequiresRebuild{true};
  bool mExternalCameraUpdates{false};
  bool mExternalTransformUpdates{false};
  bool mExternalTransformCudaInterop{false};
  bool mEffectiveExternalTransformUpdates{false};
  bool mEffectiveExternalTransformCudaInterop{false};

  vk::UniqueFence mSceneAccessFence;

#ifdef SVULKAN2_CUDA_INTEROP
  std::unique_ptr<class Denoiser> mDenoiser;
  std::string mDenoiseColorName;
  std::string mDenoiseAlbedoName;
  std::string mDenoiseNormalName;
#endif
};

} // namespace renderer
} // namespace svulkan2
