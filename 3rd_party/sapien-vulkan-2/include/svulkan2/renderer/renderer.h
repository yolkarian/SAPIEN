#pragma once
#include "gpu_scene.h"
#include "post_chain.h"
#include "renderer_base.h"
#include "svulkan2/core/context.h"
#include "svulkan2/core/descriptor_pool.h"
#include "svulkan2/resource/cubemap.h"
#include "svulkan2/resource/material.h"
#include "svulkan2/resource/render_target.h"
#include "svulkan2/resource/texture.h"
#include "svulkan2/scene/scene.h"
#include "svulkan2/shader/shader_pack_instance.h"
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace svulkan2 {
namespace core {
class Context;
class CommandPool;
} // namespace core
namespace resource {
class SVMetallicMaterial;
class SVTexture;
} // namespace resource
namespace renderer {
class Renderer : public RendererBase {
  std::shared_ptr<core::Context> mContext;
  std::shared_ptr<RendererConfig> mConfig;

  vk::UniqueDescriptorPool mDescriptorPool;
  // Dedicated pool for the scene-wide GPU scene set (holds a large bindless sampler array).
  vk::UniqueDescriptorPool mGPUScenePool;

  // std::unique_ptr<shader::ShaderManager> mShaderManager;

  std::shared_ptr<shader::ShaderPack> mShaderPack;
  std::shared_ptr<shader::ShaderPackInstance> mShaderPackInstance;
  std::unordered_map<std::string, std::shared_ptr<resource::SVRenderTarget>> mRenderTargets;
  std::unordered_map<std::string, std::shared_ptr<resource::SVRenderTarget>> mMultisampledTargets;

  // shadow targets ================================
  std::vector<uint32_t> mPointLightShadowSizes{};
  std::vector<uint32_t> mDirectionalLightShadowSizes{};
  std::vector<uint32_t> mSpotLightShadowSizes{};
  std::vector<uint32_t> mTexturedLightShadowSizes{};

  std::vector<std::shared_ptr<resource::SVRenderTarget>> mDirectionalShadowReadTargets;
  std::vector<std::shared_ptr<resource::SVRenderTarget>> mDirectionalShadowWriteTargets;

  std::vector<std::shared_ptr<resource::SVRenderTarget>> mPointShadowWriteTargets;
  std::vector<std::shared_ptr<resource::SVRenderTarget>> mPointShadowReadTargets;

  std::vector<std::shared_ptr<resource::SVRenderTarget>> mSpotShadowReadTargets;
  std::vector<std::shared_ptr<resource::SVRenderTarget>> mSpotShadowWriteTargets;

  std::vector<std::shared_ptr<resource::SVRenderTarget>> mTexturedLightShadowReadTargets;
  std::vector<std::shared_ptr<resource::SVRenderTarget>> mTexturedLightShadowWriteTargets;

  std::vector<vk::UniqueDescriptorSet> mLightSets;
  std::vector<std::unique_ptr<core::Buffer>> mLightBuffers;

  std::unique_ptr<core::Buffer> mShadowBuffer;
  // ================================================

  std::unordered_map<std::string, vk::ImageLayout> mRenderTargetFinalLayouts;

  std::vector<vk::UniqueFramebuffer> mFramebuffers;
  std::vector<vk::UniqueFramebuffer> mShadowFramebuffers;
  std::vector<uint32_t> mShadowSizes;

  std::map<std::string, std::vector<std::shared_ptr<resource::SVTexture>>> mCustomTextureArray;
  std::map<std::string, std::shared_ptr<resource::SVTexture>> mCustomTextures;
  std::map<std::string, std::shared_ptr<resource::SVCubemap>> mCustomCubemaps;

  std::shared_ptr<resource::SVCubemap> mEnvironmentMap{};

  int mWidth{};
  int mHeight{};
  bool mRequiresRebuild{true};

  std::unique_ptr<core::Buffer> mSceneBuffer;

  std::unique_ptr<core::Buffer> mCameraBufferCpu;
  std::unique_ptr<core::Buffer> mCameraBuffer;

  // ---- Scene set (common/scene.glsl): one per renderer, bound per draw ----
  // Host-visible buffers rewritten in place from the CPU: the instances (one per drawable,
  // mInstanceStride apart), materials[] and customObjectData[] (one per object). Growing one
  // rebinds the set, so it only happens before recording.
  std::unique_ptr<core::Buffer> mInstanceBuffer;
  std::unique_ptr<core::Buffer> mMaterialBuffer;
  std::unique_ptr<core::Buffer> mCustomDataBuffer;
  uint32_t mInstanceCapacity{0};
  uint32_t mMaterialCapacity{0};
  uint32_t mCustomDataCapacity{0};
  // sizeof(InstanceInfo) rounded up to minUniformBufferOffsetAlignment (a dynamic offset)
  uint32_t mInstanceStride{0};

  /** A drawable: a shape of a body, a line set or a point set. Planned before recording, in
   *  this order; every pass drawing it binds the scene set at its instance and material. */
  struct InstanceRef {
    int32_t kind{};            // 0 = shape, 1 = line set, 2 = point set
    int32_t objectIndex{};     // index into the matching Scene object array
    int32_t transformIndex{};  // object's GPU transform index
    int32_t customDataIndex{}; // -1 for line and point sets
    std::shared_ptr<resource::SVMetallicMaterial> material; // null for line and point sets
  };
  std::vector<InstanceRef> mInstances;
  std::vector<uint32_t> mObjectInstances; // instance of the first shape of each object
  uint32_t mLineInstances{};              // instance of the first line set
  uint32_t mPointInstances{};             // instance of the first point set
  std::vector<uint8_t> mInstanceData;     // CPU mirror of mInstanceBuffer
  std::vector<uint8_t> mCustomData;       // CPU mirror of mCustomDataBuffer
  // material pointer -> index into materials[] / mMaterials.
  std::unordered_map<resource::SVMetallicMaterial const *, uint32_t> mMaterialIndices;
  std::vector<std::shared_ptr<resource::SVMetallicMaterial>> mMaterials;
  // textures[] contents: slot 0 is the default texture, then the deduplicated material and
  // per-object textures; mTextureIndices maps a texture to its slot.
  std::vector<std::shared_ptr<resource::SVTexture>> mTextures;
  std::unordered_map<resource::SVTexture const *, int32_t> mTextureIndices;
  // descriptors last written to textures[] slots [0, size); later slots hold the default
  std::vector<vk::DescriptorImageInfo> mBoundTextures;
  // byte stride of customObjectData[]; 0 when the pack declares no custom data
  uint32_t mCustomDataStride{0};
  uint64_t mLastMaterialVersion{0};

  vk::UniqueDescriptorSet mGPUSceneSet;
  // transform buffer shared with all cameras (from the Scene); owned by the Scene.
  std::shared_ptr<core::Buffer> mObjectTransformBuffer;

  vk::UniqueDescriptorSet mSceneSet;
  vk::UniqueDescriptorSet mCameraSet;

  std::vector<vk::UniqueDescriptorSet> mInputTextureSets;

  bool mSpecializationConstantsChanged = true;
  std::map<std::string, SpecializationConstantValue> mSpecializationConstants;

  bool mRequiresRecord{false};
  uint64_t mLastVersion{0};
  std::shared_ptr<scene::Scene> mScene;

  // Post-processing of kRasterPostSource into kRasterPostResult (common/config.h); recorded
  // before the first pass that samples the result. Its properties are push constants, so a
  // change only re-records.
  std::unique_ptr<PostChain> mPostChain;
  bool hasPostChain() const { return mPostChain && !mPostChain->empty(); }
  bool isPostProperty(std::string const &name) const {
    return mPostChain && mPostChain->getPropertyType(name) != CustomPropertyType::eNone;
  }
  void recordPostChain();

#ifdef SVULKAN2_CUDA_INTEROP
  std::map<std::string, std::shared_ptr<core::Buffer>> mTransferBuffers;
#endif
public:
  Renderer(std::shared_ptr<core::Context> context, std::shared_ptr<RendererConfig> config);

  template <typename T> void setSpecializationConstant(std::string const &name, T value) {
    if (!mContext->isVulkanAvailable()) {
      return;
    }

    if (!mSpecializationConstants.contains(name)) {
      mSpecializationConstantsChanged = true;
      mSpecializationConstants[name] = value;
      return;
    }

    SpecializationConstantValue newConstant;
    newConstant = value;
    if (mSpecializationConstants[name] != newConstant) {
      mSpecializationConstantsChanged = true;
      mSpecializationConstants[name] = value;
    }
  }

  void resize(int width, int height) override;

  void render(scene::Camera &camera,
              vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &waitSemaphores,
              vk::ArrayProxyNoTemporaries<vk::PipelineStageFlags const> const &waitStageMasks,
              vk::ArrayProxyNoTemporaries<uint64_t const> const &waitValues,
              vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &signalSemaphores,
              vk::ArrayProxyNoTemporaries<uint64_t const> const &signalValues) override;

  /** render may be called on a thread. */
  void render(scene::Camera &camera, std::vector<vk::Semaphore> const &waitSemaphores,
              std::vector<vk::PipelineStageFlags> const &waitStages,
              std::vector<vk::Semaphore> const &signalSemaphores, vk::Fence fence) override;

  void display(std::string const &renderTargetName, vk::Image backBuffer, vk::Format format,
               uint32_t width, uint32_t height, std::vector<vk::Semaphore> const &waitSemaphores,
               std::vector<vk::PipelineStageFlags> const &waitStages,
               std::vector<vk::Semaphore> const &signalSemaphores, vk::Fence fence) override;

  void setScene(std::shared_ptr<scene::Scene> scene) override;

  std::vector<std::string> getDisplayTargetNames() const override;
  std::vector<std::string> getRenderTargetNames() const override;

  std::shared_ptr<resource::SVRenderTarget> getRenderTarget(std::string const &name) const;

  core::Image &getRenderImage(std::string const &name) override {
    return getRenderTarget(name)->getImage();
  }

  vk::ImageLayout getRenderTargetImageLayout(std::string const &name) override {
    return mRenderTargetFinalLayouts.at(name);
  }

  void setCustomTextureArray(std::string const &name,
                             std::vector<std::shared_ptr<resource::SVTexture>> texture) override;
  void setCustomTexture(std::string const &name,
                        std::shared_ptr<resource::SVTexture> texture) override;
  void setCustomCubemap(std::string const &name,
                        std::shared_ptr<resource::SVCubemap> cubemap) override;

  CustomPropertyType getCustomPropertyType(std::string const &name) const override;
  int getCustomPropertyInt(std::string const &name) const override;
  float getCustomPropertyFloat(std::string const &name) const override;
  glm::vec3 getCustomPropertyVec3(std::string const &name) const override;
  glm::vec4 getCustomPropertyVec4(std::string const &name) const override;

  void setCustomProperty(std::string const &name, int p) override;
  void setCustomProperty(std::string const &name, float p) override;
  void setCustomProperty(std::string const &name, glm::vec3 p) override;
  void setCustomProperty(std::string const &name, glm::vec4 p) override;

  core::Buffer &getCameraBuffer();

  void prepareResources(scene::Camera &camera) override;
  void uploadCpuFrameState(scene::Camera &camera) override;
  void uploadCpuCameraState(scene::Camera &camera) override;
  void uploadCpuSceneLightState() override;

  void setExternalTransformUpdatesEnabled(bool enable) { mExternalTransformUpdates = enable; }
  bool getExternalTransformUpdatesEnabled() const { return mExternalTransformUpdates; }

  void forceUploadCameraBuffer(scene::Camera &camera);

  Renderer(Renderer const &other) = delete;
  Renderer &operator=(Renderer const &other) = delete;
  Renderer(Renderer &&other) = default;
  Renderer &operator=(Renderer &&other) = default;

private:
  void prepareRenderTargets(uint32_t width, uint32_t height);
  void prepareShadowRenderTargets();
  void preparePipelines();
  void prepareFramebuffers(uint32_t width, uint32_t height);
  void prepareShadowFramebuffers();

  void prepareSceneBuffer();
  void prepareCameraBuffer();
  void prepareLightBuffers();

  // Allocate the scene set on first use; rebind the Scene's transform table when it changed.
  void prepareGPUScene();
  // Grow the instance buffer to instanceCount instances and customObjectData[] to objectCount.
  void ensureGPUBuffers(uint32_t instanceCount, uint32_t objectCount);
  // Rebuild materials[] and textures[] from the scene; true when it rebound a descriptor.
  bool refreshMaterialsAndTextures();
  // Write the instances and the per-object custom data from the CPU objects.
  void uploadInstances();
  // Bind the scene set at set index `set` with the dynamic offsets of instance `instance`.
  void bindSceneSet(vk::CommandBuffer commandBuffer, vk::PipelineLayout layout, uint32_t set,
                    uint32_t instance);

  void prepareInputTextureDescriptorSets();

  std::unordered_set<std::shared_ptr<resource::SVModel>> mModelCache;
  std::unordered_set<std::shared_ptr<resource::SVLineSet>> mLineSetCache;
  std::unordered_set<std::shared_ptr<resource::SVPointSet>> mPointSetCache;

  std::unique_ptr<core::CommandPool> mUploadCommandPool{};
  std::unique_ptr<core::CommandPool> mShadowCommandPool{};
  std::unique_ptr<core::CommandPool> mRenderCommandPool{};
  std::unique_ptr<core::CommandPool> mDisplayCommandPool{};

  vk::UniqueCommandBuffer mUploadCommandBuffer{};
  vk::UniqueCommandBuffer mShadowCommandBuffer{};
  vk::UniqueCommandBuffer mRenderCommandBuffer{};
  vk::UniqueCommandBuffer mDisplayCommandBuffer{};

  void prepareObjects();

  // record data upload commands (e.g. camera buffer upload)
  void recordUpload();

  void recordShadows();
  void recordRenderPasses();

  bool mExternalTransformUpdates{false};

  // load and uplaod required resources
  void prepareRender(scene::Camera &camera);

  // upload resources to gpu; called per frame by render() in eCpuManaged mode and
  // once by uploadCpuFrameState() when a grouped renderer takes its CPU snapshot
  void uploadGpuResources(scene::Camera &camera);

  // host-write the camera staging buffer; the recorded upload command buffer
  // copies it into the device-local camera buffer at the next submission
  void writeCameraBufferCpu(scene::Camera &camera);
  // host-write the scene/light and shadow uniform buffers
  void writeSceneLightBuffers();
};

} // namespace renderer
} // namespace svulkan2
