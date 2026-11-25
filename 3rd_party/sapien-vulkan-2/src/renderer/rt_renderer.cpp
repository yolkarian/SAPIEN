#include "svulkan2/renderer/rt_renderer.h"
#include "../common/logger.h"
#include "denoiser.h"
#include "svulkan2/common/profiler.h"
#include "svulkan2/core/context.h"
#include "svulkan2/resource/manager.h"
#include "svulkan2/shader/rt.h"
#include <map>
#include <utility>

namespace svulkan2 {
namespace renderer {
RTRenderer::RTRenderer(std::shared_ptr<core::Context> context, std::string const &shaderDir,
                       ShaderDefines const &shaderDefines,
                       std::vector<std::string> const &postPasses)
    : mContext(std::move(context)), mShaderDir(shaderDir), mShaderDefines(shaderDefines) {
  if (!mContext) {
    throw std::runtime_error("failed to create ray tracing renderer: context must not be null");
  }
  if (!mContext->isVulkanAvailable()) {
    return;
  }
  if (!mContext->isRayTracingAvailable()) {
    logger::error("The selected GPU does not support ray tracing");
    return;
  }
  mShaderPack = resource::SVResourceManager::Get()->CreateRTShaderPack(shaderDir, mShaderDefines);

  mGeometryInstanceBufferLayout = mShaderPack->getGeometryInstanceBufferLayout();
  mCameraBufferLayout = mShaderPack->getCameraBufferLayout();
  mObjectBufferLayout = mShaderPack->getObjectBufferLayout();

  // set default values
  mCustomPropertiesInt["spp"] = 4;
  mCustomPropertiesInt["maxDepth"] = 3;
  mCustomPropertiesInt["russianRoulette"] = 1;
  mCustomPropertiesInt["russianRouletteMinBounces"] = 2;
  mCustomPropertiesFloat["aperture"] = 0.f;
  mCustomPropertiesFloat["focusPlane"] = 1.f;

  mPostChain = std::make_unique<PostChain>(mContext, postPasses, mShaderDefines);
  if (mPostChain->getPropertyType("toneMapper") == CustomPropertyType::eInt) {
    mPostChain->setProperty("toneMapper", 2);
  }
  if (mPostChain->getPropertyType("exposure") == CustomPropertyType::eFloat) {
    mPostChain->setProperty("exposure", 1.f);
  }

  mSceneAccessFence =
      mContext->getDevice().createFenceUnique({vk::FenceCreateFlagBits::eSignaled});
}

void RTRenderer::resize(int width, int height) {
  if (mWidth != width || mHeight != height) {
    mWidth = width;
    mHeight = height;
    mRequiresRebuild = true;
  }
}

void RTRenderer::setScene(std::shared_ptr<scene::Scene> scene) {
  if (scene && scene->getContext() != mContext) {
    throw std::runtime_error(
        "failed to set scene: the scene renders on a different device than the renderer");
  }
  if (mScene == scene) {
    return;
  }
  if (mScene) {
    mScene->unregisterAccessFence(mSceneAccessFence.get());
  }

  mScene = scene;
  mSceneVersion = 0l;
  mSceneRenderVersion = 0l;

  if (mScene) {
    mScene->registerAccessFence(mSceneAccessFence.get());
  }
}

void RTRenderer::prepareObjects() {
  auto objects = mScene->getObjects();
  // auto objects = mScene->getVisibleObjects();
  mObjectBuffer = core::Buffer::Create(
      *mContext, std::max(size_t(1), objects.size()) * mObjectBufferLayout.size,
      vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_CPU_TO_GPU);
}

void RTRenderer::updateObjects() {
  auto objects = mScene->getObjects();
  // auto objects = mScene->getVisibleObjects();

  uint32_t segOffset = mObjectBufferLayout.elements.at("segmentation").offset;
  uint32_t transparencyOffset = mObjectBufferLayout.elements.at("transparency").offset;
  uint32_t shadeFlatOffset = mObjectBufferLayout.elements.at("shadeFlat").offset;

  uint8_t *memory = static_cast<uint8_t *>(mObjectBuffer->map());
  uint32_t stride = mObjectBufferLayout.size;

  uint32_t count = 0;
  for (auto obj : objects) {
    int shadeFlat = obj->getShadeFlat();
    glm::uvec4 seg = obj->getSegmentation();
    float transparency = obj->getTransparency();

    std::memcpy(memory + stride * count + segOffset, &seg, sizeof(glm::uvec4));
    std::memcpy(memory + stride * count + shadeFlatOffset, &shadeFlat, sizeof(int));
    std::memcpy(memory + stride * count + transparencyOffset, &transparency, sizeof(float));

    ++count;
  }
  mObjectBuffer->unmap();
}

void RTRenderer::prepareRender(scene::Camera &camera) {
  bool sceneExternallyOwned = mScene->hasExternalTransformOwnership();
  bool effectiveExternalUpdates = mExternalTransformUpdates || sceneExternallyOwned;
  bool effectiveCudaInterop = mExternalTransformCudaInterop || sceneExternallyOwned;
  if (effectiveExternalUpdates != mEffectiveExternalTransformUpdates ||
      effectiveCudaInterop != mEffectiveExternalTransformCudaInterop) {
    mEffectiveExternalTransformUpdates = effectiveExternalUpdates;
    mEffectiveExternalTransformCudaInterop = effectiveCudaInterop;
    mRequiresRebuild = true;
  }

  if (mEnvironmentMap != mScene->getEnvironmentMap()) {
    mEnvironmentMap = mScene->getEnvironmentMap();
    if (mEnvironmentMap) {
      mEnvironmentMap->load();
    }
    mRequiresRebuild = true;
  }

  // A batched camera can render a SceneGroup while remaining attached to its
  // original scene. RT resources must be built for the renderer's scene,
  // which is also the scene consumed by prepareScene().
  auto scene = mScene;
  // another renderer of the scene may have replaced its tables (a material gained a texture)
  if (mSceneVersion != scene->getVersion() || mRTTablesVersion != scene->getRTTablesVersion()) {
    mRequiresRebuild = true;
  }

  // material edits repack the scene's materials[]; one adding a texture rebuilds the tables
  bool emissionChanged = false;
  if (!mRequiresRebuild) {
    auto materials = scene->uploadRTMaterials();
    if (!materials) {
      mRequiresRebuild = true;
    } else {
      mAccumulationDirty |= materials->content != mMaterialVersions.content;
      emissionChanged = materials->emission != mMaterialVersions.emission;
      mMaterialVersions = *materials;
    }
  }

  if (mRequiresRebuild) {
    mFrameCount = 0;

    {
      SVULKAN2_PROFILE_BLOCK("Build RT acceleration structures");
      scene->buildRTResources(mGeometryInstanceBufferLayout,
                              mEffectiveExternalTransformCudaInterop);
    }
    mRTTablesVersion = scene->getRTTablesVersion();
    // zeros when another thread dropped the tables meanwhile: the next frame rebuilds
    mMaterialVersions = scene->uploadRTMaterials().value_or(scene::Scene::RTMaterialVersions{});

    mShaderPackInstance = std::make_shared<shader::RayTracingShaderPackInstance>(
        mContext,
        shader::RayTracingShaderPackInstanceDesc{
            .shaderDir = mShaderDir,
            .defines = mShaderDefines,
            .maxMeshes = static_cast<uint32_t>(mScene->getRTVertexBuffers().size()),
            .maxTextures = static_cast<uint32_t>(mScene->getRTTextures().size()),
            .maxPointSets = static_cast<uint32_t>(mScene->getRTPointSetBuffers().size())});
    prepareOutput();
    prepareCamera();
    prepareScene();
    preparePostChain();

#ifdef SVULKAN2_CUDA_INTEROP
    if (mDenoiser) {
      if (static_cast<int>(mDenoiser->getWidth()) != mWidth ||
          static_cast<int>(mDenoiser->getHeight()) != mHeight) {
        mDenoiser->allocate(mWidth, mHeight);
      }
    }
#endif

    mSceneVersion = scene->getVersion();
  } else if (emissionChanged) {
    // the emissive light-sampling tables hold the radiance of the materials
    prepareScene();
  }

  if (mSceneRenderVersion != mScene->getRenderVersion() || mRequiresRebuild) {
    if (mExecutionMode == RenderExecutionMode::eCpuManaged) {
      updateObjects();

      {
        SVULKAN2_PROFILE_BLOCK("Update RT acceleration structures");
        mScene->updateRTResources(mEffectiveExternalTransformUpdates);
      }
      camera.uploadToDevice(*mCameraBuffer, mCameraBufferLayout); // update camera
    } else {
      // Grouped-GPU mode: the render-version bump only signals transform updates and
      // accumulation reset. Object metadata refreshes only on a rebuild; light and
      // camera state upload explicitly through uploadCpuSceneLightState() /
      // uploadCpuCameraState() when dirty.
      if (mRequiresRebuild) {
        updateObjects();
      }
      SVULKAN2_PROFILE_BLOCK("Update RT acceleration structures");
      mScene->updateRTTransforms(true);
    }
    mFrameCount = 0;
    mSceneRenderVersion = mScene->getRenderVersion();
  } else if (mAccumulationDirty) {
    // a property that changes the estimate was edited: old samples no longer belong to it
    mFrameCount = 0;
  } else {
    mFrameCount += 1;
  }
  mAccumulationDirty = false;

  updatePushConstant();

  recordRender();
  recordPostprocess();

  mRequiresRebuild = false;
}

void RTRenderer::updatePushConstant() {
  auto pushConstantLayout = mShaderPack->getPushConstantLayout();
  mPushConstantBuffer.resize(pushConstantLayout->size);

  auto layout = mShaderPack->getPushConstantLayout();
  for (auto &[name, value] : mCustomPropertiesInt) {
    auto it = layout->elements.find(name);
    if (it != layout->elements.end() && it->second.dtype == DataType::INT()) {
      std::memcpy(mPushConstantBuffer.data() + it->second.offset, &value, it->second.size);
    }
  }
  for (auto &[name, value] : mCustomPropertiesFloat) {
    auto it = layout->elements.find(name);
    if (it != layout->elements.end() && it->second.dtype == DataType::FLOAT()) {
      std::memcpy(mPushConstantBuffer.data() + it->second.offset, &value, it->second.size);
    }
  }
  for (auto &[name, value] : mCustomPropertiesVec3) {
    auto it = layout->elements.find(name);
    if (it != layout->elements.end() && it->second.dtype == DataType::FLOAT3()) {
      std::memcpy(mPushConstantBuffer.data() + it->second.offset, &value, it->second.size);
    }
  }
  for (auto &[name, value] : mCustomPropertiesVec4) {
    auto it = layout->elements.find(name);
    if (it != layout->elements.end() && it->second.dtype == DataType::FLOAT4()) {
      std::memcpy(mPushConstantBuffer.data() + it->second.offset, &value, it->second.size);
    }
  }

  if (layout->elements.contains("pointLightCount")) {
    auto &elem = layout->elements.at("pointLightCount");
    if (elem.dtype != DataType::INT()) {
      throw std::runtime_error("pointLightCount must be type int");
    }
    uint32_t v = mScene->getPointLights().size();
    std::memcpy(mPushConstantBuffer.data() + elem.offset, &v, elem.size);
  }

  if (layout->elements.contains("directionalLightCount")) {
    auto &elem = layout->elements.at("directionalLightCount");
    if (elem.dtype != DataType::INT()) {
      throw std::runtime_error("directionalLightCount must be type int");
    }
    uint32_t v = mScene->getDirectionalLights().size();
    std::memcpy(mPushConstantBuffer.data() + elem.offset, &v, elem.size);
  }

  if (layout->elements.contains("spotLightCount")) {
    auto &elem = layout->elements.at("spotLightCount");
    if (elem.dtype != DataType::INT()) {
      throw std::runtime_error("spotLightCount must be type int");
    }
    uint32_t v = mScene->getSpotLights().size() + mScene->getTexturedLights().size();
    std::memcpy(mPushConstantBuffer.data() + elem.offset, &v, elem.size);
  }

  if (layout->elements.contains("parallelogramLightCount")) {
    auto &elem = layout->elements.at("parallelogramLightCount");
    if (elem.dtype != DataType::INT()) {
      throw std::runtime_error("parallelogramLightCount must be type int");
    }
    uint32_t v = mScene->getParallelogramLights().size();
    std::memcpy(mPushConstantBuffer.data() + elem.offset, &v, elem.size);
  }

  if (layout->elements.contains("frameCount")) {
    auto &elem = layout->elements.at("frameCount");
    if (elem.dtype != DataType::INT()) {
      throw std::runtime_error("frameCount must be type int");
    }
    std::memcpy(mPushConstantBuffer.data() + elem.offset, &mFrameCount, elem.size);
  }

  if (layout->elements.contains("frameSeed")) {
    auto &elem = layout->elements.at("frameSeed");
    if (elem.dtype != DataType::INT()) {
      throw std::runtime_error("frameSeed must be type int");
    }
    // Counts this renderer's renders and never resets, unlike frameCount: grouped rendering
    // restarts accumulation every frame and would otherwise repeat one noise pattern.
    int seed = static_cast<int>(mRenderCount++);
    std::memcpy(mPushConstantBuffer.data() + elem.offset, &seed, elem.size);
  }

  if (layout->elements.contains("envmap")) {
    auto &elem = layout->elements.at("envmap");
    if (elem.dtype != DataType::INT()) {
      throw std::runtime_error("envmap must be type int");
    }
    int envmap = mScene->getEnvironmentMap() != nullptr;
    std::memcpy(mPushConstantBuffer.data() + elem.offset, &envmap, elem.size);
  }

  if (layout->elements.contains("ambientLight")) {
    auto &elem = layout->elements.at("ambientLight");
    if (elem.dtype != DataType::FLOAT3()) {
      throw std::runtime_error("ambientLight must be type vec3");
    }
    glm::vec3 ambientLight = mScene->getAmbientLight();
    std::memcpy(mPushConstantBuffer.data() + elem.offset, &ambientLight, elem.size);
  }

  // integrator table sizes, written by the renderer rather than by pack.json
  for (auto [name, value] : {std::pair<char const *, int>{"envWidth", mIntegratorTables.envWidth},
                             {"emissiveCount", mIntegratorTables.emissiveCount}}) {
    auto it = layout->elements.find(name);
    if (it != layout->elements.end() && it->second.dtype == DataType::INT()) {
      std::memcpy(mPushConstantBuffer.data() + it->second.offset, &value, sizeof(value));
    }
  }
}

void RTRenderer::recordRender() {
  if (!mRenderCommandPool) {
    mRenderCommandPool = mContext->createCommandPool();
    mRenderCommandBuffer = mRenderCommandPool->allocateCommandBuffer();
    mPostprocessCommandBuffer = mRenderCommandPool->allocateCommandBuffer();
  }

  mRenderCommandBuffer->reset();

  auto pushConstantLayout = mShaderPack->getPushConstantLayout();

  mRenderCommandBuffer->begin(vk::CommandBufferBeginInfo({}, {}));

  {
    // make sure AS building is done
    vk::MemoryBarrier barrier(vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                              vk::AccessFlagBits::eAccelerationStructureReadKHR);
    mRenderCommandBuffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
        vk::PipelineStageFlagBits::eRayTracingShaderKHR, {}, barrier, {}, {});
  }

  for (auto &image : mRTImages) {
    // make sure everything is done
    image->getImage().transitionLayout(
        mRenderCommandBuffer.get(), vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
        vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
        vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eRayTracingShaderKHR);
  }

  mRenderCommandBuffer->pushConstants(
      mShaderPackInstance->getPipelineLayout(),
      vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eMissKHR |
          vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eAnyHitKHR,
      0, pushConstantLayout->size, mPushConstantBuffer.data());
  mRenderCommandBuffer->bindPipeline(vk::PipelineBindPoint::eRayTracingKHR,
                                     mShaderPackInstance->getPipeline());
  std::vector<vk::DescriptorSet> descriptorSets;
  auto const &resources = mShaderPack->getResources();
  for (uint32_t sid = 0; sid < resources.size(); ++sid) {
    if (resources.at(sid).type == shader::UniformBindingType::eRTOutput) {
      descriptorSets.push_back(mOutputSet.get());
    } else if (resources.at(sid).type == shader::UniformBindingType::eRTScene) {
      descriptorSets.push_back(mSceneSet.get());
    } else if (resources.at(sid).type == shader::UniformBindingType::eRTCamera) {
      descriptorSets.push_back(mCameraSet.get());
    } else {
      throw std::runtime_error("unrecognized descriptor set");
    }
  }

  mRenderCommandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR,
                                           mShaderPackInstance->getPipelineLayout(), 0,
                                           descriptorSets, {});

  mRenderCommandBuffer->traceRaysKHR(mShaderPackInstance->getRgenRegion(),
                                     mShaderPackInstance->getMissRegion(),
                                     mShaderPackInstance->getHitRegion(),
                                     mShaderPackInstance->getCallRegion(), mWidth, mHeight, 1);

  // make sure ray tracing is done

#ifdef SVULKAN2_CUDA_INTEROP
  if (mDenoiser) {
    mRenderImages.at(mDenoiseColorName)
        ->getImage()
        .transitionLayout(
            mRenderCommandBuffer.get(), vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
            vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead,
            vk::PipelineStageFlagBits::eRayTracingShaderKHR, vk::PipelineStageFlagBits::eTransfer);
    mRenderImages.at(mDenoiseAlbedoName)
        ->getImage()
        .transitionLayout(
            mRenderCommandBuffer.get(), vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
            vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead,
            vk::PipelineStageFlagBits::eRayTracingShaderKHR, vk::PipelineStageFlagBits::eTransfer);
    mRenderImages.at(mDenoiseNormalName)
        ->getImage()
        .transitionLayout(
            mRenderCommandBuffer.get(), vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
            vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead,
            vk::PipelineStageFlagBits::eRayTracingShaderKHR, vk::PipelineStageFlagBits::eTransfer);
  }
#endif

  mRenderCommandBuffer->end();
}

void RTRenderer::prepareOutput() {
  mRenderImages.clear();
  mRTImages.clear();

  auto &set = mShaderPack->getOutputDescription();
  if (mWidth <= 0 || mHeight <= 0) {
    return;
  }

  // create images
  std::vector<std::shared_ptr<resource::SVStorageImage>> images;
  for (auto &[bid, binding] : set.bindings) {
    if (binding.type == vk::DescriptorType::eStorageImage && binding.name.starts_with("out")) {
      std::string texName = binding.name.substr(3);
      auto image = mRenderImages[texName] = std::make_shared<resource::SVStorageImage>(
          mContext, texName, mWidth, mHeight, binding.format);
      mRTImages.push_back(image);
      images.push_back(image);
      image->createDeviceResources();
    }
  }

  auto pool = mContext->createCommandPool();
  auto commandBuffer = pool->allocateCommandBuffer();
  commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  for (auto image : images) {
    image->getImage().transitionLayout(
        commandBuffer.get(), vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, {},
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
        vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eRayTracingShaderKHR);
  }
  commandBuffer->end();
  mContext->getQueue().submitAndWait(commandBuffer.get());

  mOutputSet.reset();

  // make descriptor set
  mOutputPool = std::make_unique<core::DynamicDescriptorPool>(
      mContext->getDevice2(),
      std::vector<vk::DescriptorPoolSize>{
          {vk::DescriptorType::eStorageImage, static_cast<uint32_t>(images.size())}});
  mOutputSet = mOutputPool->allocateSet(mShaderPackInstance->getOutputSetLayout());

  // write descriptor sets
  std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
  std::vector<vk::DescriptorImageInfo> imageInfos;
  for (auto img : images) {
    imageInfos.push_back(
        vk::DescriptorImageInfo(VK_NULL_HANDLE, img->getImageView(), vk::ImageLayout::eGeneral));
  }
  for (uint32_t binding = 0; binding < imageInfos.size(); ++binding) {
    writeDescriptorSets.push_back(vk::WriteDescriptorSet(
        mOutputSet.get(), binding, 0, vk::DescriptorType::eStorageImage, imageInfos.at(binding)));
  }
  mContext->getDevice().updateDescriptorSets(writeDescriptorSets, {});
}

void RTRenderer::prepareScene() {
  prepareObjects();
  // light-sampling tables of the environment and emissive meshes (scene_set.glsl)
  mIntegratorTables.build(*mContext, *mScene, mEnvironmentMap.get());

  auto &set = mShaderPack->getSceneDescription();
  uint32_t asCount{0};
  uint32_t storageBufferCount{0};
  uint32_t textureCount{0};
  for (auto &[bid, binding] : set.bindings) {
    switch (binding.type) {
    case vk::DescriptorType::eAccelerationStructureKHR:
      ++asCount;
      break;
    case vk::DescriptorType::eStorageBuffer:
      ++storageBufferCount;
      break;
    case vk::DescriptorType::eCombinedImageSampler:
      ++textureCount;
      break;
    default:
      throw std::runtime_error("scene set should only contain acceleration "
                               "structures, storage buffers, and textures");
    }
  }

  std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
  auto as = mScene->getTLAS()->getVulkanAS();
  vk::WriteDescriptorSetAccelerationStructureKHR asWrite(as);

  vk::DescriptorBufferInfo objectBufferInfo(mObjectBuffer->getVulkanBuffer(), 0, VK_WHOLE_SIZE);
  vk::DescriptorBufferInfo geometryInstanceBufferInfo(mScene->getRTGeometryInstanceBuffer(), 0,
                                                      VK_WHOLE_SIZE);
  vk::DescriptorBufferInfo materialBufferInfo(mScene->getRTMaterialBuffer(), 0, VK_WHOLE_SIZE);
  vk::DescriptorBufferInfo pointInstanceBufferInfo(mScene->getRTPointInstanceBuffer(), 0,
                                                   VK_WHOLE_SIZE);

  std::vector<vk::DescriptorImageInfo> textureInfos;
  for (auto [view, sampler] : mScene->getRTTextures()) {
    textureInfos.push_back(
        vk::DescriptorImageInfo(sampler, view, vk::ImageLayout::eShaderReadOnlyOptimal));
  }
  std::vector<vk::DescriptorBufferInfo> vertexBufferInfos;
  for (auto buffer : mScene->getRTVertexBuffers()) {
    vertexBufferInfos.push_back({buffer, 0, VK_WHOLE_SIZE});
  }
  std::vector<vk::DescriptorBufferInfo> indexBufferInfos;
  for (auto buffer : mScene->getRTIndexBuffers()) {
    indexBufferInfos.push_back({buffer, 0, VK_WHOLE_SIZE});
  }
  std::vector<vk::DescriptorBufferInfo> pointsetBufferInfos;
  for (auto buffer : mScene->getRTPointSetBuffers()) {
    pointsetBufferInfos.push_back({buffer, 0, VK_WHOLE_SIZE});
  }

  mSceneSet.reset();
  mScenePool = std::make_unique<core::DynamicDescriptorPool>(
      mContext->getDevice2(),
      std::vector<vk::DescriptorPoolSize>{
          {vk::DescriptorType::eAccelerationStructureKHR, asCount},
          {vk::DescriptorType::eStorageBuffer,
           storageBufferCount +
               static_cast<uint32_t>(vertexBufferInfos.size() + indexBufferInfos.size())},
          {vk::DescriptorType::eCombinedImageSampler,
           textureCount + static_cast<uint32_t>(textureInfos.size())}});
  mSceneSet = mScenePool->allocateSet(mShaderPackInstance->getSceneSetLayout());

  vk::DescriptorBufferInfo pointLightBufferInfo(mScene->getRTPointLightBuffer(), 0, VK_WHOLE_SIZE);
  vk::DescriptorBufferInfo directionalLightBufferInfo(mScene->getRTDirectionalLightBuffer(), 0,
                                                      VK_WHOLE_SIZE);
  vk::DescriptorBufferInfo spotLightBufferInfo(mScene->getRTSpotLightBuffer(), 0, VK_WHOLE_SIZE);
  vk::DescriptorBufferInfo parallelogramLightBufferInfo(mScene->getRTParallelogramLightBuffer(), 0,
                                                        VK_WHOLE_SIZE);

  auto cube = mEnvironmentMap;
  if (!cube) {
    cube = resource::SVResourceManager::Get()->getDefaultCubemap();
  }
  cube->uploadToDevice(*mContext);
  vk::DescriptorImageInfo cubeMapInfo(cube->getSampler(*mContext), cube->getImageView(*mContext),
                                      vk::ImageLayout::eShaderReadOnlyOptimal);

  // the light-sampling tables, and the TLAS rows for the current transforms of emissive meshes
  auto wholeBuffer = [](core::Buffer &buffer) {
    return vk::DescriptorBufferInfo(buffer.getVulkanBuffer(), 0, VK_WHOLE_SIZE);
  };
  std::map<std::string, vk::DescriptorBufferInfo> const integratorInfos{
      {"EnvAlias", wholeBuffer(*mIntegratorTables.envAliasBuffer)},
      {"EmissiveTriangles", wholeBuffer(*mIntegratorTables.emissiveBuffer)},
      {"EmissiveAlias", wholeBuffer(*mIntegratorTables.emissiveAliasBuffer)},
      {"EmissiveBase", wholeBuffer(*mIntegratorTables.emissiveBaseBuffer)},
      {"TlasInstances", wholeBuffer(mScene->getTLAS()->getInstanceBuffer())}};

  for (auto &[bid, binding] : set.bindings) {
    if (binding.name == "tlas") {
      writeDescriptorSets.push_back(vk::WriteDescriptorSet(
          mSceneSet.get(), bid, 0, 1, vk::DescriptorType::eAccelerationStructureKHR, {}, {}, {},
          &asWrite));
    } else if (binding.name == "Objects") {
      writeDescriptorSets.push_back(vk::WriteDescriptorSet(
          mSceneSet.get(), bid, 0, vk::DescriptorType::eStorageBuffer, {}, objectBufferInfo, {}));
    } else if (binding.name == "GeometryInstances") {
      writeDescriptorSets.push_back(vk::WriteDescriptorSet(mSceneSet.get(), bid, 0,
                                                           vk::DescriptorType::eStorageBuffer, {},
                                                           geometryInstanceBufferInfo, {}));
    } else if (binding.name == "Materials") {
      writeDescriptorSets.push_back(vk::WriteDescriptorSet(mSceneSet.get(), bid, 0,
                                                           vk::DescriptorType::eStorageBuffer, {},
                                                           materialBufferInfo, {}));
    } else if (binding.name == "textures") {
      if (textureInfos.size()) {
        writeDescriptorSets.push_back(vk::WriteDescriptorSet(
            mSceneSet.get(), bid, 0, vk::DescriptorType::eCombinedImageSampler, textureInfos, {},
            {}));
      }
    } else if (binding.name == "Vertices") {
      if (vertexBufferInfos.size()) {
        writeDescriptorSets.push_back(vk::WriteDescriptorSet(mSceneSet.get(), bid, 0,
                                                             vk::DescriptorType::eStorageBuffer,
                                                             {}, vertexBufferInfos, {}));
      }
    } else if (binding.name == "Indices") {
      if (indexBufferInfos.size()) {
        writeDescriptorSets.push_back(vk::WriteDescriptorSet(mSceneSet.get(), bid, 0,
                                                             vk::DescriptorType::eStorageBuffer,
                                                             {}, indexBufferInfos, {}));
      }
    } else if (binding.name == "PointLights") {
      writeDescriptorSets.push_back(vk::WriteDescriptorSet(mSceneSet.get(), bid, 0,
                                                           vk::DescriptorType::eStorageBuffer, {},
                                                           pointLightBufferInfo, {}));
    } else if (binding.name == "DirectionalLights") {
      writeDescriptorSets.push_back(vk::WriteDescriptorSet(mSceneSet.get(), bid, 0,
                                                           vk::DescriptorType::eStorageBuffer, {},
                                                           directionalLightBufferInfo, {}));
    } else if (binding.name == "SpotLights") {
      writeDescriptorSets.push_back(vk::WriteDescriptorSet(mSceneSet.get(), bid, 0,
                                                           vk::DescriptorType::eStorageBuffer, {},
                                                           spotLightBufferInfo, {}));
    } else if (binding.name == "ParallelogramLights") {
      writeDescriptorSets.push_back(vk::WriteDescriptorSet(mSceneSet.get(), bid, 0,
                                                           vk::DescriptorType::eStorageBuffer, {},
                                                           parallelogramLightBufferInfo, {}));
    } else if (binding.name == "samplerEnvironment") {
      writeDescriptorSets.push_back(
          vk::WriteDescriptorSet(mSceneSet.get(), bid, 0,
                                 vk::DescriptorType::eCombinedImageSampler, cubeMapInfo, {}, {}));
    }

    else if (binding.name == "PointInstances") {
      writeDescriptorSets.push_back(vk::WriteDescriptorSet(mSceneSet.get(), bid, 0,
                                                           vk::DescriptorType::eStorageBuffer, {},
                                                           pointInstanceBufferInfo, {}));
    } else if (binding.name == "Points") {
      if (pointsetBufferInfos.size()) {
        writeDescriptorSets.push_back(vk::WriteDescriptorSet(mSceneSet.get(), bid, 0,
                                                             vk::DescriptorType::eStorageBuffer,
                                                             {}, pointsetBufferInfos, {}));
      }
    } else if (auto it = integratorInfos.find(binding.name); it != integratorInfos.end()) {
      writeDescriptorSets.push_back(vk::WriteDescriptorSet(
          mSceneSet.get(), bid, 0, vk::DescriptorType::eStorageBuffer, {}, it->second, {}));
    } else {
      throw std::runtime_error("unrecognized scene set binding " + binding.name);
    }
  }
  mContext->getDevice().updateDescriptorSets(writeDescriptorSets, {});
}

void RTRenderer::prepareCamera() {
  auto &set = mShaderPack->getCameraDescription();
  for (auto &[bid, binding] : set.bindings) {
    if (binding.name == "CameraBuffer") {
      if (!mCameraBuffer) {
        mCameraBuffer =
            core::Buffer::CreateUniform(*mContext, set.buffers.at(binding.arrayIndex)->size,
                                        mExternalCameraUpdates, mExternalCameraUpdates);
      }
      if (bid != 0) {
        throw std::runtime_error("CameraBuffer must have binding 0");
      }
    }
  }

  mCameraSet.reset(); // TODO: manage
  mCameraPool = std::make_unique<core::DynamicDescriptorPool>(
      mContext->getDevice2(),
      std::vector<vk::DescriptorPoolSize>{{vk::DescriptorType::eUniformBuffer, 1}}); // TODO adapt
  mCameraSet = mCameraPool->allocateSet(mShaderPackInstance->getCameraSetLayout());

  // write
  vk::DescriptorBufferInfo bufferInfo(mCameraBuffer->getVulkanBuffer(), 0, VK_WHOLE_SIZE);
  vk::WriteDescriptorSet write(mCameraSet.get(), 0, 0, vk::DescriptorType::eUniformBuffer, {},
                               bufferInfo);
  mContext->getDevice().updateDescriptorSets(write, {});
}

void RTRenderer::preparePostChain() {
  if (mPostChain->empty() || mWidth <= 0 || mHeight <= 0) {
    return;
  }
  if (!mRenderImages.contains("HdrColor")) {
    throw std::runtime_error("post-processing needs the ray tracing output outHdrColor");
  }
  if (mRenderImages.contains("Color")) {
    throw std::runtime_error("the ray tracing output outColor collides with the Color image "
                             "written by post-processing");
  }
  auto color = mRenderImages["Color"] = std::make_shared<resource::SVStorageImage>(
      mContext, "Color", mWidth, mHeight, vk::Format::eR32G32B32A32Sfloat);
  color->createDeviceResources();
  auto pool = mContext->createCommandPool();
  auto commandBuffer = pool->allocateCommandBuffer();
  commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  color->getImage().transitionLayout(
      commandBuffer.get(), vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, {},
      vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
      vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader);
  commandBuffer->end();
  mContext->getQueue().submitAndWait(commandBuffer.get());

  mPostChain->prepare(mWidth, mHeight, mRenderImages.at("HdrColor")->getImageView(),
                      color->getImageView(), [this](std::string const &name) {
                        auto it = mRenderImages.find(name);
                        if (it == mRenderImages.end()) {
                          throw std::runtime_error("a post-processing pass reads the image \"" +
                                                   name + "\", which the renderer does not have");
                        }
                        return it->second->getImageView();
                      });
}

void RTRenderer::render(scene::Camera &camera, std::vector<vk::Semaphore> const &waitSemaphores,
                        std::vector<vk::PipelineStageFlags> const &waitStages,
                        std::vector<vk::Semaphore> const &signalSemaphores, vk::Fence fence) {
  if (!mContext->isVulkanAvailable()) {
    return;
  }
  if (!mScene) {
    throw std::runtime_error("setScene must be called before rendering");
  }

  prepareRender(camera);

  mContext->getDevice().resetFences(mSceneAccessFence.get());
  mContext->getQueue().submit(mRenderCommandBuffer.get(), {}, {}, {}, mSceneAccessFence.get());

#ifdef SVULKAN2_CUDA_INTEROP
  if (mDenoiser) {
    mDenoiser->denoise(mRenderImages.at(mDenoiseColorName)->getImage(),
                       &mRenderImages.at(mDenoiseAlbedoName)->getImage(),
                       &mRenderImages.at(mDenoiseNormalName)->getImage());
  }
#endif
  mContext->getQueue().submit(mPostprocessCommandBuffer.get(), waitSemaphores, waitStages,
                              signalSemaphores, fence);
}

void RTRenderer::render(
    scene::Camera &camera, vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &waitSemaphores,
    vk::ArrayProxyNoTemporaries<vk::PipelineStageFlags const> const &waitStageMasks,
    vk::ArrayProxyNoTemporaries<uint64_t const> const &waitValues,
    vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &signalSemaphores,
    vk::ArrayProxyNoTemporaries<uint64_t const> const &signalValues) {
  if (!mContext->isVulkanAvailable()) {
    return;
  }
  if (!mScene) {
    throw std::runtime_error("setScene must be called before rendering");
  }

  prepareRender(camera);

  mContext->getQueue().submit(mRenderCommandBuffer.get(), {}, {}, {}, {});
#ifdef SVULKAN2_CUDA_INTEROP
  if (mDenoiser) {
    mDenoiser->denoise(mRenderImages.at(mDenoiseColorName)->getImage(),
                       &mRenderImages.at(mDenoiseAlbedoName)->getImage(),
                       &mRenderImages.at(mDenoiseNormalName)->getImage());
  }
#endif
  mContext->getQueue().submit(mPostprocessCommandBuffer.get(), waitSemaphores, waitStageMasks,
                              waitValues, signalSemaphores, signalValues, {});
}

void RTRenderer::display(std::string const &imageName, vk::Image backBuffer, vk::Format format,
                         uint32_t width, uint32_t height,
                         std::vector<vk::Semaphore> const &waitSemaphores,
                         std::vector<vk::PipelineStageFlags> const &waitStages,
                         std::vector<vk::Semaphore> const &signalSemaphores, vk::Fence fence) {
  if (!mContext->isPresentAvailable()) {
    throw std::runtime_error("Display failed: present is not enabled.");
  }

  if (!mDisplayCommandBuffer) {
    mDisplayCommandPool = mContext->createCommandPool();
    mDisplayCommandBuffer = mDisplayCommandPool->allocateCommandBuffer();
  }
  mDisplayCommandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  auto renderTarget = mRenderImages.at(imageName);
  auto targetFormat = renderTarget->getFormat();
  if (targetFormat != vk::Format::eR8G8B8A8Unorm &&
      targetFormat != vk::Format::eR32G32B32A32Sfloat) {
    throw std::runtime_error("failed to display: only color textures are supported in display");
  };

  renderTarget->getImage().transitionLayout(
      mDisplayCommandBuffer.get(), vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
      vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eTransferRead,
      vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
      vk::PipelineStageFlagBits::eTransfer);

  {
    vk::ImageSubresourceRange imageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    vk::ImageMemoryBarrier barrier({}, vk::AccessFlagBits::eTransferWrite,
                                   vk::ImageLayout::eUndefined,
                                   vk::ImageLayout::eTransferDstOptimal, VK_QUEUE_FAMILY_IGNORED,
                                   VK_QUEUE_FAMILY_IGNORED, backBuffer, imageSubresourceRange);
    mDisplayCommandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                           vk::PipelineStageFlagBits::eTransfer, {}, nullptr,
                                           nullptr, barrier);
  }
  vk::ImageSubresourceLayers imageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
  vk::ImageBlit imageBlit(imageSubresourceLayers,
                          {{vk::Offset3D{0, 0, 0}, vk::Offset3D{mWidth, mHeight, 1}}},
                          imageSubresourceLayers,
                          {{vk::Offset3D{0, 0, 0},
                            vk::Offset3D{static_cast<int>(width), static_cast<int>(height), 1}}});
  mDisplayCommandBuffer->blitImage(
      renderTarget->getImage().getVulkanImage(), vk::ImageLayout::eGeneral, backBuffer,
      vk::ImageLayout::eTransferDstOptimal, imageBlit, vk::Filter::eNearest);

  // transfer swap chain back
  {
    vk::ImageSubresourceRange imageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    vk::ImageMemoryBarrier barrier(
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eMemoryRead,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eColorAttachmentOptimal,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, backBuffer, imageSubresourceRange);
    mDisplayCommandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                           vk::PipelineStageFlagBits::eAllCommands, {}, nullptr,
                                           nullptr, barrier);
  }

  mDisplayCommandBuffer->end();
  mContext->getQueue().submit(mDisplayCommandBuffer.get(), waitSemaphores, waitStages,
                              signalSemaphores, fence);
}

std::vector<std::string> RTRenderer::getDisplayTargetNames() const {
  std::vector<std::string> names;
  for (auto &[bid, binding] : mShaderPack->getOutputDescription().bindings) {
    if (binding.type == vk::DescriptorType::eStorageImage && binding.name.starts_with("out")) {
      std::string texName = binding.name.substr(3);
      if (binding.format == vk::Format::eR32G32B32A32Sfloat) {
        names.push_back(texName);
      }
    }
  }
  if (mPostChain && !mPostChain->empty()) {
    names.push_back("Color");
  }
  return names;
}

int RTRenderer::getCustomPropertyInt(std::string const &name) const {
  return isPostProperty(name) ? mPostChain->getPropertyInt(name) : mCustomPropertiesInt.at(name);
}

float RTRenderer::getCustomPropertyFloat(std::string const &name) const {
  return isPostProperty(name) ? mPostChain->getPropertyFloat(name)
                              : mCustomPropertiesFloat.at(name);
}

void RTRenderer::setCustomProperty(std::string const &name, int p) {
  if (isPostProperty(name)) {
    mPostChain->setProperty(name, p);
  } else {
    setProperty(mCustomPropertiesInt, name, p);
  }
}

void RTRenderer::setCustomProperty(std::string const &name, float p) {
  if (isPostProperty(name)) {
    mPostChain->setProperty(name, p);
  } else {
    setProperty(mCustomPropertiesFloat, name, p);
  }
}

CustomPropertyType RTRenderer::getCustomPropertyType(std::string const &name) const {
  if (isPostProperty(name)) {
    return mPostChain->getPropertyType(name);
  }
  if (!mShaderPack) {
    return CustomPropertyType::eNone;
  }
  auto layout = mShaderPack->getPushConstantLayout();
  if (!layout) {
    return CustomPropertyType::eNone;
  }
  auto it = layout->elements.find(name);
  if (it == layout->elements.end()) {
    return CustomPropertyType::eNone;
  }
  if (it->second.dtype == DataType::INT()) {
    return CustomPropertyType::eInt;
  }
  if (it->second.dtype == DataType::FLOAT()) {
    return CustomPropertyType::eFloat;
  }
  if (it->second.dtype == DataType::FLOAT3()) {
    return CustomPropertyType::eVec3;
  }
  if (it->second.dtype == DataType::FLOAT4()) {
    return CustomPropertyType::eVec4;
  }
  return CustomPropertyType::eNone;
}

std::vector<std::string> RTRenderer::getRenderTargetNames() const {
  std::vector<std::string> names;
  for (auto &[bid, binding] : mShaderPack->getOutputDescription().bindings) {
    if (binding.type == vk::DescriptorType::eStorageImage && binding.name.starts_with("out")) {
      std::string texName = binding.name.substr(3);
      names.push_back(texName);
    }
  }
  if (mPostChain && !mPostChain->empty()) {
    names.push_back("Color");
  }
  return names;
}

void RTRenderer::enableDenoiser(DenoiserType type, std::string const &colorName,
                                std::string const &albedoName, std::string const &normalName) {
#ifdef SVULKAN2_CUDA_INTEROP
  if (getDenoiserType() == type) {
    return;
  }
  mRequiresRebuild = true;

  if (type == DenoiserType::eNONE) {
    disableDenoiser();
    return;
  }

  if (type == DenoiserType::eOPTIX) {
    mDenoiser = std::make_unique<DenoiserOptix>();
  } else {
    mDenoiser = std::make_unique<DenoiserOidn>();
  }
  if (!mDenoiser->init(mContext, true, true, true)) {
    logger::error("Failed to initialize denoiser");
    mDenoiser.reset();
    return;
  }

  mDenoiseColorName = colorName;
  mDenoiseAlbedoName = albedoName;
  mDenoiseNormalName = normalName;
#endif
}

RTRenderer::DenoiserType RTRenderer::getDenoiserType() const {
#ifdef SVULKAN2_CUDA_INTEROP
  if (!mDenoiser) {
    return DenoiserType::eNONE;
  }
  if (dynamic_cast<DenoiserOptix *>(mDenoiser.get())) {
    return DenoiserType::eOPTIX;
  }
  return DenoiserType::eOIDN;
#else
  return DenoiserType::eNONE;
#endif
}

void RTRenderer::disableDenoiser() {
#ifdef SVULKAN2_CUDA_INTEROP
  mRequiresRebuild = true;
  mDenoiser.reset();
#endif
}

void RTRenderer::recordPostprocess() {
  mPostprocessCommandBuffer->reset();
  mPostprocessCommandBuffer->begin(vk::CommandBufferBeginInfo({}, {}));

  // the ray tracing pass and the denoiser wrote the images the chain reads
  vk::MemoryBarrier barrier(vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eShaderRead |
                                                                  vk::AccessFlagBits::eShaderWrite);
  mPostprocessCommandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eRayTracingShaderKHR |
                                                 vk::PipelineStageFlagBits::eTransfer,
                                             vk::PipelineStageFlagBits::eComputeShader, {},
                                             barrier, {}, {});
  mPostChain->record(mPostprocessCommandBuffer.get());
  mPostprocessCommandBuffer->end();
}

core::Buffer &RTRenderer::getCameraBuffer() {
  if (!mCameraBuffer) {
    throw std::runtime_error("camera buffer is not initialized");
  }
  return *mCameraBuffer;
}

void RTRenderer::setExternalCameraUpdatesEnabled(bool enable) {
  if (mExternalCameraUpdates == enable) {
    return;
  }
  mExternalCameraUpdates = enable;
  mCameraBuffer.reset();
  mRequiresRebuild = true;
}

void RTRenderer::prepareResources(scene::Camera &camera) { prepareRender(camera); }

void RTRenderer::uploadCpuFrameState(scene::Camera &camera) {
  // requires prepared resources; object metadata is refreshed by prepareRender
  uploadCpuSceneLightState();
  uploadCpuCameraState(camera);
}

void RTRenderer::uploadCpuCameraState(scene::Camera &camera) {
  if (!mCameraBuffer) {
    throw std::runtime_error(
        "uploadCpuCameraState failed: renderer resources are not prepared");
  }
  camera.uploadToDevice(*mCameraBuffer, mCameraBufferLayout);
}

void RTRenderer::uploadCpuSceneLightState() {
  if (!mScene) {
    throw std::runtime_error("uploadCpuSceneLightState failed: scene is not set");
  }
  mScene->uploadRTLights();
}

void RTRenderer::initializeExternalTransformResources(scene::Camera &camera) {
  if (!mExternalTransformUpdates) {
    throw std::runtime_error(
        "external transform updates must be enabled before initializing their resources");
  }
  prepareRender(camera);
}

void RTRenderer::setExternalTransformUpdatesEnabled(bool enable, bool cudaInterop) {
  cudaInterop &= enable;
  if (mExternalTransformUpdates == enable &&
      mExternalTransformCudaInterop == cudaInterop) {
    return;
  }
  mExternalTransformUpdates = enable;
  mExternalTransformCudaInterop = cudaInterop;
  mRequiresRebuild = true;
}

RTRenderer::~RTRenderer() {
  if (mScene && mSceneAccessFence) {
    mScene->unregisterAccessFence(mSceneAccessFence.get());
  }
}

} // namespace renderer
} // namespace svulkan2
