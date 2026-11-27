#include "svulkan2/renderer/renderer.h"
#include "../common/logger.h"
#include "svulkan2/common/profiler.h"
#include "svulkan2/core/command_pool.h"
#include "svulkan2/resource/manager.h"
#include "svulkan2/shader/deferred.h"
#include "svulkan2/shader/gbuffer.h"
#include "svulkan2/shader/primitive.h"
#include "svulkan2/shader/primitive_shadow.h"
#include "svulkan2/shader/shadow.h"
#include <algorithm>
#include <array>
#include <optional>
#include <utility>

namespace svulkan2 {
namespace renderer {

static void
updateArrayDescriptorSets(vk::Device device, vk::DescriptorSet descriptorSet,
                          std::vector<std::tuple<vk::ImageView, vk::Sampler>> textureData,
                          uint32_t binding) {
  std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
  writeDescriptorSets.reserve(textureData.size());

  std::vector<vk::DescriptorImageInfo> imageInfos;
  imageInfos.reserve(textureData.size());
  for (auto const &tex : textureData) {
    imageInfos.push_back(vk::DescriptorImageInfo(std::get<1>(tex), std::get<0>(tex),
                                                 vk::ImageLayout::eShaderReadOnlyOptimal));
  }
  for (uint32_t idx = 0; idx < imageInfos.size(); ++idx) {
    writeDescriptorSets.push_back(vk::WriteDescriptorSet(descriptorSet, binding, idx, 1,
                                                         vk::DescriptorType::eCombinedImageSampler,
                                                         &imageInfos[idx]));
  }
  device.updateDescriptorSets(writeDescriptorSets, nullptr);
}

static void updateDescriptorSets(vk::Device device,
                                 vk::DescriptorSet descriptorSet,
                                 std::vector<std::tuple<vk::DescriptorType, vk::Buffer, vk::BufferView>> const &bufferData,
                                 std::vector<std::tuple<vk::ImageView, vk::Sampler>> textureData,
                                 uint32_t bindingOffset) {
  std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
  writeDescriptorSets.reserve(bufferData.size() + textureData.size());

  std::vector<vk::DescriptorBufferInfo> bufferInfos;
  bufferInfos.reserve(bufferData.size());
  for (auto const &bd : bufferData) {
    bufferInfos.push_back(vk::DescriptorBufferInfo(std::get<1>(bd), 0, VK_WHOLE_SIZE));
    writeDescriptorSets.push_back(
        vk::WriteDescriptorSet(descriptorSet, bindingOffset++, 0, 1, std::get<0>(bd), nullptr,
                               &bufferInfos.back(), std::get<2>(bd) ? &std::get<2>(bd) : nullptr));
  }
  std::vector<vk::DescriptorImageInfo> imageInfos;
  imageInfos.reserve(textureData.size());
  for (auto const &tex : textureData) {
    imageInfos.push_back(vk::DescriptorImageInfo(std::get<1>(tex), std::get<0>(tex),
                                                 vk::ImageLayout::eShaderReadOnlyOptimal));
    writeDescriptorSets.push_back(
        vk::WriteDescriptorSet(descriptorSet, bindingOffset++, 0, 1,
                               vk::DescriptorType::eCombinedImageSampler, &imageInfos.back()));
  }
  device.updateDescriptorSets(writeDescriptorSets, nullptr);
}

Renderer::Renderer(std::shared_ptr<core::Context> context, std::shared_ptr<RendererConfig> config)
    : mContext(std::move(context)) {
  if (!mContext) {
    throw std::runtime_error("failed to create renderer: context must not be null");
  }
  mConfig = std::make_shared<RendererConfig>(*config);

  if (!mContext->isVulkanAvailable()) {
    return;
  }

  // preload the shader pack to get vertex layouts
  mShaderPack = resource::SVResourceManager::Get()->CreateShaderPack(config->shaderDir,
                                                                  config->shaderDefines);

  mPostChain = std::make_unique<PostChain>(mContext, config->postPasses, config->shaderDefines);

  // Make reflected shader defaults visible through the custom-property API.
  for (auto const &pass : mShaderPack->getNonShadowPasses()) {
    std::shared_ptr<SpecializationConstantLayout> layout;
    if (auto parser = std::dynamic_pointer_cast<shader::GbufferPassParser>(pass)) {
      layout = parser->getSpecializationConstantLayout();
    } else if (auto parser = std::dynamic_pointer_cast<shader::DeferredPassParser>(pass)) {
      layout = parser->getSpecializationConstantLayout();
    } else if (auto parser = std::dynamic_pointer_cast<shader::PrimitivePassParser>(pass)) {
      layout = parser->getSpecializationConstantLayout();
    }
    if (!layout) {
      continue;
    }
    for (auto const &[name, element] : layout->elements) {
      SpecializationConstantValue value;
      value.dtype = element.dtype;
      std::memcpy(value.buffer, element.buffer, element.dtype.size());
      mSpecializationConstants.try_emplace(name, value);
    }
  }

  // make sure only forward renderer uses msaa
  if (mShaderPack->hasDeferredPass()) {
    mConfig->msaa = vk::SampleCountFlagBits::e1;
  }

  // Scene, camera, light (uniform buffer) and input-texture (sampler) sets all come from this
  // pool, so it must hold both descriptor types. A scene set holds 5 uniform buffers.
  vk::DescriptorPoolSize pool_sizes[] = {
      {vk::DescriptorType::eCombinedImageSampler, 256},
      {vk::DescriptorType::eUniformBuffer, 512},
  };
  auto info = vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                                           256, 2, pool_sizes);
  mDescriptorPool = mContext->getDevice().createDescriptorPoolUnique(info);

  // Dedicated pool for the one scene set of this renderer (common/scene.glsl): 3 storage
  // buffers, 2 dynamic uniform buffers and the fixed-size textures[] array.
  vk::DescriptorPoolSize gpu_pool_sizes[] = {
      {vk::DescriptorType::eStorageBuffer, 3},
      {vk::DescriptorType::eUniformBufferDynamic, 2},
      {vk::DescriptorType::eCombinedImageSampler, mContext->getMaxNumTextures()},
  };
  auto gpu_info = vk::DescriptorPoolCreateInfo(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, 3, gpu_pool_sizes);
  mGPUScenePool = mContext->getDevice().createDescriptorPoolUnique(gpu_info);
}

void Renderer::setScene(std::shared_ptr<scene::Scene> scene) {
  if (scene && scene->getContext() != mContext) {
    throw std::runtime_error(
        "failed to set scene: the scene renders on a different device than the renderer");
  }
  mScene = std::move(scene);
  mShadowMaps.reset();
  mRequiresRebuild = true;
  mRequiresRecord = true;
}

void Renderer::setRequestedTargets(std::vector<std::string> const &targets) {
  std::set<std::string> next(targets.begin(), targets.end());
  if (next == mConfig->requestedTargets) {
    return;
  }
  // mConfig is part of the ShaderPackInstance cache key; clone before mutating.
  mConfig = std::make_shared<RendererConfig>(*mConfig);
  mConfig->requestedTargets = std::move(next);
  mRequiresRebuild = true;
  mRequiresRecord = true;
}

static uint32_t scaledExtent(uint32_t dim, float scale) {
  if (scale == 1.f) {
    return dim;
  }
  return std::max(1u, static_cast<uint32_t>(std::ceil(static_cast<float>(dim) * scale)));
}

void Renderer::prepareRenderTargets(uint32_t width, uint32_t height) {
  if (!mContext->isVulkanAvailable()) {
    return;
  }

  mRenderTargets.clear();
  mMultisampledTargets.clear();

  auto renderTargetFormats = mShaderPackInstance->getRenderTargetFormats();
  auto renderTargetScales = mShaderPackInstance->getRenderTargetScales();

  for (auto &[name, format] : renderTargetFormats) {
    float scale = renderTargetScales.contains(name) ? renderTargetScales.at(name) : 1.f;
    uint32_t targetWidth = scaledExtent(width, scale);
    uint32_t targetHeight = scaledExtent(height, scale);
    mRenderTargets[name] = std::make_shared<resource::SVRenderTarget>(
        mContext, name, targetWidth, targetHeight, format, vk::SampleCountFlagBits::e1,
        hasPostChain() && name == kRasterPostSource);
    mRenderTargets[name]->createDeviceResources();
  }

  // TODO: remove single sample depth target
  if (mConfig->msaa != vk::SampleCountFlagBits::e1) {
    for (auto &[name, format] : renderTargetFormats) {
      float scale = renderTargetScales.contains(name) ? renderTargetScales.at(name) : 1.f;
      uint32_t targetWidth = scaledExtent(width, scale);
      uint32_t targetHeight = scaledExtent(height, scale);
      mMultisampledTargets[name] = std::make_shared<resource::SVRenderTarget>(
          mContext, name, targetWidth, targetHeight, format, mConfig->msaa);
      mMultisampledTargets[name]->createDeviceResources();
    }
  }

  mRenderTargetFinalLayouts = mShaderPackInstance->getRenderTargetFinalLayouts();

  if (hasPostChain()) {
    if (mConfig->msaa != vk::SampleCountFlagBits::e1) {
      throw std::runtime_error("post-processing passes are not supported with MSAA");
    }
    auto source = mRenderTargets.find(kRasterPostSource);
    if (source == mRenderTargets.end() ||
        source->second->getFormat() != vk::Format::eR32G32B32A32Sfloat) {
      throw std::runtime_error(std::string("post-processing needs the render target ") +
                               kRasterPostSource + " in the format r32g32b32a32sfloat");
    }
    auto result = mRenderTargets[kRasterPostResult] = std::make_shared<resource::SVRenderTarget>(
        mContext, kRasterPostResult, width, height, vk::Format::eR32G32B32A32Sfloat,
        vk::SampleCountFlagBits::e1, true);
    result->createDeviceResources();
    mRenderTargetFinalLayouts[kRasterPostResult] = vk::ImageLayout::eShaderReadOnlyOptimal;
    mPostChain->prepare(width, height, source->second->getImageView(), result->getImageView(),
                        [](std::string const &name) -> vk::ImageView {
                          throw std::runtime_error(
                              "a post-processing pass of a rasterization pack reads the image \"" +
                              name + "\": only inColor and outColor are available");
                        });
  }
}

void Renderer::prepareShadowRenderTargets() {
  if (!mContext->isVulkanAvailable() || !mScene) {
    return;
  }
  auto lightBufferLayout = mShaderPack->getShaderInputLayouts()->lightBufferLayout;
  mShadowMaps = mScene->getOrCreateRasterShadowMaps(mConfig->depthFormat, lightBufferLayout->size);
}

void Renderer::preparePipelines() {
  if (!mContext->isVulkanAvailable()) {
    return;
  }

  shader::ShaderPackInstanceDesc desc;
  desc.config = mConfig;
  desc.specializationConstants = mSpecializationConstants;
  auto instance =
      resource::SVResourceManager::Get()->CreateShaderPackInstance(mContext, desc);
  instance->loadAsync().get();
  if (instance->getShaderPack() != mShaderPack) {
    throw std::runtime_error("renderer corrupted! impossible error in shader pack.");
  }
  mShaderPackInstance = std::move(instance);
}

void Renderer::prepareFramebuffers(uint32_t width, uint32_t height) {
  if (!mContext->isVulkanAvailable()) {
    return;
  }
  mFramebuffers.clear();

  auto passes = mShaderPack->getNonShadowPasses();

  for (uint32_t i = 0; i < passes.size(); ++i) {
    if (!mShaderPackInstance->isPassActive(i)) {
      mFramebuffers.push_back({});
      continue;
    }
    float scale = passes[i]->getResolutionScale();
    uint32_t fbWidth = scaledExtent(width, scale);
    uint32_t fbHeight = scaledExtent(height, scale);
    auto names = passes[i]->getColorRenderTargetNames();
    std::vector<vk::ImageView> attachments;

    // color attachments and resolve attachments (matching active outputs in createRenderPass)
    if (mConfig->msaa != vk::SampleCountFlagBits::e1) {
      for (uint32_t outIdx = 0; outIdx < names.size(); ++outIdx) {
        if (mShaderPackInstance->isColorOutputActive(i, outIdx)) {
          attachments.push_back(mMultisampledTargets.at(names[outIdx])->getImageView());
        }
      }
    }
    for (uint32_t outIdx = 0; outIdx < names.size(); ++outIdx) {
      if (mShaderPackInstance->isColorOutputActive(i, outIdx)) {
        attachments.push_back(mRenderTargets.at(names[outIdx])->getImageView());
      }
    }

    auto depthName = mShaderPackInstance->getDepthRenderTargetName(*passes[i]);
    if (depthName.has_value()) {
      if (mConfig->msaa != vk::SampleCountFlagBits::e1) {
        // TODO delete the other unused depth completely
        attachments.push_back(mMultisampledTargets.at(depthName.value())->getImageView());
      } else {
        attachments.push_back(mRenderTargets.at(depthName.value())->getImageView());
      }
    }

    vk::FramebufferCreateInfo info(
        {}, mShaderPackInstance->getNonShadowPassResources().at(i).renderPass.get(),
        attachments.size(), attachments.data(), fbWidth, fbHeight, 1);
    mFramebuffers.push_back(mContext->getDevice().createFramebufferUnique(info));
  }
}

void Renderer::prepareShadowFramebuffers() {
  if (!mContext->isVulkanAvailable() || !mShadowMaps) {
    return;
  }
  mShadowFramebuffers.clear();
  for (auto const &target : mShadowMaps->allWriteTargets) {
    vk::ImageView view = target->getImageView();
    vk::FramebufferCreateInfo info({},
                                   mShaderPackInstance->getShadowPassResources().renderPass.get(),
                                   1, &view, target->getWidth(), target->getHeight(), 1);
    mShadowFramebuffers.push_back(mContext->getDevice().createFramebufferUnique(info));
  }
  assert(mShadowMaps->allShadowSizes.size() == mShadowFramebuffers.size());
}

void Renderer::resize(int width, int height) {
  if (width <= 0 || height <= 0) {
    throw std::runtime_error("failed to resize: width and height must be positive.");
  }
  mWidth = width;
  mHeight = height;
  mRequiresRebuild = true;
}

void Renderer::recordUpload() {
  if (!mContext->isVulkanAvailable()) {
    return;
  }

  mUploadCommandBuffer.reset();
  mUploadCommandPool = mContext->createCommandPool();
  mUploadCommandBuffer = mUploadCommandPool->allocateCommandBuffer();

  mUploadCommandBuffer->begin(vk::CommandBufferBeginInfo({}, {}));
  vk::BufferCopy region(0, 0, mCameraBufferCpu->getSize());
  mUploadCommandBuffer->copyBuffer(mCameraBufferCpu->getVulkanBuffer(),
                                   mCameraBuffer->getVulkanBuffer(), region);

  vk::MemoryBarrier barrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead);
  mUploadCommandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                        vk::PipelineStageFlagBits::eVertexShader |
                                            vk::PipelineStageFlagBits::eFragmentShader,
                                        {}, barrier, {}, {});

  mUploadCommandBuffer->end();
}

void Renderer::recordShadowsInto(vk::CommandBuffer cb) {
  if (!mContext->isVulkanAvailable() || !mShaderPackInstance ||
      !mShaderPackInstance->needsShadowPass()) {
    return;
  }

  uint32_t const shadowViewCount =
      mShadowMaps ? static_cast<uint32_t>(mShadowMaps->allWriteTargets.size()) : 0u;

  // render shadow passes
  if (mShaderPack->getShadowPass() && shadowViewCount > 0) {
    auto objects = mScene->getObjects();
    auto shadowPass = mShaderPack->getShadowPass();

    std::vector<vk::ClearValue> clearValues;
    clearValues.push_back(vk::ClearDepthStencilValue(1.0f, 0));

    for (uint32_t shadowIdx = 0; shadowIdx < shadowViewCount; ++shadowIdx) {
      uint32_t size = mShadowMaps->allShadowSizes[shadowIdx];

      vk::Viewport viewport{0.f, 0.f, static_cast<float>(size), static_cast<float>(size),
                            0.f, 1.f};
      vk::Rect2D scissor{vk::Offset2D{0u, 0u},
                         vk::Extent2D{static_cast<uint32_t>(size), static_cast<uint32_t>(size)}};

      vk::RenderPassBeginInfo renderPassBeginInfo{
          mShaderPackInstance->getShadowPassResources().renderPass.get(),
          mShadowFramebuffers[shadowIdx].get(),
          vk::Rect2D({0, 0}, {static_cast<uint32_t>(size), static_cast<uint32_t>(size)}),
          static_cast<uint32_t>(clearValues.size()), clearValues.data()};
      cb.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
      cb.bindPipeline(vk::PipelineBindPoint::eGraphics,
                      mShaderPackInstance->getShadowPassResources().pipeline.get());

      cb.setViewport(0, viewport);
      cb.setScissor(0, scissor);

      auto layout = mShaderPackInstance->getShadowPassResources().layout.get();
      auto types = shadowPass->getUniformBindingTypes();
      uint32_t sceneSet = 0;
      for (uint32_t setIdx = 0; setIdx < types.size(); ++setIdx) {
        switch (types[setIdx]) {
        case shader::UniformBindingType::eGPUScene:
          sceneSet = setIdx; // bound per draw
          break;
        case shader::UniformBindingType::eLight:
          cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, setIdx,
                                mLightSets[shadowIdx].get(), nullptr);
          break;
        default:
          throw std::runtime_error("shadow pass may only use the GPU scene and light buffer");
        }
      }

      for (uint32_t objIdx = 0; objIdx < objects.size(); ++objIdx) {
        if (objects[objIdx]->getTransparency() >= 1 || !objects[objIdx]->getCastShadow()) {
          continue;
        }
        auto const &shapes = objects[objIdx]->getModel()->getShapes();
        for (uint32_t shapeIdx = 0; shapeIdx < shapes.size(); ++shapeIdx) {
          auto const &shape = shapes[shapeIdx];
          bindSceneSet(cb, layout, sceneSet, mObjectInstances[objIdx] + shapeIdx);
          cb.bindVertexBuffers(0, shape->mesh->getVertexBuffer(*mContext).getVulkanBuffer(),
                               vk::DeviceSize(0));
          cb.bindIndexBuffer(shape->mesh->getIndexBuffer(*mContext).getVulkanBuffer(), 0,
                             vk::IndexType::eUint32);
          cb.drawIndexed(shape->mesh->getTriangleCount() * 3, 1, 0, 0, 0);
        }
      }
      cb.endRenderPass();
    }
  }

  if (mShaderPack->getPointShadowPass() && shadowViewCount > 0) {
    auto objects = mScene->getPointObjects();
    auto pointShadowPass = mShaderPack->getPointShadowPass();
    std::vector<vk::ClearValue> clearValues;
    clearValues.push_back(vk::ClearDepthStencilValue(1.0f, 0));

    for (uint32_t shadowIdx = 0; shadowIdx < shadowViewCount; ++shadowIdx) {
      uint32_t size = mShadowMaps->allShadowSizes[shadowIdx];

      vk::Viewport viewport{0.f, 0.f, static_cast<float>(size), static_cast<float>(size),
                            0.f, 1.f};
      vk::Rect2D scissor{vk::Offset2D{0u, 0u},
                         vk::Extent2D{static_cast<uint32_t>(size), static_cast<uint32_t>(size)}};

      vk::RenderPassBeginInfo renderPassBeginInfo{
          mShaderPackInstance->getPointShadowPassResources().renderPass.get(),
          mShadowFramebuffers[shadowIdx].get(),
          vk::Rect2D({0, 0}, {static_cast<uint32_t>(size), static_cast<uint32_t>(size)}),
          static_cast<uint32_t>(clearValues.size()), clearValues.data()};
      cb.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
      cb.bindPipeline(vk::PipelineBindPoint::eGraphics,
                      mShaderPackInstance->getPointShadowPassResources().pipeline.get());

      cb.setViewport(0, viewport);
      cb.setScissor(0, scissor);

      auto layout = mShaderPackInstance->getPointShadowPassResources().layout.get();
      auto types = pointShadowPass->getUniformBindingTypes();
      uint32_t sceneSet = 0;
      for (uint32_t setIdx = 0; setIdx < types.size(); ++setIdx) {
        switch (types[setIdx]) {
        case shader::UniformBindingType::eGPUScene:
          sceneSet = setIdx; // bound per draw
          break;
        case shader::UniformBindingType::eLight:
          cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, setIdx,
                                mLightSets[shadowIdx].get(), nullptr);
          break;
        default:
          throw std::runtime_error(
              "point shadow pass may only use the GPU scene and light buffer");
        }
      }

      for (uint32_t objIdx = 0; objIdx < objects.size(); ++objIdx) {
        if (objects[objIdx]->getTransparency() >= 1) {
          continue;
        }
        bindSceneSet(cb, layout, sceneSet, mPointInstances + objIdx);
        cb.bindVertexBuffers(0,
                             objects[objIdx]->getPointSet()->getVertexBuffer(*mContext).getVulkanBuffer(),
                             vk::DeviceSize(0));
        cb.draw(objects[objIdx]->getVertexCount(), 1, 0, 0);
      }
      cb.endRenderPass();
    }
  }
}

void Renderer::recordShadows() {
  if (!mContext->isVulkanAvailable()) {
    return;
  }

  mShadowCommandBuffer.reset();
  mShadowCommandPool = mContext->createCommandPool();
  mShadowCommandBuffer = mShadowCommandPool->allocateCommandBuffer();

  mShadowCommandBuffer->begin(vk::CommandBufferBeginInfo({}, {}));
  recordShadowsInto(mShadowCommandBuffer.get());
  mShadowCommandBuffer->end();
}

void Renderer::prepareObjects() {
  SVULKAN2_PROFILE_BLOCK_BEGIN("Prepare objects");
  auto objects = mScene->getObjects();
  auto lineObjects = mScene->getLineObjects();
  auto pointObjects = mScene->getPointObjects();
  prepareGPUScene();
  SVULKAN2_PROFILE_BLOCK_END;

  // load objects to CPU, if not already loaded
  {
    SVULKAN2_PROFILE_BLOCK("Load objects to CPU");
    std::vector<std::future<void>> futures;
    for (auto obj : objects) {
      futures.push_back(obj->getModel()->loadAsync());
    }
    for (auto &f : futures) {
      f.get();
    }
  }

  // One instance per drawable, planned before recording so every pass that draws it binds the
  // same instance: the shapes of every object, then the line sets, then the point sets.
  {
    SVULKAN2_PROFILE_BLOCK("Upload objects to GPU");
    mInstances.clear();
    mObjectInstances.resize(objects.size());
    for (uint32_t objectIndex = 0; objectIndex < objects.size(); ++objectIndex) {
      mObjectInstances[objectIndex] = mInstances.size();
      for (auto shape : objects[objectIndex]->getModel()->getShapes()) {
        shape->mesh->uploadToDevice(*mContext);
        mInstances.push_back(
            {0, static_cast<int32_t>(objectIndex), objects[objectIndex]->getInternalGpuIndex(),
             static_cast<int32_t>(objectIndex),
             std::static_pointer_cast<resource::SVMetallicMaterial>(shape->material)});
      }
    }
    mLineInstances = mInstances.size();
    if (mShaderPack->hasLinePass()) {
      for (uint32_t index = 0; index < lineObjects.size(); ++index) {
        lineObjects[index]->getLineSet()->uploadToDevice(*mContext);
        mInstances.push_back(
            {1, static_cast<int32_t>(index), lineObjects[index]->getInternalGpuIndex(), -1, {}});
      }
    }
    mPointInstances = mInstances.size();
    if (mShaderPack->hasPointPass() || mShaderPack->getPointShadowPass()) {
      for (uint32_t index = 0; index < pointObjects.size(); ++index) {
        pointObjects[index]->getPointSet()->uploadToDevice(*mContext);
        mInstances.push_back(
            {2, static_cast<int32_t>(index), pointObjects[index]->getInternalGpuIndex(), -1, {}});
      }
    }
  }

  // The command buffers recorded next bind the scene set, and updating a bound set invalidates
  // them: every binding they read must be final now.
  ensureGPUBuffers(mInstances.size(), objects.size());
  refreshMaterialsAndTextures();
}

void Renderer::recordRenderPassesInto(vk::CommandBuffer cb) {
  mModelCache.clear();
  mLineSetCache.clear();
  mPointSetCache.clear();

  // classify shapes
  uint32_t numGbufferPasses = mShaderPack->getGbufferPasses().size();

  std::vector<std::vector<std::shared_ptr<resource::SVShape>>> shapes(numGbufferPasses);
  std::vector<std::vector<uint32_t>> shapeInstances(numGbufferPasses);

  int transparencyShadingMode = numGbufferPasses > 1 ? 1 : 0;

  auto objects = mScene->getObjects();
  for (uint32_t objectIndex = 0; objectIndex < objects.size(); ++objectIndex) {
    int shadingMode = objects[objectIndex]->getShadingMode();
    if (static_cast<uint32_t>(shadingMode) >= shapes.size()) {
      continue; // do not render
    }
    if (shadingMode == 0 && objects[objectIndex]->getTransparency() != 0) {
      shadingMode = transparencyShadingMode;
    }
    if (objects[objectIndex]->getTransparency() >= 1) {
      continue;
    }
    /* HACK: hold onto the models to make sure the underlying buffers are not
     * released until command buffer reset */
    mModelCache.insert(objects[objectIndex]->getModel());

    auto const &objectShapes = objects[objectIndex]->getModel()->getShapes();
    for (uint32_t shapeIndex = 0; shapeIndex < objectShapes.size(); ++shapeIndex) {
      auto const &shape = objectShapes[shapeIndex];
      int shapeShadingMode = shadingMode;
      if (shape->material->getOpacity() == 0) {
        continue;
      }
      if (shape->material->getOpacity() != 1 && shadingMode == 0) {
        shapeShadingMode = transparencyShadingMode;
      }
      shapes[shapeShadingMode].push_back(shape);
      shapeInstances[shapeShadingMode].push_back(mObjectInstances[objectIndex] + shapeIndex);
    }
  }

  auto linesetObjects = mScene->getLineObjects();

  // classify point sets
  uint32_t numPointPasses = mShaderPack->getPointPasses().size();

  auto pointsetObjects = mScene->getPointObjects();
  std::vector<std::vector<uint32_t>> pointsets(numPointPasses);
  for (uint32_t objectIndex = 0; objectIndex < pointsetObjects.size(); ++objectIndex) {
    auto mode = pointsetObjects[objectIndex]->getShadingMode();
    if (static_cast<uint32_t>(mode) >= pointsets.size()) {
      continue;
    }
    pointsets[mode].push_back(objectIndex);
  }

  uint32_t gbufferIndex = 0;
  uint32_t pointIndex = 0;
  auto passes = mShaderPack->getNonShadowPasses();
  bool postRecorded = !hasPostChain();
  for (uint32_t pass_index = 0; pass_index < passes.size(); ++pass_index) {
    auto pass = passes[pass_index];
    auto gbufferPass = std::dynamic_pointer_cast<shader::GbufferPassParser>(pass);
    auto pointPass = std::dynamic_pointer_cast<shader::PointPassParser>(pass);
    if (!mShaderPackInstance->isPassActive(pass_index)) {
      if (gbufferPass) {
        gbufferIndex++;
      } else if (pointPass) {
        pointIndex++;
      }
      continue;
    }

    SVULKAN2_PROFILE_BLOCK("Record render pass" + std::to_string(pass_index));

    if (!postRecorded) {
      auto inputs = pass->getInputTextureNames();
      if (std::find(inputs.begin(), inputs.end(), kRasterPostResult) != inputs.end()) {
        recordPostChainInto(cb);
        postRecorded = true;
      }
    }

    float scale = pass->getResolutionScale();
    uint32_t passWidth = scaledExtent(static_cast<uint32_t>(mWidth), scale);
    uint32_t passHeight = scaledExtent(static_cast<uint32_t>(mHeight), scale);

    vk::Viewport viewport{0.f, 0.f, static_cast<float>(passWidth), static_cast<float>(passHeight),
                          0.f, 1.f};
    vk::Rect2D scissor{vk::Offset2D{0u, 0u}, vk::Extent2D{static_cast<uint32_t>(passWidth),
                                                          static_cast<uint32_t>(passHeight)}};

    uint32_t colorCount = mShaderPackInstance->getActiveColorOutputCount(pass_index);
    std::vector<vk::ClearValue> clearValues(
        mConfig->msaa == vk::SampleCountFlagBits::e1 ? colorCount : colorCount * 2,
        vk::ClearColorValue(std::array<float, 4>{0.f, 0.f, 0.f, 0.f}));
    if (mShaderPackInstance->getDepthRenderTargetName(*pass).has_value()) {
      clearValues.push_back(vk::ClearDepthStencilValue(1.0f, 0));
    }
    vk::RenderPassBeginInfo renderPassBeginInfo{
        mShaderPackInstance->getNonShadowPassResources().at(pass_index).renderPass.get(),
        mFramebuffers[pass_index].get(),
        vk::Rect2D({0, 0}, {static_cast<uint32_t>(passWidth), static_cast<uint32_t>(passHeight)}),
        static_cast<uint32_t>(clearValues.size()), clearValues.data()};
    cb.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
    cb.bindPipeline(vk::PipelineBindPoint::eGraphics,
                    mShaderPackInstance->getNonShadowPassResources().at(pass_index).pipeline.get());
    cb.setViewport(0, viewport);
    cb.setScissor(0, scissor);

    auto types = pass->getUniformBindingTypes();

    auto layout = mShaderPackInstance->getNonShadowPassResources().at(pass_index).layout.get();

    // the scene set is bound per draw at the draw's instance; a pass that draws no scene
    // geometry but reads the set sees instance 0
    std::optional<uint32_t> sceneSet;
    for (uint32_t i = 0; i < types.size(); ++i) {
      switch (types[i]) {
      case shader::UniformBindingType::eCamera:
        cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, i, mCameraSet.get(),
                              nullptr);
        break;
      case shader::UniformBindingType::eScene:
        cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, i, mSceneSet.get(),
                              nullptr);
        break;
      case shader::UniformBindingType::eTextures:
        cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, i,
                              mInputTextureSets[pass_index].get(), nullptr);
        break;
      case shader::UniformBindingType::eGPUScene:
        sceneSet = i;
        break;
      default:
        throw std::runtime_error("not implemented");
      }
    }
    auto bindInstance = [&](uint32_t instance) {
      if (sceneSet) {
        bindSceneSet(cb, layout, *sceneSet, instance);
      }
    };

    if (gbufferPass) {
      for (uint32_t i = 0; i < shapes[gbufferIndex].size(); ++i) {
        auto shape = shapes[gbufferIndex][i];
        bindInstance(shapeInstances[gbufferIndex][i]);
        cb.bindVertexBuffers(0, shape->mesh->getVertexBuffer(*mContext).getVulkanBuffer(),
                             vk::DeviceSize(0));
        cb.bindIndexBuffer(shape->mesh->getIndexBuffer(*mContext).getVulkanBuffer(), 0,
                           vk::IndexType::eUint32);
        cb.setCullMode(shape->material->getCullMode());
        cb.drawIndexed(shape->mesh->getTriangleCount() * 3, 1, 0, 0, 0);
      }
      gbufferIndex++;
    } else if (auto linePass = std::dynamic_pointer_cast<shader::LinePassParser>(pass)) {
      for (uint32_t index = 0; index < linesetObjects.size(); ++index) {
        auto &lineObj = linesetObjects[index];
        if (lineObj->getTransparency() < 1) {
          bindInstance(mLineInstances + index);
          mLineSetCache.insert(lineObj->getLineSet());
          cb.setLineWidth(lineObj->getLineWidth());
          cb.bindVertexBuffers(0,
                               lineObj->getLineSet()->getVertexBuffer(*mContext).getVulkanBuffer(),
                               vk::DeviceSize(0));
          cb.draw(lineObj->getLineSet()->getVertexCount(), 1, 0, 0);
        }
      }
    } else if (pointPass) {
      for (uint32_t i = 0; i < pointsets[pointIndex].size(); ++i) {
        uint32_t objectIndex = pointsets[pointIndex][i];
        auto &pointObj = pointsetObjects[objectIndex];
        if (pointObj->getTransparency() < 1) {
          bindInstance(mPointInstances + objectIndex);
          mPointSetCache.insert(pointObj->getPointSet());
          cb.bindVertexBuffers(
              0, pointObj->getPointSet()->getVertexBuffer(*mContext).getVulkanBuffer(),
              vk::DeviceSize(0));
          cb.draw(pointObj->getVertexCount(), 1, 0, 0);
        }
      }
      pointIndex++;
    } else {
      bindInstance(0);
      cb.draw(3, 1, 0, 0);
    }
    cb.endRenderPass();
  }
  if (!postRecorded) {
    throw std::runtime_error(std::string("the shader pack lists post-processing passes but no "
                                         "pass samples their result, sampler") +
                             kRasterPostResult);
  }
}

void Renderer::recordRenderPasses() {
  mRenderCommandBuffer.reset();
  mRenderCommandPool = mContext->createCommandPool();
  mRenderCommandBuffer = mRenderCommandPool->allocateCommandBuffer();

  mRenderCommandBuffer->begin(vk::CommandBufferBeginInfo({}, {}));
  recordRenderPassesInto(mRenderCommandBuffer.get());
  mRenderCommandBuffer->end();
}

void Renderer::prepareRender(scene::Camera &camera) {
  if (mLastVersion != mScene->getVersion()) {
    logger::info("Scene updated");
    mRequiresRecord = true;
  }

  // Material edits rewrite materials[] in place, in grouped mode too. A grown material table or
  // a changed texture list rebinds the scene set, which invalidates the recorded command buffers.
  if (!mRequiresRecord && mGPUSceneSet &&
      mLastMaterialVersion != resource::SVMetallicMaterial::getGlobalVersion() &&
      refreshMaterialsAndTextures()) {
    mRequiresRecord = true;
  }

  if (mEnvironmentMap != mScene->getEnvironmentMap()) {
    mEnvironmentMap = mScene->getEnvironmentMap();
    if (mEnvironmentMap) {
      mEnvironmentMap->load();
    }
    mRequiresRebuild = true;
  }

  if (mWidth <= 0 || mHeight <= 0) {
    throw std::runtime_error("failed to render: resize must be called before rendering.");
  }

  if (mRequiresRecord) {
    auto pointLights = mScene->getPointLights();
    auto directionalLights = mScene->getDirectionalLights();
    auto spotLights = mScene->getSpotLights();
    auto texturedLights = mScene->getTexturedLights();

    mPointLightShadowSizes.clear();
    mDirectionalLightShadowSizes.clear();
    mSpotLightShadowSizes.clear();
    mTexturedLightShadowSizes.clear();

    for (auto l : pointLights) {
      if (l->isShadowEnabled()) {
        mPointLightShadowSizes.push_back(l->getShadowMapSize());
      }
    }
    for (auto l : directionalLights) {
      if (l->isShadowEnabled()) {
        mDirectionalLightShadowSizes.push_back(l->getShadowMapSize());
      }
    }
    for (auto l : spotLights) {
      if (l->isShadowEnabled()) {
        mSpotLightShadowSizes.push_back(l->getShadowMapSize());
      }
    }
    for (auto l : texturedLights) {
      mTexturedLightShadowSizes.push_back(l->getShadowMapSize());
    }

    {
      // load custom textures
      std::vector<std::future<void>> futures;
      for (auto t : mCustomTextures) {
        futures.push_back(t.second->loadAsync());
      }
      for (auto a : mCustomTextureArray) {
        for (auto t : a.second) {
          futures.push_back(t->loadAsync());
        }
      }
      for (auto t : mCustomCubemaps) {
        futures.push_back(t.second->loadAsync());
      }
      for (auto l : mScene->getTexturedLights()) {
        if (l->getTexture()) {
          futures.push_back(l->getTexture()->loadAsync());
        }
      }
      for (auto &f : futures) {
        f.get();
      }
      for (auto l : mScene->getTexturedLights()) {
        if (l->getTexture()) {
          l->getTexture()->uploadToDevice(*mContext);
        }
      }
    }

    if (mShaderPack->getShadowPass()) {
      if (mDirectionalLightShadowSizes.size() > mShaderPack->getMaxDirectionalShadowCount()) {
        throw std::runtime_error("The scene contains too many directional "
                                 "lights that cast shadows.");
      }
      if (mPointLightShadowSizes.size() > mShaderPack->getMaxPointShadowCount()) {
        throw std::runtime_error("The scene contains too many point "
                                 "lights that cast shadows.");
      }
      if (mSpotLightShadowSizes.size() > mShaderPack->getMaxSpotShadowCount()) {
        throw std::runtime_error("The scene contains too many spot "
                                 "lights that cast shadows.");
      }
      if (mTexturedLightShadowSizes.size() > mShaderPack->getMaxTexturedShadowCount()) {
        throw std::runtime_error("The scene contains too many textured lights.");
      }
    }

    // the light counts size the scene set's light arrays and the shaders' loops; changing one
    // rebuilds the pipelines and the light buffers
    setSpecializationConstant<int>("NUM_POINT_LIGHTS", pointLights.size());
    setSpecializationConstant<int>("NUM_DIRECTIONAL_LIGHTS", directionalLights.size());
    setSpecializationConstant<int>("NUM_SPOT_LIGHTS", spotLights.size());
    setSpecializationConstant<int>("NUM_POINT_LIGHT_SHADOWS", mPointLightShadowSizes.size());
    setSpecializationConstant<int>("NUM_DIRECTIONAL_LIGHT_SHADOWS",
                                   mDirectionalLightShadowSizes.size());
    setSpecializationConstant<int>("NUM_SPOT_LIGHT_SHADOWS", mSpotLightShadowSizes.size());
    setSpecializationConstant<int>("NUM_TEXTURED_LIGHT_SHADOWS", mTexturedLightShadowSizes.size());

    if (mShaderPack->getShadowPass() &&
        (!mShaderPackInstance || mShaderPackInstance->needsShadowPass())) {
      auto lightBufferLayout = mShaderPack->getShaderInputLayouts()->lightBufferLayout;
      auto activeShadowMaps = mScene->getRasterShadowMaps(mConfig->depthFormat);
      if (!mShadowMaps || mShadowMaps != activeShadowMaps ||
          !mShadowMaps->matches(mConfig->depthFormat, lightBufferLayout->size,
                                mDirectionalLightShadowSizes, mPointLightShadowSizes,
                                mSpotLightShadowSizes, mTexturedLightShadowSizes)) {
        mRequiresRebuild = true;
      }
    }
  }

  bool rebuilt = false;
  if (mRequiresRebuild || mSpecializationConstantsChanged) {
    SVULKAN2_PROFILE_BLOCK("Rebuilding Pipeline");
    mRequiresRecord = true;
    rebuilt = true;

    preparePipelines();

    prepareRenderTargets(mWidth, mHeight);
    if (mShaderPackInstance->needsShadowPass()) {
      prepareShadowRenderTargets();
      prepareShadowFramebuffers();
    } else {
      mShadowMaps.reset();
      mShadowFramebuffers.clear();
    }

    prepareFramebuffers(mWidth, mHeight);
    prepareInputTextureDescriptorSets();

    if (mShaderPackInstance->needsShadowPass()) {
      prepareLightBuffers();
    } else {
      mLightSets.clear();
    }
    if (mShaderPackInstance->needsSceneSet()) {
      prepareSceneBuffer();
    } else {
      mShadowBuffer.reset();
      mSceneBuffer.reset();
      mSceneSet.reset();
      mPointLightBuffer.buffer.reset();
      mDirectionalLightBuffer.buffer.reset();
      mSpotLightBuffer.buffer.reset();
    }
    prepareCameraBuffer();

    if (camera.getWidth() != mWidth || camera.getHeight() != mHeight) {
      throw std::runtime_error("Camera size and renderer size does not match. "
                               "Please resize camera first.");
    }
    // only a complete rebuild clears the flags: one that threw (too many lights for the
    // uniform buffers, a camera of another size) is retried by the next frame
    mSpecializationConstantsChanged = false;
    mRequiresRebuild = false;
  }

  if (mRequiresRecord) {
    prepareObjects();

    recordUpload();

    if (!mGroupedSingleSubmit) {
      if (mShaderPackInstance->needsShadowPass()) {
        SVULKAN2_PROFILE_BLOCK("Record shadow draw calls");
        recordShadows();
      } else {
        mShadowCommandBuffer.reset();
      }
      {
        SVULKAN2_PROFILE_BLOCK("Record render draw calls");
        recordRenderPasses();
      }
    } else {
      mShadowCommandBuffer.reset();
      mRenderCommandBuffer.reset();
    }

    if (mExecutionMode == RenderExecutionMode::eCpuManaged) {
      // classic CPU flow: seed the freshly recorded resources immediately. In
      // eGroupedGpu mode the owning group performs one explicit CPU snapshot at
      // initialization; record-time uploads would overwrite GPU-owned buffers.
      uploadGpuResources(camera);
      mContext->getQueue().waitIdle();
    } else {
      // the instance table follows the draws just recorded and holds no GPU-owned state;
      // a mid-run rebuild also reallocates the CPU-owned scene/light buffers.
      uploadInstances();
      if (rebuilt) {
        writeSceneLightBuffers();
      }
    }
  }

  mRequiresRecord = false;
  mLastVersion = mScene->getVersion();

  for (auto &[name, target] : mRenderTargets) {
    target->getImage().setCurrentLayout(mRenderTargetFinalLayouts[name]);
  }

  // when using multisampling, the original layout is always transfer src
  for (auto &[name, target] : mMultisampledTargets) {
    mRenderTargets.at(name)->getImage().setCurrentLayout(vk::ImageLayout::eTransferSrcOptimal);
    target->getImage().setCurrentLayout(mRenderTargetFinalLayouts[name]);
  }
}

void Renderer::writeCameraBufferCpu(scene::Camera &camera) {
  camera.uploadToDevice(*mCameraBufferCpu,
                        *mShaderPack->getShaderInputLayouts()->cameraBufferLayout);
}

void Renderer::writeSceneLightBuffers() {
  if (!mSceneBuffer) {
    return;
  }
  packLights();
  auto const &layout = *mShaderPack->getShaderInputLayouts()->sceneBufferLayout;
  glm::vec4 ambientLight = mScene->getAmbientLight();
  mSceneBuffer->upload(&ambientLight, sizeof(ambientLight),
                       layout.elements.at("ambientLight").offset);

  auto writeLights = [](LightBufferBinding &light, auto const &lights) {
    if (!light.buffer || lights.empty()) {
      return;
    }
    if (lights.size() > light.capacity) {
      // the light counts change only with a rebuild, which reallocates the buffers
      throw std::runtime_error("the scene has more lights than its light buffers hold");
    }
    light.buffer->upload(lights.data(), lights.size() * sizeof(lights[0]));
  };
  writeLights(mPointLightBuffer, mLights.point);
  writeLights(mDirectionalLightBuffer, mLights.directional);
  writeLights(mSpotLightBuffer, mLights.spot);
  if (mShaderPackInstance && mShaderPackInstance->needsShadowPass() && mShadowMaps &&
      mShadowBuffer) {
    mScene->uploadShadowToDevice(*mShadowBuffer, *mShadowMaps,
                                 *mShaderPack->getShaderInputLayouts()->shadowBufferLayout);
  }
}

void Renderer::uploadCpuCameraState(scene::Camera &camera) {
  if (!mContext->isVulkanAvailable()) {
    return;
  }
  if (!mCameraBufferCpu || !mUploadCommandBuffer) {
    throw std::runtime_error(
        "uploadCpuCameraState failed: renderer resources are not prepared");
  }
  writeCameraBufferCpu(camera);
  // Wait for the copy to finish before returning. mUploadCommandBuffer is recorded
  // without eSimultaneousUse and reads mCameraBufferCpu, so a caller that uploads on
  // consecutive frames without an intervening render would otherwise resubmit a
  // pending command buffer and rewrite the staging buffer while it is still in use.
  mContext->getQueue().submitAndWait(mUploadCommandBuffer.get());
}

void Renderer::uploadCpuSceneLightState() {
  if (!mContext->isVulkanAvailable()) {
    return;
  }
  if (!mShaderPackInstance) {
    throw std::runtime_error(
        "uploadCpuSceneLightState failed: renderer resources are not prepared");
  }
  if (!mSceneBuffer) {
    return;
  }
  writeSceneLightBuffers();
}

void Renderer::uploadGpuResources(scene::Camera &camera) {
  {
    SVULKAN2_PROFILE_BLOCK("Update camera & scene");
    writeCameraBufferCpu(camera);
    writeSceneLightBuffers();
  }

  {
    SVULKAN2_PROFILE_BLOCK("Update GPU scene");
    // GPU interop can own object transforms; the instance and custom-data tables stay CPU-owned
    if (!mExternalTransformUpdates && !mScene->hasExternalTransformOwnership()) {
      mScene->uploadObjectTransforms();
    }
    uploadInstances();
  }

  mContext->getQueue().submit(mUploadCommandBuffer.get(), {});
}

void Renderer::forceUploadCameraBuffer(scene::Camera &camera) {
  if (mCameraBufferCpu && mUploadCommandBuffer) {
    camera.uploadToDevice(*mCameraBufferCpu,
                          *mShaderPack->getShaderInputLayouts()->cameraBufferLayout);
    mContext->getQueue().submitAndWait(mUploadCommandBuffer.get());
  }
}

bool Renderer::hasActiveShadows() const {
  return mShaderPackInstance && mShaderPackInstance->needsShadowPass() && mShadowMaps &&
         !mShadowMaps->allWriteTargets.empty();
}

bool Renderer::wantsPointShadowPass() const {
  return mShaderPack->getPointShadowPass() != nullptr && mScene &&
         !mScene->getPointObjects().empty();
}

bool Renderer::prepareGroupedFrame(scene::Camera &camera) {
  if (!mContext->isVulkanAvailable()) {
    return false;
  }
  if (!mScene) {
    throw std::runtime_error("setScene must be called before rendering");
  }
  bool needsPrepare = mRequiresRebuild || mSpecializationConstantsChanged || mRequiresRecord ||
                      !mShaderPackInstance || mLastVersion != mScene->getVersion() ||
                      mEnvironmentMap != mScene->getEnvironmentMap();
  if (!needsPrepare && mGPUSceneSet &&
      mLastMaterialVersion != resource::SVMetallicMaterial::getGlobalVersion()) {
    if (refreshMaterialsAndTextures()) {
      mRequiresRecord = true;
      needsPrepare = true;
    }
  }
  if (!needsPrepare) {
    return false;
  }
  prepareRender(camera);
  return true;
}

bool Renderer::shouldRenderShadows(bool &wantsPointShadow) const {
  wantsPointShadow = false;
  if (!hasActiveShadows()) {
    return false;
  }
  wantsPointShadow = wantsPointShadowPass();
  // the maps are shared by every camera of the scene: reuse them only when they hold exactly
  // what this pack draws, point-cloud shadows included or not, whichever camera rendered first
  if (mShadowMaps->renderedShadowEpoch == mScene->getShadowEpoch() &&
      mShadowMaps->renderedWithPointShadow == wantsPointShadow) {
    return false;
  }
  return true;
}

void Renderer::render(scene::Camera &camera, std::vector<vk::Semaphore> const &waitSemaphores,
                      std::vector<vk::PipelineStageFlags> const &waitStages,
                      std::vector<vk::Semaphore> const &signalSemaphores, vk::Fence fence) {
  if (!mContext->isVulkanAvailable()) {
    return;
  }
  if (!mScene) {
    throw std::runtime_error("setScene must be called before rendering");
  }
  SVULKAN2_PROFILE_BLOCK("Record & Submit");
  prepareRender(camera);
  if (!mRenderCommandBuffer) {
    if (mShaderPackInstance->needsShadowPass()) {
      recordShadows();
    }
    recordRenderPasses();
  }

  if (mExecutionMode == RenderExecutionMode::eCpuManaged) {
    uploadGpuResources(camera);
  }

  bool wantsPointShadow = false;
  if (shouldRenderShadows(wantsPointShadow)) {
    if (!mShadowCommandBuffer) {
      recordShadows();
    }
    std::array<vk::CommandBuffer, 2> cbs = {mShadowCommandBuffer.get(),
                                            mRenderCommandBuffer.get()};
    mContext->getQueue().submit(cbs, waitSemaphores, waitStages, signalSemaphores, fence);
    mShadowMaps->renderedShadowEpoch = mScene->getShadowEpoch();
    mShadowMaps->renderedWithPointShadow = wantsPointShadow;
    mScene->noteRasterShadowsRendered();
  } else {
    mContext->getQueue().submit(mRenderCommandBuffer.get(), waitSemaphores, waitStages,
                                signalSemaphores, fence);
  }
}

void Renderer::render(
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
  SVULKAN2_PROFILE_BLOCK("Record & Submit");
  prepareRender(camera);
  if (!mRenderCommandBuffer) {
    if (mShaderPackInstance->needsShadowPass()) {
      recordShadows();
    }
    recordRenderPasses();
  }

  if (mExecutionMode == RenderExecutionMode::eCpuManaged) {
    uploadGpuResources(camera);
  }

  bool wantsPointShadow = false;
  if (shouldRenderShadows(wantsPointShadow)) {
    if (!mShadowCommandBuffer) {
      recordShadows();
    }
    std::array<vk::CommandBuffer, 2> cbs = {mShadowCommandBuffer.get(),
                                            mRenderCommandBuffer.get()};
    mContext->getQueue().submit(cbs, waitSemaphores, waitStageMasks, waitValues, signalSemaphores,
                                signalValues, {});
    mShadowMaps->renderedShadowEpoch = mScene->getShadowEpoch();
    mShadowMaps->renderedWithPointShadow = wantsPointShadow;
    mScene->noteRasterShadowsRendered();
  } else {
    mContext->getQueue().submit(mRenderCommandBuffer.get(), waitSemaphores, waitStageMasks,
                                waitValues, signalSemaphores, signalValues, {});
  }
}

std::vector<std::string> Renderer::getDisplayTargetNames() const {
  if (!mContext->isVulkanAvailable()) {
    return {};
  }

  std::unordered_set<std::string> names;
  for (auto pass : mShaderPack->getNonShadowPasses()) {
    for (auto &elem : pass->getTextureOutputLayout()->elements) {
      std::string texName = shader::getOutTextureName(elem.second.name);
      if (texName.ends_with("Depth")) {
        throw std::runtime_error("You are not allowed to name your texture \"*Depth\"");
      }
      if (elem.second.dtype == DataType::FLOAT4()) {
        names.insert(texName);
      }
    }
  }

  return std::vector(names.begin(), names.end());
}

std::vector<std::string> Renderer::getRenderTargetNames() const {
  if (!mContext->isVulkanAvailable()) {
    return {};
  }

  if (mShaderPackInstance && !mConfig->requestedTargets.empty()) {
    std::vector<std::string> names;
    auto formats = mShaderPackInstance->getRenderTargetFormats();
    names.reserve(formats.size());
    for (auto const &[name, _] : formats) {
      names.push_back(name);
    }
    return names;
  }

  std::unordered_set<std::string> names;
  for (auto pass : mShaderPack->getNonShadowPasses()) {
    std::optional<std::string> depthName =
        (mConfig->shareGbufferDepths && dynamic_cast<shader::GbufferPassParser const *>(pass.get()))
            ? std::optional<std::string>("GbufferDepth")
            : pass->getDepthRenderTargetName();
    if (depthName.has_value()) {
      names.insert(depthName.value());
    }
    for (auto &elem : pass->getTextureOutputLayout()->elements) {
      std::string texName = shader::getOutTextureName(elem.second.name);
      if (texName.ends_with("Depth")) {
        throw std::runtime_error("You are not allowed to name your texture \"*Depth\"");
      }
      names.insert(texName);
    }
  }

  return std::vector(names.begin(), names.end());
}

void Renderer::display(std::string const &renderTargetName, vk::Image backBuffer,
                       vk::Format format, uint32_t width, uint32_t height,
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

  if (!mRenderTargets.contains(renderTargetName)) {
    throw std::runtime_error("failed to find render target with name " + renderTargetName +
                             ". Did you forget to take picture?");
  }

  auto renderTarget = mRenderTargets.at(renderTargetName);

  auto targetFormat = renderTarget->getFormat();
  if (targetFormat != vk::Format::eR8G8B8A8Unorm &&
      targetFormat != vk::Format::eR32G32B32A32Sfloat) {
    throw std::runtime_error("failed to display: only color textures are supported in display");
  };
  auto layout = renderTarget->getImage().getCurrentLayout(0);

  if (layout != vk::ImageLayout::eTransferSrcOptimal) {
    vk::AccessFlags sourceAccessMask;
    vk::PipelineStageFlags sourceStage;
    if (layout == vk::ImageLayout::eColorAttachmentOptimal) {
      sourceAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
      sourceStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    } else if (layout == vk::ImageLayout::eShaderReadOnlyOptimal) {
      sourceAccessMask = {};
      sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
    } else {
      throw std::runtime_error("invalid layout");
    }
    // transfer render target
    renderTarget->getImage().transitionLayout(mDisplayCommandBuffer.get(), layout,
                                              vk::ImageLayout::eTransferSrcOptimal,
                                              sourceAccessMask, vk::AccessFlagBits::eTransferRead,
                                              sourceStage, vk::PipelineStageFlagBits::eTransfer);
  }

  // transfer swap chain
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
      renderTarget->getImage().getVulkanImage(), vk::ImageLayout::eTransferSrcOptimal, backBuffer,
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

  renderTarget->getImage().setCurrentLayout(vk::ImageLayout::eTransferSrcOptimal);
}

void Renderer::prepareSceneBuffer() {
  if (mShaderPackInstance->needsShadowPass()) {
    mShadowBuffer = core::Buffer::CreateUniform(
        *mContext, mShaderPack->getShaderInputLayouts()->shadowBufferLayout->size);
  } else {
    mShadowBuffer.reset();
  }
  mSceneBuffer = core::Buffer::CreateUniform(
      *mContext, mShaderPack->getShaderInputLayouts()->sceneBufferLayout->size);
  // the light buffers are allocated for the new set once its bindings are known
  auto layout = mShaderPackInstance->getSceneDescriptorSetLayout();

  mSceneSet = std::move(mContext->getDevice()
                            .allocateDescriptorSetsUnique(
                                vk::DescriptorSetAllocateInfo(mDescriptorPool.get(), 1, &layout))
                            .front());

  auto setDesc = mShaderPack->getShaderInputLayouts()->sceneSetDescription;

  for (uint32_t bindingIndex = 0; bindingIndex < setDesc.bindings.size(); ++bindingIndex) {
    auto binding = setDesc.bindings.at(bindingIndex);
    if (binding.name == "SceneBuffer") {
      updateDescriptorSets(
          mContext->getDevice(), mSceneSet.get(),
          {{vk::DescriptorType::eUniformBuffer, mSceneBuffer->getVulkanBuffer(), nullptr}}, {},
          bindingIndex);
    } else if (binding.name == "PointLights") {
      mPointLightBuffer.binding = static_cast<int>(bindingIndex);
    } else if (binding.name == "DirectionalLights") {
      mDirectionalLightBuffer.binding = static_cast<int>(bindingIndex);
    } else if (binding.name == "SpotLights") {
      mSpotLightBuffer.binding = static_cast<int>(bindingIndex);
    } else if (binding.name == "ShadowBuffer") {
      updateDescriptorSets(
          mContext->getDevice(), mSceneSet.get(),
          {{vk::DescriptorType::eUniformBuffer, mShadowBuffer->getVulkanBuffer(), nullptr}}, {},
          bindingIndex);
    } else if (binding.name == "samplerPointLightDepths") {
      std::vector<std::tuple<vk::ImageView, vk::Sampler>> textures;
      if (mShadowMaps) {
        for (auto const &t : mShadowMaps->pointReadTargets) {
          textures.push_back({t->getImageView(), t->getSampler()});
        }
      }
      updateArrayDescriptorSets(mContext->getDevice(), mSceneSet.get(), textures, bindingIndex);
    } else if (binding.name == "samplerDirectionalLightDepths") {
      std::vector<std::tuple<vk::ImageView, vk::Sampler>> textures;
      if (mShadowMaps) {
        for (auto const &t : mShadowMaps->directionalReadTargets) {
          textures.push_back({t->getImageView(), t->getSampler()});
        }
      }
      updateArrayDescriptorSets(mContext->getDevice(), mSceneSet.get(), textures, bindingIndex);
    } else if (binding.name == "samplerSpotLightDepths") {
      std::vector<std::tuple<vk::ImageView, vk::Sampler>> textures;
      if (mShadowMaps) {
        for (auto const &t : mShadowMaps->spotReadTargets) {
          textures.push_back({t->getImageView(), t->getSampler()});
        }
      }
      updateArrayDescriptorSets(mContext->getDevice(), mSceneSet.get(), textures, bindingIndex);
    } else if (binding.name == "samplerTexturedLightDepths") {
      std::vector<std::tuple<vk::ImageView, vk::Sampler>> textures;
      if (mShadowMaps) {
        for (auto const &t : mShadowMaps->texturedReadTargets) {
          textures.push_back({t->getImageView(), t->getSampler()});
        }
      }
      updateArrayDescriptorSets(mContext->getDevice(), mSceneSet.get(), textures, bindingIndex);
    } else if (binding.name == "samplerTexturedLightTextures") {
      auto lights = mScene->getTexturedLights();
      std::vector<std::tuple<vk::ImageView, vk::Sampler>> textures;
      for (auto l : lights) {
        auto t = l->getTexture();
        if (!t) {
          logger::error("A textured light does not have a texture attached");
          t = resource::SVResourceManager::Get()->getDefaultTexture();
          t->loadAsync().get();
          t->uploadToDevice(*mContext);
        }
        textures.push_back({t->getImageView(*mContext), t->getSampler(*mContext)});
      }
      updateArrayDescriptorSets(mContext->getDevice(), mSceneSet.get(), textures, bindingIndex);
    } else if (binding.type == vk::DescriptorType::eCombinedImageSampler &&
               binding.name.substr(0, 7) == "sampler") {
      std::string customTextureName = binding.name.substr(7);
      if (customTextureName.substr(0, 6) == "Random") {
        auto randomTex =
            resource::SVResourceManager::Get()->CreateRandomTexture(customTextureName);
        randomTex->uploadToDevice(*mContext);
        updateDescriptorSets(
            mContext->getDevice(), mSceneSet.get(), {},
            {{randomTex->getImageView(*mContext), randomTex->getSampler(*mContext)}},
            bindingIndex);
      } else if (mCustomTextures.find(customTextureName) != mCustomTextures.end()) {
        mCustomTextures[customTextureName]->uploadToDevice(*mContext);
        updateDescriptorSets(mContext->getDevice(), mSceneSet.get(), {},
                             {{mCustomTextures[customTextureName]->getImageView(*mContext),
                               mCustomTextures[customTextureName]->getSampler(*mContext)}},
                             bindingIndex);
      } else if (mCustomTextureArray.contains(customTextureName)) {
        auto textures = mCustomTextureArray.at(customTextureName);
        std::vector<std::tuple<vk::ImageView, vk::Sampler>> ts;
        for (auto t : textures) {
          t->uploadToDevice(*mContext);
          ts.push_back({t->getImageView(*mContext), t->getSampler(*mContext)});
        }
        updateDescriptorSets(mContext->getDevice(), mSceneSet.get(), {}, ts, bindingIndex);
      } else if (customTextureName == "BRDFLUT") {
        // generate if BRDFLUT is not supplied
        auto tex = resource::SVResourceManager::Get()->getDefaultBRDFLUT(*mContext);
        updateDescriptorSets(mContext->getDevice(), mSceneSet.get(), {},
                             {{tex->getImageView(*mContext), tex->getSampler(*mContext)}},
                             bindingIndex);
      } else if (mCustomCubemaps.find(customTextureName) != mCustomCubemaps.end()) {
        mCustomCubemaps[customTextureName]->uploadToDevice(*mContext);
        updateDescriptorSets(mContext->getDevice(), mSceneSet.get(), {},
                             {{mCustomCubemaps[customTextureName]->getImageView(*mContext),
                               mCustomCubemaps[customTextureName]->getSampler(*mContext)}},
                             bindingIndex);
      } else if (customTextureName == "Environment") {
        auto cube = mEnvironmentMap;
        if (!cube) {
          cube = resource::SVResourceManager::Get()->getDefaultCubemap();
        }
        cube->uploadToDevice(*mContext);
        updateDescriptorSets(mContext->getDevice(), mSceneSet.get(), {},
                             {{cube->getImageView(*mContext), cube->getSampler(*mContext)}},
                             bindingIndex);
      } else {
        throw std::runtime_error("custom sampler \"" + customTextureName +
                                 "\" is not set in the renderer");
      }
    } else {
      throw std::runtime_error("unrecognized uniform binding in scene \"" + binding.name + "\"");
    }
  }
  allocateLightBuffers();
} // namespace renderer

// A host-visible buffer of the scene set, rewritten in place from the CPU.
static std::unique_ptr<core::Buffer>
createSceneBuffer(core::Context &context, vk::DeviceSize size, vk::BufferUsageFlags usage) {
  return core::Buffer::Create(context, size, usage, VMA_MEMORY_USAGE_CPU_TO_GPU);
}

static void writeBufferBinding(vk::Device device, vk::DescriptorSet set, uint32_t binding,
                               vk::DescriptorType type, core::Buffer &buffer,
                               vk::DeviceSize range = VK_WHOLE_SIZE) {
  vk::DescriptorBufferInfo info{buffer.getVulkanBuffer(), 0, range};
  device.updateDescriptorSets(vk::WriteDescriptorSet(set, binding, 0, type, {}, info, {}),
                              nullptr);
}

// The textures set_texture (int field) or set_texture_array (int array field, cut to its
// length) assigned to a CustomObjectData field of an object.
static std::vector<std::shared_ptr<resource::SVTexture>>
customTextures(scene::Object const &object, std::string const &name,
               StructDataLayout::Element const &field) {
  if (field.array.empty()) {
    return {object.getCustomTexture(name)};
  }
  auto textures = object.getCustomTextureArray(name);
  textures.resize(std::min<size_t>(textures.size(), field.size / sizeof(int32_t)));
  return textures;
}

void Renderer::prepareGPUScene() {
  if (!mGPUSceneSet) {
    auto layout = mShaderPackInstance->getGPUSceneDescriptorSetLayout();
    mGPUSceneSet = std::move(mContext->getDevice()
                                 .allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(
                                     mGPUScenePool.get(), 1, &layout))
                                 .front());
    mCustomDataStride = mShaderPack->getShaderInputLayouts()->customObjectDataStride;
    // each instance is bound at its own dynamic offset
    auto alignment = static_cast<uint32_t>(
        mContext->getPhysicalDeviceLimits().minUniformBufferOffsetAlignment);
    mInstanceStride = (sizeof(InstanceInfo) + alignment - 1) / alignment * alignment;

    // Every slot of the fixed-size textures[] must hold a valid descriptor. Slot 0 is the
    // default texture for good; the other slots hold it while unused.
    auto def = resource::SVResourceManager::Get()->getDefaultTexture();
    def->loadAsync().get();
    def->uploadToDevice(*mContext);
    mBoundTextures = {{def->getSampler(*mContext), def->getImageView(*mContext),
                       vk::ImageLayout::eShaderReadOnlyOptimal}};
    std::vector<vk::DescriptorImageInfo> slots(mContext->getMaxNumTextures(), mBoundTextures[0]);
    mContext->getDevice().updateDescriptorSets(
        vk::WriteDescriptorSet(mGPUSceneSet.get(), 3, 0,
                               vk::DescriptorType::eCombinedImageSampler, slots, {}, {}),
        nullptr);
    mTextures = {def};
  }

  // The transform table is the Scene's (CPU uploads or CUDA pose writes); the Scene replaces it
  // when it grows.
  auto transformBuffer = mScene->getObjectTransformBuffer();
  if (mObjectTransformBuffer != transformBuffer) {
    mObjectTransformBuffer = std::move(transformBuffer);
    writeBufferBinding(mContext->getDevice(), mGPUSceneSet.get(), 0,
                       vk::DescriptorType::eStorageBuffer, *mObjectTransformBuffer);
  }
}

void Renderer::ensureGPUBuffers(uint32_t instanceCount, uint32_t objectCount) {
  // a buffer holds at least one element, so its binding is always valid
  instanceCount = std::max(instanceCount, 1u);
  if (instanceCount > mInstanceCapacity) {
    mInstanceCapacity = std::max(instanceCount, mInstanceCapacity * 2);
    mInstanceBuffer =
        createSceneBuffer(*mContext, vk::DeviceSize(mInstanceStride) * mInstanceCapacity,
                          vk::BufferUsageFlagBits::eUniformBuffer);
    writeBufferBinding(mContext->getDevice(), mGPUSceneSet.get(), 1,
                       vk::DescriptorType::eUniformBufferDynamic, *mInstanceBuffer,
                       sizeof(InstanceInfo));
  }
  objectCount = std::max(objectCount, 1u);
  if (mCustomDataStride && objectCount > mCustomDataCapacity) {
    mCustomDataCapacity = std::max(objectCount, mCustomDataCapacity * 2);
    mCustomDataBuffer =
        createSceneBuffer(*mContext, vk::DeviceSize(mCustomDataStride) * mCustomDataCapacity,
                          vk::BufferUsageFlagBits::eStorageBuffer);
    writeBufferBinding(mContext->getDevice(), mGPUSceneSet.get(), 4,
                       vk::DescriptorType::eStorageBuffer, *mCustomDataBuffer);
  }
}

void Renderer::packLights() {
  mLights.clear();
  // textured light i samples samplerTexturedLightTextures[i]
  int textured = 0;
  mLights.append(*mScene,
                 [&](std::shared_ptr<resource::SVTexture> const &) { return textured++; });
}

void Renderer::allocateLightBuffers() {
  packLights();
  auto allocate = [&](LightBufferBinding &light, char const *name, size_t count, size_t stride) {
    light.buffer.reset();
    if (light.binding < 0) {
      return;
    }
    // the shader arrays hold one light more than the count, so none is empty
    light.capacity = count + 1;
    vk::DeviceSize size = light.capacity * stride;
    if (size > mContext->getPhysicalDeviceLimits().maxUniformBufferRange) {
      throw std::runtime_error(std::string("the scene has ") + std::to_string(count) + " " +
                               name + " lights; their uniform buffer of " +
                               std::to_string(size) + " bytes exceeds the device limit of " +
                               std::to_string(mContext->getPhysicalDeviceLimits()
                                                  .maxUniformBufferRange) +
                               " bytes");
    }
    light.buffer = createSceneBuffer(*mContext, size, vk::BufferUsageFlagBits::eUniformBuffer);
    writeBufferBinding(mContext->getDevice(), mSceneSet.get(), light.binding,
                       vk::DescriptorType::eUniformBuffer, *light.buffer);
  };
  allocate(mPointLightBuffer, "point", mLights.point.size(), sizeof(scene::GpuPointLight));
  allocate(mDirectionalLightBuffer, "directional", mLights.directional.size(),
           sizeof(scene::GpuDirectionalLight));
  allocate(mSpotLightBuffer, "spot and textured", mLights.spot.size(),
           sizeof(scene::GpuSpotLight));
}

bool Renderer::refreshMaterialsAndTextures() {
  uint64_t version = resource::SVMetallicMaterial::getGlobalVersion();
  auto objects = mScene->getObjects();
  mMaterials.clear();
  mMaterialIndices.clear();
  mTextures.resize(1); // slot 0: the default texture
  mTextureIndices.clear();

  // Register a texture in textures[] and return its slot.
  auto textureIndex = [&](std::shared_ptr<resource::SVTexture> const &t) -> int32_t {
    auto [it, inserted] =
        mTextureIndices.try_emplace(t.get(), static_cast<int32_t>(mTextures.size()));
    if (inserted) {
      t->loadAsync().get();
      t->uploadToDevice(*mContext);
      mTextures.push_back(t);
    }
    return it->second;
  };

  std::vector<scene::GpuMaterial> gpuMaterials;
  std::vector<std::pair<uint8_t, vk::CullModeFlagBits>> drawStates;
  for (auto obj : objects) {
    for (auto const &shape : obj->getModel()->getShapes()) {
      auto mat = std::static_pointer_cast<resource::SVMetallicMaterial>(shape->material);
      if (!mMaterialIndices.try_emplace(mat.get(), static_cast<uint32_t>(gpuMaterials.size()))
               .second) {
        continue;
      }
      gpuMaterials.push_back(scene::packMaterial(*mat, textureIndex));
      float opacity = mat->getOpacity();
      uint8_t opacityBucket = opacity == 0.f ? 0 : (opacity != 1.f ? 1 : 2);
      drawStates.emplace_back(opacityBucket, mat->getCullMode());
      mMaterials.push_back(mat);
    }
  }

  // Per-object custom textures live in textures[] too: set_texture fills an int field and
  // set_texture_array an int array field of CustomObjectData with their slots.
  if (auto customLayout = mShaderPack->getShaderInputLayouts()->customObjectDataLayout) {
    for (auto obj : objects) {
      for (auto const &[name, elem] : customLayout->elements) {
        if (elem.dtype != DataType::INT()) {
          continue;
        }
        for (auto const &t : customTextures(*obj, name, elem)) {
          if (!t) {
            continue;
          }
          if (t->getDescription().dim != 2) {
            throw std::runtime_error("render body texture \"" + name +
                                     "\" must be 2D: shaders sample per-object textures "
                                     "from textures[] (sampler2D)");
          }
          textureIndex(t);
        }
      }
    }
  }

  if (mTextures.size() > mContext->getMaxNumTextures()) {
    throw std::runtime_error(
        "the scene uses " + std::to_string(mTextures.size() - 1) + " textures, more than the " +
        std::to_string(mContext->getMaxNumTextures() - 1) +
        " that fit in textures[]; raise max_num_textures in sapien.render.set_global_config()");
  }

  bool rebound = false;
  uint32_t materialCount = std::max<uint32_t>(gpuMaterials.size(), 1);
  if (materialCount > mMaterialCapacity) {
    mMaterialCapacity =
        std::max({materialCount, mMaterialCapacity * 2, mContext->getMaxNumMaterials()});
    mMaterialBuffer = createSceneBuffer(
        *mContext, sizeof(scene::GpuMaterial) * mMaterialCapacity,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eUniformBuffer);
    writeBufferBinding(mContext->getDevice(), mGPUSceneSet.get(), 2,
                       vk::DescriptorType::eStorageBuffer, *mMaterialBuffer);
    writeBufferBinding(mContext->getDevice(), mGPUSceneSet.get(), 5,
                       vk::DescriptorType::eUniformBufferDynamic, *mMaterialBuffer,
                       sizeof(scene::GpuMaterial));
    rebound = true;
  }
  if (!gpuMaterials.empty()) {
    mMaterialBuffer->upload(gpuMaterials);
  }

  // Rewrite the textures[] slots whose texture changed; slots past the list get the default.
  std::vector<vk::DescriptorImageInfo> textures{mBoundTextures[0]};
  for (size_t i = 1; i < mTextures.size(); ++i) {
    textures.push_back({mTextures[i]->getSampler(*mContext),
                        mTextures[i]->getImageView(*mContext),
                        vk::ImageLayout::eShaderReadOnlyOptimal});
  }
  if (textures != mBoundTextures) {
    auto slots = textures;
    slots.resize(std::max(textures.size(), mBoundTextures.size()), mBoundTextures[0]);
    mContext->getDevice().updateDescriptorSets(
        vk::WriteDescriptorSet(mGPUSceneSet.get(), 3, 0,
                               vk::DescriptorType::eCombinedImageSampler, slots, {}, {}),
        nullptr);
    mBoundTextures = std::move(textures);
    rebound = true;
  }

  if (drawStates != mMaterialDrawStates) {
    mMaterialDrawStates = std::move(drawStates);
    rebound = true;
  }

  mLastMaterialVersion = version;
  return rebound;
}

void Renderer::uploadInstances() {
  auto objects = mScene->getObjects();
  auto lineObjects = mScene->getLineObjects();
  auto pointObjects = mScene->getPointObjects();

  mInstanceData.assign(size_t(mInstanceStride) * mInstances.size(), 0);
  for (size_t i = 0; i < mInstances.size(); ++i) {
    auto const &ref = mInstances[i];
    InstanceInfo info{};
    info.transformIndex = ref.transformIndex;
    info.customDataIndex = ref.customDataIndex;
    info.materialIndex = -1;
    if (ref.kind == 0) {
      auto obj = objects.at(ref.objectIndex);
      info.materialIndex = mMaterialIndices.at(ref.material.get());
      info.segmentation = obj->getSegmentation();
      info.transparency = obj->getTransparency();
      info.shadeFlat = obj->getShadeFlat();
    } else if (ref.kind == 1) {
      auto obj = lineObjects.at(ref.objectIndex);
      info.segmentation = obj->getSegmentation();
      info.transparency = obj->getTransparency();
    } else {
      auto obj = pointObjects.at(ref.objectIndex);
      info.segmentation = obj->getSegmentation();
      info.transparency = obj->getTransparency();
    }
    std::memcpy(mInstanceData.data() + i * mInstanceStride, &info, sizeof(info));
  }
  if (!mInstanceData.empty()) {
    mInstanceBuffer->upload(mInstanceData);
  }

  // Per-object custom data, one element per object: set_property values by name and
  // textures[] slots from set_texture/set_texture_array in int fields. Unset fields stay 0,
  // which for a texture field is the default texture.
  auto customLayout = mShaderPack->getShaderInputLayouts()->customObjectDataLayout;
  if (!mCustomDataStride || !customLayout || objects.empty()) {
    return;
  }
  mCustomData.assign(size_t(mCustomDataStride) * objects.size(), 0);
  for (size_t i = 0; i < objects.size(); ++i) {
    uint8_t *dst = mCustomData.data() + i * mCustomDataStride;
    auto const &values = objects[i]->getCustomData();
    for (auto const &[name, elem] : customLayout->elements) {
      if (auto it = values.find(name); it != values.end()) {
        if (it->second.dtype != elem.dtype) {
          throw std::runtime_error("Upload object failed: object attribute \"" + name +
                                   "\" does not match declared type.");
        }
        std::memcpy(dst + elem.offset, &it->second.floatValue, elem.size);
      } else if (elem.dtype == DataType::INT()) {
        auto textures = customTextures(*objects[i], name, elem);
        for (size_t k = 0; k < textures.size(); ++k) {
          if (textures[k]) {
            int32_t slot = mTextureIndices.at(textures[k].get());
            std::memcpy(dst + elem.offset + k * sizeof(int32_t), &slot, sizeof(slot));
          }
        }
      }
    }
  }
  mCustomDataBuffer->upload(mCustomData);
}

void Renderer::bindSceneSet(vk::CommandBuffer commandBuffer, vk::PipelineLayout layout,
                            uint32_t set, uint32_t instance) {
  // dynamic offsets of binding 1 (the instance) and binding 5 (its material), in binding order
  std::array<uint32_t, 2> offsets{0, 0};
  if (instance < mInstances.size()) {
    offsets[0] = instance * mInstanceStride;
    if (auto const &material = mInstances[instance].material) {
      offsets[1] = mMaterialIndices.at(material.get()) * sizeof(scene::GpuMaterial);
    }
  }
  commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, set,
                                   mGPUSceneSet.get(), offsets);
}

void Renderer::prepareLightBuffers() {
  uint32_t numShadows =
      mShadowMaps ? static_cast<uint32_t>(mShadowMaps->lightBuffers.size()) : 0u;

  if (numShadows * 2 < mLightSets.size()) {
    mLightSets.resize(numShadows);
  }

  auto layout = mShaderPackInstance->getLightDescriptorSetLayout();
  while (mLightSets.size() < numShadows) {
    auto shadowSet = std::move(mContext->getDevice()
                                   .allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(
                                       mDescriptorPool.get(), 1, &layout))
                                   .front());
    mLightSets.push_back(std::move(shadowSet));
  }

  for (uint32_t i = 0; i < numShadows; ++i) {
    updateDescriptorSets(
        mContext->getDevice(), mLightSets[i].get(),
        {{vk::DescriptorType::eUniformBuffer, mShadowMaps->lightBuffers[i]->getVulkanBuffer(),
          nullptr}},
        {}, 0);
  }
}

void Renderer::prepareInputTextureDescriptorSets() {
  auto layouts = mShaderPackInstance->getInputTextureLayouts();
  auto passes = mShaderPack->getNonShadowPasses();

  mInputTextureSets.clear();
  for (uint32_t i = 0; i < passes.size(); ++i) {
    if (!mShaderPackInstance->isPassActive(i)) {
      mInputTextureSets.push_back({});
      continue;
    }
    auto layout = layouts[i];
    if (layout) {
      mInputTextureSets.push_back(
          std::move(mContext->getDevice()
                        .allocateDescriptorSetsUnique(
                            vk::DescriptorSetAllocateInfo(mDescriptorPool.get(), 1, &layout))
                        .front()));

      std::vector<std::tuple<vk::ImageView, vk::Sampler>> textureData;
      auto names = passes[i]->getInputTextureNames();
      for (auto name : names) {
        if (mRenderTargets.find(name) != mRenderTargets.end()) {
          textureData.push_back(
              {mRenderTargets[name]->getImageView(), mRenderTargets[name]->getSampler()});
        } else if (name.substr(0, 6) == "Random") {
          auto randomTex = resource::SVResourceManager::Get()->CreateRandomTexture(name);
          randomTex->uploadToDevice(*mContext);
          textureData.push_back(
              {randomTex->getImageView(*mContext), randomTex->getSampler(*mContext)});
        } else if (mCustomTextures.find(name) != mCustomTextures.end()) {
          mCustomTextures[name]->uploadToDevice(*mContext);
          textureData.push_back({mCustomTextures[name]->getImageView(*mContext),
                                 mCustomTextures[name]->getSampler(*mContext)});
        } else if (name == "BRDFLUT") {
          // generate if BRDFLUT is not supplied
          auto tex = resource::SVResourceManager::Get()->getDefaultBRDFLUT(*mContext);
          textureData.push_back({tex->getImageView(*mContext), tex->getSampler(*mContext)});
        } else if (mCustomCubemaps.find(name) != mCustomCubemaps.end()) {
          mCustomCubemaps[name]->uploadToDevice(*mContext);
          textureData.push_back({mCustomCubemaps[name]->getImageView(*mContext),
                                 mCustomCubemaps[name]->getSampler(*mContext)});
        } else if (name == "Environment") {
          auto cube = mEnvironmentMap;
          if (!cube) {
            cube = resource::SVResourceManager::Get()->getDefaultCubemap();
          }
          cube->uploadToDevice(*mContext);
          textureData.push_back({cube->getImageView(*mContext), cube->getSampler(*mContext)});
        } else {
          throw std::runtime_error("custom sampler \"" + name + "\" is not set in the renderer");
        }
      }
      updateDescriptorSets(mContext->getDevice(), mInputTextureSets.back().get(), {}, textureData,
                           0);
    } else {
      mInputTextureSets.push_back({});
    }
  }
}

void Renderer::prepareCameraBuffer() {
  if (mCameraBuffer) {
    // camera buffer can always be reused
    return;
  }

  mCameraBufferCpu = core::Buffer::Create(
      *mContext, mShaderPack->getShaderInputLayouts()->cameraBufferLayout->size,
      vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_ONLY);
  mCameraBuffer = core::Buffer::CreateUniform(
      *mContext, mShaderPack->getShaderInputLayouts()->cameraBufferLayout->size, true, true);

  auto layout = mShaderPackInstance->getCameraDescriptorSetLayout();
  mCameraSet = std::move(mContext->getDevice()
                             .allocateDescriptorSetsUnique(
                                 vk::DescriptorSetAllocateInfo(mDescriptorPool.get(), 1, &layout))
                             .front());
  updateDescriptorSets(
      mContext->getDevice(), mCameraSet.get(),
      {{vk::DescriptorType::eUniformBuffer, mCameraBuffer->getVulkanBuffer(), nullptr}}, {}, 0);
}

void Renderer::setCustomTextureArray(std::string const &name,
                                     std::vector<std::shared_ptr<resource::SVTexture>> textures) {
  mCustomTextureArray[name] = textures;
}
void Renderer::setCustomTexture(std::string const &name,
                                std::shared_ptr<resource::SVTexture> texture) {
  mCustomTextures[name] = texture;
}

void Renderer::setCustomCubemap(std::string const &name,
                                std::shared_ptr<resource::SVCubemap> cubemap) {
  mCustomCubemaps[name] = cubemap;
}

void Renderer::recordPostChainInto(vk::CommandBuffer commandBuffer) {
  auto &source = mRenderTargets.at(kRasterPostSource)->getImage();
  auto &result = mRenderTargets.at(kRasterPostResult)->getImage();
  // the pass sampling the result counts as a reader of the source (shader_pack_instance.cpp),
  // so the source is read-only here; the result is rewritten every frame
  source.transitionLayout(commandBuffer, vk::ImageLayout::eShaderReadOnlyOptimal,
                          vk::ImageLayout::eGeneral, vk::AccessFlagBits::eColorAttachmentWrite,
                          vk::AccessFlagBits::eShaderRead,
                          vk::PipelineStageFlagBits::eColorAttachmentOutput,
                          vk::PipelineStageFlagBits::eComputeShader);
  result.transitionLayout(commandBuffer, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
                          vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eShaderWrite,
                          vk::PipelineStageFlagBits::eFragmentShader,
                          vk::PipelineStageFlagBits::eComputeShader);
  mPostChain->record(commandBuffer);
  source.transitionLayout(commandBuffer, vk::ImageLayout::eGeneral,
                          vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead,
                          vk::AccessFlagBits::eShaderRead,
                          vk::PipelineStageFlagBits::eComputeShader,
                          vk::PipelineStageFlagBits::eFragmentShader);
  result.transitionLayout(commandBuffer, vk::ImageLayout::eGeneral,
                          vk::ImageLayout::eShaderReadOnlyOptimal,
                          vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
                          vk::PipelineStageFlagBits::eComputeShader,
                          vk::PipelineStageFlagBits::eFragmentShader);
}

CustomPropertyType Renderer::getCustomPropertyType(std::string const &name) const {
  if (isPostProperty(name)) {
    return mPostChain->getPropertyType(name);
  }
  auto it = mSpecializationConstants.find(name);
  if (it == mSpecializationConstants.end()) {
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

int Renderer::getCustomPropertyInt(std::string const &name) const {
  if (isPostProperty(name)) {
    return mPostChain->getPropertyInt(name);
  }
  if (mSpecializationConstants.contains(name)) {
    auto c = mSpecializationConstants.at(name);
    if (c.dtype == DataType::INT()) {
      int v;
      std::memcpy(&v, c.buffer, sizeof(int));
      return v;
    }
  }
  throw std::runtime_error("invalid property " + name);
}

float Renderer::getCustomPropertyFloat(std::string const &name) const {
  if (isPostProperty(name)) {
    return mPostChain->getPropertyFloat(name);
  }
  if (mSpecializationConstants.contains(name)) {
    auto c = mSpecializationConstants.at(name);
    if (c.dtype == DataType::FLOAT()) {
      float v;
      std::memcpy(&v, c.buffer, sizeof(float));
      return v;
    }
  }
  throw std::runtime_error("invalid property " + name);
}

glm::vec3 Renderer::getCustomPropertyVec3(std::string const &name) const {
  if (mSpecializationConstants.contains(name)) {
    auto c = mSpecializationConstants.at(name);
    if (c.dtype == DataType::FLOAT3()) {
      glm::vec3 v;
      std::memcpy(&v[0], c.buffer, sizeof(float) * 3);
      return v;
    }
  }
  throw std::runtime_error("invalid property " + name);
}
glm::vec4 Renderer::getCustomPropertyVec4(std::string const &name) const {
  if (mSpecializationConstants.contains(name)) {
    auto c = mSpecializationConstants.at(name);
    if (c.dtype == DataType::FLOAT4()) {
      glm::vec4 v;
      std::memcpy(&v[0], c.buffer, sizeof(float) * 4);
      return v;
    }
  }
  throw std::runtime_error("invalid property " + name);
}

void Renderer::setCustomProperty(std::string const &name, int p) {
  if (!isPostProperty(name)) {
    setSpecializationConstant(name, p);
  } else if (mPostChain->getPropertyInt(name) != p) {
    mPostChain->setProperty(name, p);
    mRequiresRecord = true;
  }
}

void Renderer::setCustomProperty(std::string const &name, float p) {
  if (!isPostProperty(name)) {
    setSpecializationConstant(name, p);
  } else if (mPostChain->getPropertyFloat(name) != p) {
    mPostChain->setProperty(name, p);
    mRequiresRecord = true;
  }
}

void Renderer::setCustomProperty(std::string const &name, glm::vec3 p) {
  setSpecializationConstant(name, p);
}

void Renderer::setCustomProperty(std::string const &name, glm::vec4 p) {
  setSpecializationConstant(name, p);
}

std::shared_ptr<resource::SVRenderTarget>
Renderer::getRenderTarget(std::string const &name) const {
  auto it = mRenderTargets.find(name);
  if (it == mRenderTargets.end()) {
    throw std::runtime_error("render target \"" + name + "\" is not available");
  }
  return it->second;
}

core::Buffer &Renderer::getCameraBuffer() {
  prepareCameraBuffer();
  return *mCameraBuffer;
}

void Renderer::prepareResources(scene::Camera &camera) { prepareRender(camera); }

void Renderer::uploadCpuFrameState(scene::Camera &camera) { uploadGpuResources(camera); }

} // namespace renderer
} // namespace svulkan2
