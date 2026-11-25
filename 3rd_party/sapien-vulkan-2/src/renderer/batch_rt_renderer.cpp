#include "svulkan2/renderer/batch_rt_renderer.h"
#include "../common/logger.h"
#include "svulkan2/common/format.h"
#include "svulkan2/core/as.h"
#include "svulkan2/core/context.h"
#include "svulkan2/resource/manager.h"
#include "svulkan2/scene/light.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <sstream>
#include <unordered_map>

namespace svulkan2 {
namespace renderer {

namespace {

/** pictures of batch.comp, in the binding order the shader declares them */
const std::vector<std::string> BATCH_PICTURES = {"Color",       "Albedo", "Depth",
                                                "Normal",      "Position", "Segmentation"};

/** push constant members the renderer writes itself */
const std::vector<std::string> BATCH_RESERVED_PROPERTIES = {"cameraCount", "width", "height"};

std::string upperCase(std::string const &text) {
  std::string result = text;
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return result;
}

/** binding of `name` in batch.comp, or throws */
uint32_t pictureBinding(std::string const &name) {
  auto it = std::find(BATCH_PICTURES.begin(), BATCH_PICTURES.end(), name);
  if (it == BATCH_PICTURES.end()) {
    throw std::runtime_error(
        "batched ray tracing picture \"" + name +
        "\" is not a picture of batch.comp; the pack declares Color, Albedo, Depth, Normal, "
        "Position and Segmentation");
  }
  return static_cast<uint32_t>(std::distance(BATCH_PICTURES.begin(), it));
}

void copyField(std::vector<uint8_t> &data, size_t base, uint32_t offset, void const *value,
               size_t size) {
  std::memcpy(data.data() + base + offset, value, size);
}

} // namespace

BatchRTRenderer::BatchRTRenderer(std::shared_ptr<core::Context> context,
                                 std::string const &shaderDir, ShaderDefines const &defines,
                                 int width, int height, std::vector<std::string> const &pictures,
                                 std::map<std::string, vk::Format> const &formats)
    : mContext(context), mShaderDir(shaderDir), mWidth(width), mHeight(height) {
  if (!mContext || !mContext->isVulkanAvailable()) {
    throw std::runtime_error("failed to create the batched ray tracing renderer: no Vulkan "
                             "device");
  }
  if (!mContext->isRayTracingAvailable()) {
    throw std::runtime_error("failed to create the batched ray tracing renderer: the device has "
                             "no ray tracing support");
  }
  if (mWidth <= 0 || mHeight <= 0) {
    throw std::runtime_error(
        "failed to create the batched ray tracing renderer: invalid resolution");
  }
  if (pictures.empty()) {
    throw std::runtime_error(
        "failed to create the batched ray tracing renderer: no picture is requested");
  }

  // Each requested picture is a define, so the reflection of the compiled shader already
  // describes exactly the pictures this camera group asks for.
  for (std::string const &name : pictures) {
    pictureBinding(name);
  }
  mPictureNames = pictures;
  std::sort(mPictureNames.begin(), mPictureNames.end());
  mPictureNames.erase(std::unique(mPictureNames.begin(), mPictureNames.end()),
                      mPictureNames.end());

  ShaderDefines packDefines = defines;
  for (auto const &name : mPictureNames) {
    packDefines["OUTPUT_" + upperCase(name)] = "";
  }
  for (auto const &[name, format] : formats) {
    if (format == vk::Format::eUndefined) {
      continue;
    }
    if (std::find(mPictureNames.begin(), mPictureNames.end(), name) == mPictureNames.end()) {
      throw std::runtime_error("batched ray tracing picture format \"" + name +
                               "\" names a picture this camera group does not render; request "
                               "the picture among the group's pictures first");
    }
    packDefines[upperCase(name) + "_FORMAT_" + shader::getBatchFormatDefineName(format)] = "";
  }

  mShaderPack = resource::SVResourceManager::Get()->CreateBatchShaderPack(mShaderDir, packDefines);

  for (auto const &name : mPictureNames) {
    vk::Format declared = mShaderPack->getOutputFormat(name);
    if (declared == vk::Format::eUndefined) {
      throw std::runtime_error("batched ray tracing shader does not declare the picture \"" +
                               name + "\": add a storage buffer block named Out" + name);
    }
    auto it = formats.find(name);
    if (it != formats.end() && it->second != vk::Format::eUndefined && it->second != declared) {
      throw std::runtime_error("batched ray tracing picture \"" + name + "\" is declared as " +
                               vk::to_string(declared) + " by the shader, not the requested " +
                               vk::to_string(it->second) +
                               "; in batch.comp declare uint for 8-bit colour, vec4 for "
                               "float32, float for r32f and uvec4 for uint32");
    }
    mPictureFormats[name] = declared;
    mPictureBytes = static_cast<vk::DeviceSize>(mWidth) * mHeight * getFormatSize(declared);
  }

}

BatchRTRenderer::~BatchRTRenderer() = default;

void BatchRTRenderer::buildPipeline() {
  vk::Device device = mContext->getDevice();

  std::vector<vk::PushConstantRange> ranges;
  if (mShaderPack->getPushConstantLayout() && mShaderPack->getPushConstantLayout()->size) {
    ranges.emplace_back(vk::ShaderStageFlagBits::eCompute, 0,
                        mShaderPack->getPushConstantLayout()->size);
    mPushConstantBuffer.assign(mShaderPack->getPushConstantLayout()->size, 0);
  }

  std::vector<vk::DescriptorSetLayoutBinding> outputBindings;
  for (auto const &[bid, binding] : mShaderPack->getOutputDescription().bindings) {
    if (binding.type != vk::DescriptorType::eStorageBuffer) {
      throw std::runtime_error("batched ray tracing outputs must be storage buffers");
    }
    outputBindings.emplace_back(bid, vk::DescriptorType::eStorageBuffer, 1,
                                vk::ShaderStageFlagBits::eCompute);
  }
  mOutputLayout = device.createDescriptorSetLayoutUnique(
      vk::DescriptorSetLayoutCreateInfo({}, outputBindings));

  // The scene set holds the environment acceleration structures and the group tables. The
  // arrays are sized by the group's tables (built by initialize()) and partially bound, so a
  // table may be empty (no textures, no point sets) without a dummy descriptor.
  std::vector<vk::DescriptorSetLayoutBinding> sceneBindings;
  std::vector<vk::DescriptorBindingFlags> sceneFlags;
  for (auto const &[bid, binding] : mShaderPack->getSceneDescription().bindings) {
    if (binding.type != vk::DescriptorType::eAccelerationStructureKHR &&
        binding.type != vk::DescriptorType::eStorageBuffer &&
        binding.type != vk::DescriptorType::eCombinedImageSampler) {
      throw std::runtime_error("the batched ray tracing scene set cannot hold " +
                               vk::to_string(binding.type));
    }
    sceneBindings.emplace_back(bid, binding.type, getSceneBindingCount(binding.name),
                               vk::ShaderStageFlagBits::eCompute);
    sceneFlags.push_back(binding.dim > 0 ? vk::DescriptorBindingFlags(
                                               vk::DescriptorBindingFlagBits::ePartiallyBound)
                                         : vk::DescriptorBindingFlags{});
  }
  vk::DescriptorSetLayoutBindingFlagsCreateInfo sceneFlagInfo(sceneFlags);
  vk::DescriptorSetLayoutCreateInfo sceneLayoutInfo({}, sceneBindings);
  sceneLayoutInfo.setPNext(&sceneFlagInfo);
  mSceneLayout = device.createDescriptorSetLayoutUnique(sceneLayoutInfo);

  std::vector<vk::DescriptorSetLayout> layouts(mShaderPack->getResources().size());
  layouts.at(mShaderPack->getOutputSetId()) = mOutputLayout.get();
  layouts.at(mShaderPack->getSceneSetId()) = mSceneLayout.get();
  mPipelineLayout =
      device.createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo({}, layouts, ranges));

  auto module =
      device.createShaderModuleUnique(vk::ShaderModuleCreateInfo({}, mShaderPack->getCode()));
  auto result = device.createComputePipelineUnique(
      {}, vk::ComputePipelineCreateInfo(
              {}, vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eCompute,
                                                   module.get(), "main"),
              mPipelineLayout.get()));
  if (result.result != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create the batched ray tracing pipeline");
  }
  mPipeline = std::move(result.value);

  mCommandPool = mContext->createCommandPool();
  mCommandBuffer = mCommandPool->allocateCommandBuffer();
}

void BatchRTRenderer::initialize(std::vector<std::shared_ptr<scene::Scene>> const &environments,
                                 int cameraCount, std::vector<int> const &cameraEnvironments) {
  if (mInitialized) {
    throw std::runtime_error("the batched ray tracing renderer is already initialized");
  }
  if (environments.empty()) {
    throw std::runtime_error("the batched ray tracing renderer needs at least one environment");
  }
  if (cameraCount <= 0 ||
      cameraEnvironments.size() != static_cast<size_t>(cameraCount)) {
    throw std::runtime_error("the batched ray tracing renderer needs one environment per camera");
  }

  mEnvironments.clear();
  mEnvironmentRows.assign(environments.size(), BatchEnvironmentRow{});
  std::vector<scene::Scene::AccelerationStructureBuild> builds;
  for (auto const &environment : environments) {
    if (!environment) {
      throw std::runtime_error("a batched ray tracing environment has no scene");
    }
    if (environment->getContext() != mContext) {
      throw std::runtime_error("a batched ray tracing environment renders on another device");
    }
    // an externally owned scene gets its instance rows from CUDA
    builds.push_back({environment.get(), environment->hasExternalTransformOwnership()});
    mEnvironments.push_back(environment);
  }
  // every environment's BLASes and TLAS together; the group keeps its own scene tables, so the
  // per-camera ray tracing storage buffers are never built
  scene::Scene::BuildAccelerationStructures(builds);
  for (int index : cameraEnvironments) {
    if (index < 0 || index >= static_cast<int>(environments.size())) {
      throw std::runtime_error("batched ray tracing camera environment out of range");
    }
  }
  mCameraCount = cameraCount;

  // group tables, concatenated in environment order
  std::vector<uint8_t> objectData;
  std::vector<uint8_t> geometryData;
  std::vector<int> pointInstanceData;
  uint32_t objectStride = mShaderPack->getObjectBufferLayout().size;
  std::vector<uint8_t> textureIndexData;
  uint32_t textureIndexStride = mShaderPack->getTextureIndexBufferLayout().size;

  std::unordered_map<resource::SVMesh *, uint32_t> meshIds;
  std::unordered_map<resource::SVMaterial *, uint32_t> materialIds;
  std::unordered_map<resource::SVTexture *, uint32_t> textureIds;
  std::unordered_map<resource::SVPointSet *, uint32_t> pointSetIds;
  std::vector<std::shared_ptr<resource::SVMesh>> meshes;
  std::vector<std::shared_ptr<resource::SVMetallicMaterial>> materials;
  std::vector<std::shared_ptr<resource::SVTexture>> textures;
  std::vector<std::shared_ptr<resource::SVPointSet>> pointSets;

  uint32_t geometryStride = mShaderPack->getGeometryInstanceBufferLayout().size;
  uint32_t geometryIndexOffset =
      mShaderPack->getGeometryInstanceBufferLayout().elements.at("geometryIndex").offset;
  uint32_t materialIndexOffset =
      mShaderPack->getGeometryInstanceBufferLayout().elements.at("materialIndex").offset;
  uint32_t segmentationOffset =
      mShaderPack->getObjectBufferLayout().elements.at("segmentation").offset;
  uint32_t transparencyOffset =
      mShaderPack->getObjectBufferLayout().elements.at("transparency").offset;
  uint32_t shadeFlatOffset = mShaderPack->getObjectBufferLayout().elements.at("shadeFlat").offset;

  for (size_t envIndex = 0; envIndex < mEnvironments.size(); ++envIndex) {
    auto const &scene = mEnvironments.at(envIndex);
    BatchEnvironmentRow &row = mEnvironmentRows.at(envIndex);
    row.objectOffset = static_cast<int>(objectData.size() / objectStride);
    row.geometryInstanceOffset = static_cast<int>(geometryData.size() / geometryStride);
    row.pointInstanceOffset = static_cast<int>(pointInstanceData.size());

    auto appendObject = [&](glm::uvec4 segmentation, float transparency, int shadeFlat) {
      size_t base = objectData.size();
      objectData.resize(base + objectStride, 0);
      copyField(objectData, base, segmentationOffset, &segmentation, sizeof(glm::uvec4));
      copyField(objectData, base, transparencyOffset, &transparency, sizeof(float));
      copyField(objectData, base, shadeFlatOffset, &shadeFlat, sizeof(int));
    };

    // triangles first, then point objects: the scene's TLAS numbers its instances that way
    for (auto *object : scene->getObjects()) {
      appendObject(object->getSegmentation(), object->getTransparency(),
                   object->getShadeFlat() ? 1 : 0);
      for (auto &shape : object->getModel()->getShapes()) {
        shape->mesh->uploadToDevice(*mContext);
        shape->material->uploadToDevice(*mContext);

        uint32_t meshId;
        auto meshIt = meshIds.find(shape->mesh.get());
        if (meshIt == meshIds.end()) {
          meshId = static_cast<uint32_t>(meshes.size());
          meshIds[shape->mesh.get()] = meshId;
          meshes.push_back(shape->mesh);
        } else {
          meshId = meshIt->second;
        }

        uint32_t materialId;
        auto materialIt = materialIds.find(shape->material.get());
        if (materialIt == materialIds.end()) {
          auto material = std::static_pointer_cast<resource::SVMetallicMaterial>(shape->material);
          materialId = static_cast<uint32_t>(materials.size());
          materialIds[shape->material.get()] = materialId;
          materials.push_back(material);
          std::array candidateTextures{material->getEmissionTexture(),
                                       material->getBaseColorTexture(),
                                       material->getNormalTexture(),
                                       material->getMetallicTexture(),
                                       material->getRoughnessTexture(),
                                       material->getTransmissionTexture()};
          for (auto const &texture : candidateTextures) {
            if (texture && !textureIds.contains(texture.get())) {
              texture->uploadToDevice(*mContext);
              textureIds[texture.get()] = static_cast<uint32_t>(textures.size());
              textures.push_back(texture);
            }
          }
        } else {
          materialId = materialIt->second;
        }

        size_t base = geometryData.size();
        geometryData.resize(base + geometryStride, 0);
        copyField(geometryData, base, geometryIndexOffset, &meshId, sizeof(uint32_t));
        copyField(geometryData, base, materialIndexOffset, &materialId, sizeof(uint32_t));
      }
    }

    for (auto *object : scene->getPointObjects()) {
      appendObject(object->getSegmentation(), object->getTransparency(), 0);
      auto pointSet = object->getPointSet();
      pointSet->uploadToDevice(*mContext);
      uint32_t pointSetId;
      auto it = pointSetIds.find(pointSet.get());
      if (it == pointSetIds.end()) {
        pointSetId = static_cast<uint32_t>(pointSets.size());
        pointSetIds[pointSet.get()] = pointSetId;
        pointSets.push_back(pointSet);
      } else {
        pointSetId = it->second;
      }
      pointInstanceData.push_back(static_cast<int>(pointSetId));
    }
  }

  // per material texture indices; the layout comes from the shader
  textureIndexData.assign(materials.size() * textureIndexStride, 0);
  auto const &textureIndexLayout = mShaderPack->getTextureIndexBufferLayout();
  auto textureOffset = [&](char const *name) {
    auto it = textureIndexLayout.elements.find(name);
    if (it == textureIndexLayout.elements.end()) {
      throw std::runtime_error(std::string("batch.comp TextureIndex must declare \"") + name +
                               "\"");
    }
    return it->second.offset;
  };
  mMaterialBuffers.clear();
  for (size_t index = 0; index < materials.size(); ++index) {
    auto const &material = materials.at(index);
    mMaterialBuffers.push_back(material->getDeviceBuffer(*mContext).getVulkanBuffer());
    auto id = [&](std::shared_ptr<resource::SVTexture> const &texture) {
      return texture ? static_cast<int>(textureIds.at(texture.get())) : -1;
    };
    int emission = id(material->getEmissionTexture());
    int diffuse = id(material->getBaseColorTexture());
    int normal = id(material->getNormalTexture());
    int metallic = id(material->getMetallicTexture());
    int roughness = id(material->getRoughnessTexture());
    int transmission = id(material->getTransmissionTexture());
    size_t base = index * textureIndexStride;
    copyField(textureIndexData, base, textureOffset("emission"), &emission, sizeof(int));
    copyField(textureIndexData, base, textureOffset("diffuse"), &diffuse, sizeof(int));
    copyField(textureIndexData, base, textureOffset("normal"), &normal, sizeof(int));
    copyField(textureIndexData, base, textureOffset("metallic"), &metallic, sizeof(int));
    copyField(textureIndexData, base, textureOffset("roughness"), &roughness, sizeof(int));
    copyField(textureIndexData, base, textureOffset("transmission"), &transmission, sizeof(int));
  }

  mVertexBuffers.clear();
  mIndexBuffers.clear();
  for (auto const &mesh : meshes) {
    mVertexBuffers.push_back(mesh->getVertexBuffer(*mContext).getVulkanBuffer());
    mIndexBuffers.push_back(mesh->getIndexBuffer(*mContext).getVulkanBuffer());
  }
  mPointSetBuffers.clear();
  for (auto const &pointSet : pointSets) {
    mPointSetBuffers.push_back(pointSet->getVertexBuffer(*mContext).getVulkanBuffer());
  }

  mTextures.clear();
  for (auto const &texture : textures) {
    mTextures.emplace_back(texture->getImageView(*mContext), texture->getSampler(*mContext));
  }

  auto createStorage = [&](void const *data, size_t size) {
    auto buffer = core::Buffer::Create(
        *mContext, std::max<size_t>(size, 1),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_GPU_ONLY);
    if (size) {
      buffer->upload(data, size);
    }
    return buffer;
  };

  mGeometryInstanceBuffer = createStorage(geometryData.data(), geometryData.size());
  mTextureIndexBuffer = createStorage(textureIndexData.data(), textureIndexData.size());
  mObjectBuffer = createStorage(objectData.data(), objectData.size());
  mPointInstanceBuffer =
      createStorage(pointInstanceData.data(), pointInstanceData.size() * sizeof(int));

  // camera rows sorted by environment: neighbouring workgroups trace the same TLAS
  std::vector<int> cameraOfRow(cameraCount);
  for (int camera = 0; camera < cameraCount; ++camera) {
    cameraOfRow[camera] = camera;
  }
  std::stable_sort(cameraOfRow.begin(), cameraOfRow.end(), [&](int a, int b) {
    return cameraEnvironments.at(a) < cameraEnvironments.at(b);
  });
  mCameraOrder = cameraOfRow;

  mCameraSlotSize = sizeof(BatchCameraRow);
  std::vector<BatchCameraRow> rows(cameraCount);
  for (int row = 0; row < cameraCount; ++row) {
    rows.at(row).envIndex = cameraEnvironments.at(cameraOfRow.at(row));
  }
  // device local and exported: the CUDA pose kernel writes view matrices into the rows in place
  mCameraBuffer = core::Buffer::Create(
      *mContext, mCameraSlotSize * mCameraCount,
      vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_GPU_ONLY, VmaAllocationCreateFlags{}, true);
  mCameraBuffer->upload(rows);

  mCameraOrderBuffer = core::Buffer::Create(
      *mContext, sizeof(int) * std::max<size_t>(1, mCameraOrder.size()),
      vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_CPU_TO_GPU);
  mCameraOrderBuffer->upload(mCameraOrder);

  // Environment and light tables. Their sizes are fixed from here on: the camera group rejects
  // topology changes, so later light edits rewrite rows in place (uploadSceneLightState).
  refreshEnvironmentRows();
  refreshLightRows();
  mEnvironmentBuffer = createStorage(mEnvironmentRows.data(),
                                     mEnvironmentRows.size() * sizeof(BatchEnvironmentRow));
  mPointLightBuffer = createStorage(mLights.point.data(),
                                    mLights.point.size() * sizeof(scene::GpuPointLight));
  mDirectionalLightBuffer = createStorage(
      mLights.directional.data(), mLights.directional.size() * sizeof(scene::GpuDirectionalLight));
  mSpotLightBuffer =
      createStorage(mLights.spot.data(), mLights.spot.size() * sizeof(scene::GpuSpotLight));
  mParallelogramLightBuffer =
      createStorage(mLights.parallelogram.data(),
                    mLights.parallelogram.size() * sizeof(scene::GpuParallelogramLight));

  allocatePictures();
  buildPipeline();
  writeDescriptors();
  mInitialized = true;
}

void BatchRTRenderer::allocatePictures() {
  for (auto const &name : mPictureNames) {
    // exported to CUDA: the group writes each picture in place, so take_picture copies nothing
    mPictureBuffers[name] = core::Buffer::Create(
        *mContext, mPictureBytes * mCameraCount,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
            vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_GPU_ONLY, VmaAllocationCreateFlags{}, true);
  }
}

void BatchRTRenderer::writeDescriptors() {
  vk::Device device = mContext->getDevice();

  mOutputPool = std::make_unique<core::DynamicDescriptorPool>(
      mContext->getDevice2(),
      std::vector<vk::DescriptorPoolSize>{{vk::DescriptorType::eStorageBuffer,
                                           static_cast<uint32_t>(mPictureNames.size())}});
  mOutputSet = mOutputPool->allocateSet(mOutputLayout.get());

  std::map<vk::DescriptorType, uint32_t> sceneCounts;
  for (auto const &[bid, binding] : mShaderPack->getSceneDescription().bindings) {
    sceneCounts[binding.type] += getSceneBindingCount(binding.name);
  }
  std::vector<vk::DescriptorPoolSize> sceneSizes;
  for (auto const &[type, count] : sceneCounts) {
    sceneSizes.emplace_back(type, count);
  }
  mScenePool = std::make_unique<core::DynamicDescriptorPool>(mContext->getDevice2(), sceneSizes);
  mSceneSet = mScenePool->allocateSet(mSceneLayout.get());

  // Each write owns its payload. Buffer and image infos live on the heap and survive the
  // vector growing; the acceleration structure chain lives in the element itself, so pNext is
  // linked only after the last element was added.
  struct BatchDescriptorWrite {
    vk::WriteDescriptorSet write;
    std::vector<vk::DescriptorBufferInfo> buffers;
    std::vector<vk::DescriptorImageInfo> images;
    std::vector<vk::AccelerationStructureKHR> accelerationStructures;
    vk::WriteDescriptorSetAccelerationStructureKHR accelerationStructureWrite;
  };
  std::vector<BatchDescriptorWrite> pending;

  auto addBuffers = [&](vk::DescriptorSet set, uint32_t binding,
                        std::vector<vk::DescriptorBufferInfo> infos) {
    pending.emplace_back();
    BatchDescriptorWrite &entry = pending.back();
    entry.buffers = std::move(infos);
    entry.write = vk::WriteDescriptorSet(set, binding, 0,
                                         static_cast<uint32_t>(entry.buffers.size()),
                                         vk::DescriptorType::eStorageBuffer, nullptr,
                                         entry.buffers.data());
  };
  auto addImages = [&](vk::DescriptorSet set, uint32_t binding,
                       std::vector<vk::DescriptorImageInfo> infos) {
    pending.emplace_back();
    BatchDescriptorWrite &entry = pending.back();
    entry.images = std::move(infos);
    entry.write = vk::WriteDescriptorSet(set, binding, 0,
                                         static_cast<uint32_t>(entry.images.size()),
                                         vk::DescriptorType::eCombinedImageSampler,
                                         entry.images.data());
  };

  // output set: one storage buffer per picture
  for (auto const &[bid, binding] : mShaderPack->getOutputDescription().bindings) {
    std::string name = binding.name.substr(3);
    addBuffers(mOutputSet.get(), bid,
               {{mPictureBuffers.at(name)->getVulkanBuffer(), 0, VK_WHOLE_SIZE}});
  }

  // scene set, bound by name
  std::vector<vk::AccelerationStructureKHR> tlas;
  for (auto const &scene : mEnvironments) {
    tlas.push_back(scene->getTLAS()->getVulkanAS());
  }

  auto buffersOf = [&](std::vector<vk::Buffer> const &buffers) {
    std::vector<vk::DescriptorBufferInfo> infos;
    for (auto buffer : buffers) {
      infos.emplace_back(buffer, 0, VK_WHOLE_SIZE);
    }
    return infos;
  };

  for (auto const &[bid, binding] : mShaderPack->getSceneDescription().bindings) {
    std::string const &name = binding.name;
    if (name == "tlas") {
      pending.emplace_back();
      BatchDescriptorWrite &entry = pending.back();
      entry.accelerationStructures = tlas;
      entry.accelerationStructureWrite =
          vk::WriteDescriptorSetAccelerationStructureKHR(
              static_cast<uint32_t>(entry.accelerationStructures.size()),
              entry.accelerationStructures.data());
      entry.write = vk::WriteDescriptorSet(mSceneSet.get(), bid, 0,
                                           static_cast<uint32_t>(tlas.size()),
                                           vk::DescriptorType::eAccelerationStructureKHR);
    } else if (name == "Cameras") {
      addBuffers(mSceneSet.get(), bid, {{mCameraBuffer->getVulkanBuffer(), 0, VK_WHOLE_SIZE}});
    } else if (name == "Environments") {
      addBuffers(mSceneSet.get(), bid, {{mEnvironmentBuffer->getVulkanBuffer(), 0, VK_WHOLE_SIZE}});
    } else if (name == "CameraOrder") {
      addBuffers(mSceneSet.get(), bid, {{mCameraOrderBuffer->getVulkanBuffer(), 0, VK_WHOLE_SIZE}});
    } else if (name == "GeometryInstances") {
      addBuffers(mSceneSet.get(), bid,
                 {{mGeometryInstanceBuffer->getVulkanBuffer(), 0, VK_WHOLE_SIZE}});
    } else if (name == "Materials") {
      addBuffers(mSceneSet.get(), bid, buffersOf(mMaterialBuffers));
    } else if (name == "TextureIndices") {
      addBuffers(mSceneSet.get(), bid,
                 {{mTextureIndexBuffer->getVulkanBuffer(), 0, VK_WHOLE_SIZE}});
    } else if (name == "textures") {
      std::vector<vk::DescriptorImageInfo> infos;
      for (auto const &[view, sampler] : mTextures) {
        infos.emplace_back(sampler, view, vk::ImageLayout::eShaderReadOnlyOptimal);
      }
      addImages(mSceneSet.get(), bid, std::move(infos));
    } else if (name == "environmentMaps") {
      addImages(mSceneSet.get(), bid, environmentMapInfos());
    } else if (name == "Objects") {
      addBuffers(mSceneSet.get(), bid, {{mObjectBuffer->getVulkanBuffer(), 0, VK_WHOLE_SIZE}});
    } else if (name == "Vertices") {
      addBuffers(mSceneSet.get(), bid, buffersOf(mVertexBuffers));
    } else if (name == "Indices") {
      addBuffers(mSceneSet.get(), bid, buffersOf(mIndexBuffers));
    } else if (name == "PointLights") {
      addBuffers(mSceneSet.get(), bid, {{mPointLightBuffer->getVulkanBuffer(), 0, VK_WHOLE_SIZE}});
    } else if (name == "DirectionalLights") {
      addBuffers(mSceneSet.get(), bid,
                 {{mDirectionalLightBuffer->getVulkanBuffer(), 0, VK_WHOLE_SIZE}});
    } else if (name == "SpotLights") {
      addBuffers(mSceneSet.get(), bid, {{mSpotLightBuffer->getVulkanBuffer(), 0, VK_WHOLE_SIZE}});
    } else if (name == "ParallelogramLights") {
      addBuffers(mSceneSet.get(), bid,
                 {{mParallelogramLightBuffer->getVulkanBuffer(), 0, VK_WHOLE_SIZE}});
    } else if (name == "Points") {
      addBuffers(mSceneSet.get(), bid, buffersOf(mPointSetBuffers));
    } else if (name == "PointInstances") {
      addBuffers(mSceneSet.get(), bid,
                 {{mPointInstanceBuffer->getVulkanBuffer(), 0, VK_WHOLE_SIZE}});
    } else {
      throw std::runtime_error("unrecognized batched ray tracing scene binding \"" + name + "\"");
    }
  }

  std::vector<vk::WriteDescriptorSet> writes;
  writes.reserve(pending.size());
  for (auto const &entry : pending) {
    writes.push_back(entry.write);
    if (entry.write.descriptorType == vk::DescriptorType::eAccelerationStructureKHR) {
      writes.back().setPNext(&entry.accelerationStructureWrite);
    }
  }
  device.updateDescriptorSets(writes, {});
}

uint32_t BatchRTRenderer::getSceneBindingCount(std::string const &name) const {
  size_t count = 1;
  if (name == "tlas") {
    count = mEnvironments.size();
  } else if (name == "Vertices" || name == "Indices") {
    count = mVertexBuffers.size();
  } else if (name == "Materials") {
    count = mMaterialBuffers.size();
  } else if (name == "textures") {
    count = mTextures.size();
  } else if (name == "Points") {
    count = mPointSetBuffers.size();
  } else if (name == "environmentMaps") {
    count = mEnvironments.size() + 1; // the default cube plus at most one per environment
  }
  return static_cast<uint32_t>(std::max<size_t>(count, 1));
}

std::vector<vk::DescriptorImageInfo> BatchRTRenderer::environmentMapInfos() {
  // slot 0 is the default black cube, so a zero index means "no environment map"
  std::vector<vk::DescriptorImageInfo> infos;
  auto black = resource::SVResourceManager::Get()->getDefaultCubemap();
  black->uploadToDevice(*mContext);
  infos.emplace_back(black->getSampler(*mContext), black->getImageView(*mContext),
                     vk::ImageLayout::eShaderReadOnlyOptimal);
  for (auto const &map : mEnvironmentMaps) {
    infos.emplace_back(map->getSampler(*mContext), map->getImageView(*mContext),
                       vk::ImageLayout::eShaderReadOnlyOptimal);
  }
  return infos;
}

void BatchRTRenderer::refreshEnvironmentRows() {
  // environmentMap is 1 + the slot of the environment's cubemap in the sampler array
  mEnvironmentMaps.clear();
  for (size_t index = 0; index < mEnvironments.size(); ++index) {
    BatchEnvironmentRow &row = mEnvironmentRows.at(index);
    auto const &scene = mEnvironments.at(index);
    row.ambientLight = scene->getAmbientLight();
    row.environmentMap = 0;
    auto cubemap = scene->getEnvironmentMap();
    if (!cubemap) {
      continue;
    }
    cubemap->load();
    cubemap->uploadToDevice(*mContext);
    auto it = std::find(mEnvironmentMaps.begin(), mEnvironmentMaps.end(), cubemap);
    if (it == mEnvironmentMaps.end()) {
      mEnvironmentMaps.push_back(cubemap);
      it = std::prev(mEnvironmentMaps.end());
    }
    row.environmentMap = static_cast<int>(std::distance(mEnvironmentMaps.begin(), it)) + 1;
  }
}

void BatchRTRenderer::refreshLightRows() {
  // lights concatenate in environment order; each environment names its [begin, end) range
  mLights.clear();
  auto textureIndex = [&](std::shared_ptr<resource::SVTexture> const &texture) {
    texture->loadAsync().get();
    texture->uploadToDevice(*mContext);
    auto it = std::find_if(mTextures.begin(), mTextures.end(), [&](auto const &entry) {
      return entry.first == texture->getImageView(*mContext);
    });
    if (it == mTextures.end()) {
      mTextures.emplace_back(texture->getImageView(*mContext), texture->getSampler(*mContext));
      it = std::prev(mTextures.end());
    }
    return static_cast<int>(std::distance(mTextures.begin(), it));
  };
  for (size_t index = 0; index < mEnvironments.size(); ++index) {
    BatchEnvironmentRow &row = mEnvironmentRows.at(index);
    row.pointLightBegin = static_cast<int>(mLights.point.size());
    row.directionalLightBegin = static_cast<int>(mLights.directional.size());
    row.spotLightBegin = static_cast<int>(mLights.spot.size());
    row.parallelogramLightBegin = static_cast<int>(mLights.parallelogram.size());
    mLights.append(*mEnvironments.at(index), textureIndex);
    row.pointLightEnd = static_cast<int>(mLights.point.size());
    row.directionalLightEnd = static_cast<int>(mLights.directional.size());
    row.spotLightEnd = static_cast<int>(mLights.spot.size());
    row.parallelogramLightEnd = static_cast<int>(mLights.parallelogram.size());
  }
}

void BatchRTRenderer::uploadSceneLightState() {
  checkInitialized();
  auto previousMaps = mEnvironmentMaps;
  refreshEnvironmentRows();
  refreshLightRows();
  if (mEnvironmentMaps != previousMaps) {
    // the caller waits for the previous frame, and every frame re-records its command buffer
    auto const &bindings = mShaderPack->getSceneDescription().bindings;
    for (auto const &[bid, binding] : bindings) {
      if (binding.name == "environmentMaps") {
        auto infos = environmentMapInfos();
        vk::WriteDescriptorSet write(mSceneSet.get(), bid, 0, static_cast<uint32_t>(infos.size()),
                                     vk::DescriptorType::eCombinedImageSampler, infos.data());
        mContext->getDevice().updateDescriptorSets(write, {});
      }
    }
  }

  mPointLightBuffer->upload(mLights.point.data(),
                            mLights.point.size() * sizeof(scene::GpuPointLight));
  mDirectionalLightBuffer->upload(mLights.directional.data(),
                                  mLights.directional.size() * sizeof(scene::GpuDirectionalLight));
  mSpotLightBuffer->upload(mLights.spot.data(), mLights.spot.size() * sizeof(scene::GpuSpotLight));
  mParallelogramLightBuffer->upload(mLights.parallelogram.data(),
                                    mLights.parallelogram.size() *
                                        sizeof(scene::GpuParallelogramLight));
  mEnvironmentBuffer->upload(mEnvironmentRows.data(),
                             mEnvironmentRows.size() * sizeof(BatchEnvironmentRow));
}

void BatchRTRenderer::updatePushConstant() {
  auto layout = mShaderPack->getPushConstantLayout();
  if (!layout) {
    return;
  }
  std::fill(mPushConstantBuffer.begin(), mPushConstantBuffer.end(), 0);
  auto set = [&](std::string const &name, void const *value, uint32_t size) {
    auto it = layout->elements.find(name);
    if (it == layout->elements.end() || it->second.size < size) {
      return;
    }
    std::memcpy(mPushConstantBuffer.data() + it->second.offset, value, size);
  };
  set("cameraCount", &mCameraCount, sizeof(int));
  set("width", &mWidth, sizeof(int));
  set("height", &mHeight, sizeof(int));
  for (auto const &[name, value] : mProperties) {
    auto it = layout->elements.find(name);
    if (it == layout->elements.end()) {
      continue;
    }
    if (std::holds_alternative<int>(value) && it->second.dtype == DataType::INT()) {
      int v = std::get<int>(value);
      std::memcpy(mPushConstantBuffer.data() + it->second.offset, &v, sizeof(int));
    } else if (std::holds_alternative<float>(value) && it->second.dtype == DataType::FLOAT()) {
      float v = std::get<float>(value);
      std::memcpy(mPushConstantBuffer.data() + it->second.offset, &v, sizeof(float));
    }
  }
}

void BatchRTRenderer::render(vk::Semaphore wait, uint64_t waitValue, vk::Semaphore signal,
                             uint64_t signalValue) {
  checkInitialized();

  mCommandBuffer->reset();
  mCommandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

  // Order the CUDA-written instance transforms and the light uploads of this frame before the
  // refits that read them. Every refit of this submit shares the single barrier.
  {
    vk::MemoryBarrier barrier(vk::AccessFlagBits::eMemoryWrite,
                              vk::AccessFlagBits::eAccelerationStructureWriteKHR |
                                  vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                  vk::AccessFlagBits::eShaderRead);
    mCommandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                   vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR |
                                       vk::PipelineStageFlagBits::eComputeShader,
                                   {}, barrier, {}, {});
  }

  // deformable and point-set BLAS refits first, then every environment's TLAS refit in one
  // build command: the refits are independent, so no barrier separates them
  bool blasRefits = false;
  std::vector<core::TLAS *> tlases;
  tlases.reserve(mEnvironments.size());
  for (auto const &scene : mEnvironments) {
    blasRefits |= scene->recordBLASRefits(mCommandBuffer.get());
    tlases.push_back(scene->getTLAS());
  }
  if (blasRefits) {
    vk::MemoryBarrier barrier(vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                              vk::AccessFlagBits::eAccelerationStructureReadKHR);
    mCommandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                                   vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {},
                                   barrier, {}, {});
  }
  core::TLAS::RecordUpdates(mCommandBuffer.get(), tlases);

  {
    vk::MemoryBarrier barrier(vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                              vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                  vk::AccessFlagBits::eShaderRead);
    mCommandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                                   vk::PipelineStageFlagBits::eComputeShader, {}, barrier, {},
                                   {});
  }

  mCommandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, mPipeline.get());
  auto layout = mShaderPack->getPushConstantLayout();

  // one dispatch over every camera row; each row selects its environment's TLAS
  updatePushConstant();
  std::vector<vk::DescriptorSet> sets(mShaderPack->getResources().size());
  sets.at(mShaderPack->getOutputSetId()) = mOutputSet.get();
  sets.at(mShaderPack->getSceneSetId()) = mSceneSet.get();
  mCommandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, mPipelineLayout.get(), 0,
                                    sets, {});
  if (layout) {
    mCommandBuffer->pushConstants(mPipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0,
                                 layout->size, mPushConstantBuffer.data());
  }
  mCommandBuffer->dispatch((mWidth + 7) / 8, (mHeight + 7) / 8,
                           static_cast<uint32_t>(mCameraCount));

  mCommandBuffer->end();

  // same argument shape as BatchedCamera::takePicture (a stage mask is a PipelineStageFlags)
  vk::PipelineStageFlags stages = vk::PipelineStageFlagBits::eAllCommands;
  mContext->getQueue().submit(mCommandBuffer.get(), wait, stages, waitValue, signal, signalValue,
                             {});
}

void BatchRTRenderer::waitIdle() { mContext->waitIdle(); }

void BatchRTRenderer::checkInitialized() const {
  if (!mInitialized) {
    throw std::runtime_error("the batched ray tracing renderer is not initialized: call "
                             "initialize() first");
  }
}

vk::Format BatchRTRenderer::getPictureFormat(std::string const &name) const {
  auto it = mPictureFormats.find(name);
  if (it == mPictureFormats.end()) {
    throw std::runtime_error("the batched ray tracing renderer does not render the picture \"" +
                             name + "\"");
  }
  return it->second;
}

vk::DeviceSize BatchRTRenderer::getPictureBytes(std::string const &name) const {
  auto it = mPictureFormats.find(name);
  if (it == mPictureFormats.end()) {
    throw std::runtime_error("the batched ray tracing renderer does not render the picture \"" +
                             name + "\"");
  }
  return static_cast<vk::DeviceSize>(mWidth) * mHeight * getFormatSize(it->second);
}

core::Buffer &BatchRTRenderer::getPictureBuffer(std::string const &name) {
  checkInitialized();
  auto it = mPictureBuffers.find(name);
  if (it == mPictureBuffers.end()) {
    throw std::runtime_error("the batched ray tracing renderer does not render the picture \"" +
                             name + "\"");
  }
  return *it->second;
}

void BatchRTRenderer::checkCameraRow(int row) const {
  checkInitialized();
  if (row < 0 || row >= mCameraCount) {
    throw std::runtime_error("batched ray tracing camera row out of range");
  }
}

#ifdef SVULKAN2_CUDA_INTEROP
void *BatchRTRenderer::getCameraRowCudaPtr(int row) {
  checkCameraRow(row);
  return static_cast<uint8_t *>(mCameraBuffer->getCudaPtr()) +
         static_cast<size_t>(row) * mCameraSlotSize;
}
#endif

void BatchRTRenderer::uploadCamera(int row, glm::mat4 const &viewMatrix,
                                   glm::mat4 const &projectionMatrix) {
  checkCameraRow(row);
  // the four matrices are contiguous at the start of the row: one upload
  std::array<glm::mat4, 4> matrices{viewMatrix, projectionMatrix, glm::affineInverse(viewMatrix),
                                    glm::inverse(projectionMatrix)};
  static_assert(offsetof(BatchCameraRow, projectionMatrixInverse) == 3 * sizeof(glm::mat4));
  mCameraBuffer->upload(matrices.data(), sizeof(matrices),
                        static_cast<size_t>(row) * mCameraSlotSize);
}

CustomPropertyType BatchRTRenderer::getCustomPropertyType(std::string const &name) const {
  auto layout = mShaderPack->getPushConstantLayout();
  if (!layout) {
    return CustomPropertyType::eNone;
  }
  if (std::find(BATCH_RESERVED_PROPERTIES.begin(), BATCH_RESERVED_PROPERTIES.end(), name) !=
      BATCH_RESERVED_PROPERTIES.end()) {
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
  return CustomPropertyType::eNone;
}

void BatchRTRenderer::setCustomProperty(std::string const &name, int value) {
  if (getCustomPropertyType(name) != CustomPropertyType::eInt) {
    throw std::runtime_error("batched ray tracing property \"" + name +
                             "\" is not an int member of the push constant");
  }
  mProperties[name] = value;
}

void BatchRTRenderer::setCustomProperty(std::string const &name, float value) {
  if (getCustomPropertyType(name) != CustomPropertyType::eFloat) {
    throw std::runtime_error("batched ray tracing property \"" + name +
                             "\" is not a float member of the push constant");
  }
  mProperties[name] = value;
}

std::string BatchRTRenderer::summary() const {
  std::ostringstream stream;
  stream << "BatchRTRenderer(" << getWidth() << "x" << getHeight() << ", " << mCameraCount
         << " cameras, " << mEnvironments.size() << " environments, pictures";
  for (auto const &name : mPictureNames) {
    stream << " " << name << ":" << vk::to_string(mPictureFormats.at(name));
  }
  stream << ")";
  return stream.str();
}

} // namespace renderer
} // namespace svulkan2
