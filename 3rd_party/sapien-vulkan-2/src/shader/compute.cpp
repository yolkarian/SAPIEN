#include "svulkan2/shader/compute.h"
#include "svulkan2/core/context.h"
#include "svulkan2/shader/brdf_lut.h"
#include "svulkan2/shader/latlong_to_cubemap.h"
#include "svulkan2/shader/glsl_compiler.h"
#include <algorithm>
#include <string>

namespace svulkan2 {
namespace shader {

std::unique_ptr<core::Image> generateBRDFLUT(core::Context &context, uint32_t size) {
  auto image = std::make_unique<core::Image>(
      context, vk::ImageType::e2D, vk::Extent3D{size, size, 1}, vk::Format::eR16G16Sfloat,
      vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc |
          vk::ImageUsageFlagBits::eStorage,
      VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);

  auto device = context.getDevice();

  vk::DescriptorSetLayoutBinding binding(0, vk::DescriptorType::eStorageImage, 1,
                                         vk::ShaderStageFlagBits::eCompute);
  auto descriptorSetLayout =
      device.createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo({}, binding));
  auto pipelineLayout = device.createPipelineLayoutUnique(
      vk::PipelineLayoutCreateInfo({}, descriptorSetLayout.get()));

  auto descriptorSet = context.getDescriptorPool().allocateSet(descriptorSetLayout.get());

  vk::ImageViewCreateInfo viewInfo(
      {}, image->getVulkanImage(), vk::ImageViewType::e2D, image->getFormat(),
      vk::ComponentMapping(vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
                           vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA),
      vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
  auto imageView = device.createImageViewUnique(viewInfo);

  auto imageInfo = vk::DescriptorImageInfo({}, imageView.get(), vk::ImageLayout::eGeneral);
  vk::WriteDescriptorSet write(descriptorSet.get(), 0, 0, vk::DescriptorType::eStorageImage,
                               imageInfo);
  device.updateDescriptorSets(write, {});

  std::vector<uint32_t> code(reinterpret_cast<const uint32_t *>(brdf_lut_code),
                             reinterpret_cast<const uint32_t *>(brdf_lut_code) +
                                 brdf_lut_size / 4);

  auto shaderModule = device.createShaderModuleUnique(vk::ShaderModuleCreateInfo({}, code));
  auto shaderStageInfo = vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eCompute,
                                                           shaderModule.get(), "main");
  vk::ComputePipelineCreateInfo computePipelineCreateInfo({}, shaderStageInfo,
                                                          pipelineLayout.get());
  auto pipelinCache = device.createPipelineCacheUnique({});
  auto pipeline =
      device.createComputePipelineUnique(pipelinCache.get(), computePipelineCreateInfo).value;

  auto pool = context.createCommandPool();
  auto cb = pool->allocateCommandBuffer();
  cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  image->transitionLayout(cb.get(), vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, {},
                          vk::AccessFlagBits::eShaderWrite, vk::PipelineStageFlagBits::eTopOfPipe,
                          vk::PipelineStageFlagBits::eComputeShader);

  cb->bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.get());
  cb->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0,
                         descriptorSet.get(), {});
  cb->dispatch(size, size, 1);
  image->transitionLayout(
      cb.get(), vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
      vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eFragmentShader);
  cb->end();
  image->setCurrentLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

  context.getQueue().submitAndWait(cb.get());
  return image;
}

// Split-sum specular prefilter, one dispatch per mip level. `OUTPUT_FORMAT` is replaced by the
// cube's format before compiling. Samples are drawn from the GGX lobe and read from a box-filtered
// mip chain of the source at the level matching each sample's solid angle (filtered importance
// sampling), so HDR sources keep their energy without the sparkle a few very bright texels would
// otherwise leave.
static const char *const PREFILTER_GLSL = R"glsl(#version 450
layout(set = 0, binding = 0) uniform samplerCube EnvMap;
layout(set = 1, binding = 0, OUTPUT_FORMAT) uniform writeonly imageCube OutEnvMap;

layout(push_constant) uniform Constants { float roughness; } constants;

const float PI = 3.1415926535897932384626433832795;

// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
float radicalInverse_VdC(uint bits) {
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 Hammersley(uint i, uint N) { return vec2(float(i) / float(N), radicalInverse_VdC(i)); }

// tangent-space half vector distributed as GGX (pdf = D * cosTheta)
vec3 ImportanceSampleGGX(vec2 Xi, float a2) {
  float Phi = 2 * PI * Xi.x;
  float CosTheta = sqrt((1 - Xi.y) / (1 + (a2 - 1) * Xi.y));
  float SinTheta = sqrt(1 - CosTheta * CosTheta);
  return vec3(SinTheta * cos(Phi), SinTheta * sin(Phi), CosTheta);
}

vec3 PrefilterEnvMap(float Roughness, vec3 R) {
  vec3 N = R;
  vec3 V = R;
  vec3 UpVector = abs(N.z) < 0.999 ? vec3(0, 0, 1) : vec3(1, 0, 0);
  vec3 TangentX = normalize(cross(UpVector, N));
  vec3 TangentY = cross(N, TangentX);

  float a = Roughness * Roughness;
  float a2 = a * a;
  float resolution = float(textureSize(EnvMap, 0).x);
  float saTexel = 4.0 * PI / (6.0 * resolution * resolution);
  float maxLod = float(textureQueryLevels(EnvMap) - 1);

  vec3 PrefilteredColor = vec3(0);
  float TotalWeight = 0;

  const uint NumSamples = 1024;
  for (uint i = 0; i < NumSamples; ++i) {
    vec3 Ht = ImportanceSampleGGX(Hammersley(i, NumSamples), a2);
    vec3 H = TangentX * Ht.x + TangentY * Ht.y + N * Ht.z;
    vec3 L = 2 * dot(V, H) * H - V;
    float dotNL = dot(N, L);
    if (dotNL > 0) {
      // with N == V the reflected-direction pdf D * NoH / (4 * VoH) reduces to D / 4
      float d = Ht.z * Ht.z * (a2 - 1) + 1;
      float pdf = a2 / (PI * d * d) * 0.25;
      float saSample = 1.0 / (float(NumSamples) * pdf + 1e-6);
      float lod = clamp(0.5 * log2(saSample / saTexel) + 1.0, 0.0, maxLod);
      PrefilteredColor += textureLod(EnvMap, L, lod).rgb * dotNL;
      TotalWeight += dotNL;
    }
  }
  return PrefilteredColor / max(TotalWeight, 1e-6);
}

// http://alinloghin.com/articles/compute_ibl.html
vec3 cubeCoordToWorld(ivec3 cubeCoord, vec2 cubemapSize) {
  vec2 texCoord = (vec2(cubeCoord.xy) + 0.5) / cubemapSize;
  texCoord = texCoord * 2.0 - 1.0; // -1..1
  switch (cubeCoord.z) {
  case 0: return vec3(1.0, -texCoord.yx);             // posx
  case 1: return vec3(-1.0, -texCoord.y, texCoord.x); // negx
  case 2: return vec3(texCoord.x, 1.0, texCoord.y);   // posy
  case 3: return vec3(texCoord.x, -1.0, -texCoord.y); // negy
  case 4: return vec3(texCoord.x, -texCoord.y, 1.0);  // posz
  case 5: return vec3(-texCoord.xy, -1.0);            // negz
  }
  return vec3(0.0);
}

void main() {
  ivec2 s = imageSize(OutEnvMap);
  ivec3 cubeCoord = ivec3(gl_GlobalInvocationID);
  if (cubeCoord.x >= s.x || cubeCoord.y >= s.y) {
    return;
  }
  vec3 R = normalize(cubeCoordToWorld(cubeCoord, vec2(s)));
  imageStore(OutEnvMap, cubeCoord, vec4(PrefilterEnvMap(constants.roughness, R), 1));
}
)glsl";

static char const *storageFormatQualifier(vk::Format format) {
  switch (format) {
  case vk::Format::eR8G8B8A8Unorm:
  case vk::Format::eR8G8B8A8Srgb:
    return "rgba8";
  case vk::Format::eR16G16B16A16Sfloat:
    return "rgba16f";
  case vk::Format::eR32G32B32A32Sfloat:
    return "rgba32f";
  default:
    throw std::runtime_error("failed to prefilter cube map: unsupported image format " +
                             vk::to_string(format));
  }
}

void prefilterCubemap(core::Context &context, core::Image &image) {
  uint32_t levels = image.getMipLevels();
  if (levels == 1) {
    return;
  }

  auto device = context.getDevice();

  auto binding0 = vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eCombinedImageSampler, 1,
                                                 vk::ShaderStageFlagBits::eCompute);
  auto binding1 = vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageImage, 1,
                                                 vk::ShaderStageFlagBits::eCompute);
  auto descriptorSetLayout0 =
      device.createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo({}, binding0));
  auto descriptorSetLayout1 =
      device.createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo({}, binding1));
  std::vector<vk::DescriptorSetLayout> layouts = {descriptorSetLayout0.get(),
                                                  descriptorSetLayout1.get()};
  auto pushConstantRange =
      vk::PushConstantRange(vk::ShaderStageFlagBits::eCompute, 0, sizeof(float));
  auto pipelineLayout = device.createPipelineLayoutUnique(
      vk::PipelineLayoutCreateInfo({}, layouts, pushConstantRange));

  std::string glsl = PREFILTER_GLSL;
  {
    std::string const placeholder = "OUTPUT_FORMAT";
    glsl.replace(glsl.find(placeholder), placeholder.size(),
                 storageFormatQualifier(image.getFormat()));
  }
  auto code = GLSLCompiler::compileToSpirv(vk::ShaderStageFlagBits::eCompute, glsl);
  auto shaderModule = device.createShaderModuleUnique(vk::ShaderModuleCreateInfo({}, code));
  auto shaderStageInfo = vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eCompute,
                                                           shaderModule.get(), "main");
  vk::ComputePipelineCreateInfo computePipelineCreateInfo({}, shaderStageInfo,
                                                          pipelineLayout.get());
  auto pipelinCache = device.createPipelineCacheUnique({});
  auto pipeline =
      device.createComputePipelineUnique(pipelinCache.get(), computePipelineCreateInfo).value;

  // The pass reads the whole mip chain of the source while it produces levels >= 1, so it
  // writes into a scratch cube and copies the result back afterwards.
  auto filtered = std::make_unique<core::Image>(
      context, vk::ImageType::e2D, image.getExtent(), image.getFormat(),
      vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eStorage,
      VMA_MEMORY_USAGE_GPU_ONLY, vk::SampleCountFlagBits::e1, levels, 6,
      vk::ImageTiling::eOptimal, vk::ImageCreateFlagBits::eCubeCompatible);

  auto levelRange = [&](uint32_t baseLevel, uint32_t count) {
    return vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, baseLevel, count, 0, 6);
  };
  auto transition = [&](vk::CommandBuffer cb, vk::Image target, vk::ImageSubresourceRange range,
                        vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                        vk::AccessFlags srcAccess, vk::AccessFlags dstAccess,
                        vk::PipelineStageFlags srcStage, vk::PipelineStageFlags dstStage) {
    cb.pipelineBarrier(srcStage, dstStage, {}, {}, {},
                       vk::ImageMemoryBarrier(srcAccess, dstAccess, oldLayout, newLayout,
                                              VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                              target, range));
  };

  auto pool = context.createCommandPool();
  auto cb = pool->allocateCommandBuffer();

  // 1. box-filter the base level down the source's own chain
  cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  transition(cb.get(), image.getVulkanImage(), levelRange(0, 1),
             vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferDstOptimal,
             vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eTransferWrite,
             vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer);
  transition(cb.get(), image.getVulkanImage(), levelRange(1, levels - 1),
             vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, {},
             vk::AccessFlagBits::eTransferWrite, vk::PipelineStageFlagBits::eTopOfPipe,
             vk::PipelineStageFlagBits::eTransfer);
  for (uint32_t layer = 0; layer < 6; ++layer) {
    image.generateMipmaps(cb.get(), layer);
  }
  transition(cb.get(), filtered->getVulkanImage(), levelRange(0, levels),
             vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, {},
             vk::AccessFlagBits::eShaderWrite, vk::PipelineStageFlagBits::eTopOfPipe,
             vk::PipelineStageFlagBits::eComputeShader);
  cb->end();
  context.getQueue().submitAndWait(cb.get());

  // source: every level, trilinear
  auto descriptorSet = context.getDescriptorPool().allocateSet(descriptorSetLayout0.get());
  auto imageView = device.createImageViewUnique(vk::ImageViewCreateInfo(
      {}, image.getVulkanImage(), vk::ImageViewType::eCube, image.getFormat(),
      vk::ComponentSwizzle::eIdentity, levelRange(0, levels)));
  auto sampler = device.createSamplerUnique(vk::SamplerCreateInfo(
      {}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
      vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
      vk::SamplerAddressMode::eClampToEdge, 0.f, false, 0.f, false, vk::CompareOp::eNever, 0.f,
      static_cast<float>(levels), vk::BorderColor::eFloatOpaqueWhite));
  auto imageInfo = vk::DescriptorImageInfo(sampler.get(), imageView.get(),
                                           vk::ImageLayout::eShaderReadOnlyOptimal);
  device.updateDescriptorSets(
      vk::WriteDescriptorSet(descriptorSet.get(), 0, 0,
                             vk::DescriptorType::eCombinedImageSampler, imageInfo),
      {});

  std::vector<vk::UniqueImageView> levelViews;
  std::vector<core::DescriptorSet> levelSets;
  for (uint32_t level = 1; level < levels; ++level) {
    levelViews.push_back(device.createImageViewUnique(vk::ImageViewCreateInfo(
        {}, filtered->getVulkanImage(), vk::ImageViewType::eCube, image.getFormat(),
        vk::ComponentSwizzle::eIdentity, levelRange(level, 1))));
    levelSets.push_back(context.getDescriptorPool().allocateSet(descriptorSetLayout1.get()));
    auto levelInfo =
        vk::DescriptorImageInfo({}, levelViews.back().get(), vk::ImageLayout::eGeneral);
    device.updateDescriptorSets(vk::WriteDescriptorSet(levelSets.back().get(), 0, 0,
                                                       vk::DescriptorType::eStorageImage,
                                                       levelInfo),
                                {});
  }

  // 2. prefilter each level; one submit per level, a long compute pass can lose the device
  uint32_t size = image.getExtent().width;
  for (uint32_t level = 1; level < levels; ++level) {
    size = std::max(size / 2, 1u);
    float roughness = static_cast<float>(level) / (levels - 1);
    cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cb->bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.get());
    cb->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0,
                           descriptorSet.get(), {});
    cb->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(float),
                      &roughness);
    cb->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 1,
                           levelSets[level - 1].get(), {});
    cb->dispatch(size, size, 6);
    cb->end();
    context.getQueue().submitAndWait(cb.get());
  }

  // 3. replace the box-filtered levels with the prefiltered ones
  cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  transition(cb.get(), filtered->getVulkanImage(), levelRange(1, levels - 1),
             vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal,
             vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead,
             vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer);
  transition(cb.get(), image.getVulkanImage(), levelRange(1, levels - 1),
             vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferDstOptimal,
             vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eTransferWrite,
             vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer);
  size = image.getExtent().width;
  for (uint32_t level = 1; level < levels; ++level) {
    size = std::max(size / 2, 1u);
    vk::ImageSubresourceLayers layers(vk::ImageAspectFlagBits::eColor, level, 0, 6);
    cb->copyImage(filtered->getVulkanImage(), vk::ImageLayout::eTransferSrcOptimal,
                  image.getVulkanImage(), vk::ImageLayout::eTransferDstOptimal,
                  vk::ImageCopy(layers, {}, layers, {}, vk::Extent3D{size, size, 1}));
  }
  transition(cb.get(), image.getVulkanImage(), levelRange(1, levels - 1),
             vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
             vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
             vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands);
  cb->end();
  context.getQueue().submitAndWait(cb.get());

  image.setCurrentLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
}

std::unique_ptr<core::Image> latlongToCube(core::Context &context, core::Image &latlong,
                                           int mipLevels) {
  uint32_t width = latlong.getExtent().width;
  uint32_t height = latlong.getExtent().height;
  if (width != height * 2) {
    throw std::runtime_error("invalid latlong map");
  }

  auto device = context.getDevice();

  auto cubemap = std::make_unique<core::Image>(
      context, vk::ImageType::e2D, vk::Extent3D{height, height, 1}, latlong.getFormat(),
      vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
          vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
      VMA_MEMORY_USAGE_GPU_ONLY, vk::SampleCountFlagBits::e1, mipLevels, 6,
      vk::ImageTiling::eOptimal, vk::ImageCreateFlagBits::eCubeCompatible);

  auto binding0 = vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eCombinedImageSampler, 1,
                                                 vk::ShaderStageFlagBits::eCompute);
  auto binding1 = vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageImage, 1,
                                                 vk::ShaderStageFlagBits::eCompute);
  auto descriptorSetLayout0 =
      device.createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo({}, binding0));
  auto descriptorSetLayout1 =
      device.createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo({}, binding1));
  std::vector<vk::DescriptorSetLayout> layouts = {descriptorSetLayout0.get(),
                                                  descriptorSetLayout1.get()};
  auto pipelineLayout =
      device.createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo({}, layouts));

  // 2D image
  auto descriptorSet = context.getDescriptorPool().allocateSet(descriptorSetLayout0.get());
  vk::ImageViewCreateInfo viewInfo(
      {}, latlong.getVulkanImage(), vk::ImageViewType::e2D, latlong.getFormat(),
      vk::ComponentSwizzle::eIdentity,
      vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
  auto imageView = device.createImageViewUnique(viewInfo);
  auto sampler = device.createSamplerUnique(vk::SamplerCreateInfo(
      {}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eNearest,
      vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode::eRepeat, 0.f, false, 0.f, false, vk::CompareOp::eNever, 0.f, 0.f,
      vk::BorderColor::eFloatOpaqueWhite));
  auto imageInfo = vk::DescriptorImageInfo(sampler.get(), imageView.get(),
                                           vk::ImageLayout::eShaderReadOnlyOptimal);
  vk::WriteDescriptorSet write(descriptorSet.get(), 0, 0,
                               vk::DescriptorType::eCombinedImageSampler, imageInfo);
  device.updateDescriptorSets(write, {});

  std::vector<uint32_t> code(reinterpret_cast<const uint32_t *>(latlong_to_cubemap_code),
                             reinterpret_cast<const uint32_t *>(latlong_to_cubemap_code) +
                                 latlong_to_cubemap_size / 4);
  auto shaderModule = device.createShaderModuleUnique(vk::ShaderModuleCreateInfo({}, code));
  auto shaderStageInfo = vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eCompute,
                                                           shaderModule.get(), "main");
  vk::ComputePipelineCreateInfo computePipelineCreateInfo({}, shaderStageInfo,
                                                          pipelineLayout.get());
  auto pipelinCache = device.createPipelineCacheUnique({});
  auto pipeline =
      device.createComputePipelineUnique(pipelinCache.get(), computePipelineCreateInfo).value;

  // cube image
  vk::ImageViewCreateInfo cubeBaseViewInfo(
      {}, cubemap->getVulkanImage(), vk::ImageViewType::eCube, cubemap->getFormat(),
      vk::ComponentSwizzle::eIdentity,
      vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6));
  auto cubeView = device.createImageViewUnique(cubeBaseViewInfo);
  auto cubeSet = context.getDescriptorPool().allocateSet(descriptorSetLayout1.get());
  auto cubeImageInfo = vk::DescriptorImageInfo({}, cubeView.get(), vk::ImageLayout::eGeneral);
  vk::WriteDescriptorSet write2(cubeSet.get(), 0, 0, vk::DescriptorType::eStorageImage,
                                cubeImageInfo);
  device.updateDescriptorSets(write2, {});

  auto pool = context.createCommandPool();
  auto cb = pool->allocateCommandBuffer();

  cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  cubemap->transitionLayout(cb.get(), vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, {},
                            vk::AccessFlagBits::eShaderWrite,
                            vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eComputeShader);

  cb->bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.get());
  cb->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0,
                         descriptorSet.get(), {});
  cb->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 1, cubeSet.get(),
                         {});
  cb->dispatch(height, height, 6);

  cubemap->transitionLayout(
      cb.get(), vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
      vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader);

  cb->end();
  context.getQueue().submitAndWait(cb.get());

  return cubemap;
}

} // namespace shader
} // namespace svulkan2
