#include "svulkan2/renderer/post_chain.h"
#include "../shader/reflect.h"
#include "svulkan2/core/command_pool.h"
#include "svulkan2/core/context.h"
#include "svulkan2/shader/base_parser.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace svulkan2 {
namespace renderer {

namespace {
char const *const kInput = "inColor";
char const *const kOutput = "outColor";
constexpr vk::Format kColorFormat = vk::Format::eR32G32B32A32Sfloat;

// Neutral values of the built-in post passes, keyed by push-constant member name and seeded
// before any pack.json value is applied. A pack may list a pass without declaring its
// parameters, and the push constants then stay at their zero-filled value: color_grade with
// cdlSlope and cdlPower 0 renders black, and fxaa with threshold 0 smooths every pixel. Members
// whose neutral value is zero are absent.
struct FloatDefault {
  char const *name;
  float value;
};
constexpr FloatDefault kFloatDefaults[] = {
    {"exposure", 1.f},
    {"bloomThreshold", 1.f},
    {"bloomRadius", 4.f},
    {"bloomKnee", .5f},
    {"whiteBalanceTemp", 6500.f},
    {"contrast", 1.f},
    {"saturation", 1.f},
    {"cdlSlope", 1.f},
    {"cdlPower", 1.f},
    {"vignetteSmoothness", .5f},
    {"fxaaEdgeThreshold", .125f},
    {"fxaaSubpix", .75f},
};
struct IntDefault {
  char const *name;
  int value;
};
constexpr IntDefault kIntDefaults[] = {
    {"toneMapper", 2},
};
} // namespace

PostChain::PostChain(std::shared_ptr<core::Context> context,
                     std::vector<std::string> const &files, ShaderDefines const &defines)
    : mContext(std::move(context)) {
  vk::Device device = mContext->getDevice();
  for (auto const &file : files) {
    auto fail = [&](std::string const &message) {
      return std::runtime_error("post-processing pass " + file + ": " + message);
    };
    auto const &code =
        GLSLCompiler::compileGlslFileCached(vk::ShaderStageFlagBits::eCompute, file, defines);
    spirv_cross::Compiler compiler(code);
    auto resources = compiler.get_shader_resources();

    Pass pass;
    pass.file = file;
    if (!resources.uniform_buffers.empty() || !resources.storage_buffers.empty() ||
        !resources.sampled_images.empty()) {
      throw fail("only storage images and one push constant block are supported");
    }
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    for (auto &r : resources.storage_images) {
      if (compiler.get_decoration(r.id, spv::Decoration::DecorationDescriptorSet) != 0) {
        throw fail("image \"" + r.name + "\" must be in set 0");
      }
      auto const &type = compiler.get_type(r.type_id);
      if (!type.array.empty()) {
        throw fail("image \"" + r.name + "\" must not be an array");
      }
      if ((r.name == kInput || r.name == kOutput) &&
          get_image_format(type.image.format) != kColorFormat) {
        throw fail("image \"" + r.name + "\" must be declared rgba32f");
      }
      uint32_t binding = compiler.get_decoration(r.id, spv::Decoration::DecorationBinding);
      pass.images[binding] = r.name;
      bindings.push_back({binding, vk::DescriptorType::eStorageImage, 1,
                          vk::ShaderStageFlagBits::eCompute});
    }
    auto declares = [&](char const *name) {
      for (auto const &[binding, image] : pass.images) {
        if (image == name) {
          return true;
        }
      }
      return false;
    };
    if (!declares(kInput) || !declares(kOutput)) {
      throw fail("a pass must declare the storage images inColor and outColor");
    }

    std::vector<vk::PushConstantRange> ranges;
    for (auto &r : resources.push_constant_buffers) {
      pass.parameters = shader::parseBuffer(compiler, compiler.get_type(r.base_type_id));
      for (auto const &[name, element] : pass.parameters->elements) {
        if (!element.array.empty() ||
            (element.dtype != DataType::INT() && element.dtype != DataType::FLOAT())) {
          throw fail("parameter \"" + name + "\" must be a scalar int or float");
        }
        auto [it, inserted] = mProperties.try_emplace(name, Property{element.dtype});
        if (!inserted && it->second.dtype != element.dtype) {
          throw fail("parameter \"" + name + "\" has another type in an earlier pass");
        }
        if (inserted) {
          // the neutral value of a built-in pass parameter; a pack.json value overrides it
          for (auto const &d : kFloatDefaults) {
            if (name == d.name && element.dtype == DataType::FLOAT()) {
              std::memcpy(it->second.value, &d.value, sizeof(d.value));
            }
          }
          for (auto const &d : kIntDefaults) {
            if (name == d.name && element.dtype == DataType::INT()) {
              std::memcpy(it->second.value, &d.value, sizeof(d.value));
            }
          }
        }
      }
      ranges.push_back({vk::ShaderStageFlagBits::eCompute, 0, pass.parameters->size});
    }

    for (uint32_t axis = 0; axis < 2; ++axis) {
      pass.localSize[axis] = std::max(
          1u, compiler.get_execution_mode_argument(spv::ExecutionModeLocalSize, axis));
    }

    pass.setLayout =
        device.createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo({}, bindings));
    pass.pipelineLayout = device.createPipelineLayoutUnique(
        vk::PipelineLayoutCreateInfo({}, pass.setLayout.get(), ranges));
    auto module = device.createShaderModuleUnique(vk::ShaderModuleCreateInfo({}, code));
    vk::PipelineShaderStageCreateInfo stage({}, vk::ShaderStageFlagBits::eCompute, module.get(),
                                            "main");
    pass.pipeline = device
                        .createComputePipelineUnique(
                            {}, vk::ComputePipelineCreateInfo({}, stage, pass.pipelineLayout.get()))
                        .value;
    mPasses.push_back(std::move(pass));
  }
}

PostChain::~PostChain() {
  // sets go back to the pool before it is destroyed
  for (auto &pass : mPasses) {
    pass.set.reset();
  }
}

CustomPropertyType PostChain::getPropertyType(std::string const &name) const {
  auto it = mProperties.find(name);
  if (it == mProperties.end()) {
    return CustomPropertyType::eNone;
  }
  return it->second.dtype == DataType::INT() ? CustomPropertyType::eInt
                                             : CustomPropertyType::eFloat;
}

template <typename T>
void PostChain::store(std::string const &name, DataType const &dtype, T value) {
  auto it = mProperties.find(name);
  if (it == mProperties.end() || it->second.dtype != dtype) {
    throw std::runtime_error("invalid post-processing property " + name);
  }
  std::memcpy(it->second.value, &value, sizeof(value));
}

void PostChain::setProperty(std::string const &name, int value) {
  store(name, DataType::INT(), value);
}
void PostChain::setProperty(std::string const &name, float value) {
  store(name, DataType::FLOAT(), value);
}

int PostChain::getPropertyInt(std::string const &name) const {
  auto it = mProperties.find(name);
  if (it == mProperties.end() || it->second.dtype != DataType::INT()) {
    throw std::runtime_error("invalid post-processing property " + name);
  }
  int value;
  std::memcpy(&value, it->second.value, sizeof(value));
  return value;
}

float PostChain::getPropertyFloat(std::string const &name) const {
  auto it = mProperties.find(name);
  if (it == mProperties.end() || it->second.dtype != DataType::FLOAT()) {
    throw std::runtime_error("invalid post-processing property " + name);
  }
  float value;
  std::memcpy(&value, it->second.value, sizeof(value));
  return value;
}

void PostChain::prepare(uint32_t width, uint32_t height, vk::ImageView source,
                        vk::ImageView destination,
                        std::function<vk::ImageView(std::string const &)> const &lookup) {
  if (mPasses.empty()) {
    return;
  }

  // the images between passes
  size_t const scratchCount = std::min<size_t>(mPasses.size() - 1, 2);
  if (width != mWidth || height != mHeight || mScratch.size() != scratchCount) {
    mScratch.clear();
    auto pool = mContext->createCommandPool();
    auto commandBuffer = pool->allocateCommandBuffer();
    commandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    for (size_t i = 0; i < scratchCount; ++i) {
      auto image = std::make_shared<resource::SVStorageImage>(
          mContext, "PostColor" + std::to_string(i), width, height, kColorFormat);
      image->createDeviceResources();
      image->getImage().transitionLayout(
          commandBuffer.get(), vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, {},
          vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
          vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader);
      mScratch.push_back(std::move(image));
    }
    commandBuffer->end();
    mContext->getQueue().submitAndWait(commandBuffer.get());
    mWidth = width;
    mHeight = height;
  }

  uint32_t imageCount = 0;
  for (auto &pass : mPasses) {
    pass.set.reset();
    imageCount += pass.images.size();
  }
  mPool = std::make_unique<core::DynamicDescriptorPool>(
      mContext->getDevice2(),
      std::vector<vk::DescriptorPoolSize>{{vk::DescriptorType::eStorageImage, imageCount}});

  for (size_t index = 0; index < mPasses.size(); ++index) {
    auto &pass = mPasses[index];
    pass.set = mPool->allocateSet(pass.setLayout.get());
    std::vector<vk::DescriptorImageInfo> infos;
    infos.reserve(pass.images.size());
    std::vector<vk::WriteDescriptorSet> writes;
    for (auto const &[binding, name] : pass.images) {
      vk::ImageView view;
      if (name == kInput) {
        view = index == 0 ? source : mScratch[(index - 1) % 2]->getImageView();
      } else if (name == kOutput) {
        view = index + 1 == mPasses.size() ? destination : mScratch[index % 2]->getImageView();
      } else {
        view = lookup(name);
      }
      infos.push_back({VK_NULL_HANDLE, view, vk::ImageLayout::eGeneral});
      writes.push_back(vk::WriteDescriptorSet(pass.set.get(), binding, 0,
                                              vk::DescriptorType::eStorageImage, infos.back()));
    }
    mContext->getDevice().updateDescriptorSets(writes, {});
  }
}

void PostChain::record(vk::CommandBuffer commandBuffer) const {
  std::vector<char> parameters;
  for (size_t index = 0; index < mPasses.size(); ++index) {
    auto const &pass = mPasses[index];
    if (pass.parameters) {
      parameters.assign(pass.parameters->size, 0);
      for (auto const &[name, element] : pass.parameters->elements) {
        std::memcpy(parameters.data() + element.offset, mProperties.at(name).value,
                    sizeof(Property::value));
      }
      commandBuffer.pushConstants(pass.pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0,
                                  static_cast<uint32_t>(parameters.size()), parameters.data());
    }
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pass.pipeline.get());
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pass.pipelineLayout.get(), 0,
                                     pass.set.get(), {});
    commandBuffer.dispatch((mWidth + pass.localSize[0] - 1) / pass.localSize[0],
                           (mHeight + pass.localSize[1] - 1) / pass.localSize[1], 1);
    if (index + 1 < mPasses.size()) {
      vk::MemoryBarrier barrier(vk::AccessFlagBits::eShaderWrite,
                                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
      commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                    vk::PipelineStageFlagBits::eComputeShader, {}, barrier, {},
                                    {});
    }
  }
}

} // namespace renderer
} // namespace svulkan2
