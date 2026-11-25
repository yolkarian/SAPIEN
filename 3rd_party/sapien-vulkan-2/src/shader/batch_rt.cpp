#include "svulkan2/shader/batch_rt.h"
#include "../common/logger.h"
#include "svulkan2/common/fs.h"
#include "reflect.h"
#include <algorithm>
#include <cctype>
#include <sstream>

namespace svulkan2 {
namespace shader {

vk::Format getBatchOutputFormat(DataType const &dtype) {
  if (dtype.bytes != 4) {
    throw std::runtime_error(
        "batch shader output buffer element must be vec4, uvec4, uint or float");
  }
  if (dtype.kind == TypeKind::eFloat) {
    if (dtype.shape == 4) {
      return vk::Format::eR32G32B32A32Sfloat;
    }
    if (dtype.shape == 1) {
      return vk::Format::eR32Sfloat;
    }
  }
  if (dtype.kind == TypeKind::eUint) {
    if (dtype.shape == 4) {
      return vk::Format::eR32G32B32A32Uint;
    }
    if (dtype.shape == 1) {
      return vk::Format::eR8G8B8A8Unorm;
    }
  }
  throw std::runtime_error("batch shader output buffer element must be vec4 (rgba32f), float "
                           "(r32f), uvec4 (rgba32ui) or uint (rgba8)");
}

std::string getBatchFormatDefineName(vk::Format format) {
  std::string name = vk::to_string(format);
  std::string result;
  for (char c : name) {
    result.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  return result;
}

BatchShaderPack::BatchShaderPack(std::string const &shaderDir, ShaderDefines const &defines)
    : mShaderDir(shaderDir) {
  fs::path path(shaderDir);
  auto shaderFile = path / "batch.comp";
  if (!fs::exists(shaderFile)) {
    throw std::runtime_error("batched ray tracing shader directory must contain batch.comp: " +
                             shaderDir);
  }
  logger::info("Compiling: " + shaderFile.string());
  mSPVCode = GLSLCompiler::compileGlslFileCached(vk::ShaderStageFlagBits::eCompute,
                                                shaderFile, defines);
  logger::info("Compiled: " + shaderFile.string());
  reflectSPV();
}

void BatchShaderPack::reflectSPV() {
  mResources.clear();

  spirv_cross::Compiler compiler(mSPVCode);
  auto resources = compiler.get_shader_resources();

  auto setOf = [&compiler](spirv_cross::Resource const &r) {
    return compiler.get_decoration(r.id, spv::Decoration::DecorationDescriptorSet);
  };
  auto bindingOf = [&compiler](spirv_cross::Resource const &r) {
    return compiler.get_decoration(r.id, spv::Decoration::DecorationBinding);
  };

  for (auto &r : resources.uniform_buffers) {
    auto &set = mResources[setOf(r)];
    if (set.bindings.contains(bindingOf(r))) {
      throw std::runtime_error("batch shader: duplicated set " + std::to_string(setOf(r)) +
                               " binding " + std::to_string(bindingOf(r)));
    }
    set.buffers.push_back(parseBuffer(compiler, r));
    set.bindings[bindingOf(r)] = {.name = r.name,
                                  .type = vk::DescriptorType::eUniformBuffer,
                                  .dim = 0,
                                  .arraySize = 0,
                                  .arrayIndex =
                                      static_cast<uint32_t>(set.buffers.size() - 1)};
  }
  for (auto &r : resources.storage_buffers) {
    auto &set = mResources[setOf(r)];
    if (set.bindings.contains(bindingOf(r))) {
      throw std::runtime_error("batch shader: duplicated set " + std::to_string(setOf(r)) +
                               " binding " + std::to_string(bindingOf(r)));
    }
    int dim = compiler.get_type(r.type_id).array.size();
    set.buffers.push_back(parseBuffer(compiler, r));
    set.bindings[bindingOf(r)] = {.name = r.name,
                                  .type = vk::DescriptorType::eStorageBuffer,
                                  .dim = dim,
                                  .arraySize = dim == 1 ? compiler.get_type(r.type_id).array[0]
                                                       : 0,
                                  .arrayIndex =
                                      static_cast<uint32_t>(set.buffers.size() - 1)};
  }
  for (auto &r : resources.acceleration_structures) {
    auto &set = mResources[setOf(r)];
    int dim = compiler.get_type(r.type_id).array.size();
    set.bindings[bindingOf(r)] = {.name = r.name,
                                  .type = vk::DescriptorType::eAccelerationStructureKHR,
                                  .dim = dim,
                                  .arraySize = dim == 1 ? compiler.get_type(r.type_id).array[0]
                                                       : 0,
                                  .arrayIndex = 0};
  }
  for (auto &r : resources.sampled_images) {
    auto &set = mResources[setOf(r)];
    int dim = compiler.get_type(r.type_id).array.size();
    set.samplers.push_back(r.name);
    set.bindings[bindingOf(r)] = {.name = r.name,
                                  .type = vk::DescriptorType::eCombinedImageSampler,
                                  .dim = dim,
                                  .arraySize = dim == 1 ? compiler.get_type(r.type_id).array[0]
                                                       : 0,
                                  .arrayIndex =
                                      static_cast<uint32_t>(set.samplers.size() - 1)};
  }
  for (auto &r : resources.storage_images) {
    auto &set = mResources[setOf(r)];
    set.images.push_back(r.name);
    set.bindings[bindingOf(r)] = {.name = r.name,
                                  .type = vk::DescriptorType::eStorageImage,
                                  .dim = 0,
                                  .arraySize = 0,
                                  .arrayIndex =
                                      static_cast<uint32_t>(set.images.size() - 1),
                                  .format = get_image_format(compiler.get_type(r.type_id)
                                                                 .image.format)};
  }
  for (auto &r : resources.push_constant_buffers) {
    auto const &type = compiler.get_type(r.type_id);
    if (type.basetype != spirv_cross::SPIRType::Struct) {
      throw std::runtime_error("batch shader: the push constant must be a struct");
    }
    mPushConstantLayout = parseBuffer(compiler, type);
  }

  // identify the output set (all storage buffers named Out<Name>) and the scene set (the rest)
  int outputSetId = -1;
  for (auto &[sid, set] : mResources) {
    bool onlyOutputs = !set.bindings.empty();
    bool anyStorageImage = false;
    for (auto &[bid, binding] : set.bindings) {
      if (binding.type == vk::DescriptorType::eStorageImage) {
        anyStorageImage = true;
      }
      if (binding.name.starts_with("Out") &&
          binding.type == vk::DescriptorType::eStorageBuffer) {
        continue;
      }
      onlyOutputs = false;
    }
    if (anyStorageImage) {
      throw std::runtime_error("batch shader: storage images are not supported; declare a "
                               "storage buffer named Out<Name> instead");
    }
    if (onlyOutputs && outputSetId >= 0) {
      throw std::runtime_error("batch shader: exactly one descriptor set may hold the Out<Name> "
                               "buffers");
    }
    if (onlyOutputs) {
      outputSetId = static_cast<int>(sid);
    }
  }
  if (outputSetId < 0) {
    throw std::runtime_error("batch shader declares no Out<Name> storage buffer");
  }
  mOutputSetId = static_cast<uint32_t>(outputSetId);

  for (auto &[sid, set] : mResources) {
    if (static_cast<int>(sid) != outputSetId) {
      mSceneSetId = sid;
    }
  }
  if (mResources.size() != 2) {
    throw std::runtime_error("batch shader must use exactly two descriptor sets: outputs in set " +
                             std::to_string(mOutputSetId) + " and the scene in set " +
                             std::to_string(mSceneSetId) + ", got " +
                             std::to_string(mResources.size()));
  }

  mOutputs.clear();
  for (auto &[bid, binding] : mResources.at(mOutputSetId).bindings) {
    if (!binding.name.starts_with("Out")) {
      throw std::runtime_error("batch shader output set binding " + std::to_string(bid) +
                               " must be named Out<Name>, got \"" + binding.name + "\"");
    }
    auto const &layout = mResources.at(mOutputSetId).buffers.at(binding.arrayIndex);
    if (layout->elements.size() != 1) {
      throw std::runtime_error("batch shader output \"" + binding.name +
                               "\" must declare exactly one array member, e.g. \"vec4 c[];\"");
    }
    auto const &element = layout->elements.begin()->second;
    if (element.array.size() != 1 || element.dtype.kind == TypeKind::eStruct) {
      throw std::runtime_error("batch shader output \"" + binding.name +
                               "\" must declare an unsized array");
    }
    mOutputs.push_back({.name = binding.name.substr(3),
                        .blockName = binding.name,
                        .format = getBatchOutputFormat(element.dtype),
                        .binding = bid});
  }
  std::sort(mOutputs.begin(), mOutputs.end(),
            [](Output const &a, Output const &b) { return a.binding < b.binding; });

  logger::info("\n" + summary());
}

DescriptorSetDescription const &BatchShaderPack::getOutputDescription() const {
  return mResources.at(mOutputSetId);
}

DescriptorSetDescription const &BatchShaderPack::getSceneDescription() const {
  return mResources.at(mSceneSetId);
}

vk::Format BatchShaderPack::getOutputFormat(std::string const &name) const {
  for (auto &output : mOutputs) {
    if (output.name == name) {
      return output.format;
    }
  }
  return vk::Format::eUndefined;
}

/** the struct a scene buffer block holds: `buffer Objects { Object o[]; }` -> Object */
static StructDataLayout const &
findElementStruct(std::unordered_map<uint32_t, DescriptorSetDescription> const &resources,
                  std::string const &bindingName) {
  for (auto &[sid, set] : resources) {
    for (auto &[bid, binding] : set.bindings) {
      if (binding.name != bindingName) {
        continue;
      }
      if (binding.type != vk::DescriptorType::eStorageBuffer) {
        throw std::runtime_error("batch shader: \"" + bindingName + "\" must be a storage buffer");
      }
      auto const &block = *set.buffers.at(binding.arrayIndex);
      if (block.elements.size() != 1 ||
          block.elements.begin()->second.dtype.kind != TypeKind::eStruct ||
          !block.elements.begin()->second.member) {
        throw std::runtime_error("batch shader: \"" + bindingName +
                                 "\" must hold a single struct or array of structs");
      }
      return *block.elements.begin()->second.member;
    }
  }
  throw std::runtime_error("batch shader does not declare \"" + bindingName + "\"");
}

StructDataLayout const &BatchShaderPack::getObjectBufferLayout() const {
  return findElementStruct(mResources, "Objects");
}

StructDataLayout const &BatchShaderPack::getGeometryInstanceBufferLayout() const {
  return findElementStruct(mResources, "GeometryInstances");
}

std::string BatchShaderPack::summary() const {
  std::stringstream ss;
  ss << "Batch ray tracing shader pack " << mShaderDir << "\n";
  ss << "Push constant size " << (mPushConstantLayout ? mPushConstantLayout->size : 0) << "\n";
  for (auto &[sid, set] : mResources) {
    ss << "Set " << sid << (static_cast<int>(sid) == static_cast<int>(mOutputSetId)
                                ? " (outputs)"
                                : " (scene)")
       << "\n";
    for (auto &[bid, binding] : set.bindings) {
      ss << "  Binding " << bid << " " << binding.name << " "
         << vk::to_string(binding.type) << "\n";
    }
  }
  for (auto &output : mOutputs) {
    ss << "Output " << output.name << " " << vk::to_string(output.format) << "\n";
  }
  return ss.str();
}

} // namespace shader
} // namespace svulkan2
