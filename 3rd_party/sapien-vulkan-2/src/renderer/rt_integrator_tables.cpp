#include "svulkan2/renderer/rt_integrator_tables.h"
#include "svulkan2/common/format.h"
#include "svulkan2/core/buffer.h"
#include "svulkan2/core/context.h"
#include "svulkan2/core/image.h"
#include "svulkan2/resource/cubemap.h"
#include "svulkan2/resource/material.h"
#include "svulkan2/resource/mesh.h"
#include "svulkan2/resource/model.h"
#include "svulkan2/scene/object.h"
#include "svulkan2/scene/scene.h"

#include "alias_table.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <glm/gtc/packing.hpp>
#include <stdexcept>
#include <vector>

namespace svulkan2 {
namespace renderer {
namespace {

constexpr float kPi = 3.14159265358979323846f;
// The equal-solid-angle environment grid has width x width / 2 cells. Splatting texels into
// cells needs several texels per cell, or cells that receive no texel centre are sampled only
// through the floor below and light sampling becomes very noisy: the width follows the cube's
// face size (6 s^2 texels against width^2 / 2 cells: width <= 1.7 s keeps about 4 per cell).
constexpr int kEnvMinWidth = 16;
constexpr int kEnvMaxWidth = 256;

int envGridWidth(uint32_t faceSize) {
  int width = kEnvMinWidth;
  while (width * 2 <= kEnvMaxWidth && width * 2 <= 1.7f * faceSize) {
    width *= 2;
  }
  return width;
}

// every environment cell keeps at least this fraction of the mean weight, so light sampling
// alone (lightSampling "nee") still reaches every direction the filtered lookup can return
constexpr float kEnvWeightFloor = 1e-3f;

float luminance(glm::vec3 c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }

// Linear value of channel `c` of a texel; formats other than float and normalized integer
// read as 0, which only lowers that texel's sampling weight.
float readChannel(uint8_t const *texel, ComponentFormat format, uint32_t elementSize, uint32_t c) {
  switch (format) {
  case ComponentFormat::eSfloat:
    if (elementSize == 4) {
      float v;
      std::memcpy(&v, texel + 4 * c, 4);
      return v;
    }
    if (elementSize == 2) {
      uint16_t h;
      std::memcpy(&h, texel + 2 * c, 2);
      return glm::unpackHalf1x16(h);
    }
    return 0.f;
  case ComponentFormat::eUnorm:
    if (elementSize == 1) {
      return texel[c] / 255.f;
    }
    if (elementSize == 2) {
      uint16_t v;
      std::memcpy(&v, texel + 2 * c, 2);
      return v / 65535.f;
    }
    return 0.f;
  default:
    return 0.f;
  }
}

// Direction through (u, v) in [-1, 1]^2 of cube face `face`, in the cube's lookup space
// (Vulkan face selection: +X, -X, +Y, -Y, +Z, -Z).
glm::vec3 cubeDirection(uint32_t face, float u, float v) {
  switch (face) {
  case 0:
    return {1.f, -v, -u};
  case 1:
    return {-1.f, -v, u};
  case 2:
    return {u, 1.f, v};
  case 3:
    return {u, -1.f, -v};
  case 4:
    return {u, -v, 1.f};
  default:
    return {-u, -v, -1.f};
  }
}

// Solid angle of the cube-face rectangle [u0, u1] x [v0, v1].
float faceSolidAngle(float u0, float v0, float u1, float v1) {
  auto area = [](float u, float v) { return std::atan2(u * v, std::sqrt(u * u + v * v + 1.f)); };
  return std::abs(area(u0, v0) - area(u0, v1) - area(u1, v0) + area(u1, v1));
}

// Cell of world direction d (unit length); mirrors envDirectionCell in integrator_common.glsl.
uint32_t envCell(glm::vec3 d, int width) {
  int height = width / 2;
  float phi = std::atan2(d.y, d.x);
  if (phi < 0.f) {
    phi += 2.f * kPi;
  }
  int i = std::clamp(int(phi / (2.f * kPi) * width), 0, width - 1);
  int j = std::clamp(int((1.f - std::clamp(d.z, -1.f, 1.f)) * 0.5f * height), 0, height - 1);
  return j * width + i;
}

std::unique_ptr<core::Buffer> makeStorage(core::Context &context, void const *data, size_t size) {
  auto buffer = core::Buffer::Create(context, std::max<size_t>(size, 16),
                                     vk::BufferUsageFlagBits::eStorageBuffer,
                                     VMA_MEMORY_USAGE_CPU_TO_GPU);
  if (size) {
    buffer->upload(data, size);
  }
  return buffer;
}

SapienAliasTable buildAlias(std::vector<float> const &weights) {
  SapienAliasTable table{};
  char error[256] = {};
  if (sapien_alias_table_build(weights.data(), weights.size(), &table, error, sizeof(error))) {
    throw std::runtime_error(std::string("failed to build an alias table: ") + error);
  }
  return table;
}

} // namespace

void IntegratorTables::build(core::Context &context, scene::Scene &scene,
                             resource::SVCubemap *environment) {
  // ---- environment: splat every base-level texel (luminance x solid angle) into its cell ----
  envWidth = 0;
  std::vector<uint32_t> envEntries(4, 0);
  core::Image *image = nullptr;
  if (environment) {
    environment->uploadToDevice(context);
    image = environment->getDeviceCubeImage(context);
  }
  if (image) {
    int const width = envGridWidth(image->getExtent().width);
    std::vector<float> weights(width * (width / 2), 0.f);
    vk::Format format = image->getFormat();
    uint32_t w = image->getExtent().width;
    uint32_t h = image->getExtent().height;
    uint32_t texelSize = getFormatSize(format);
    ComponentFormat componentFormat = getFormatComponentFormat(format);
    uint32_t elementSize = getFormatElementSize(format);
    uint32_t channels = std::min(getFormatChannels(format), 3u);
    std::vector<uint8_t> face(size_t(w) * h * texelSize);
    for (uint32_t f = 0; f < 6; ++f) {
      image->download(face.data(), face.size(), {0, 0, 0}, {w, h, 1}, f, 0);
      for (uint32_t y = 0; y < h; ++y) {
        float v0 = 2.f * y / h - 1.f;
        float v1 = 2.f * (y + 1) / h - 1.f;
        for (uint32_t x = 0; x < w; ++x) {
          float u0 = 2.f * x / w - 1.f;
          float u1 = 2.f * (x + 1) / w - 1.f;
          uint8_t const *texel = face.data() + (size_t(y) * w + x) * texelSize;
          glm::vec3 radiance(0.f);
          for (uint32_t c = 0; c < channels; ++c) {
            radiance[c] = readChannel(texel, componentFormat, elementSize, c);
          }
          float weight = luminance(radiance) * faceSolidAngle(u0, v0, u1, v1);
          if (!(weight > 0.f) || !std::isfinite(weight)) {
            continue;
          }
          // the shaders look the cube up at worldDirection.xzy
          glm::vec3 c = glm::normalize(cubeDirection(f, 0.5f * (u0 + u1), 0.5f * (v0 + v1)));
          weights[envCell({c.x, c.z, c.y}, width)] += weight;
        }
      }
    }
    double total = 0.0;
    for (float &weight : weights) {
      if (!std::isfinite(weight) || weight < 0.f) {
        weight = 0.f;
      }
      total += weight;
    }
    if (std::isfinite(total) && total > 0.0) {
      float floor = float(kEnvWeightFloor * total / weights.size());
      for (float &weight : weights) {
        weight += floor;
      }
      SapienAliasTable table = buildAlias(weights);
      if (std::isfinite(table.total) && table.total > 0.f) {
        float cellSolidAngle = 4.f * kPi / weights.size();
        envEntries.assign(4 * weights.size(), 0);
        for (size_t c = 0; c < weights.size(); ++c) {
          float pdf = weights[c] / table.total / cellSolidAngle;
          envEntries[4 * c + 0] = table.entries[c].alias;
          envEntries[4 * c + 1] = table.entries[c].threshold;
          std::memcpy(&envEntries[4 * c + 2], &pdf, sizeof(pdf));
        }
        envWidth = width;
      }
      sapien_alias_table_free(&table);
    }
  }
  envAliasBuffer =
      makeStorage(context, envEntries.data(), envEntries.size() * sizeof(uint32_t));

  // ---- emissive meshes: one entry per triangle, in the object space of its TLAS row ----
  // Triangles of a geometry instance are consecutive, so a hit finds its entry as
  // base + primitive. Emission textures are not light-sampled: such meshes are left to BSDF
  // sampling (base -1), which also keeps their contribution unbiased.
  std::vector<float> triangles; // 16 floats per triangle
  std::vector<float> weights;
  std::vector<int32_t> bases;
  int32_t row = 0; // TLAS instance rows follow scene.getObjects()
  for (auto *object : scene.getObjects()) {
    glm::mat4 model = object->getTransform().worldModelMatrix;
    for (auto const &shape : object->getModel()->getShapes()) {
      auto material = std::static_pointer_cast<resource::SVMetallicMaterial>(shape->material);
      glm::vec4 emission = material->getEmission();
      glm::vec3 radiance = glm::vec3(emission) * emission.w;
      auto const &mesh = shape->mesh;
      // Emission textures, base-color textures (which may carry alpha cutouts), layered emission
      // (a coat or fuzz dims it by the emission factor) and a partly transparent base (stochastic
      // coverage) make the visible emission vary per direction or per sample: such meshes are left
      // to BSDF sampling (base -1), where the closest-hit shader applies the same factor, which
      // also keeps their contribution unbiased.
      bool modulated = material->getTexture(resource::SVMetallicMaterial::eEmission) ||
                       material->getTexture(resource::SVMetallicMaterial::eBaseColor) ||
                       material->getCoatWeight() > 0.f || material->getFuzzWeight() > 0.f ||
                       material->getBaseColor().a < 1.f || object->getTransparency() > 0.f;
      bool emissive = luminance(radiance) > 0.f && !modulated && mesh &&
                      mesh->getTriangleCount() > 0 && mesh->hasVertexAttribute("position");
      bases.push_back(emissive ? int32_t(weights.size()) : -1);
      if (!emissive) {
        continue;
      }
      auto const &indices = mesh->getIndices();
      auto const &positions = mesh->getVertexAttribute("position");
      auto vertex = [&](uint32_t i) {
        return glm::vec3(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]);
      };
      int32_t rowBits = row;
      float rowFloat;
      std::memcpy(&rowFloat, &rowBits, sizeof(rowFloat));
      // Match camera.rahit back-face visibility: only thin-walled or transmissive meshes emit
      // from both sides; opaque single-sided meshes emit from their front face only.
      float doubleSided =
          (material->getThinWalled() || material->getTransmission() >= 1e-4f ||
           material->getTexture(resource::SVMetallicMaterial::eTransmission) != nullptr)
              ? 1.f
              : 0.f;
      for (uint32_t t = 0; t < mesh->getTriangleCount(); ++t) {
        glm::vec3 p0 = vertex(indices[3 * t]);
        glm::vec3 e1 = vertex(indices[3 * t + 1]) - p0;
        glm::vec3 e2 = vertex(indices[3 * t + 2]) - p0;
        float worldArea = 0.5f * glm::length(glm::cross(glm::vec3(model * glm::vec4(e1, 0.f)),
                                                        glm::vec3(model * glm::vec4(e2, 0.f))));
        float w = worldArea * luminance(radiance);
        if (!std::isfinite(w) || w < 0.f) {
          w = 0.f;
        }
        weights.push_back(w);
        float data[16] = {p0.x, p0.y, p0.z, 0.f, e1.x,       e1.y,       e1.z,       rowFloat,
                          e2.x, e2.y, e2.z, 0.f, radiance.x, radiance.y, radiance.z, doubleSided};
        triangles.insert(triangles.end(), data, data + 16);
      }
    }
    ++row;
  }

  emissiveCount = 0;
  std::vector<uint32_t> aliasEntries(2, 0);
  double total = 0.0;
  for (float weight : weights) {
    total += weight;
  }
  if (std::isfinite(total) && total > 0.0) {
    SapienAliasTable table = buildAlias(weights);
    if (std::isfinite(table.total) && table.total > 0.f) {
      aliasEntries.assign(2 * weights.size(), 0);
      for (size_t t = 0; t < weights.size(); ++t) {
        aliasEntries[2 * t] = table.entries[t].alias;
        aliasEntries[2 * t + 1] = table.entries[t].threshold;
        triangles[16 * t + 3] = weights[t] / table.total; // selection probability
      }
      emissiveCount = int(weights.size());
    } else {
      triangles.clear();
      std::fill(bases.begin(), bases.end(), -1);
    }
    sapien_alias_table_free(&table);
  } else {
    triangles.clear();
    std::fill(bases.begin(), bases.end(), -1);
  }
  emissiveBuffer = makeStorage(context, triangles.data(), triangles.size() * sizeof(float));
  emissiveAliasBuffer =
      makeStorage(context, aliasEntries.data(), aliasEntries.size() * sizeof(uint32_t));
  emissiveBaseBuffer = makeStorage(context, bases.data(), bases.size() * sizeof(int32_t));
}

} // namespace renderer
} // namespace svulkan2
