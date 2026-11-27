#include "assimp_material.h"
#include "internal.h"
#include "../common/logger.h"
#include "svulkan2/common/image.h"
#include "svulkan2/resource/manager.h"
#include <assimp/GltfMaterial.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <unordered_map>

namespace fs = std::filesystem;

namespace svulkan2 {
namespace resource {

// glTF's infinite index of refraction (KHR_materials_ior 0): normal reflectance rounds to 1
static constexpr float kInfiniteIor = 1e4f;

static std::vector<uint8_t> loadCompressedTexture(aiTexture const *texture, int &width,
                                                  int &height, int &channels,
                                                  int desiredChannels = 0) {
  return loadImageFromMemory(reinterpret_cast<unsigned char *>(texture->pcData), texture->mWidth,
                             width, height, channels, desiredChannels);
}

/** a texture description for an image sampled with linear filtering */
static SVTextureDescription makeTextureDescription(SVImage const &image, bool srgb,
                                                   int desiredChannels = 0) {
  SVTextureDescription description{};
  description.source = SVTextureDescription::SourceType::eCUSTOM;
  description.format = image.getFormat();
  description.desiredChannels = desiredChannels;
  description.mipLevels = image.getMipLevels();
  description.magFilter = vk::Filter::eLinear;
  description.minFilter = vk::Filter::eLinear;
  description.addressModeU = vk::SamplerAddressMode::eRepeat;
  description.addressModeV = vk::SamplerAddressMode::eRepeat;
  description.addressModeW = vk::SamplerAddressMode::eRepeat;
  description.srgb = srgb;
  description.dim = 2;
  return description;
}

/** load an embedded texture to host memory; no device is touched */
static std::shared_ptr<SVTexture> loadEmbededTexture(aiTexture const *texture, uint32_t mipLevels,
                                                     int desiredChannels = 0, bool srgb = false) {
  if (strcmp(texture->achFormatHint, "kx2") == 0) {
    auto image = internal::imageFromMemory(texture->pcData, texture->mWidth);
    auto description = makeTextureDescription(*image, srgb, desiredChannels);
    if (description.mipLevels != mipLevels) {
      logger::warn("embedded ktx texture has {} levels but {} levels requested",
                   description.mipLevels, mipLevels);
    }
    return SVTexture::FromImage(std::move(image), description);
  }

  if (texture->mHeight == 0) {
    int width, height, channels;
    std::vector<uint8_t> data =
        loadCompressedTexture(texture, width, height, channels, desiredChannels);

    vk::Format format;
    switch (channels) {
    case 1:
      format = vk::Format::eR8Unorm;
      break;
    case 2:
      format = vk::Format::eR8G8Unorm;
      break;
    case 3:
      format = vk::Format::eR8G8B8Unorm;
      break;
    case 4:
      format = vk::Format::eR8G8B8A8Unorm;
      break;
    default:
      throw std::runtime_error("invalid image channels");
    }

    return SVTexture::FromRawData(width, height, 1, format, toRawBytes(data), 2, mipLevels,
                                  vk::Filter::eLinear, vk::Filter::eLinear,
                                  vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
                                  vk::SamplerAddressMode::eRepeat, srgb);
  }

  if (strcmp(texture->achFormatHint, "rgba8888") != 0) {
    throw std::runtime_error("unsupported texture: only rgba8888 format is supported");
  }
  std::vector<char> rawData(reinterpret_cast<char *>(texture->pcData),
                            reinterpret_cast<char *>(texture->pcData) +
                                texture->mWidth * texture->mHeight * 4);

  return SVTexture::FromRawData(texture->mWidth, texture->mHeight, 1, vk::Format::eR8G8B8A8Unorm,
                                rawData, 2, mipLevels, vk::Filter::eLinear, vk::Filter::eLinear,
                                vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
                                vk::SamplerAddressMode::eRepeat, srgb);
}

/** the roughness (green) and metallic (blue) channels of one image, sampled through two views
 *  that swizzle a single channel each; both textures share the image */
static std::tuple<std::shared_ptr<SVTexture>, std::shared_ptr<SVTexture>>
loadEmbededRoughnessMetallicTexture(aiTexture const *texture) {
  auto image = loadEmbededTexture(texture, 1, 4, false)->getImage();

  auto makeView = [](vk::ComponentSwizzle component) {
    return vk::ComponentMapping(component, component, component, component);
  };

  auto roughnessDescription = makeTextureDescription(*image, false, 4);
  roughnessDescription.components = makeView(vk::ComponentSwizzle::eG);

  auto metallicDescription = roughnessDescription;
  metallicDescription.components = makeView(vk::ComponentSwizzle::eB);

  return {SVTexture::FromImage(image, roughnessDescription),
          SVTexture::FromImage(image, metallicDescription)};
}

std::vector<std::shared_ptr<SVMaterial>>
loadAssimpMaterials(aiScene const *scene, std::string const &filename, bool honorTwoSided,
                    std::vector<std::future<void>> &futures) {
  auto manager = SVResourceManager::Get();
  fs::path parentDir = fs::path(filename).parent_path();

  const uint32_t MIP_LEVEL = manager->getDefaultMipLevels();
  const bool loadTextures = !manager->shouldNotLoadTexture();
  std::vector<std::shared_ptr<SVMaterial>> materials;
  std::unordered_map<std::string, std::shared_ptr<SVTexture>> textureCache;
  std::unordered_map<std::string,
                     std::tuple<std::shared_ptr<SVTexture>, std::shared_ptr<SVTexture>>>
      roughnessMetallicTextureCache;

  // glTF keeps transmission thin-walled unless KHR_materials_volume gives it a thickness
  std::string extension = fs::path(filename).extension().string();
  bool gltf = extension == ".gltf" || extension == ".glb";

  for (uint32_t mat_idx = 0; mat_idx < scene->mNumMaterials; ++mat_idx) {
    auto *m = scene->mMaterials[mat_idx];
    auto material = std::make_shared<SVMetallicMaterial>();

    int twoSided{0};
    m->Get(AI_MATKEY_TWOSIDED, twoSided);
    if (twoSided && honorTwoSided) {
      material->setCullMode(vk::CullModeFlagBits::eNone);
    }

    float alpha = 1.f;
    if (m->Get(AI_MATKEY_OPACITY, alpha) == AI_SUCCESS) {
      if (alpha < 1e-5) {
        logger::warn("The file {} has a fully transparent material. This is "
                     "probably due to modeling error. Setting opacity to 1. If "
                     "it is not an error, please remove the object entirely.",
                     filename);
        alpha = 1.f;
      }
    } else if (m->Get(AI_MATKEY_TRANSPARENCYFACTOR, alpha) == AI_SUCCESS) {
      alpha = 1 - alpha;
    }

    aiColor3D emission{0, 0, 0};
    float emissionStrength = 1.f;
    m->Get(AI_MATKEY_COLOR_EMISSIVE, emission);
    m->Get(AI_MATKEY_EMISSIVE_INTENSITY, emissionStrength);
    material->setEmission({emission.r, emission.g, emission.b, emissionStrength});

    aiColor3D diffuse{0, 0, 0};
    m->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse);
    aiColor4D baseColor = {0, 0, 0, 1};
    if (m->Get(AI_MATKEY_BASE_COLOR, baseColor) == AI_SUCCESS) {
      diffuse = {baseColor.r, baseColor.g, baseColor.b};
      alpha = baseColor.a;
    }
    material->setBaseColor({diffuse.r, diffuse.g, diffuse.b, alpha});

    float metallic = 0.f;
    m->Get(AI_MATKEY_METALLIC_FACTOR, metallic);
    material->setMetallic(metallic);

    float roughness = 1.f;
    if (m->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness) != AI_SUCCESS) {
      // pbrSpecularGlossiness and legacy formats: roughness from the shininess
      aiColor4D specularColor;
      ai_real shininess;
      if (m->Get(AI_MATKEY_COLOR_SPECULAR, specularColor) == AI_SUCCESS &&
          m->Get(AI_MATKEY_SHININESS, shininess) == AI_SUCCESS) {
        float specularIntensity =
            specularColor[0] * 0.2125f + specularColor[1] * 0.7154f + specularColor[2] * 0.0721f;
        float normalizedShininess = std::sqrt(shininess / 1000);
        normalizedShininess = std::min(std::max(normalizedShininess, 0.0f), 1.0f);
        roughness = 1 - normalizedShininess * specularIntensity;
      }
    }
    material->setRoughness(roughness);

    // KHR_materials_specular is OpenPBR's specular_weight and specular_color, with two
    // differences: the extension leaves the metal BRDF alone, and its colour may exceed 1 (the
    // excess moves into the weight, which then raises the dielectric reflectance the same way)
    float specularFactor = 1.f;
    bool metal = metallic >= 1.f && m->GetTextureCount(aiTextureType_METALNESS) == 0;
    if (m->Get(AI_MATKEY_SPECULAR_FACTOR, specularFactor) == AI_SUCCESS && !metal) {
      aiColor3D specularColor{1, 1, 1};
      m->Get(AI_MATKEY_COLOR_SPECULAR, specularColor);
      glm::vec3 color{specularColor.r, specularColor.g, specularColor.b};
      float peak = std::max({color.r, color.g, color.b});
      if (peak > 1.f) {
        specularFactor *= peak;
        color /= peak;
      }
      material->setSpecularWeight(specularFactor);
      material->setSpecularColor(color);
    }

    // KHR_materials_ior: 0 is the specular-glossiness compatibility mode, an infinite index
    // whose reflectance is 1; 1 is legal and reflects nothing. Other formats write "Ni 1" for
    // "unset", and an index of 1 would remove every reflection, so they fall back to 1.5.
    float ior = 1.5f;
    if (m->Get(AI_MATKEY_REFRACTI, ior) == AI_SUCCESS) {
      if (gltf && ior == 0.f) {
        ior = kInfiniteIor;
      } else if (gltf ? ior < 1.f : ior <= 1.001f) {
        ior = 1.5f;
      }
    }
    material->setIor(ior);

    float transmission = 0.f;
    m->Get(AI_MATKEY_TRANSMISSION_FACTOR, transmission);
    material->setTransmission(transmission);

    // KHR_materials_volume: a thickness makes the body solid; the attenuation colour is reached
    // at the attenuation distance (OpenPBR's transmission_color at transmission_depth). Without
    // one, KHR_materials_transmission tints the transmitted light by the base colour at the
    // surface: transmission_color at depth 0.
    float thickness = 0.f;
    bool volume = m->Get(AI_MATKEY_VOLUME_THICKNESS_FACTOR, thickness) == AI_SUCCESS;
    material->setThinWalled(transmission > 0.f && (volume ? thickness <= 0.f : gltf));
    float attenuationDistance = 0.f;
    aiColor3D attenuationColor{1, 1, 1};
    if (m->Get(AI_MATKEY_VOLUME_ATTENUATION_DISTANCE, attenuationDistance) == AI_SUCCESS &&
        std::isfinite(attenuationDistance) && attenuationDistance > 0.f &&
        m->Get(AI_MATKEY_VOLUME_ATTENUATION_COLOR, attenuationColor) == AI_SUCCESS) {
      material->setTransmissionDepth(attenuationDistance);
      material->setTransmissionColor({attenuationColor.r, attenuationColor.g, attenuationColor.b});
    } else if (gltf && transmission > 0.f) {
      material->setTransmissionColor({diffuse.r, diffuse.g, diffuse.b});
    }

    // KHR_materials_clearcoat is the coat (its IOR is fixed at 1.5)
    float coat = 0.f;
    if (m->Get(AI_MATKEY_CLEARCOAT_FACTOR, coat) == AI_SUCCESS && coat > 0.f) {
      float coatRoughness = 0.f;
      m->Get(AI_MATKEY_CLEARCOAT_ROUGHNESS_FACTOR, coatRoughness);
      material->setCoatWeight(coat);
      material->setCoatRoughness(coatRoughness);
      material->setCoatIor(1.5f);
    }

    // KHR_materials_sheen is the fuzz
    aiColor3D sheenColor{0, 0, 0};
    if (m->Get(AI_MATKEY_SHEEN_COLOR_FACTOR, sheenColor) == AI_SUCCESS &&
        std::max({sheenColor.r, sheenColor.g, sheenColor.b}) > 0.f) {
      float sheenRoughness = 0.f;
      m->Get(AI_MATKEY_SHEEN_ROUGHNESS_FACTOR, sheenRoughness);
      material->setFuzzWeight(1.f);
      material->setFuzzColor({sheenColor.r, sheenColor.g, sheenColor.b});
      material->setFuzzRoughness(sheenRoughness);
    }

    // texture `type` into `slot`: an embedded image of the file or a file next to it, cached
    // per path; colour textures (base colour, emission) are sRGB
    auto loadTexture = [&](aiTextureType type, SVMetallicMaterial::TextureSlot slot, bool color) {
      aiString path;
      if (m->GetTextureCount(type) == 0 || m->GetTexture(type, 0, &path) != AI_SUCCESS) {
        return;
      }
      if (!loadTextures) {
        logger::info("Texture ignored {}", path.C_Str());
      } else if (auto embedded = scene->GetEmbeddedTexture(path.C_Str())) {
        auto [entry, added] =
            textureCache.try_emplace(std::string(path.C_Str()) + (color ? "#srgb" : ""));
        if (added) {
          logger::info("Loading embeded texture {}", path.C_Str());
          entry->second = loadEmbededTexture(embedded, MIP_LEVEL, color ? 4 : 0, color);
        }
        material->setTexture(slot, entry->second);
      } else {
        std::string fullPath = (parentDir / path.C_Str()).string();
        auto texture = color ? manager->CreateTextureFromFile(
                                   fullPath, MIP_LEVEL, vk::Filter::eLinear, vk::Filter::eLinear,
                                   vk::SamplerAddressMode::eRepeat,
                                   vk::SamplerAddressMode::eRepeat, true, 4)
                             : manager->CreateTextureFromFile(fullPath, MIP_LEVEL);
        futures.push_back(texture->loadAsync());
        material->setTexture(slot, texture);
      }
      aiUVTransform transform;
      if (m->Get(AI_MATKEY_UVTRANSFORM(type, 0), transform) == AI_SUCCESS) {
        material->setTextureTransform(
            slot, {transform.mTranslation.x, transform.mTranslation.y + 1 - transform.mScaling.y,
                   transform.mScaling.x, transform.mScaling.y});
      }
    };
    loadTexture(aiTextureType_DIFFUSE, SVMetallicMaterial::eBaseColor, true);
    loadTexture(aiTextureType_METALNESS, SVMetallicMaterial::eMetallic, false);
    loadTexture(aiTextureType_NORMALS, SVMetallicMaterial::eNormal, false);
    loadTexture(aiTextureType_EMISSIVE, SVMetallicMaterial::eEmission, true);
    loadTexture(aiTextureType_DIFFUSE_ROUGHNESS, SVMetallicMaterial::eRoughness, false);
    loadTexture(aiTextureType_TRANSMISSION, SVMetallicMaterial::eTransmission, false);
    // glTF's occlusion texture arrives as a light map with its strength; other formats' light
    // maps are not occlusion (Collada's ambient texture is one)
    if (gltf) {
      loadTexture(aiTextureType_LIGHTMAP, SVMetallicMaterial::eOcclusion, false);
      float strength = 1.f;
      if (m->Get("$tex.file.strength", aiTextureType_LIGHTMAP, 0, strength) == AI_SUCCESS) {
        material->setOcclusionStrength(strength);
      }
    }
    loadTexture(aiTextureType_AMBIENT_OCCLUSION, SVMetallicMaterial::eOcclusion, false);

    // glTF's metallic-roughness texture: one image, roughness in green and metallic in blue
    aiString path;
    if (m->GetTexture(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE, &path) ==
        AI_SUCCESS) {
      if (!loadTextures) {
        logger::info("Texture ignored {}", path.C_Str());
      } else if (auto texture = scene->GetEmbeddedTexture(path.C_Str())) {
        auto [entry, added] = roughnessMetallicTextureCache.try_emplace(path.C_Str());
        if (added) {
          logger::info("Loading embeded roughness metallic texture {}", path.C_Str());
          entry->second = loadEmbededRoughnessMetallicTexture(texture);
        }
        material->setTexture(SVMetallicMaterial::eRoughness, std::get<0>(entry->second));
        material->setTexture(SVMetallicMaterial::eMetallic, std::get<1>(entry->second));
      } else {
        logger::warn("Loading non-embeded roughness metallic texture is "
                     "currently not supported");
      }
      aiUVTransform transform;
      if (m->Get(AI_MATKEY_UVTRANSFORM(aiTextureType_UNKNOWN, 0), transform) == AI_SUCCESS) {
        glm::vec4 uv{transform.mTranslation.x,
                     transform.mTranslation.y + 1 - transform.mScaling.y, transform.mScaling.x,
                     transform.mScaling.y};
        material->setTextureTransform(SVMetallicMaterial::eRoughness, uv);
        material->setTextureTransform(SVMetallicMaterial::eMetallic, uv);
      }
    }
    materials.push_back(material);
  }

  return materials;
}

} // namespace resource
} // namespace svulkan2
