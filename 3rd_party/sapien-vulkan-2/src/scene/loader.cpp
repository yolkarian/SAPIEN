#include "svulkan2/scene/loader.h"
#include "../common/logger.h"
#include "../resource/internal.h"
#include "svulkan2/common/image.h"
#include "svulkan2/core/context.h"
#include "svulkan2/resource/manager.h"
#include "svulkan2/scene/scene.h"

#include <assimp/GltfMaterial.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <cstring>
#include <filesystem>
#include <queue>

namespace fs = std::filesystem;

namespace svulkan2 {
namespace scene {

static std::vector<uint8_t> loadCompressedTexture(aiTexture const *texture, int &width,
                                                  int &height, int &channels,
                                                  int desiredChannels = 0) {
  return loadImageFromMemory(reinterpret_cast<unsigned char *>(texture->pcData), texture->mWidth,
                             width, height, channels, desiredChannels);
}

/** a texture description for an image sampled with linear filtering */
static resource::SVTextureDescription makeTextureDescription(resource::SVImage const &image,
                                                             bool srgb, int desiredChannels = 0) {
  resource::SVTextureDescription description{};
  description.source = resource::SVTextureDescription::SourceType::eCUSTOM;
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
static std::shared_ptr<resource::SVTexture> loadEmbededTexture(aiTexture const *texture,
                                                               uint32_t mipLevels,
                                                               int desiredChannels = 0,
                                                               bool srgb = false) {
  if (strcmp(texture->achFormatHint, "kx2") == 0) {
    auto image = resource::internal::imageFromMemory(texture->pcData, texture->mWidth);
    auto description = makeTextureDescription(*image, srgb, desiredChannels);
    if (description.mipLevels != mipLevels) {
      logger::warn("embedded ktx texture has {} levels but {} levels requested",
                   description.mipLevels, mipLevels);
    }
    return resource::SVTexture::FromImage(std::move(image), description);
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

    return resource::SVTexture::FromRawData(
        width, height, 1, format, toRawBytes(data), 2, mipLevels, vk::Filter::eLinear,
        vk::Filter::eLinear, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode::eRepeat, srgb);
  }

  if (strcmp(texture->achFormatHint, "rgba8888") != 0) {
    throw std::runtime_error("unsupported texture: only rgba8888 format is supported");
  }
  std::vector<char> rawData(reinterpret_cast<char *>(texture->pcData),
                            reinterpret_cast<char *>(texture->pcData) +
                                texture->mWidth * texture->mHeight * 4);

  return resource::SVTexture::FromRawData(
      texture->mWidth, texture->mHeight, 1, vk::Format::eR8G8B8A8Unorm, rawData, 2, mipLevels,
      vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, srgb);
}

/** the roughness (green) and metallic (blue) channels of one image, sampled through two views
 *  that swizzle a single channel each; both textures share the image */
static std::tuple<std::shared_ptr<resource::SVTexture>, std::shared_ptr<resource::SVTexture>>
loadEmbededRoughnessMetallicTexture(aiTexture const *texture) {
  auto image = loadEmbededTexture(texture, 1, 4, false)->getImage();

  auto makeView = [](vk::ComponentSwizzle component) {
    return vk::ComponentMapping(component, component, component, component);
  };

  auto roughnessDescription = makeTextureDescription(*image, false, 4);
  roughnessDescription.components = makeView(vk::ComponentSwizzle::eG);

  auto metallicDescription = roughnessDescription;
  metallicDescription.components = makeView(vk::ComponentSwizzle::eB);

  return {resource::SVTexture::FromImage(image, roughnessDescription),
          resource::SVTexture::FromImage(image, metallicDescription)};
}

std::shared_ptr<Scene> LoadScene(std::shared_ptr<core::Context> context, std::string const &path) {
  if (!context) {
    throw std::runtime_error("failed to load scene: invalid context");
  }
  std::shared_ptr<Scene> result = std::make_shared<Scene>(std::move(context));

  Assimp::Importer importer;
  uint32_t flags = aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_GenNormals |
                   aiProcess_FlipUVs;
  importer.SetPropertyBool(AI_CONFIG_IMPORT_COLLADA_IGNORE_UP_DIRECTION, true);
  const aiScene *scene = importer.ReadFile(path, flags);
  if (!scene) {
    throw std::runtime_error("failed to parse scene " + path);
  }
  fs::path parentDir = fs::path(path).parent_path();

  std::vector<std::future<void>> futures; // futures of sub tasks

  auto manager = resource::SVResourceManager::Get();

  const uint32_t MIP_LEVEL = manager->getDefaultMipLevels();
  const bool loadTextures = !manager->shouldNotLoadTexture();
  std::vector<std::shared_ptr<resource::SVMaterial>> materials;
  std::unordered_map<std::string, std::shared_ptr<resource::SVTexture>> textureCache;
  std::unordered_map<std::string, std::tuple<std::shared_ptr<resource::SVTexture>,
                                             std::shared_ptr<resource::SVTexture>>>
      roughnessMetallicTextureCache;

  // load materials
  for (uint32_t mat_idx = 0; mat_idx < scene->mNumMaterials; ++mat_idx) {
    auto *m = scene->mMaterials[mat_idx];
    aiColor3D emission{0, 0, 0};
    float emissionStrength = 1.f;

    aiColor3D diffuse{0, 0, 0};
    float specular = 0.5f;

    float alpha = 1.f;
    float metallic = 0.f;
    float transmission = 0.f;
    float ior = 1.01f;

    float roughness = 1.f;

    if (m->Get(AI_MATKEY_OPACITY, alpha) == AI_SUCCESS) {
      if (alpha < 1e-5) {
        logger::warn("The file {} has a fully transparent material. This is "
                     "probably due to modeling error. Setting opacity to 1. If "
                     "it is not an error, please remove the object entirely.",
                     path);
        alpha = 1.f;
      }
    } else {
      if (m->Get(AI_MATKEY_TRANSPARENCYFACTOR, alpha) == AI_SUCCESS) {
        alpha = 1 - alpha;
      }
    }

    m->Get(AI_MATKEY_COLOR_EMISSIVE, emission);
    m->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse);
    m->Get(AI_MATKEY_EMISSIVE_INTENSITY, emissionStrength);

    {
      aiColor4D baseColor = {0, 0, 0, 1};
      if (m->Get(AI_MATKEY_BASE_COLOR, baseColor) == AI_SUCCESS) {
        diffuse = {baseColor.r, baseColor.g, baseColor.b};
        alpha = baseColor.a;
      }
    }

    // CHECK GLTF_PBRMETALLICROUGHNESS_METALLIC_FACTOR

    // assimp code for reading roughness, metallic, and glossiness
    if (m->Get(AI_MATKEY_METALLIC_FACTOR, metallic) != AI_SUCCESS) {
      metallic = 0.f;
    }

    // pbrSpecularGlossiness
    if (m->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness) != AI_SUCCESS) {
      aiColor4D specularColor;
      ai_real shininess;
      if (m->Get(AI_MATKEY_COLOR_SPECULAR, specularColor) == AI_SUCCESS &&
          m->Get(AI_MATKEY_SHININESS, shininess) == AI_SUCCESS) {
        float specularIntensity =
            specularColor[0] * 0.2125f + specularColor[1] * 0.7154f + specularColor[2] * 0.0721f;
        float normalizedShininess = std::sqrt(shininess / 1000);
        normalizedShininess = std::min(std::max(normalizedShininess, 0.0f), 1.0f);
        normalizedShininess = normalizedShininess * specularIntensity;
        roughness = 1 - normalizedShininess;
      }
    }

    // KHR_materials_specular
    {
      float specularFactor = 0.f;
      aiColor4D specularColor;
      if (m->Get(AI_MATKEY_SPECULAR_FACTOR, specularFactor) == AI_SUCCESS &&
          m->Get(AI_MATKEY_COLOR_SPECULAR, specularFactor) == AI_SUCCESS) {
        specular =
            (specularColor[0] * 0.2125f + specularColor[1] * 0.7154f + specularColor[2] * 0.0721f);
        specular *= 0.5f;
      }
    }

    // if (m->Get(AI_MATKEY_GLOSSINESS_FACTOR, specular) != AI_SUCCESS) {
    //   float shininess;
    //   if (m->Get(AI_MATKEY_SHININESS, shininess)) {
    //     specular = shininess / 1000;
    //   }
    // }

    if (m->Get(AI_MATKEY_TRANSMISSION_FACTOR, transmission) != AI_SUCCESS) {
      transmission = 0.f;
    }
    if (m->Get(AI_MATKEY_REFRACTI, ior) != AI_SUCCESS) {
      ior = 1.01f;
    }

    std::shared_ptr<resource::SVTexture> baseColorTexture{};
    std::shared_ptr<resource::SVTexture> normalTexture{};
    std::shared_ptr<resource::SVTexture> roughnessTexture{};
    std::shared_ptr<resource::SVTexture> metallicTexture{};
    std::shared_ptr<resource::SVTexture> emissionTexture{};
    std::shared_ptr<resource::SVTexture> transmissionTexture{};

    glm::vec4 baseColorTransform{0, 0, 1, 1};
    glm::vec4 normalTransform{0, 0, 1, 1};
    glm::vec4 roughnessTransform{0, 0, 1, 1};
    glm::vec4 metallicTransform{0, 0, 1, 1};
    glm::vec4 emissionTransform{0, 0, 1, 1};
    glm::vec4 transmissionTransform{0, 0, 1, 1};

    aiUVTransform transform;

    aiString path;
    if (m->GetTextureCount(aiTextureType_DIFFUSE) > 0 &&
        m->GetTexture(aiTextureType_DIFFUSE, 0, &path) == AI_SUCCESS) {
      if (!loadTextures) {
        logger::info("Texture ignored {}", path.C_Str());
      } else {
        logger::info("Trying to load texture {}", path.C_Str());
        if (auto texture = scene->GetEmbeddedTexture(path.C_Str())) {
          if (textureCache.contains(std::string(path.C_Str()))) {
            baseColorTexture = textureCache[std::string(path.C_Str())];
          } else {
            logger::info("Loading embeded texture {}", path.C_Str());
            textureCache[std::string(path.C_Str())] = baseColorTexture =
                loadEmbededTexture(texture, MIP_LEVEL, 4, true);
          }
        } else {
          std::string p = std::string(path.C_Str());
          std::string fullPath = (parentDir / p).string();
          baseColorTexture = manager->CreateTextureFromFile(
              fullPath, MIP_LEVEL, vk::Filter::eLinear, vk::Filter::eLinear,
              vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, true, 4);
          futures.push_back(baseColorTexture->loadAsync());
        }
      }
      if (m->Get(AI_MATKEY_UVTRANSFORM_DIFFUSE(0), transform) == AI_SUCCESS) {
        baseColorTransform = {transform.mTranslation.x,
                              transform.mTranslation.y + 1 - transform.mScaling.y,
                              transform.mScaling.x, transform.mScaling.y};
      }
    }

    if (m->GetTextureCount(aiTextureType_METALNESS) > 0 &&
        m->GetTexture(aiTextureType_METALNESS, 0, &path) == AI_SUCCESS) {
      if (!loadTextures) {
        logger::info("Texture ignored {}", path.C_Str());
      } else {
        if (auto texture = scene->GetEmbeddedTexture(path.C_Str())) {
          if (textureCache.contains(std::string(path.C_Str()))) {
            metallicTexture = textureCache[std::string(path.C_Str())];
          } else {
            logger::info("Loading embeded texture {}", path.C_Str());
            textureCache[std::string(path.C_Str())] = metallicTexture =
                loadEmbededTexture(texture, MIP_LEVEL);
          }
        } else {
          std::string p = std::string(path.C_Str());
          std::string fullPath = (parentDir / p).string();
          metallicTexture = manager->CreateTextureFromFile(fullPath, MIP_LEVEL);
          futures.push_back(metallicTexture->loadAsync());
        }
      }

      if (m->Get(AI_MATKEY_UVTRANSFORM(aiTextureType_METALNESS, 0), transform) == AI_SUCCESS) {
        metallicTransform = {transform.mTranslation.x,
                             transform.mTranslation.y + 1 - transform.mScaling.y,
                             transform.mScaling.x, transform.mScaling.y};
      }
    }

    if (m->GetTextureCount(aiTextureType_NORMALS) > 0 &&
        m->GetTexture(aiTextureType_NORMALS, 0, &path) == AI_SUCCESS) {
      if (!loadTextures) {
        logger::info("Texture ignored {}", path.C_Str());
      } else {
        if (auto texture = scene->GetEmbeddedTexture(path.C_Str())) {
          if (textureCache.contains(std::string(path.C_Str()))) {
            normalTexture = textureCache[std::string(path.C_Str())];
          } else {
            logger::info("Loading embeded texture {}", path.C_Str());
            textureCache[std::string(path.C_Str())] = normalTexture =
                loadEmbededTexture(texture, MIP_LEVEL);
          }
        } else {
          std::string p = std::string(path.C_Str());
          std::string fullPath = (parentDir / p).string();
          normalTexture = manager->CreateTextureFromFile(fullPath, MIP_LEVEL);
          futures.push_back(normalTexture->loadAsync());
        }
      }
      if (m->Get(AI_MATKEY_UVTRANSFORM_NORMALS(0), transform) == AI_SUCCESS) {
        normalTransform = {transform.mTranslation.x,
                           transform.mTranslation.y + 1 - transform.mScaling.y,
                           transform.mScaling.x, transform.mScaling.y};
      }
    }

    if (m->GetTextureCount(aiTextureType_EMISSIVE) > 0 &&
        m->GetTexture(aiTextureType_EMISSIVE, 0, &path) == AI_SUCCESS) {
      if (!loadTextures) {
        logger::info("Texture ignored {}", path.C_Str());
      } else {
        if (auto texture = scene->GetEmbeddedTexture(path.C_Str())) {
          if (textureCache.contains(std::string(path.C_Str()))) {
            emissionTexture = textureCache[std::string(path.C_Str())];
          } else {
            logger::info("Loading embeded texture {}", path.C_Str());
            textureCache[std::string(path.C_Str())] = emissionTexture =
                loadEmbededTexture(texture, MIP_LEVEL, 4, true);
          }
        } else {
          std::string p = std::string(path.C_Str());
          std::string fullPath = (parentDir / p).string();
          emissionTexture = manager->CreateTextureFromFile(
              fullPath, MIP_LEVEL, vk::Filter::eLinear, vk::Filter::eLinear,
              vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, true, 4);
          futures.push_back(emissionTexture->loadAsync());
        }
      }
      if (m->Get(AI_MATKEY_UVTRANSFORM_EMISSIVE(0), transform) == AI_SUCCESS) {
        emissionTransform = {transform.mTranslation.x,
                             transform.mTranslation.y + 1 - transform.mScaling.y,
                             transform.mScaling.x, transform.mScaling.y};
      }
    }

    if (m->GetTextureCount(aiTextureType_DIFFUSE_ROUGHNESS) > 0 &&
        m->GetTexture(aiTextureType_DIFFUSE_ROUGHNESS, 0, &path) == AI_SUCCESS) {
      if (!loadTextures) {
        logger::info("Texture ignored {}", path.C_Str());
      } else {
        if (auto texture = scene->GetEmbeddedTexture(path.C_Str())) {
          if (textureCache.contains(std::string(path.C_Str()))) {
            roughnessTexture = textureCache[std::string(path.C_Str())];
          } else {
            logger::info("Loading embeded texture {}", path.C_Str());
            textureCache[std::string(path.C_Str())] = roughnessTexture =
                loadEmbededTexture(texture, MIP_LEVEL);
          }
        } else {
          std::string p = std::string(path.C_Str());
          std::string fullPath = (parentDir / p).string();
          roughnessTexture = manager->CreateTextureFromFile(fullPath, MIP_LEVEL);
          futures.push_back(roughnessTexture->loadAsync());
        }
      }
      if (m->Get(AI_MATKEY_UVTRANSFORM(aiTextureType_DIFFUSE_ROUGHNESS, 0), transform) ==
          AI_SUCCESS) {
        roughnessTransform = {transform.mTranslation.x,
                              transform.mTranslation.y + 1 - transform.mScaling.y,
                              transform.mScaling.x, transform.mScaling.y};
      }
    }

    if (m->GetTexture(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE, &path) ==
        AI_SUCCESS) {
      if (!loadTextures) {
        logger::info("Texture ignored {}", path.C_Str());
      } else {
        if (auto texture = scene->GetEmbeddedTexture(path.C_Str())) {
          if (roughnessMetallicTextureCache.contains(std::string(path.C_Str()))) {
            std::tie(roughnessTexture, metallicTexture) =
                roughnessMetallicTextureCache[std::string(path.C_Str())];
          } else {
            logger::info("Loading embeded roughness metallic texture {}", path.C_Str());
            std::tie(roughnessTexture, metallicTexture) =
                roughnessMetallicTextureCache[std::string(path.C_Str())] =
                    loadEmbededRoughnessMetallicTexture(texture);
          }
        } else {
          logger::warn("Loading non-embeded roughness metallic texture is "
                       "currently not supported");
        }
      }
      // TODO: check if this actually does anything
      if (m->Get(AI_MATKEY_UVTRANSFORM(aiTextureType_UNKNOWN, 0), transform) == AI_SUCCESS) {
        roughnessTransform = metallicTransform = {
            transform.mTranslation.x, transform.mTranslation.y + 1 - transform.mScaling.y,
            transform.mScaling.x, transform.mScaling.y};
      }
    }

    if (m->GetTextureCount(aiTextureType_TRANSMISSION) > 0 &&
        m->GetTexture(aiTextureType_TRANSMISSION, 0, &path) == AI_SUCCESS) {
      if (!loadTextures) {
        logger::info("Texture ignored {}", path.C_Str());
      } else {
        if (auto texture = scene->GetEmbeddedTexture(path.C_Str())) {
          if (textureCache.contains(std::string(path.C_Str()))) {
            transmissionTexture = textureCache[std::string(path.C_Str())];
          } else {
            logger::info("Loading embeded texture {}", path.C_Str());
            textureCache[std::string(path.C_Str())] = transmissionTexture =
                loadEmbededTexture(texture, MIP_LEVEL);
          }
        } else {
          std::string p = std::string(path.C_Str());
          std::string fullPath = (parentDir / p).string();
          transmissionTexture = manager->CreateTextureFromFile(fullPath, MIP_LEVEL);
          futures.push_back(transmissionTexture->loadAsync());
        }
      }
      if (m->Get(AI_MATKEY_UVTRANSFORM(aiTextureType_TRANSMISSION, 0), transform) == AI_SUCCESS) {
        transmissionTransform = {transform.mTranslation.x,
                                 transform.mTranslation.y + 1 - transform.mScaling.y,
                                 transform.mScaling.x, transform.mScaling.y};
      }
    }

    auto material = std::make_shared<resource::SVMetallicMaterial>(
        glm::vec4{emission.r, emission.g, emission.b, emissionStrength},
        glm::vec4{diffuse.r, diffuse.g, diffuse.b, alpha}, specular, roughness, metallic,
        transmission, ior);
    material->setTextures(baseColorTexture, roughnessTexture, normalTexture, metallicTexture,
                          emissionTexture, transmissionTexture);
    material->setBaseColorTextureTransform(baseColorTransform);
    material->setNormalTextureTransform(normalTransform);
    material->setRoughnessTextureTransform(roughnessTransform);
    material->setMetallicTextureTransform(metallicTransform);
    material->setTransmissionTextureTransform(transmissionTransform);
    material->setEmissionTextureTransform(emissionTransform);
    materials.push_back(material);
  }

  std::vector<std::shared_ptr<resource::SVMesh>> meshes;
  for (uint32_t mesh_idx = 0; mesh_idx < scene->mNumMeshes; ++mesh_idx) {
    auto mesh = scene->mMeshes[mesh_idx];
    if (!mesh->HasFaces()) {
      meshes.push_back(nullptr);
      continue;
    }

    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> texcoords;
    std::vector<float> tangents;
    std::vector<float> bitangents;
    std::vector<float> colors;

    for (uint32_t v = 0; v < mesh->mNumVertices; v++) {
      glm::vec4 color = glm::vec4(1, 1, 1, 1);
      glm::vec3 normal = glm::vec3(0);
      glm::vec2 texcoord = glm::vec2(0);
      glm::vec3 position =
          glm::vec3(mesh->mVertices[v].x, mesh->mVertices[v].y, mesh->mVertices[v].z);
      glm::vec3 tangent = glm::vec3(0);
      glm::vec3 bitangent = glm::vec3(0);
      if (mesh->HasNormals()) {
        normal = glm::vec3{mesh->mNormals[v].x, mesh->mNormals[v].y, mesh->mNormals[v].z};
      }
      if (mesh->HasTextureCoords(0)) {
        texcoord = glm::vec2{mesh->mTextureCoords[0][v].x, mesh->mTextureCoords[0][v].y};
      }
      if (mesh->HasTangentsAndBitangents()) {
        tangent = glm::vec3{mesh->mTangents[v].x, mesh->mTangents[v].y, mesh->mTangents[v].z};
        bitangent =
            glm::vec3{mesh->mBitangents[v].x, mesh->mBitangents[v].y, mesh->mBitangents[v].z};
      }
      if (mesh->HasVertexColors(0)) {
        color = glm::vec4{mesh->mColors[0][v].r, mesh->mColors[0][v].g, mesh->mColors[0][v].b,
                          mesh->mColors[0][v].a};
      }
      positions.push_back(position.x);
      positions.push_back(position.y);
      positions.push_back(position.z);

      normals.push_back(normal.x);
      normals.push_back(normal.y);
      normals.push_back(normal.z);

      texcoords.push_back(texcoord.x);
      texcoords.push_back(texcoord.y);

      tangents.push_back(tangent.x);
      tangents.push_back(tangent.y);
      tangents.push_back(tangent.z);

      bitangents.push_back(bitangent.x);
      bitangents.push_back(bitangent.y);
      bitangents.push_back(bitangent.z);

      colors.push_back(color.r);
      colors.push_back(color.g);
      colors.push_back(color.b);
      colors.push_back(color.a);
    }

    std::vector<uint32_t> indices;
    for (uint32_t f = 0; f < mesh->mNumFaces; f++) {
      auto face = mesh->mFaces[f];
      if (face.mNumIndices != 3) {
        continue;
      }
      indices.push_back(face.mIndices[0]);
      indices.push_back(face.mIndices[1]);
      indices.push_back(face.mIndices[2]);
    }

    if (positions.size() == 0 || indices.size() == 0) {
      logger::warn("A mesh in the file has no triangles: {}", path);
      meshes.push_back(nullptr);
      continue;
    }
    auto svmesh = std::make_shared<resource::SVMeshRigid>();
    svmesh->setIndices(indices);
    svmesh->setVertexAttribute("position", positions);
    svmesh->setVertexAttribute("normal", normals);
    svmesh->setVertexAttribute("tangent", tangents);
    svmesh->setVertexAttribute("bitangent", bitangents);
    svmesh->setVertexAttribute("uv", texcoords);
    svmesh->setVertexAttribute("color", colors);

    meshes.push_back(svmesh);
  }

  std::vector<std::shared_ptr<resource::SVShape>> shapes;

  std::queue<aiNode *> q;
  q.push(scene->mRootNode);
  std::queue<Node *> qp;
  qp.push(nullptr);

  while (!q.empty()) {
    aiNode *node = q.front();
    Node *parent = qp.front();
    q.pop();
    qp.pop();

    aiVector3t<float> scaling;
    aiQuaterniont<float> rotation;
    aiVector3t<float> position;
    node->mTransformation.Decompose(scaling, rotation, position);
    Transform transform = {.position = {position.x, position.y, position.z},
                           .rotation = {rotation.w, rotation.x, rotation.y, rotation.z},
                           .scale = {scaling.x, scaling.y, scaling.z}};

    Node &newNode = parent ? result->addNode(*parent, transform) : result->getRootNode();
    newNode.setName(std::string(node->mName.C_Str()));

    for (uint32_t i = 0; i < node->mNumMeshes; ++i) {
      uint32_t meshIdx = node->mMeshes[i];

      auto svmesh = meshes.at(meshIdx);
      if (!svmesh) {
        continue;
      }

      auto shape = std::make_shared<resource::SVShape>();
      shape->mesh = svmesh;
      shape->material = materials[scene->mMeshes[meshIdx]->mMaterialIndex];
      shape->name = std::string(scene->mMeshes[meshIdx]->mName.C_Str());
      shapes.push_back(shape);

      auto &obj = result->addObject(newNode, resource::SVModel::FromData({shape}));
      obj.setName(std::string(scene->mMeshes[meshIdx]->mName.C_Str()));
    }

    for (uint32_t i = 0; i < node->mNumChildren; ++i) {
      q.push(node->mChildren[i]);
      qp.push(&newNode);
    }
  }
  for (auto &f : futures) {
    f.get();
  }

  // load lights
  for (uint32_t i = 0; i < scene->mNumLights; ++i) {
    switch (scene->mLights[i]->mType) {
    case aiLightSource_POINT: {
      auto &l = result->addPointLight();
      l.setColor(glm::vec3{scene->mLights[i]->mColorDiffuse.r, scene->mLights[i]->mColorDiffuse.g,
                           scene->mLights[i]->mColorDiffuse.b} /
                 5.f);
      l.setPosition({scene->mLights[i]->mPosition.x, scene->mLights[i]->mPosition.y,
                     scene->mLights[i]->mPosition.z});
      break;
    }
    case aiLightSource_DIRECTIONAL: {
      auto &l = result->addDirectionalLight();
      l.setColor(glm::vec3{scene->mLights[i]->mColorDiffuse.r, scene->mLights[i]->mColorDiffuse.g,
                           scene->mLights[i]->mColorDiffuse.b} /
                 5.f);
      l.setPosition({scene->mLights[i]->mPosition.x, scene->mLights[i]->mPosition.y,
                     scene->mLights[i]->mPosition.z});
      l.setDirection({scene->mLights[i]->mDirection.x, scene->mLights[i]->mDirection.y,
                      scene->mLights[i]->mDirection.z});
      break;
    }
    case aiLightSource_SPOT: {
      auto &l = result->addSpotLight();
      l.setColor(glm::vec3{scene->mLights[i]->mColorDiffuse.r, scene->mLights[i]->mColorDiffuse.g,
                           scene->mLights[i]->mColorDiffuse.b} /
                 5.f);
      l.setPosition({scene->mLights[i]->mPosition.x, scene->mLights[i]->mPosition.y,
                     scene->mLights[i]->mPosition.z});
      l.setDirection({scene->mLights[i]->mDirection.x, scene->mLights[i]->mDirection.y,
                      scene->mLights[i]->mDirection.z});
      l.setFovSmall(scene->mLights[i]->mAngleInnerCone);
      l.setFov(scene->mLights[i]->mAngleOuterCone);
      break;
    }
    case aiLightSource_AREA: {
      auto &l = result->addParallelogramLight();
      l.setColor(glm::vec3{scene->mLights[i]->mColorDiffuse.r, scene->mLights[i]->mColorDiffuse.g,
                           scene->mLights[i]->mColorDiffuse.b} /
                 5.f);
      l.setShape({scene->mLights[i]->mSize.x, scene->mLights[i]->mSize.y});

      l.setRotation(glm::quatLookAt(glm::vec3{scene->mLights[i]->mDirection.x,
                                              scene->mLights[i]->mDirection.y,
                                              scene->mLights[i]->mDirection.z},
                                    glm::vec3{scene->mLights[i]->mUp.x, scene->mLights[i]->mUp.y,
                                              scene->mLights[i]->mUp.z}));
      l.setPosition({scene->mLights[i]->mPosition.x, scene->mLights[i]->mPosition.y,
                     scene->mLights[i]->mPosition.z});
      break;
    }
    case aiLightSource_AMBIENT: {
      result->setAmbientLight({scene->mLights[i]->mColorAmbient.r,
                               scene->mLights[i]->mColorAmbient.g,
                               scene->mLights[i]->mColorAmbient.b, 1.f});
      break;
    }
    default:
      logger::warn("failed to load light: cannot recognize light type");
      break;
    }
  }

  return result;
}

} // namespace scene
} // namespace svulkan2
