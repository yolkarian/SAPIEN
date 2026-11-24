#include "svulkan2/resource/model.h"
#include "../common/logger.h"
#include "assimp_material.h"
#include "internal.h"
#include "svulkan2/common/image.h"
#include "svulkan2/common/launch_policy.h"
#include "svulkan2/core/context.h"
#include "svulkan2/resource/manager.h"
#include <assimp/GltfMaterial.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

namespace svulkan2 {
namespace resource {

std::shared_ptr<SVModel> SVModel::FromPrototype(std::shared_ptr<SVModel> prototype) {
  auto model = std::shared_ptr<SVModel>(new SVModel);
  model->mPrototype = prototype;
  model->mDescription = prototype->mDescription;
  return model;
}

std::shared_ptr<SVModel> SVModel::FromFile(std::string const &filename) {
  auto model = std::shared_ptr<SVModel>(new SVModel);
  model->mDescription = {.source = ModelDescription::SourceType::eFILE, .filename = filename};
  return model;
}

std::shared_ptr<SVModel> SVModel::FromData(std::vector<std::shared_ptr<SVShape>> shapes) {
  auto model = std::shared_ptr<SVModel>(new SVModel);
  model->mDescription = {.source = ModelDescription::SourceType::eCUSTOM, .filename = {}};
  model->mShapes = shapes;
  model->mLoaded = true;
  return model;
}

std::vector<std::shared_ptr<SVShape>> const &SVModel::getShapes() {
  loadAsync().get();
  return mShapes;
}

std::future<void> SVModel::loadAsync() {
  if (mLoaded) {
    return std::async(std::launch::deferred, []() {});
  }
  if (mDescription.source != ModelDescription::SourceType::eFILE) {
    throw std::runtime_error("loading failed: only mesh created from files can be loaded");
  }
  return std::async(LAUNCH_ASYNC, [this]() {
    std::lock_guard<std::mutex> lock(mLoadingMutex);
    if (mLoaded) {
      return;
    }

    if (mPrototype) {
      auto shapes = mPrototype->getShapes();
      for (auto s : shapes) {
        auto newShape = std::make_shared<SVShape>();

        auto mat = std::dynamic_pointer_cast<SVMetallicMaterial>(s->material);
        auto newMat = std::make_shared<SVMetallicMaterial>(
            mat->getEmission(), mat->getBaseColor(), mat->getFresnel(), mat->getRoughness(),
            mat->getMetallic(), mat->getTransmission(), mat->getIor());
        newMat->setTextures(mat->getBaseColorTexture(), mat->getRoughnessTexture(),
                            mat->getNormalTexture(), mat->getMetallicTexture(),
                            mat->getEmissionTexture(), mat->getTransmissionTexture());
        newMat->setBaseColorTextureTransform(mat->getBaseColorTextureTransform());
        newMat->setRoughnessTextureTransform(mat->getRoughnessTextureTransform());
        newMat->setNormalTextureTransform(mat->getNormalTextureTransform());
        newMat->setMetallicTextureTransform(mat->getMetallicTextureTransform());
        newMat->setEmissionTextureTransform(mat->getEmissionTextureTransform());
        newMat->setTransmissionTextureTransform(mat->getTransmissionTextureTransform());
        newMat->setCullMode(mat->getCullMode());

        newShape->mesh = s->mesh;
        newShape->material = newMat;
        mShapes.push_back(newShape);
      }
      mLoaded = true;
      return;
    }

    logger::info("Loading: {}", mDescription.filename);

    std::string path = mDescription.filename;
    Assimp::Importer importer;
    uint32_t flags = aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_GenNormals |
                     aiProcess_FlipUVs | aiProcess_PreTransformVertices;

    importer.SetPropertyBool(AI_CONFIG_IMPORT_COLLADA_IGNORE_UP_DIRECTION, true);

    const aiScene *scene = importer.ReadFile(path, flags);

    if (!scene) {
      throw std::runtime_error("Failed to load scene: " + std::string(importer.GetErrorString()) +
                               ", " + path);
    }
    if (scene->mRootNode->mMetaData) {
      throw std::runtime_error("Failed to load mesh file: file contains unsupported metadata, " +
                               path);
    }

    std::vector<std::future<void>> futures; // futures of sub tasks

    auto materials = loadAssimpMaterials(scene, path, true, futures);

    for (uint32_t mesh_idx = 0; mesh_idx < scene->mNumMeshes; ++mesh_idx) {
      auto mesh = scene->mMeshes[mesh_idx];
      if (!mesh->HasFaces()) {
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
        continue;
      }
      auto svmesh = std::make_shared<SVMeshRigid>();
      svmesh->setIndices(indices);
      svmesh->setVertexAttribute("position", positions);
      svmesh->setVertexAttribute("normal", normals);
      svmesh->setVertexAttribute("tangent", tangents);
      svmesh->setVertexAttribute("bitangent", bitangents);
      svmesh->setVertexAttribute("uv", texcoords);
      svmesh->setVertexAttribute("color", colors);

      auto shape = std::make_shared<SVShape>();
      shape->mesh = svmesh;
      shape->material = materials[mesh->mMaterialIndex];
      shape->name = mesh->mName.C_Str();
      mShapes.push_back(shape);
    }
    for (auto &f : futures) {
      f.get();
    }
    mLoaded = true;
    logger::info("Loaded: {}", mDescription.filename);
  });
}

core::BLAS *SVModel::createBLAS(core::Context &context, bool update) {
  std::vector<vk::AccelerationStructureGeometryKHR> geometries;
  std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges;
  std::vector<uint32_t> maxPrimitiveCount;
  for (auto shape : getShapes()) {
    geometries.push_back(shape->mesh->getASGeometry(context));
    ranges.push_back({shape->mesh->getTriangleCount(), 0, 0, 0});
    maxPrimitiveCount.push_back(shape->mesh->getTriangleCount());
  }

  auto &state = mDeviceStates.get(context);
  state.blas = std::make_unique<core::BLAS>(context, geometries, ranges, maxPrimitiveCount, update);
  return state.blas.get();
}

void SVModel::recordUpdateBLAS(core::Context &context, vk::CommandBuffer commandBuffer) {
  auto state = mDeviceStates.find(context);
  if (!state || !state->blas) {
    throw std::runtime_error("failed to update BLAS: the model has no BLAS on this device");
  }
  std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges;
  for (auto shape : getShapes()) {
    ranges.push_back({shape->mesh->getTriangleCount(), 0, 0, 0});
  }
  state->blas->recordUpdate(commandBuffer, ranges);
}

core::BLAS *SVModel::getBLAS(core::Context const &context) {
  auto state = mDeviceStates.find(context);
  return state && state->blas ? state->blas.get() : nullptr;
}

SVModel::SVModel() {}

SVModel::~SVModel() = default;

} // namespace resource
} // namespace svulkan2
