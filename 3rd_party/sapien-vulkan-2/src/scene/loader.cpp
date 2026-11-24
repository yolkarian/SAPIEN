#include "svulkan2/scene/loader.h"
#include "../resource/assimp_material.h"
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
  std::vector<std::future<void>> futures; // futures of sub tasks

  auto materials = resource::loadAssimpMaterials(scene, path, false, futures);

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
