#include "sapien/physx/mesh_manager.h"
#include "../logger.h"
#include "sapien/physx/physx_default.h"
#include "sapien/physx/physx_system.h"
#include <assimp/Exporter.hpp>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <filesystem>
#include <set>

namespace fs = std::filesystem;
using namespace physx;

namespace sapien {
namespace physx {

static bool sameSDFConfig(PhysxSDFShapeConfig const &lhs, PhysxSDFShapeConfig const &rhs) {
  return lhs.spacing == rhs.spacing && lhs.subgridSize == rhs.subgridSize &&
         lhs.numThreadsForConstruction == rhs.numThreadsForConstruction &&
         lhs.resolution == rhs.resolution && lhs.bitsPerSubgridPixel == rhs.bitsPerSubgridPixel &&
         lhs.narrowBandThickness == rhs.narrowBandThickness && lhs.margin == rhs.margin &&
         lhs.enableRemeshing == rhs.enableRemeshing &&
         lhs.triangleCountReductionFactor == rhs.triangleCountReductionFactor;
}

static std::shared_ptr<MeshManager> gManager;
std::shared_ptr<MeshManager> MeshManager::Get() {
  if (!gManager) {
    gManager = std::make_shared<MeshManager>();
  }
  return gManager;
}

void MeshManager::Clear() {
  if (gManager) {
    gManager->mTriangleMeshWithSDFRegistry.clear();
    gManager->mTriangleMeshRegistry.clear();
    gManager->mConvexMeshRegistry.clear();
    gManager->mConvexMeshGroupRegistry.clear();
  }
}

size_t MeshManager::CachedMeshCount() {
  if (!gManager) {
    return 0;
  }
  size_t count = gManager->mTriangleMeshWithSDFRegistry.size() +
                 gManager->mTriangleMeshRegistry.size() + gManager->mConvexMeshRegistry.size();
  for (auto const &[key, meshes] : gManager->mConvexMeshGroupRegistry) {
    count += meshes.size();
  }
  return count;
}

size_t MeshManager::ExternallyHeldMeshCount() {
  if (!gManager) {
    return 0;
  }
  size_t count = 0;
  auto addIfExternal = [&count](auto const &mesh) {
    if (mesh && mesh.use_count() > 1) {
      ++count;
    }
  };
  for (auto const &[key, mesh] : gManager->mTriangleMeshWithSDFRegistry) {
    addIfExternal(mesh);
  }
  for (auto const &[key, mesh] : gManager->mTriangleMeshRegistry) {
    addIfExternal(mesh);
  }
  for (auto const &[key, mesh] : gManager->mConvexMeshRegistry) {
    addIfExternal(mesh);
  }
  for (auto const &[key, meshes] : gManager->mConvexMeshGroupRegistry) {
    for (auto const &mesh : meshes) {
      addIfExternal(mesh);
    }
  }
  return count;
}

static std::string getFullPath(std::string const &filename) {
  if (!fs::is_regular_file(fs::path(filename))) {
    logger::error("File not found: {}", filename);
    return nullptr;
  }
  return fs::canonical(filename).string();
}

MeshManager::MeshManager() {}

std::shared_ptr<PhysxTriangleMesh> MeshManager::loadTriangleMesh(const std::string &filename) {
  std::string fullPath = getFullPath(filename);

  auto it = mTriangleMeshRegistry.find(fullPath);
  if (it != mTriangleMeshRegistry.end()) {
    logger::info("Using loaded mesh: {}", filename);
    return it->second;
  }

  auto mesh = std::make_shared<PhysxTriangleMesh>(fullPath, false);
  mTriangleMeshRegistry[fullPath] = mesh;

  return mesh;
}

std::shared_ptr<PhysxTriangleMesh>
MeshManager::loadTriangleMeshWithSDF(const std::string &filename,
                                     std::optional<PhysxSDFShapeConfig> config) {
  std::string fullPath = getFullPath(filename);
  auto effectiveConfig = config.value_or(PhysxDefault::getSDFShapeConfig());

  auto it = mTriangleMeshWithSDFRegistry.find(fullPath);
  if (it != mTriangleMeshWithSDFRegistry.end()) {
    if (sameSDFConfig(it->second->getSDFConfig(), effectiveConfig)) {
      logger::info("Using loaded mesh with SDF: {}", filename);
      return it->second;
    } else {
      logger::info("Reloading mesh with different SDF parameters: {}", filename);
    }
  }

  auto mesh = std::make_shared<PhysxTriangleMesh>(fullPath, true, effectiveConfig);
  mTriangleMeshWithSDFRegistry[fullPath] = mesh;

  return mesh;
}

std::vector<std::shared_ptr<PhysxConvexMesh>>
MeshManager::loadConvexMeshGroup(const std::string &filename) {
  std::string fullPath = getFullPath(filename);
  auto it = mConvexMeshGroupRegistry.find(fullPath);
  if (it != mConvexMeshGroupRegistry.end()) {
    logger::info("Using loaded mesh group: {}", filename);
    return it->second;
  }

  auto meshes = PhysxConvexMesh::LoadByConnectedParts(fullPath);
  mConvexMeshGroupRegistry[fullPath] = meshes;

  return meshes;
}

std::shared_ptr<PhysxConvexMesh> MeshManager::loadConvexMesh(const std::string &filename) {

  std::string fullPath = getFullPath(filename);
  auto it = mConvexMeshRegistry.find(fullPath);
  if (it != mConvexMeshRegistry.end()) {
    logger::info("Using loaded mesh: {}", filename);
    return it->second;
  }

  auto mesh = std::make_shared<PhysxConvexMesh>(fullPath);
  mConvexMeshRegistry[fullPath] = mesh;
  return mesh;
}

} // namespace physx
} // namespace sapien
