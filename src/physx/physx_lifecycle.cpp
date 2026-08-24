#include "sapien/physx/physx_lifecycle.h"

#include "sapien/entity.h"
#include "sapien/physx/mesh_manager.h"
#include "sapien/physx/physx_default.h"
#include "sapien/physx/physx_engine.h"
#include "sapien/scene.h"

namespace sapien {
namespace physx {

bool PhysxLiveResources::canShutdown() const {
  // Library-owned MeshManager entries are cleared by shutdown() itself, but cached
  // meshes also held by a caller still block it. The default material cache is weak,
  // so an alive default material is always owned by a caller (typically a live shape).
  uint64_t externalObjects = physxObjects > cachedMeshes ? physxObjects - cachedMeshes : 0;
  return systems == 0 && externallyHeldCachedMeshes == 0 && externalObjects == 0;
}

std::string PhysxLiveResources::describe() const {
  std::ostringstream ss;
  ss << "scenes=" << scenes << ", entities=" << entities << ", systems=" << systems
     << " (gpu=" << gpuSystems << ")"
     << ", physx_objects=" << physxObjects << ", cached_meshes=" << cachedMeshes
     << " (externally_held=" << externallyHeldCachedMeshes << ")"
     << ", default_material=" << (defaultMaterial ? "alive" : "none")
     << ", engine=" << (engineExists ? "exists" : "none");
  return ss.str();
}

PhysxLiveResources physxLiveResources() {
  PhysxLiveResources out;
  out.scenes = Scene::liveCount();
  out.entities = Entity::liveCount();
  if (auto engine = PhysxEngine::GetIfExists()) {
    out.engineExists = true;
    out.systems = engine->liveSystemCount();
    out.gpuSystems = engine->liveGpuSystemCount();
    out.physxObjects = engine->liveObjectCount();
  }
  out.cachedMeshes = static_cast<uint64_t>(MeshManager::CachedMeshCount());
  out.externallyHeldCachedMeshes = static_cast<uint64_t>(MeshManager::ExternallyHeldMeshCount());
  out.defaultMaterial = PhysxDefault::HasDefaultMaterial();
  return out;
}

bool physxCanShutdown() { return physxLiveResources().canShutdown(); }

void physxShutdown() {
  PhysxLiveResources snapshot = physxLiveResources();
  if (!snapshot.canShutdown()) {
    throw std::runtime_error(
        "failed to shut down SAPIEN PhysX: caller-owned resources are still alive (" +
        snapshot.describe() +
        "). Close every scene/system and drop every material, mesh, shape, "
        "articulation, component, builder and CUDA view reference first.");
  }

  auto engine = PhysxEngine::GetIfExists();
  if (!engine) {
    PhysxDefault::Reset();
    return;
  }

  // Commit phase: the side-effect-free snapshot established that every live PhysX
  // object is library-owned by one of these caches, so clearing them cannot invalidate
  // a caller. Release Px* cache objects while the engine is still alive.
  MeshManager::Clear();
  PhysxDefault::Reset();

  if (engine->liveSystemCount() != 0 || engine->liveObjectCount() != 0) {
    throw std::runtime_error(
        "internal lifecycle accounting error: PhysX resources remained after a successful "
        "shutdown preflight");
  }
  if (!engine->isShutdown()) {
    engine->shutdown();
  }
}

} // namespace physx
} // namespace sapien
