#pragma once
#include <cstdint>
#include <sstream>
#include <string>

namespace sapien {
namespace physx {

/** Snapshot of the resources that must be gone before a job-scope shutdown. */
struct PhysxLiveResources {
  /** Live SAPIEN Scene and Entity handles (informational; closed/detached handles do
      not themselves block PhysX shutdown). */
  uint64_t scenes{0};
  uint64_t entities{0};
  /** Live CPU and GPU PhysX systems. */
  uint64_t systems{0};
  uint64_t gpuSystems{0};
  /** Live SAPIEN objects holding Px* resources (materials, shapes, meshes,
      articulations, components). */
  uint64_t physxObjects{0};
  /** Meshes cached in the MeshManager singleton. */
  uint64_t cachedMeshes{0};
  /** Cached meshes also held by a caller (shape/builder/Python reference). */
  uint64_t externallyHeldCachedMeshes{0};
  /** Whether the lazy default material is still alive. */
  bool defaultMaterial{false};
  /** Whether a PhysxEngine exists (incl. an already shut down zombie). */
  bool engineExists{false};

  /** Everything except the zombie engine itself blocks shutdown. */
  bool canShutdown() const;

  /** Human-readable multi-line description for error messages and logging. */
  std::string describe() const;
};

/** Collect the full preflight snapshot without mutating any state. */
PhysxLiveResources physxLiveResources();

/** True when physxShutdown() would succeed right now. */
bool physxCanShutdown();

/** Job-scope terminal shutdown of PhysX: clears SAPIEN-owned caches, releases every
 *  CUDA context manager whose lease was not returned (none must exist), destroys the
 *  PhysxEngine (PxCloseExtensions, PxPhysics, PxFoundation) and restores module-level
 *  defaults, so a new job can call enable_gpu()/PhysxEngine::Get() into a fresh
 *  runtime. Throws std::runtime_error and leaves state untouched when any caller-owned
 *  resource is still alive. Idempotent. */
void physxShutdown();

} // namespace physx
} // namespace sapien
