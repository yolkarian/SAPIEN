#include "sapien/physx/physx_default.h"
#include "./broadphase_env_id.hpp"
#include "sapien/physx/material.h"
#include "sapien/physx/physx_system.h"
#include <PxPhysicsAPI.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#ifndef _WIN32
#include <pwd.h>
#include <unistd.h>
#endif

namespace sapien {
namespace physx {

void PhysxSceneConfig::setGpuBroadPhaseEnvIdBits(uint8_t bitsX, uint8_t bitsY, uint8_t bitsZ) {
  for (auto const &[bits, axis] :
       {std::pair{bitsX, 'x'}, std::pair{bitsY, 'y'}, std::pair{bitsZ, 'z'}}) {
    if (bits > kMaxBroadphaseEnvIdBits) {
      throw std::runtime_error(std::string("gpu broadphase env ID bits on axis ") + axis +
                               " must be in [0, " + std::to_string(kMaxBroadphaseEnvIdBits) + "]");
    }
  }
  mBitsEnvIDX = bitsX;
  mBitsEnvIDY = bitsY;
  mBitsEnvIDZ = bitsZ;
}

static float gStaticFriction{0.3};
static float gDynamicFriction{0.3};
static float gRestitution{0.1};
static std::weak_ptr<PhysxMaterial> gDefaultMaterial;
// guards gDefaultMaterial and the friction/restitution defaults it is built from
static std::mutex gDefaultMaterialMutex;
static bool gGPUEnabled{false};
static PhysxSceneConfig gSceneConfig{};
static PhysxBodyConfig gBodyConfig{};
static PhysxShapeConfig gShapeConfig{};
static PhysxSDFShapeConfig gSDFConfig{};

static ::physx::PxGpuDynamicsMemoryConfig gGpuMemoryConfig{};

void PhysxDefault::SetDefaultMaterial(float staticFriction, float dynamicFriction,
                                      float restitution) {
  std::lock_guard lock(gDefaultMaterialMutex);
  gDefaultMaterial.reset();
  gStaticFriction = staticFriction;
  gDynamicFriction = dynamicFriction;
  gRestitution = restitution;
}

std::shared_ptr<PhysxMaterial> PhysxDefault::GetDefaultMaterial() {
  std::lock_guard lock(gDefaultMaterialMutex);
  auto m = gDefaultMaterial.lock();
  if (m) {
    return m;
  }
  gDefaultMaterial = m =
      std::make_shared<PhysxMaterial>(gStaticFriction, gDynamicFriction, gRestitution);
  return m;
}

void PhysxDefault::setGpuMemoryConfig(uint64_t tempBufferCapacity, uint32_t maxRigidContactCount,
                                      uint32_t maxRigidPatchCount, uint32_t heapCapacity,
                                      uint32_t foundLostPairsCapacity,
                                      uint32_t foundLostAggregatePairsCapacity,
                                      uint32_t totalAggregatePairsCapacity,
                                      uint32_t collisionStackSize) {
  gGpuMemoryConfig.tempBufferCapacity = tempBufferCapacity;
  gGpuMemoryConfig.maxRigidContactCount = maxRigidContactCount;
  gGpuMemoryConfig.maxRigidPatchCount = maxRigidPatchCount;
  gGpuMemoryConfig.heapCapacity = heapCapacity;
  gGpuMemoryConfig.foundLostPairsCapacity = foundLostPairsCapacity;
  gGpuMemoryConfig.foundLostAggregatePairsCapacity = foundLostAggregatePairsCapacity;
  gGpuMemoryConfig.totalAggregatePairsCapacity = totalAggregatePairsCapacity;
  gGpuMemoryConfig.collisionStackSize = collisionStackSize;
}

::physx::PxGpuDynamicsMemoryConfig const &PhysxDefault::getGpuMemoryConfig() {
  return gGpuMemoryConfig;
}

void PhysxDefault::setSceneConfig(Vec3 gravity, float bounceThreshold, bool enablePCM,
                                  bool enableTGS, bool enableCCD, bool enableEnhancedDeterminism,
                                  bool enableFrictionEveryIteration, float frictionOffsetThreshold,
                                  float frictionCorrelationDistance, uint32_t cpuWorkers) {
  gSceneConfig.gravity = gravity;
  gSceneConfig.bounceThreshold = bounceThreshold;
  gSceneConfig.enablePCM = enablePCM;
  gSceneConfig.enableTGS = enableTGS;
  gSceneConfig.enableCCD = enableCCD;
  gSceneConfig.enableEnhancedDeterminism = enableEnhancedDeterminism;
  gSceneConfig.enableFrictionEveryIteration = enableFrictionEveryIteration;
  gSceneConfig.frictionOffsetThreshold = frictionOffsetThreshold;
  gSceneConfig.frictionCorrelationDistance = frictionCorrelationDistance;
  gSceneConfig.cpuWorkers = cpuWorkers;
}
void PhysxDefault::setSceneConfig(PhysxSceneConfig const &config) { gSceneConfig = config; }
PhysxSceneConfig const &PhysxDefault::getSceneConfig() { return gSceneConfig; }

void PhysxDefault::setBodyConfig(uint32_t solverIterations, uint32_t solverVelocityIterations,
                                 float sleepThreshold) {
  gBodyConfig.solverPositionIterations = solverIterations;
  gBodyConfig.solverVelocityIterations = solverVelocityIterations;
  gBodyConfig.sleepThreshold = sleepThreshold;
}
void PhysxDefault::setBodyConfig(PhysxBodyConfig const &config) { gBodyConfig = config; }
PhysxBodyConfig const &PhysxDefault::getBodyConfig() { return gBodyConfig; }

void PhysxDefault::setShapeConfig(float contactOffset, float restOffset) {
  gShapeConfig.contactOffset = contactOffset;
  gShapeConfig.restOffset = restOffset;
}
void PhysxDefault::setShapeConfig(PhysxShapeConfig const &config) { gShapeConfig = config; }
PhysxShapeConfig const &PhysxDefault::getShapeConfig() { return gShapeConfig; }

namespace {

namespace fs = std::filesystem;

#if PX_SUPPORT_GPU_PHYSX
/** Names the PhysX GPU runtime PhysX loads (PxLoadPhysxGPUModule). PhysX keeps the returned
 *  pointer, so the path is stored here for the life of the process. */
class PhysxGpuLibraryHook : public ::PxGpuLoadHook {
public:
  std::string path;
  char const *getPhysXGpuDllName() const override { return path.c_str(); }
};
PhysxGpuLibraryHook gPhysxGpuLibraryHook;
#endif

fs::path homeDirectory() {
#ifdef _WIN32
  if (char const *profile = std::getenv("USERPROFILE"); profile && *profile) {
    return profile;
  }
#else
  if (char const *home = std::getenv("HOME"); home && *home) {
    return home;
  }
  if (passwd const *entry = getpwuid(getuid()); entry && entry->pw_dir) {
    return entry->pw_dir;
  }
#endif
  throw std::runtime_error("failed to find the home directory holding the PhysX GPU runtime");
}

} // namespace

std::string PhysxDefault::GetPhysxGpuCacheDirectory() {
  return (homeDirectory() / ".sapien" / "physx" / getPhysxVersion()).string();
}

std::vector<std::string> PhysxDefault::GetPhysxGpuLibraryCandidates() {
  // the layout of the release packages sapien.physx.enable_gpu() downloads
  fs::path dir = GetPhysxGpuCacheDirectory();
#ifdef _WIN32
  return {(dir / "PhysXGpu_64.dll").string(),
          (dir / "physxgpu-windows-vc17win64" / "PhysX" / "bin" / "win.x86_64.vc143.mt" /
           "release" / "PhysXGpu_64.dll")
              .string()};
#else
  return {(dir / "libPhysXGpu_64.so").string(),
          (dir / "physxgpu-linux-clang" / "PhysX" / "bin" / "linux.x86_64" / "release" /
           "libPhysXGpu_64.so")
              .string()};
#endif
}

void PhysxDefault::EnableGPU(std::string const &physxGpuLibrary) {
  // A zombie engine left by sapien.physx.shutdown() must not block a fresh job. Only
  // a live engine means GPU PhysX state may have already been committed elsewhere.
  auto engine = PhysxEngine::GetIfExists();
  if (engine && !engine->isShutdown()) {
    throw std::runtime_error(
        "GPU PhysX can only be enabled once before any other code involving PhysX");
  }

  fs::path library = physxGpuLibrary;
  if (library.empty()) {
    auto candidates = GetPhysxGpuLibraryCandidates();
    auto found = std::find_if(candidates.begin(), candidates.end(),
                              [](std::string const &path) { return fs::is_regular_file(path); });
    if (found == candidates.end()) {
      std::string searched;
      for (auto const &path : candidates) {
        searched += "\n  " + path;
      }
      throw std::runtime_error(
          "failed to enable GPU PhysX: no PhysX GPU runtime found; call "
          "sapien.physx.enable_gpu() once from Python to download it, or pass its path. "
          "Searched:" +
          searched);
    }
    library = *found;
  } else if (!fs::is_regular_file(library)) {
    throw std::runtime_error("failed to enable GPU PhysX: the PhysX GPU runtime " +
                             library.string() + " does not exist");
  }
  library = fs::absolute(library);

#if PX_SUPPORT_GPU_PHYSX
  // PhysX loads the runtime once per process and never unloads it, so a different file would be
  // silently ignored after the first GPU system.
  if (!gPhysxGpuLibraryHook.path.empty() && gPhysxGpuLibraryHook.path != library.string()) {
    throw std::runtime_error("failed to enable GPU PhysX with " + library.string() +
                             ": this process already uses the PhysX GPU runtime " +
                             gPhysxGpuLibraryHook.path);
  }
  gPhysxGpuLibraryHook.path = library.string();
  // PhysX itself loads the CUDA driver (libcuda.so.1 / nvcuda.dll) before this runtime.
  ::PxSetPhysXGpuLoadHook(&gPhysxGpuLibraryHook);
#endif
  gGPUEnabled = true;
}

bool PhysxDefault::HasDefaultMaterial() { return static_cast<bool>(gDefaultMaterial.lock()); }

void PhysxDefault::Reset() {
  gDefaultMaterial.reset();
  gStaticFriction = 0.3f;
  gDynamicFriction = 0.3f;
  gRestitution = 0.1f;
  gGPUEnabled = false;
  gSceneConfig = PhysxSceneConfig{};
  gBodyConfig = PhysxBodyConfig{};
  gShapeConfig = PhysxShapeConfig{};
  gSDFConfig = PhysxSDFShapeConfig{};
  gGpuMemoryConfig = ::physx::PxGpuDynamicsMemoryConfig{};
}

void PhysxDefault::setSDFShapeConfig(float spacing, uint32_t subgridSize,
                                     uint32_t numThreadsForConstruction, uint32_t resolution,
                                     uint32_t bitsPerSubgridPixel, float narrowBandThickness,
                                     float margin, bool enableRemeshing,
                                     float triangleCountReductionFactor) {
  gSDFConfig.spacing = spacing;
  gSDFConfig.subgridSize = subgridSize;
  gSDFConfig.numThreadsForConstruction = numThreadsForConstruction;
  gSDFConfig.resolution = resolution;
  gSDFConfig.bitsPerSubgridPixel = bitsPerSubgridPixel;
  gSDFConfig.narrowBandThickness = narrowBandThickness;
  gSDFConfig.margin = margin;
  gSDFConfig.enableRemeshing = enableRemeshing;
  gSDFConfig.triangleCountReductionFactor = triangleCountReductionFactor;
}
void PhysxDefault::setSDFShapeConfig(PhysxSDFShapeConfig const &c) { gSDFConfig = c; }
PhysxSDFShapeConfig PhysxDefault::getSDFShapeConfig() { return gSDFConfig; }

bool PhysxDefault::GetGPUEnabled() { return gGPUEnabled; }
std::string PhysxDefault::getPhysxVersion() { return PHYSX_VERSION; }

} // namespace physx
} // namespace sapien
