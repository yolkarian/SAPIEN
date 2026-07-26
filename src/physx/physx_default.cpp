#include "sapien/physx/physx_default.h"
#include "sapien/physx/material.h"
#include "sapien/physx/physx_system.h"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace sapien {
namespace physx {

namespace {
// PhysX gives objects shared by all environments (PX_INVALID_U32) this fixed encoded interval
// instead of a banded one, so bands outside it never collide with them. The constants are
// gEncodedMinExtent / gEncodedMaxExtent in PhysX source/gpubroadphase/src/CUDA/broadphase.cu.
constexpr uint32_t kSharedEncodedMin = 0x01800000u;
constexpr uint32_t kSharedEncodedMax = 0xfe7fffffu;

// Bands on one axis whose whole range overlaps the shared interval. The coordinate part fills a
// band from below (`base | (encodeFloat(coord) >> bits)`), so requiring the band itself to
// overlap keeps the environment safe for any coordinate.
std::pair<uint32_t, uint32_t> safeBandRange(uint8_t bits) {
  const uint64_t width = uint64_t{1} << (32 - bits);
  const uint32_t first = static_cast<uint32_t>((kSharedEncodedMin + width - 1) / width);
  const uint32_t last = static_cast<uint32_t>((uint64_t{kSharedEncodedMax} + 1) / width - 1);
  return {first, last};
}
} // namespace

BroadphaseEnvIdWindow broadphaseEnvIdWindow(uint8_t bitsX, uint8_t bitsY, uint8_t bitsZ) {
  // The widest axis alone decides the banding -- every axis shifts the same environment ID, so
  // the narrower ones only keep a subset of the bits the widest one keeps -- and therefore
  // alone decides which bands still reach the shared interval.
  const uint8_t bits = std::max({bitsX, bitsY, bitsZ});
  if (bits == 0 || bits > kMaxBroadphaseEnvIdBits) {
    return {0, kMaxSceneEnvironmentId};
  }
  auto const [first, last] = safeBandRange(bits);
  return {first, last < first ? 0u : last - first + 1};
}

uint32_t broadphaseEnvBandCount(uint8_t bitsX, uint8_t bitsY, uint8_t bitsZ) {
  // Every axis shifts the same environment ID, so each keeps the low bits of one value and the
  // widest axis alone decides how many environments land in distinct bands.
  const uint8_t widest = std::max({bitsX, bitsY, bitsZ});
  return widest == 0 ? 0u : 1u << widest;
}

uint8_t broadphaseEnvIdBitsForEnvCount(uint32_t envCount, bool withSharedObjects) {
  // Pick the width that gives every environment its own band. Wider is not needed for
  // correctness -- environments past the window wrap into the high bits of the ID, which PhysX
  // filters on exactly -- but a private band is free and keeps the broadphase spread out.
  for (uint8_t bits = 1; bits <= kMaxBroadphaseEnvIdBits; ++bits) {
    const uint32_t usable =
        withSharedObjects ? broadphaseEnvIdWindow(bits, bits, bits).capacity : 1u << bits;
    if (usable >= envCount) {
      return bits;
    }
  }
  return envCount <= maxBroadphaseEnvCount(withSharedObjects) ? kMaxBroadphaseEnvIdBits : 0;
}

uint32_t maxBroadphaseEnvCount(bool withSharedObjects) {
  constexpr uint8_t b = kMaxBroadphaseEnvIdBits;
  if (!withSharedObjects) {
    return kMaxSceneEnvironmentId;
  }
  // Only the band -- the low `b` bits -- has to stay inside the window. The rest of the count
  // rides above it, so the reachable total is the window scaled by how many times a full band
  // range fits below the largest ID PhysX may be handed.
  const uint64_t capacity = broadphaseEnvIdWindow(b, b, b).capacity;
  return static_cast<uint32_t>(capacity * (uint64_t{kMaxSceneEnvironmentId} / (uint64_t{1} << b)));
}

void PhysxSceneConfig::setGpuBroadPhaseEnvIdBits(uint8_t bits) {
  if (bits > kMaxBroadphaseEnvIdBits) {
    throw std::runtime_error("gpu_broadphase_env_id_bits must be in [0, " +
                             std::to_string(kMaxBroadphaseEnvIdBits) + "]");
  }
  // Z alone: the band count is 2**max(x, y, z) either way, so spreading the bits would only
  // cost coordinate precision on the extra axes without buying a single band.
  gpuBroadPhaseNbBitsEnvIDX = 0;
  gpuBroadPhaseNbBitsEnvIDY = 0;
  gpuBroadPhaseNbBitsEnvIDZ = bits;
}

void PhysxSceneConfig::setGpuBroadPhaseEnvCount(uint32_t envCount, bool withSharedObjects) {
  if (envCount == 0) {
    setGpuBroadPhaseEnvIdBits(0);
    return;
  }
  const uint8_t bits = broadphaseEnvIdBitsForEnvCount(envCount, withSharedObjects);
  if (bits == 0) {
    throw std::runtime_error(
        "failed to configure GPU broadphase environment IDs: " + std::to_string(envCount) +
        " environments exceed the maximum of " +
        std::to_string(maxBroadphaseEnvCount(withSharedObjects)) +
        (withSharedObjects ? ". PhysX reserves the outermost bands for objects shared by all "
                             "environments; pass with_shared_objects=False if this simulation "
                             "has none."
                           : "."));
  }
  setGpuBroadPhaseEnvIdBits(bits);
}

static float gStaticFriction{0.3};
static float gDynamicFriction{0.3};
static float gRestitution{0.1};
static std::weak_ptr<PhysxMaterial> gDefaultMaterial;
static bool gGPUEnabled{false};
static PhysxSceneConfig gSceneConfig{};
static PhysxBodyConfig gBodyConfig{};
static PhysxShapeConfig gShapeConfig{};
static PhysxSDFShapeConfig gSDFConfig{};

static ::physx::PxGpuDynamicsMemoryConfig gGpuMemoryConfig{};

void PhysxDefault::SetDefaultMaterial(float staticFriction, float dynamicFriction,
                                      float restitution) {
  gDefaultMaterial.reset();
  gStaticFriction = staticFriction;
  gDynamicFriction = dynamicFriction;
  gRestitution = restitution;
}

std::shared_ptr<PhysxMaterial> PhysxDefault::GetDefaultMaterial() {
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

void PhysxDefault::EnableGPU() {
  if (PhysxEngine::GetIfExists()) {
    throw std::runtime_error(
        "GPU PhysX can only be enabled once before any other code involving PhysX");
  }
  gGPUEnabled = true;
}

void PhysxDefault::setSDFShapeConfig(float spacing, uint32_t subgridSize,
                                     uint32_t numThreadsForConstruction,
                                     uint32_t resolution, uint32_t bitsPerSubgridPixel,
                                     float narrowBandThickness, float margin,
                                     bool enableRemeshing,
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
