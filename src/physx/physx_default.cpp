#include "sapien/physx/physx_default.h"
#include "./broadphase_env_id.hpp"
#include "sapien/physx/material.h"
#include "sapien/physx/physx_system.h"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace sapien {
namespace physx {

void PhysxSceneConfig::setGpuBroadPhaseEnvIdBits(uint8_t bitsX, uint8_t bitsY, uint8_t bitsZ) {
  for (auto const &[bits, axis] :
       {std::pair{bitsX, 'x'}, std::pair{bitsY, 'y'}, std::pair{bitsZ, 'z'}}) {
    if (bits > kMaxBroadphaseEnvIdBits) {
      throw std::runtime_error(std::string("gpu broadphase env ID bits on axis ") + axis +
                               " must be in [0, " + std::to_string(kMaxBroadphaseEnvIdBits) +
                               "]");
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
