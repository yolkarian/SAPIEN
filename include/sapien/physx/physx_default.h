#pragma once
#include "sapien/math/vec3.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace physx {
struct PxGpuDynamicsMemoryConfig;
};

namespace sapien {
namespace physx {
class PhysxMaterial;

/** Default per-axis environment-ID bit count, on Z alone. Four matches PhysX's own "snap to
 *  grid" shift, so the banding costs no coordinate precision at all. */
constexpr uint8_t kDefaultBroadphaseEnvIdBits = 4;

struct PhysxSceneConfig {
  Vec3 gravity = {0, 0, -9.81};           // default gravity
  float bounceThreshold = 2.f;            // relative velocity below this will not bounce
  bool enablePCM = true;                  // Use persistent contact manifold solver for contact
  bool enableTGS = true;                  // use TGS solver
  bool enableCCD = false;                 // use continuous collision detection
  bool enableEnhancedDeterminism = false; // improve determinism
  bool enableFrictionEveryIteration =
      true;                // better friction calculation, recommended for robotics
  float frictionOffsetThreshold = 0.04f;   
  float frictionCorrelationDistance = 0.025f;
  uint32_t cpuWorkers = 0; // CPU workers, 0 for using main thread

  /** How many ordinary, non-shared scenes the simulation will hold. When set, SAPIEN assigns
   *  each scene a unique environment ID, refuses to hand out more than this many, and derives
   *  the broadphase bit counts, ignoring any explicit ones. Unset, SAPIEN assigns nothing:
   *  every scene is environment 0 unless separated with `setSceneOffset`. Must be positive. */
  std::optional<uint32_t> numScenes{};

  /** Whether the simulation has a shared scene, such as one holding a ground plane, that every
   *  environment must collide with. PhysX gives shared objects a fixed encoded interval that the
   *  outermost bands fall outside of, so SAPIEN shifts every environment ID into the bands that
   *  still overlap it and stamps the ID into the collision-group scene field for its own filter
   *  shader. */
  bool withSharedScene = false;

  /** GPU broadphase environment ID bits, per axis. When non-zero on an axis, the environment ID
   *  is merged into that axis's broadphase bounds, spreading environments apart so sweep-and-
   *  prune has fewer candidate pairs to reject. Only used when broadPhaseType is eGPU.
   *
   *  Read-only: the bits are written through `setGpuBroadPhaseEnvIdBits`, or derived from
   *  `numScenes`. Read them to see what the system will use. */
  uint8_t getGpuBroadPhaseNbBitsEnvIDX() const { return mBitsEnvIDX; }
  uint8_t getGpuBroadPhaseNbBitsEnvIDY() const { return mBitsEnvIDY; }
  uint8_t getGpuBroadPhaseNbBitsEnvIDZ() const { return mBitsEnvIDZ; }

  /** The only way to write the bit counts. All three must be in [0, 16].
   *
   *  The axes are not independent: PhysX shifts the same environment ID on each of them
   *  (`envId << (32 - bits_axis)`), so every axis keeps the *low* bits of one value and the
   *  distinct band count is `2**max(x, y, z)`, never the product. Raising a count also raises
   *  PhysX's "snap to grid" shift on that axis, spending coordinate precision, so anything above
   *  the default is only worth it on axes that can afford it. With a shared scene the non-zero
   *  counts must all be equal, because a narrower axis wraps the shifted ID back out of the
   *  band that reaches the shared object. */
  void setGpuBroadPhaseEnvIdBits(uint8_t bitsX, uint8_t bitsY, uint8_t bitsZ);

private:
  // Written only by setGpuBroadPhaseEnvIdBits, which PhysxSystemGpu also calls to install the
  // counts it derives from `numScenes`.

  uint8_t mBitsEnvIDX = 0;
  uint8_t mBitsEnvIDY = 0;
  uint8_t mBitsEnvIDZ = kDefaultBroadphaseEnvIdBits;
};

struct PhysxBodyConfig {
  uint32_t solverPositionIterations = 10; // solver position iterations, helps reduce jittering
  uint32_t solverVelocityIterations = 1;  // solver velocity iterations
  float sleepThreshold = 0.005f;          // put to sleep if (kinetic energy/(mass) falls below
};

struct PhysxShapeConfig {
  float contactOffset = 0.01f; // how close should contacts be generated
  float restOffset = 0.f;
};

struct PhysxSDFShapeConfig {
  float spacing = 0.01f;
  uint32_t subgridSize = 6;
  uint32_t numThreadsForConstruction = 4;
  uint32_t resolution = 0;
  uint32_t bitsPerSubgridPixel = 16;
  float narrowBandThickness = 0.01f;
  float margin = 0.f; // currently ignored by SAPIEN's plain PhysX SDF cooking path
  bool enableRemeshing =
      false; // currently ignored by SAPIEN's plain PhysX SDF cooking path
  float triangleCountReductionFactor =
      1.f; // currently ignored by SAPIEN's plain PhysX SDF cooking path
};

class PhysxDefault {
public:
  static std::shared_ptr<PhysxMaterial> GetDefaultMaterial();
  static void SetDefaultMaterial(float staticFriction, float dynamicFriction, float restitution);
  static void setGpuMemoryConfig(uint64_t tempBufferCapacity, uint32_t maxRigidContactCount,
                                 uint32_t maxRigidPatchCount, uint32_t heapCapacity,
                                 uint32_t foundLostPairsCapacity,
                                 uint32_t foundLostAggregatePairsCapacity,
                                 uint32_t totalAggregatePairsCapacity,
                                 uint32_t collisionStackSize);
  static ::physx::PxGpuDynamicsMemoryConfig const &getGpuMemoryConfig();

  static void setSceneConfig(Vec3 gravity, float bounceThreshold, bool enablePCM, bool enableTGS,
                             bool enableCCD, bool enableEnhancedDeterminism,
                             bool enableFrictionEveryIteration, float frictionOffsetThreshold,
                             float frictionCorrelationDistance, uint32_t cpuWorkers);
  static void setSceneConfig(PhysxSceneConfig const &);
  static PhysxSceneConfig const &getSceneConfig();

  static void setBodyConfig(uint32_t solverIterations, uint32_t solverVelocityIterations,
                            float sleepThreshold);
  static void setBodyConfig(PhysxBodyConfig const &);
  static PhysxBodyConfig const &getBodyConfig();

  static void setShapeConfig(float contactOffset, float restOffset);
  static void setShapeConfig(PhysxShapeConfig const &);
  static PhysxShapeConfig const &getShapeConfig();

  static void setSDFShapeConfig(float spacing = 0.01f, uint32_t subgridSize = 6,
                                uint32_t numThreadsForConstruction = 4,
                                uint32_t resolution = 0,
                                uint32_t bitsPerSubgridPixel = 16,
                                float narrowBandThickness = 0.01f, float margin = 0.f,
                                bool enableRemeshing = false,
                                float triangleCountReductionFactor = 1.f);
  static void setSDFShapeConfig(PhysxSDFShapeConfig const &);
  static PhysxSDFShapeConfig getSDFShapeConfig();

  // enable GPU simulation, may not be disabled
  static void EnableGPU();
  static bool GetGPUEnabled();

  static std::string getPhysxVersion();
};

} // namespace physx
} // namespace sapien
