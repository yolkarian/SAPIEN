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


/** Largest per-axis environment-ID bit count PhysX accepts. */
constexpr uint8_t kMaxBroadphaseEnvIdBits = 16;

/** Largest environment ID SAPIEN accepts for a scene. */
constexpr uint32_t kMaxSceneEnvironmentId = 1u << 24;

/** Default per-axis environment-ID bit count. Five is the widest that needs no shifting: every
 *  band it produces still reaches the fixed encoded interval PhysX gives shared objects, so the
 *  out-of-the-box configuration spreads environments apart without reserving anything. */
constexpr uint8_t kDefaultBroadphaseEnvIdBits = 5;
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

  // GPU broadphase environment ID bits, per axis. When non-zero on an axis, the environment ID
  // is merged into that axis's broadphase bounds, spreading environments apart so sweep-and-
  // prune has fewer candidate pairs to reject. Only used when broadPhaseType is eGPU.
  //
  // The axes are NOT independent: PhysX shifts the same environment ID on each of them
  // (`envId << (32 - bits_axis)`), so every axis keeps the *low* bits of one value and the
  // distinct band count is 2**max(x, y, z), never the product. Spreading bits across axes
  // therefore buys nothing and costs coordinate precision on each one, because SAPIEN has to
  // raise PhysX's "snap to grid" shift to match. The default puts them on Z alone, leaving X
  // and Y at PhysX's own shift; move them to another axis if that one matters least to you.
  uint8_t gpuBroadPhaseNbBitsEnvIDX = 0;
  uint8_t gpuBroadPhaseNbBitsEnvIDY = 0;
  uint8_t gpuBroadPhaseNbBitsEnvIDZ = kDefaultBroadphaseEnvIdBits;

  /** How many scenes the system will hold. When set, SAPIEN sizes the bit count from it, hands
   *  unset scenes a fresh unique environment ID, and derives what `PxActor::setEnvironmentID`
   *  receives. Unset, SAPIEN invents nothing: an untouched scene is environment 0 and the IDs
   *  reach PhysX exactly as given. */
  std::optional<uint32_t> gpuBroadPhaseNumScenes{};

  /** Whether a scene will be marked shared (environment ID -1), such as one holding a ground
   *  plane. PhysX gives shared objects a fixed encoded interval that the outermost bands fall
   *  outside of, so SAPIEN offsets every ID into the bands that still overlap it and stamps the
   *  raw ID into the collision-group scene field for its own filter shader. */
  bool gpuBroadPhaseWithSharedScene = false;

  /** Put `bits` on Z alone and clear X and Y -- the layout that costs the least coordinate
   *  precision for a given band count. Use the per-axis fields directly for any other layout. */
  void setGpuBroadPhaseEnvIdBits(uint8_t bits);

  /** Give every one of `envCount` environments its own broadphase band, when one is available,
   *  by sizing the Z-axis bit count. `withSharedObjects` reserves the bands PhysX makes
   *  unusable, about 1.2% of them, so a count that exactly fits is one bit too small.
   *
   *  A private band is a broadphase-spreading optimisation, not a correctness requirement:
   *  environments past the window wrap into the high bits of the ID, which PhysX still filters
   *  on exactly. Throws only past `maxBroadphaseEnvCount`. */
  void setGpuBroadPhaseEnvCount(uint32_t envCount, bool withSharedObjects = true);
};

/** The range of environment IDs whose broadphase band still overlaps shared objects.
 *
 *  PhysX relocates environment `e` into the encoded band `[e << (32 - b), (e + 1) << (32 - b))`
 *  but gives objects shared by all environments (`PX_INVALID_U32`) a fixed encoded interval that
 *  does not follow the banding. The bands at either end of the range fall outside it, and bodies
 *  there silently stop colliding with shared objects. `offset` is the first band that overlaps
 *  it, `capacity` how many consecutive environments fit above.
 *
 *  Only applied when a shared scene is declared, since without one there is no fixed interval
 *  to miss. */
struct BroadphaseEnvIdWindow {
  uint32_t offset{0};
  uint32_t capacity{0};
};

/** Compute the usable environment-ID window for a per-axis bit configuration.
 *
 *  Derived from `max(bitsX, bitsY, bitsZ)`: that axis alone decides how the environments are
 *  banded, so it alone decides which bands reach the shared interval. All-zero bits disable the
 *  banding, which makes the whole ID range usable. */
BroadphaseEnvIdWindow broadphaseEnvIdWindow(uint8_t bitsX, uint8_t bitsY, uint8_t bitsZ);

/** How many environments land in distinct broadphase bands.
 *
 *  PhysX shifts the same environment ID on every axis, so each axis keeps the low bits of one
 *  value: the widest axis alone decides the count, which is `2**max(x, y, z)` rather than the
 *  product. Environments beyond it share bands, which costs broadphase spreading but stays
 *  correct, because PhysX filters candidate pairs on the full environment ID. */
uint32_t broadphaseEnvBandCount(uint8_t bitsX, uint8_t bitsY, uint8_t bitsZ);

/** The smallest uniform per-axis bit count that gives `envCount` environments a private band,
 *  falling back to `kMaxBroadphaseEnvIdBits` when they only fit by wrapping, or 0 past
 *  `maxBroadphaseEnvCount`. See setGpuBroadPhaseEnvCount for `withSharedObjects`. */
uint8_t broadphaseEnvIdBitsForEnvCount(uint32_t envCount, bool withSharedObjects = true);

/** How many environments the banding can address in total.
 *
 *  Only the band -- the low `bits` of the ID -- has to stay inside the usable window; PhysX
 *  discards the higher bits when it places the box but compares the full 32-bit value when it
 *  filters the pair. So environments past the window ride above it and the reachable total is
 *  the window scaled by how many band ranges fit under `kMaxSceneEnvironmentId`. */
uint32_t maxBroadphaseEnvCount(bool withSharedObjects = true);

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
