#pragma once
// Internal helpers behind the declarative GPU broadphase environment-ID config. Not part of the
// public C++ API: callers declare `numScenes` / `withSharedScene` on PhysxSceneConfig and SAPIEN
// derives everything below.
#include <cstdint>

namespace sapien {
namespace physx {

/** Largest per-axis environment-ID bit count PhysX accepts. */
constexpr uint8_t kMaxBroadphaseEnvIdBits = 16;

/** Largest environment ID SAPIEN accepts for a scene. */
constexpr uint32_t kMaxSceneEnvironmentId = 1u << 24;

/** Largest number of ordinary environments a shared scene can coexist with. SAPIEN's own filter
 *  shader keeps the scene ID in 16 bits of a collision group word and reserves 0xffff for the
 *  shared scene itself. */
constexpr uint32_t kMaxSharedSceneEnvCount = 0xffffu;

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
 *  banded, so it alone decides which bands reach the shared interval. Callers must have already
 *  rejected mixed non-zero counts, which would wrap a narrower axis out of that band. All-zero
 *  bits disable the banding, which makes the whole ID range usable. */
BroadphaseEnvIdWindow broadphaseEnvIdWindow(uint8_t bitsX, uint8_t bitsY, uint8_t bitsZ);

/** How many environments land in distinct broadphase bands.
 *
 *  PhysX shifts the same environment ID on every axis, so each axis keeps the low bits of one
 *  value: the widest axis alone decides the count, which is `2**max(x, y, z)` rather than the
 *  product. Environments beyond it share bands, which costs broadphase spreading but stays
 *  correct, because PhysX filters candidate pairs on the full environment ID. */
uint32_t broadphaseEnvBandCount(uint8_t bitsX, uint8_t bitsY, uint8_t bitsZ);

/** Whether a bit configuration is safe to pair with a shared scene: at most one distinct
 *  non-zero count. A narrower axis keeps fewer of the shifted ID's low bits, so an ID inside the
 *  widest axis's safe band can wrap back out of it on that axis and lose the shared object. */
bool broadphaseEnvIdBitsUniform(uint8_t bitsX, uint8_t bitsY, uint8_t bitsZ);

/** The smallest uniform per-axis bit count that gives `envCount` environments a private band,
 *  falling back to `kMaxBroadphaseEnvIdBits` when they only fit by wrapping, or 0 past
 *  `maxBroadphaseEnvCount`. */
uint8_t broadphaseEnvIdBitsForEnvCount(uint32_t envCount, bool withSharedObjects = true);

/** How many environments the banding can address in total.
 *
 *  Only the band -- the low `bits` of the ID -- has to stay inside the usable window; PhysX
 *  discards the higher bits when it places the box but compares the full 32-bit value when it
 *  filters the pair. So environments past the window ride above it and the reachable total is
 *  the window scaled by how many band ranges fit under `kMaxSceneEnvironmentId`. */
uint32_t maxBroadphaseEnvCount(bool withSharedObjects = true);

} // namespace physx
} // namespace sapien
