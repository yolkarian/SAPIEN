#include "./broadphase_env_id.hpp"
#include <algorithm>
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
  if (widest == 0 || widest > kMaxBroadphaseEnvIdBits) {
    return 0u;
  }
  return 1u << widest;
}

bool broadphaseEnvIdBitsUniform(uint8_t bitsX, uint8_t bitsY, uint8_t bitsZ) {
  // A narrower axis keeps fewer of the shifted ID's low bits, so an ID placed inside the widest
  // axis's safe band wraps back out of it there and stops reaching the shared object. Zero
  // disables an axis outright, which is fine; two different non-zero widths are not.
  uint8_t seen = 0;
  for (uint8_t b : {bitsX, bitsY, bitsZ}) {
    if (b == 0) {
      continue;
    }
    if (seen != 0 && b != seen) {
      return false;
    }
    seen = b;
  }
  return true;
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

} // namespace physx
} // namespace sapien
