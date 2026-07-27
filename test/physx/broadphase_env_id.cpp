// Coverage for the broadphase environment-ID banding math. The wrap path -- environments past
// the usable window riding in the high bits of the ID -- only engages above 64768 shared
// environments, which no Python test can reach at a sane cost, so it is exercised here against
// the helpers directly.
#include "../../src/physx/broadphase_env_id.hpp"
#include <array>
#include <cstdint>
#include <set>
#include <gtest/gtest.h>

using namespace sapien::physx;

namespace {

// gEncodedMinExtent / gEncodedMaxExtent in PhysX
// source/gpubroadphase/src/CUDA/broadphase.cu: the fixed encoded interval PhysX gives objects
// shared by all environments, instead of a banded one.
constexpr uint64_t kSharedEncodedMin = 0x01800000ull;
constexpr uint64_t kSharedEncodedMax = 0xfe7fffffull;

// Mirrors PhysxSystemGpu::getBroadphaseEnvironmentId: the band stays inside the window and the
// count above it rides in the high bits, which PhysX discards when it places the box but
// compares exactly when it filters the pair.
uint64_t effectiveId(uint32_t envId, const BroadphaseEnvIdWindow &window, uint32_t bandCount) {
  const uint32_t band = window.offset + envId % window.capacity;
  const uint32_t wrap = envId / window.capacity;
  return uint64_t{band} + uint64_t{wrap} * bandCount;
}

} // namespace

TEST(BroadphaseEnvId, WindowIsDerivedFromTheWidestAxis) {
  // Every axis shifts the same environment ID, so the narrower ones only keep a subset of the
  // widest one's bits: (8, 0, 0), (0, 0, 8) and (8, 8, 8) must all band identically.
  for (uint8_t bits = 1; bits <= kMaxBroadphaseEnvIdBits; ++bits) {
    const auto reference = broadphaseEnvIdWindow(bits, bits, bits);
    for (auto const &axes : {std::array<uint8_t, 3>{bits, 0, 0},
                             std::array<uint8_t, 3>{0, bits, 0},
                             std::array<uint8_t, 3>{0, 0, bits}}) {
      const auto window = broadphaseEnvIdWindow(axes[0], axes[1], axes[2]);
      EXPECT_EQ(window.offset, reference.offset) << "bits=" << int(bits);
      EXPECT_EQ(window.capacity, reference.capacity) << "bits=" << int(bits);
    }
    EXPECT_EQ(broadphaseEnvBandCount(bits, 0, 0), 1u << bits);
    EXPECT_EQ(broadphaseEnvBandCount(bits, bits, bits), 1u << bits);
  }
}

TEST(BroadphaseEnvId, EveryBandInTheWindowOverlapsTheSharedInterval) {
  // The whole point of the window: a body in one of these bands still generates contacts with
  // an object shared by all environments, such as a ground plane.
  for (uint8_t bits = 2; bits <= kMaxBroadphaseEnvIdBits; ++bits) {
    const auto window = broadphaseEnvIdWindow(bits, bits, bits);
    ASSERT_GT(window.capacity, 0u) << "bits=" << int(bits);

    const uint64_t width = uint64_t{1} << (32 - bits);
    for (uint32_t envId : {0u, window.capacity / 2, window.capacity - 1}) {
      const uint64_t bandLow = uint64_t{window.offset + envId} * width;
      const uint64_t bandHigh = bandLow + width - 1;
      EXPECT_GE(bandLow, kSharedEncodedMin) << "bits=" << int(bits) << " env=" << envId;
      EXPECT_LE(bandHigh, kSharedEncodedMax) << "bits=" << int(bits) << " env=" << envId;
    }

    // The bands immediately outside are exactly the ones that do not fit.
    if (window.offset > 0) {
      const uint64_t belowLow = uint64_t{window.offset - 1} * width;
      EXPECT_LT(belowLow, kSharedEncodedMin);
    }
    const uint64_t aboveHigh = uint64_t{window.offset + window.capacity} * width + width - 1;
    EXPECT_GT(aboveHigh, kSharedEncodedMax);
  }
}

TEST(BroadphaseEnvId, OneBitLeavesNoUsableBand) {
  // Two bands, and the boundary between them falls inside the shared interval, so neither is
  // wholly contained. PhysxSystemGpu rejects this at construction.
  EXPECT_EQ(broadphaseEnvIdWindow(1, 1, 1).capacity, 0u);
}

TEST(BroadphaseEnvId, WrappedEnvironmentsStayDistinctAndInBand) {
  // Past the window the count rides in the high bits. PhysX discards those when it places the
  // box -- so the body still lands in a band that reaches shared objects -- but compares the
  // full 32-bit value when it filters, so the environments stay separate.
  constexpr uint8_t bits = 4;
  const auto window = broadphaseEnvIdWindow(bits, bits, bits);
  const uint32_t bandCount = broadphaseEnvBandCount(bits, bits, bits);
  ASSERT_GT(window.capacity, 0u);
  ASSERT_LT(window.capacity, bandCount) << "the window must be narrower than the band range";

  std::set<uint64_t> seen;
  const uint32_t total = window.capacity * 3 + 1;
  for (uint32_t envId = 0; envId < total; ++envId) {
    const uint64_t effective = effectiveId(envId, window, bandCount);
    EXPECT_TRUE(seen.insert(effective).second) << "collision at env=" << envId;

    // The low `bits` -- the part PhysX bands on -- must land inside the window.
    const uint32_t band = static_cast<uint32_t>(effective) & (bandCount - 1);
    EXPECT_GE(band, window.offset) << "env=" << envId;
    EXPECT_LT(band, window.offset + window.capacity) << "env=" << envId;
  }

  // The first environment past the window is the boundary case: same band as environment 0,
  // one band range higher.
  EXPECT_EQ(effectiveId(0, window, bandCount) + bandCount,
            effectiveId(window.capacity, window, bandCount));
}

TEST(BroadphaseEnvId, BitsForEnvCountGivesEveryEnvironmentAPrivateBand) {
  for (uint32_t count : {1u, 2u, 15u, 16u, 17u, 4096u}) {
    const uint8_t shared = broadphaseEnvIdBitsForEnvCount(count, true);
    ASSERT_GT(shared, 0u) << "count=" << count;
    EXPECT_GE(broadphaseEnvIdWindow(shared, shared, shared).capacity, count) << "count=" << count;
    // One bit narrower would not have held them, so the sizing is not wasteful.
    if (shared > 1) {
      const uint8_t narrower = shared - 1;
      EXPECT_LT(broadphaseEnvIdWindow(narrower, narrower, narrower).capacity, count)
          << "count=" << count;
    }

    const uint8_t plain = broadphaseEnvIdBitsForEnvCount(count, false);
    ASSERT_GT(plain, 0u) << "count=" << count;
    EXPECT_GE(1u << plain, count) << "count=" << count;
    // One bit is the search floor, so minimality only has meaning above it.
    if (plain > 1) {
      EXPECT_LT(1u << (plain - 1), count) << "count=" << count;
    }
  }
}

TEST(BroadphaseEnvId, CountsPastTheAddressableTotalAreRejected) {
  const uint32_t maxShared = maxBroadphaseEnvCount(true);
  EXPECT_GT(broadphaseEnvIdBitsForEnvCount(maxShared, true), 0u);
  EXPECT_EQ(broadphaseEnvIdBitsForEnvCount(maxShared + 1, true), 0u);
}

TEST(BroadphaseEnvId, UniformityRejectsMixedNonZeroCounts) {
  // A narrower axis keeps fewer of the shifted ID's low bits, so an environment inside the
  // widest axis's safe band wraps back out of it there and loses the shared object.
  EXPECT_TRUE(broadphaseEnvIdBitsUniform(0, 0, 0));
  EXPECT_TRUE(broadphaseEnvIdBitsUniform(0, 0, 5));
  EXPECT_TRUE(broadphaseEnvIdBitsUniform(5, 5, 5));
  EXPECT_TRUE(broadphaseEnvIdBitsUniform(5, 0, 5));
  EXPECT_FALSE(broadphaseEnvIdBitsUniform(12, 8, 0));
  EXPECT_FALSE(broadphaseEnvIdBitsUniform(2, 0, 5));
}
