#pragma once
#include <PxFiltering.h>

using namespace physx;
namespace sapien {
namespace physx {

inline constexpr PxU32 kCollisionGroupSceneIdMask = 0xffff0000u;
inline constexpr PxU32 kCollisionGroupIgnoreIdMask = 0x0000ffffu;
inline constexpr PxU32 kCollisionGroupSharedId = 0xffffu;

inline PxU32 getCollisionGroupSceneId(PxFilterData const &filterData) {
  return (filterData.word3 & kCollisionGroupSceneIdMask) >> 16;
}

inline PxU32 getCollisionGroupIgnoreId(PxFilterData const &filterData) {
  return filterData.word3 & kCollisionGroupIgnoreIdMask;
}

inline bool collisionGroupSceneIdsCanCollide(PxFilterData const &filterData0,
                                             PxFilterData const &filterData1) {
  PxU32 sceneId0 = getCollisionGroupSceneId(filterData0);
  PxU32 sceneId1 = getCollisionGroupSceneId(filterData1);
  return sceneId0 == sceneId1 || sceneId0 == kCollisionGroupSharedId ||
         sceneId1 == kCollisionGroupSharedId;
}

inline bool collisionGroupIgnoreIdsMatch(PxFilterData const &filterData0,
                                         PxFilterData const &filterData1) {
  PxU32 ignoreId0 = getCollisionGroupIgnoreId(filterData0);
  PxU32 ignoreId1 = getCollisionGroupIgnoreId(filterData1);
  return ignoreId0 == ignoreId1 && ignoreId0 != kCollisionGroupSharedId;
}

inline PxFilterFlags
TypeAffinityIgnoreFilterShader(PxFilterObjectAttributes attributes0, PxFilterData filterData0,
                               PxFilterObjectAttributes attributes1, PxFilterData filterData1,
                               PxPairFlags &pairFlags, const void *constantBlock,
                               PxU32 constantBlockSize) {

  if (PxFilterObjectIsTrigger(attributes0) || PxFilterObjectIsTrigger(attributes1)) {
    pairFlags = PxPairFlag::eTRIGGER_DEFAULT;
    return PxFilterFlag::eDEFAULT;
  }

  // If the top 16 bits of word3 are different, the shapes will never collide,
  // e.g. they are in different scenes. Scene ID 0xffff is shared and collides
  // with all scene IDs, matching PhysX GPU environment-ID shared semantics.
  if (!collisionGroupSceneIdsCanCollide(filterData0, filterData1)) {
    return PxFilterFlag::eKILL;
  }

  // If the lower 16 bits of word3 are the same (e.g. articulation ID) and word2
  // has a matching bit (e.g. door and frame both set the same bit), the shapes
  // will not collide. Ignore ID 0xffff is shared and does not match any ID.
  if ((filterData0.word2 & filterData1.word2) &&
      collisionGroupIgnoreIdsMatch(filterData0, filterData1)) {
    return PxFilterFlag::eKILL;
  }

  // Otherwise, apply MuJoCo's collision model to word0 and word1
  if ((filterData0.word0 & filterData1.word1) || (filterData1.word0 & filterData0.word1)) {
    pairFlags = PxPairFlag::eCONTACT_DEFAULT | PxPairFlag::eNOTIFY_CONTACT_POINTS |
                PxPairFlag::eNOTIFY_TOUCH_PERSISTS | PxPairFlag::eNOTIFY_TOUCH_FOUND |
                PxPairFlag::eNOTIFY_TOUCH_LOST | PxPairFlag::ePRE_SOLVER_VELOCITY |
                PxPairFlag::ePOST_SOLVER_VELOCITY | PxPairFlag::eDETECT_CCD_CONTACT;

    return PxFilterFlag::eDEFAULT;
  }
  return PxFilterFlag::eKILL;
}

inline PxFilterFlags
TypeAffinityIgnoreFilterShaderGpu(PxFilterObjectAttributes attributes0, PxFilterData filterData0,
                                  PxFilterObjectAttributes attributes1, PxFilterData filterData1,
                                  PxPairFlags &pairFlags, const void *constantBlock,
                                  PxU32 constantBlockSize) {

  if (PxFilterObjectIsTrigger(attributes0) || PxFilterObjectIsTrigger(attributes1)) {
    pairFlags = PxPairFlag::eTRIGGER_DEFAULT;
    return PxFilterFlag::eDEFAULT;
  }

  // If the top 16 bits of word3 are different, the shapes will never collide,
  // e.g. they are in different scenes. Scene ID 0xffff is shared and collides
  // with all scene IDs, matching PhysX GPU environment-ID shared semantics.
  if (!collisionGroupSceneIdsCanCollide(filterData0, filterData1)) {
    return PxFilterFlag::eKILL;
  }

  // If the lower 16 bits of word3 are the same (e.g. articulation ID) and word2
  // has a matching bit (e.g. door and frame both set the same bit), the shapes
  // will not collide. Ignore ID 0xffff is shared and does not match any ID.
  if ((filterData0.word2 & filterData1.word2) &&
      collisionGroupIgnoreIdsMatch(filterData0, filterData1)) {
    return PxFilterFlag::eKILL;
  }

  // Otherwise, apply MuJoCo's collision model to word0 and word1
  if ((filterData0.word0 & filterData1.word1) || (filterData1.word0 & filterData0.word1)) {
    pairFlags = PxPairFlag::eCONTACT_DEFAULT;
    return PxFilterFlag::eDEFAULT;
  }
  return PxFilterFlag::eKILL;
}

} // namespace physx
} // namespace sapien
