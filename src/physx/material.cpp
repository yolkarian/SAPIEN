#include "sapien/physx/material.h"
#include "sapien/physx/physx_system.h"

using namespace physx;
namespace sapien::physx {

PhysxMaterial::PhysxMaterial(float staticFriction, float dynamicFriction, float restitution)
    : mEngine(PhysxEngine::Get()) {
  mMaterial =
      mEngine->getPxPhysics()->createMaterial(staticFriction, dynamicFriction, restitution);
}

PhysxMaterial::~PhysxMaterial() { mMaterial->release(); }

} // namespace sapien::physx
