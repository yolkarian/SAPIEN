#pragma once
#include "./physx_engine.h"
#include <PxPhysicsAPI.h>
#include <memory>

namespace sapien {
namespace physx {
class PhysxEngine;

class PhysxMaterial : public std::enable_shared_from_this<PhysxMaterial> {
public:
  PhysxMaterial() : PhysxMaterial(0.f, 0.f, 0.f) {}
  PhysxMaterial(float staticFriction, float dynamicFriction, float restitution);

  inline ::physx::PxMaterial *getPxMaterial() const { return mMaterial; };

  inline float getStaticFriction() const { return mMaterial->getStaticFriction(); }
  inline float getDynamicFriction() const { return mMaterial->getDynamicFriction(); }
  inline float getRestitution() const { return mMaterial->getRestitution(); }

  inline void setStaticFriction(float coef) const { mMaterial->setStaticFriction(coef); }
  inline void setDynamicFriction(float coef) const { mMaterial->setDynamicFriction(coef); }
  inline void setRestitution(float coef) const { mMaterial->setRestitution(coef); }

  /** Contact pairs select the higher-priority mode: average < min < multiply < max.
   * Friction applies to both static and dynamic coefficients; restitution is independent.
   * Mutations affect all shapes sharing this material. Modify only between simulation steps;
   * on GPU, configure before gpu_init() (later propagation is not guaranteed).
   * Only the four valid PxCombineMode values may be supplied.
   */
  inline ::physx::PxCombineMode::Enum getFrictionCombineMode() const {
    return mMaterial->getFrictionCombineMode();
  }
  inline void setFrictionCombineMode(::physx::PxCombineMode::Enum mode) const {
    mMaterial->setFrictionCombineMode(mode);
  }
  inline ::physx::PxCombineMode::Enum getRestitutionCombineMode() const {
    return mMaterial->getRestitutionCombineMode();
  }
  inline void setRestitutionCombineMode(::physx::PxCombineMode::Enum mode) const {
    mMaterial->setRestitutionCombineMode(mode);
  }

  PhysxMaterial(PhysxMaterial const &other) = delete;
  PhysxMaterial &operator=(PhysxMaterial const &other) = delete;
  PhysxMaterial(PhysxMaterial &&other) = default;
  PhysxMaterial &operator=(PhysxMaterial &&other) = default;

  ~PhysxMaterial();

private:
  PhysxLiveObjectGuard mLiveGuard;
  std::shared_ptr<PhysxEngine> mEngine;
  ::physx::PxMaterial *mMaterial;
};

} // namespace physx
} // namespace sapien
