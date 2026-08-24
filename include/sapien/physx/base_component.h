#pragma once

#include "../component.h"
#include "./physx_engine.h"
#include <memory>

namespace sapien {
namespace physx {
class PhysxEngine;

class PhysxBaseComponent : public Component {
public:
  PhysxBaseComponent();

protected:
  std::shared_ptr<PhysxEngine> mEngine;

private:
  /** Live-object accounting for the job-scope shutdown preflight. */
  PhysxLiveObjectGuard mLiveGuard;
};

} // namespace physx
} // namespace sapien
