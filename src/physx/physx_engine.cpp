#include "sapien/physx/physx_engine.h"
#include "../logger.h"
#include "sapien/physx/physx_default.h"
#include <algorithm>
#include <utility>

#ifdef SAPIEN_CUDA
#include "../utils/cuda_lib.h"
#include "sapien/utils/cuda.h"
#endif

using namespace physx;

namespace sapien {
namespace physx {

static PxDefaultAllocator gDefaultAllocatorCallback;

class SapienErrorCallback : public PxErrorCallback {
  PxErrorCode::Enum mLastErrorCode = PxErrorCode::eNO_ERROR;

public:
  void reportError(PxErrorCode::Enum code, const char *message, const char *file,
                   int line) override {
    mLastErrorCode = code;

#ifdef NDEBUG
    logger::critical("{}", message);
#else
    logger::critical("{}:{}: {}", file, line, message);
#endif
  }
  PxErrorCode::Enum getLastErrorCode() {
    auto code = mLastErrorCode;
    mLastErrorCode = PxErrorCode::eNO_ERROR;
    return code;
  }
};

static SapienErrorCallback gDefaultErrorCallback;

static std::weak_ptr<PhysxEngine> gEngine;
std::shared_ptr<PhysxEngine> PhysxEngine::Get(float toleranceLength, float toleranceSpeed) {
  auto engine = gEngine.lock();
  if (engine && engine->isShutdown()) {
    // A previous engine was shut down under sapien.physx.shutdown() but an external
    // shared_ptr kept the object alive. Replace the registry entry so future code gets
    // a fresh engine; the zombie refuses all operations through its accessors.
    engine.reset();
    gEngine.reset();
  }
  if (!engine) {
    gEngine = engine = std::make_shared<PhysxEngine>(toleranceLength, toleranceSpeed);
  }
  return engine;
}

std::shared_ptr<PhysxEngine> PhysxEngine::GetIfExists() { return gEngine.lock(); }

::physx::PxPhysics *PhysxEngine::getPxPhysics() const {
  if (mShutdown) {
    throw std::runtime_error("failed to use PhysX engine: the engine was shut down");
  }
  return mPxPhysics;
}

PhysxEngine::PhysxEngine(float toleranceLength, float toleranceSpeed) {
  logger::getLogger();

  mPxFoundation =
      PxCreateFoundation(PX_PHYSICS_VERSION, gDefaultAllocatorCallback, gDefaultErrorCallback);
  if (!mPxFoundation) {
    throw std::runtime_error("PhysX foundation creation failed");
  }

  PxTolerancesScale toleranceScale(toleranceLength, toleranceSpeed);

  mPxPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *mPxFoundation, toleranceScale);
  if (!mPxPhysics) {
    throw std::runtime_error("PhysX creation failed");
  }
  if (!PxInitExtensions(*mPxPhysics, nullptr)) {
    throw std::runtime_error("PhysX extension initialization failed");
  }
}

::physx::PxCudaContextManager *PhysxEngine::getCudaContextManager(int cudaId) {
  if (!PhysxDefault::GetGPUEnabled()) {
    throw std::runtime_error("Using CUDA is not allowed when PhysX GPU is not enabled.");
  }
  if (mShutdown) {
    throw std::runtime_error(
        "failed to get PhysX CUDA context manager: the PhysX engine was shut down");
  }
#ifdef SAPIEN_CUDA
  std::lock_guard lock(mCudaContextManagerMutex);
  if (auto it = mCudaContextManagers.find(cudaId); it != mCudaContextManagers.end()) {
    return it->second;
  }

  // One PxCudaContextManager per device, each wrapping that device's primary context (the one
  // the CUDA runtime, PyTorch, CuPy and JAX share), like omni.physx's scene multi-GPU mode, so
  // GPU systems on different devices coexist in one process. PhysX reads the device from the
  // provided context, which must therefore be current while the manager is created; the scope
  // restores the caller's device afterwards. PhysX never destroys a provided context.
  CudaDeviceScope deviceScope(cudaId);
  CUcontext context = cudaPrimaryContext(cudaId);
  PxCudaContextManagerDesc cudaContextManagerDesc;
  cudaContextManagerDesc.ctx = &context;
  auto manager =
      PxCreateCudaContextManager(*mPxFoundation, cudaContextManagerDesc, PxGetProfilerCallback());
  if (!manager || !manager->contextIsValid()) {
    if (manager) {
      manager->release();
    }
    throw std::runtime_error("failed to create PhysX CUDA context manager on cuda:" +
                             std::to_string(cudaId));
  }
  mCudaContextManagers[cudaId] = manager;
  return manager;
#else
  return nullptr;
#endif
}

void PhysxEngine::releaseCudaContextManagers() {
  std::lock_guard lock(mCudaContextManagerMutex);
  for (auto const &[cudaId, manager] : mCudaContextManagers) {
    manager->release();
  }
  mCudaContextManagers.clear();
}

void PhysxEngine::shutdown() {
  if (mShutdown) {
    return;
  }
  if (liveSystemCount() != 0) {
    throw std::runtime_error("failed to shut down PhysX engine: " +
                             std::to_string(liveSystemCount()) + " systems are still open");
  }
  if (liveObjectCount() != 0) {
    throw std::runtime_error(
        "failed to shut down PhysX engine: " + std::to_string(liveObjectCount()) +
        " PhysX-backed objects are still alive");
  }
  // No system is left, so no PxScene uses a manager any more.
  releaseCudaContextManagers();
  PxCloseExtensions();
  if (mPxPhysics) {
    mPxPhysics->release();
    mPxPhysics = nullptr;
  }
  if (mPxFoundation) {
    mPxFoundation->release();
    mPxFoundation = nullptr;
  }
  mShutdown = true;
}

void PhysxEngine::registerSystem() { mLiveSystems.fetch_add(1, std::memory_order_acq_rel); }
void PhysxEngine::unregisterSystem() { mLiveSystems.fetch_sub(1, std::memory_order_acq_rel); }
void PhysxEngine::registerGpuSystem() { mLiveGpuSystems.fetch_add(1, std::memory_order_acq_rel); }
void PhysxEngine::unregisterGpuSystem() {
  mLiveGpuSystems.fetch_sub(1, std::memory_order_acq_rel);
}
void PhysxEngine::noteObjectCreated() { mLiveObjects.fetch_add(1, std::memory_order_acq_rel); }
void PhysxEngine::noteObjectDestroyed() { mLiveObjects.fetch_sub(1, std::memory_order_acq_rel); }

PhysxLiveObjectGuard::PhysxLiveObjectGuard() : mEngine(PhysxEngine::Get()) {
  mEngine->noteObjectCreated();
}

PhysxLiveObjectGuard::PhysxLiveObjectGuard(PhysxLiveObjectGuard &&other) noexcept
    : mEngine(std::move(other.mEngine)) {}

PhysxLiveObjectGuard &PhysxLiveObjectGuard::operator=(PhysxLiveObjectGuard &&other) noexcept {
  if (this != &other) {
    if (mEngine) {
      mEngine->noteObjectDestroyed();
    }
    mEngine = std::move(other.mEngine);
  }
  return *this;
}

PhysxLiveObjectGuard::~PhysxLiveObjectGuard() {
  if (mEngine) {
    mEngine->noteObjectDestroyed();
  }
}

PhysxEngine::~PhysxEngine() {
  if (mShutdown) {
    return;
  }
  // Every system holds the engine, so none is left and no PxScene uses a manager.
  releaseCudaContextManagers();
  PxCloseExtensions();
  if (mPxPhysics) {
    mPxPhysics->release();
    mPxPhysics = nullptr;
  }
  if (mPxFoundation) {
    mPxFoundation->release();
    mPxFoundation = nullptr;
  }
}

} // namespace physx
} // namespace sapien
