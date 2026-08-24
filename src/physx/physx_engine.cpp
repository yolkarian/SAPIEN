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

std::shared_ptr<PhysxCudaContextLease> PhysxEngine::acquireCudaContextLease(int cudaId) {
  if (!PhysxDefault::GetGPUEnabled()) {
    throw std::runtime_error("Using CUDA is not allowed when PhysX GPU is not enabled.");
  }
  if (mShutdown) {
    throw std::runtime_error(
        "failed to acquire PhysX CUDA context: the PhysX engine was shut down");
  }
#ifdef SAPIEN_CUDA
  {
    auto it = mCudaContextLeases.find(cudaId);
    if (it != mCudaContextLeases.end()) {
      if (auto lease = it->second.lock()) {
        return lease;
      }
      mCudaContextLeases.erase(it);
    }
    std::erase_if(mCudaContextLeases, [](auto const &entry) { return entry.second.expired(); });
  }

  PxCudaContextManagerDesc cudaContextManagerDesc;
  CUcontext context{};

  checkCudaErrors(cudaSetDevice(cudaId));

  // NOTE: context initialization seems inconsistent across cuda versions
  // cudaFree(0) is guaranteed to establish context
  checkCudaErrors(cudaFree(0));

  checkCudaDriverErrors(CudaLib::Get().cuCtxGetCurrent(&context));
  if (!context) {
    throw std::runtime_error("failed to get CUDA context.");
  }

  // NOTE: PhysX API really suggests it supports multiple GPUs, but no it doesn't.
  if (!mCudaContextLeases.empty() && mCudaContextLeases.begin()->first != cudaId) {
    int existingCudaId = mCudaContextLeases.begin()->first;
    throw std::runtime_error(
        "failed to create PhysX on cuda:" + std::to_string(cudaId) +
        ". PhysX only supports a single GPU and a scene has previously been created on cuda:" +
        std::to_string(existingCudaId) + ".");
  }

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
  auto lease = std::make_shared<PhysxCudaContextLease>(shared_from_this(), manager, cudaId);
  mCudaContextLeases[cudaId] = lease;
  return lease;
#else
  return nullptr;
#endif
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
  for (auto const &[cudaId, weakLease] : mCudaContextLeases) {
    if (auto lease = weakLease.lock()) {
      throw std::runtime_error("failed to shut down PhysX engine: a CUDA context lease on cuda:" +
                               std::to_string(cudaId) +
                               " is still held; close the owning GPU system first");
    }
  }
  mCudaContextLeases.clear();
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
