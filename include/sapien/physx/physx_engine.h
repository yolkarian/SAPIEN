#pragma once
#include <PxPhysicsAPI.h>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>

namespace sapien {
namespace physx {

class PhysxSystem;
class PhysxSystemCpu;
class PhysxSystemGpu;
class PhysxEngine;

/** RAII registration of one live PhysX-backed object. A shared_ptr move transfers
 *  the count without double-counting, unlike manual ctor/dtor hooks. Declare it as
 *  the first member so it is destroyed after the owning object has released its Px*
 *  resources in its destructor body. */
class PhysxLiveObjectGuard {
public:
  PhysxLiveObjectGuard();
  PhysxLiveObjectGuard(PhysxLiveObjectGuard const &) = delete;
  PhysxLiveObjectGuard &operator=(PhysxLiveObjectGuard const &) = delete;
  PhysxLiveObjectGuard(PhysxLiveObjectGuard &&other) noexcept;
  PhysxLiveObjectGuard &operator=(PhysxLiveObjectGuard &&other) noexcept;
  ~PhysxLiveObjectGuard();

private:
  std::shared_ptr<PhysxEngine> mEngine;
};

/** RAII ownership of one PxCudaContextManager. A GPU system acquires a lease when it
 *  is constructed; the manager is released when the last lease on it is destroyed,
 *  after the PxScene of that system has been released. The lease map lives per CUDA
 *  device so future multi-GPU usage manages one context per device independently. */
class PhysxCudaContextLease {
public:
  PhysxCudaContextLease(std::shared_ptr<PhysxEngine> engine,
                        ::physx::PxCudaContextManager *manager, int cudaId)
      : mEngine(std::move(engine)), mManager(manager), mCudaId(cudaId) {}
  ~PhysxCudaContextLease() {
    if (mManager) {
      mManager->release();
    }
  }
  PhysxCudaContextLease(PhysxCudaContextLease const &) = delete;
  PhysxCudaContextLease &operator=(PhysxCudaContextLease const &) = delete;

  ::physx::PxCudaContextManager *get() const { return mManager; }
  int cudaId() const { return mCudaId; }

private:
  std::shared_ptr<PhysxEngine> mEngine;
  ::physx::PxCudaContextManager *mManager;
  int mCudaId;
};

class PhysxEngine : public std::enable_shared_from_this<PhysxEngine> {
public:
  static std::shared_ptr<PhysxEngine> Get(float toleranceLength = 0.1f,
                                          float toleranceSpeed = 0.2f);

  static std::shared_ptr<PhysxEngine> GetIfExists();

  PhysxEngine(float toleranceLength, float toleranceSpeed);

  /** PxPhysics of this engine; throws after shutdown(). */
  ::physx::PxPhysics *getPxPhysics() const;

  /** Acquire the RAII lease for this device's PhysX CUDA context manager, creating it
   *  on first use. Callers release the PxScene using the manager before releasing the
   *  lease; the manager is released when the last lease holder gives it up. */
  std::shared_ptr<PhysxCudaContextLease> acquireCudaContextLease(int cudaId);

  /** True after shutdown(): every PhysX resource of this engine has been released. A
   *  shutdown engine is a zombie; Get()/GetIfExists() replace it. */
  bool isShutdown() const { return mShutdown; }

  /** Terminal shutdown of the whole engine: extensions, PxPhysics, PxFoundation.
   *  Only legal with no live systems and no live PhysX objects; enforceable with
   *  liveSystemCount()/liveObjectCount(). Idempotent (shutdown zombie is a no-op). */
  void shutdown();

  // ---- lifecycle accounting used by module-level can_shutdown()/shutdown() ----
  uint64_t liveSystemCount() const { return mLiveSystems.load(std::memory_order_acquire); }
  uint64_t liveGpuSystemCount() const { return mLiveGpuSystems.load(std::memory_order_acquire); }
  uint64_t liveObjectCount() const { return mLiveObjects.load(std::memory_order_acquire); }

  ~PhysxEngine();

private:
  friend class PhysxSystem;
  friend class PhysxSystemCpu;
  friend class PhysxSystemGpu;
  friend class PhysxLiveObjectGuard;
  void registerSystem();
  void unregisterSystem();
  void registerGpuSystem();
  void unregisterGpuSystem();
  void noteObjectCreated();
  void noteObjectDestroyed();

  ::physx::PxPhysics *mPxPhysics{};
  ::physx::PxFoundation *mPxFoundation{};

  // Per-device weak cache of the PhysX CUDA context manager lease. An alive lease
  // means a GPU system still owns the manager; expired entries own nothing.
  std::map<int, std::weak_ptr<PhysxCudaContextLease>> mCudaContextLeases;

  std::atomic<uint64_t> mLiveSystems{0};
  std::atomic<uint64_t> mLiveGpuSystems{0};
  std::atomic<uint64_t> mLiveObjects{0};
  bool mShutdown{false};

  // Object classes that hold PhysX resources account themselves here.
  friend class PhysxMaterial;
  friend class PhysxCollisionShape;
  friend class PhysxConvexMesh;
  friend class PhysxHeightField;
  friend class PhysxTriangleMesh;
  friend class PhysxArticulation;
  friend class PhysxBaseComponent;
};

} // namespace physx
} // namespace sapien
