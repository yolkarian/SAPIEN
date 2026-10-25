#pragma once
#include <PxPhysicsAPI.h>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
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

class PhysxEngine {
public:
  static std::shared_ptr<PhysxEngine> Get(float toleranceLength = 0.1f,
                                          float toleranceSpeed = 0.2f);

  static std::shared_ptr<PhysxEngine> GetIfExists();

  PhysxEngine(float toleranceLength, float toleranceSpeed);

  /** PxPhysics of this engine; throws after shutdown(). */
  ::physx::PxPhysics *getPxPhysics() const;

  /** This device's PhysX CUDA context manager, created on first use around the device's
   *  primary CUDA context; each device has its own. The engine keeps every manager until
   *  shutdown()/destruction: PhysX caches a kernel table per manager pointer for the
   *  PxPhysics lifetime, so a released manager replaced by one at the same address would
   *  launch stale kernels. PxScenes using a manager are released before it because
   *  shutdown() requires every system to be closed. */
  ::physx::PxCudaContextManager *getCudaContextManager(int cudaId);

  /** True after shutdown(): every PhysX resource of this engine has been released. A
   *  shutdown engine is a zombie; Get()/GetIfExists() replace it. */
  bool isShutdown() const { return mShutdown; }

  /** Terminal shutdown of the whole engine: CUDA context managers, extensions,
   *  PxPhysics, PxFoundation. Only legal with no live systems and no live PhysX
   *  objects; enforceable with liveSystemCount()/liveObjectCount(). Idempotent
   *  (shutdown zombie is a no-op). */
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
  void releaseCudaContextManagers();

  ::physx::PxPhysics *mPxPhysics{};
  ::physx::PxFoundation *mPxFoundation{};

  // One manager per CUDA device, owned until shutdown()/destruction.
  std::map<int, ::physx::PxCudaContextManager *> mCudaContextManagers;
  std::mutex mCudaContextManagerMutex;

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
