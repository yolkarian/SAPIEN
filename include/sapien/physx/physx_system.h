#pragma once
#include "../array.h"
#include "../component.h"
#include "../device.h"
#include "../system.h"
#include "./physx_default.h"
#include "./physx_engine.h"
#include "mesh_manager.h"
#include "sapien/scene.h"
#include "scene_query.h"
#include "simulation_callback.hpp"
#include <PxPhysicsAPI.h>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#ifdef SAPIEN_CUDA
#include "sapien/utils/cuda.h"
#endif

namespace sapien {
namespace physx {
class PhysxArticulation;
class PhysxMaterial;
class PhysxRigidBodyComponent;
class PhysxRigidDynamicComponent;
class PhysxRigidStaticComponent;
class PhysxArticulationLinkComponent;
// Defined in src/physx/broadphase_env_id.hpp: implementation detail of the declarative
// environment-ID config, never handed to callers.
struct BroadphaseEnvIdWindow;

class PhysxSystem : public System {

public:
  std::shared_ptr<PhysxEngine> getEngine() const { return mEngine; }

  PhysxSceneConfig const &getSceneConfig() const { return mSceneConfig; };

  virtual ::physx::PxScene *getPxScene() const { return mPxScene; }
  virtual void registerComponent(std::shared_ptr<PhysxRigidDynamicComponent> component) = 0;
  virtual void registerComponent(std::shared_ptr<PhysxRigidStaticComponent> component) = 0;
  virtual void registerComponent(std::shared_ptr<PhysxArticulationLinkComponent> component) = 0;
  virtual void unregisterComponent(std::shared_ptr<PhysxRigidDynamicComponent> component) = 0;
  virtual void unregisterComponent(std::shared_ptr<PhysxRigidStaticComponent> component) = 0;
  virtual void unregisterComponent(std::shared_ptr<PhysxArticulationLinkComponent> component) = 0;
  virtual std::vector<std::shared_ptr<PhysxRigidDynamicComponent>>
  getRigidDynamicComponents() const = 0;
  virtual std::vector<std::shared_ptr<PhysxRigidStaticComponent>>
  getRigidStaticComponents() const = 0;
  virtual std::vector<std::shared_ptr<PhysxArticulationLinkComponent>>
  getArticulationLinkComponents() const = 0;

  void setTimestep(float step) { mTimestep = step; };
  float getTimestep() const { return mTimestep; }

  std::string getName() const override { return "physx"; }
  virtual bool isGpu() const = 0;

  /** get articulation count directly from PhysX */
  int getArticulationCount() const;

  /** get articulation max dof directly from PhysX */
  int computeArticulationMaxDof() const;

  /** get articulation max link count directly from PhysX */
  int computeArticulationMaxLinkCount() const;

  void setSceneCollisionId(int id) { mSceneCollisionId = id; }
  int getSceneCollisionId() const { return mSceneCollisionId; }

  ~PhysxSystem();

protected:
  PhysxSystem();

  PhysxSceneConfig mSceneConfig;
  std::shared_ptr<PhysxEngine> mEngine;

  ::physx::PxScene *mPxScene;
  float mTimestep{0.01f};

  ::physx::PxDefaultCpuDispatcher *mPxCPUDispatcher;

  int mSceneCollisionId{0};

};

class PhysxSystemCpu : public PhysxSystem {
public:
  PhysxSystemCpu();

  void registerComponent(std::shared_ptr<PhysxRigidDynamicComponent> component) override;
  void registerComponent(std::shared_ptr<PhysxRigidStaticComponent> component) override;
  void registerComponent(std::shared_ptr<PhysxArticulationLinkComponent> component) override;
  void unregisterComponent(std::shared_ptr<PhysxRigidDynamicComponent> component) override;
  void unregisterComponent(std::shared_ptr<PhysxRigidStaticComponent> component) override;
  void unregisterComponent(std::shared_ptr<PhysxArticulationLinkComponent> component) override;
  std::vector<std::shared_ptr<PhysxRigidDynamicComponent>>
  getRigidDynamicComponents() const override;
  std::vector<std::shared_ptr<PhysxRigidStaticComponent>>
  getRigidStaticComponents() const override;
  std::vector<std::shared_ptr<PhysxArticulationLinkComponent>>
  getArticulationLinkComponents() const override;

  std::unique_ptr<PhysxHitInfo> raycast(Vec3 const &origin, Vec3 const &direction, float distance);

  void step() override;
  bool isGpu() const override { return false; }

  std::string packState() const;
  void unpackState(std::string const &data);

  std::vector<Contact *> getContacts() const { return mSimulationCallback.getContacts(); }

  ~PhysxSystemCpu();

private:
  DefaultEventCallback mSimulationCallback;

  std::set<std::shared_ptr<PhysxRigidDynamicComponent>, comp_cmp> mRigidDynamicComponents;
  std::set<std::shared_ptr<PhysxRigidStaticComponent>, comp_cmp> mRigidStaticComponents;
  std::set<std::shared_ptr<PhysxArticulationLinkComponent>, comp_cmp> mArticulationLinkComponents;
};

#ifdef SAPIEN_CUDA

struct PhysxGpuContactPairImpulseQuery {
  CudaArray query;
  CudaArray buffer;
};

struct PhysxGpuContactBodyImpulseQuery {
  CudaArray query;
  CudaArray buffer;
};

class PhysxSystemGpu : public PhysxSystem {
public:
  /** Reads the environment-ID declaration from `PhysxSceneConfig`: `numScenes`,
   *  `withSharedScene` and the per-axis bit counts are snapshotted here and frozen.
   *  This is the only point where they apply -- PhysX bakes the bit count into the scene at
   *  `createScene` and refuses `PxActor::setEnvironmentID` once an actor is in a scene. */
  PhysxSystemGpu(std::shared_ptr<Device> device);

  void registerComponent(std::shared_ptr<PhysxRigidDynamicComponent> component) override;
  void registerComponent(std::shared_ptr<PhysxRigidStaticComponent> component) override;
  void registerComponent(std::shared_ptr<PhysxArticulationLinkComponent> component) override;
  void unregisterComponent(std::shared_ptr<PhysxRigidDynamicComponent> component) override;
  void unregisterComponent(std::shared_ptr<PhysxRigidStaticComponent> component) override;
  void unregisterComponent(std::shared_ptr<PhysxArticulationLinkComponent> component) override;
  std::vector<std::shared_ptr<PhysxRigidDynamicComponent>>
  getRigidDynamicComponents() const override;
  std::vector<std::shared_ptr<PhysxRigidStaticComponent>>
  getRigidStaticComponents() const override;
  std::vector<std::shared_ptr<PhysxArticulationLinkComponent>>
  getArticulationLinkComponents() const override;

  void step() override;

  void stepStart();
  void stepFinish();

  bool isGpu() const override { return true; }

  void gpuInit();
  bool isInitialized() const { return mGpuInitialized; }
  void checkGpuInitialized() const;

  /** Set the CUDA stream for all GPU operations.
   *  gpuQuery* and gpuApply* will be synchronized with the stream
   *  If not set, gpuQuery* and gpuApply* synchronizes with the default stream */
  void gpuSetCudaStream(uintptr_t stream);
  uintptr_t gpuGetCudaStream() const { return reinterpret_cast<uintptr_t>(mCudaStream); }

  /** handle to the pose-vel buffer for rigid dynamic bodies and links */
  CudaArrayHandle gpuGetRigidBodyCudaHandle() const { return mCudaRigidBodyBuffer.handle(); }
  CudaArrayHandle gpuGetRigidDynamicCudaHandle() const { return mCudaRigidDynamicHandle; }
  CudaArrayHandle gpuGetArticulationLinkCudaHandle() const { return mCudaLinkHandle; }

  CudaArrayHandle gpuGetRigidBodyForceCudaHandle() const {
    return mCudaRigidBodyForceBuffer.handle();
  }
  CudaArrayHandle gpuGetRigidDynamicForceCudaHandle() const {
    return mCudaRigidDynamicForceHandle;
  }
  CudaArrayHandle gpuGetRigidBodyTorqueCudaHandle() const {
    return mCudaRigidBodyTorqueBuffer.handle();
  }
  CudaArrayHandle gpuGetRigidDynamicTorqueCudaHandle() const {
    return mCudaRigidDynamicTorqueHandle;
  }
  CudaArrayHandle gpuGetArticulationLinkForceCudaHandle() const {
    return mCudaArticulationLinkForceHandle;
  }
  CudaArrayHandle gpuGetArticulationLinkTorqueCudaHandle() const {
    return mCudaArticulationLinkTorqueHandle;
  }

  CudaArrayHandle gpuGetArticulationQposCudaHandle() const { return mCudaQposHandle; }
  CudaArrayHandle gpuGetArticulationQvelCudaHandle() const { return mCudaQvelHandle; }
  CudaArrayHandle gpuGetArticulationQaccCudaHandle() const { return mCudaQaccHandle; }
  CudaArrayHandle gpuGetArticulationQfCudaHandle() const { return mCudaQfHandle; }
  CudaArrayHandle gpuGetArticulationGravityCompensationCudaHandle() const {
    return mCudaArticulationGravityCompensationHandle;
  }
  CudaArrayHandle gpuGetArticulationCoriolisAndCentrifugalCompensationCudaHandle() const {
    return mCudaArticulationCoriolisAndCentrifugalCompensationHandle;
  }

  CudaArrayHandle gpuGetArticulationQTargetPosCudaHandle() const { return mCudaQTargetPosHandle; }
  CudaArrayHandle gpuGetArticulationQTargetVelCudaHandle() const { return mCudaQTargetVelHandle; }

  /** Padded dense Jacobians with shape
   *  [articulation_count, 6 + (max_links - 1) * 6, 6 + max_dofs]. */
  CudaArrayHandle gpuGetArticulationJacobianCudaHandle() const {
    return mCudaArticulationJacobianHandle;
  }

  /** Valid dense Jacobian shape per articulation, [rows, cols], indexed by gpu_index. */
  CudaArrayHandle gpuGetArticulationJacobianShapeCudaHandle() const {
    return mCudaArticulationJacobianShapeBuffer.handle();
  }

  CudaArrayHandle gpuGetArticulationLinkIncomingJointForceHandle() const {
    return mCudaArticulationLinkIncomingJointForceBuffer.handle();
  }

  void gpuFetchRigidDynamicData();
  void gpuFetchRigidDynamicDataIfNeeded();
  void gpuFetchArticulationLinkPose();
  void gpuFetchArticulationLinkPoseIfNeeded();
  void gpuFetchArticulationLinkVel();
  void gpuFetchArticulationQpos();
  void gpuFetchArticulationQvel();
  void gpuFetchArticulationQacc();
  void gpuFetchArticulationQTargetPos();
  void gpuFetchArticulationQTargetVel();

  /** Compute dense articulation Jacobians into gpuGetArticulationJacobianCudaHandle(). */
  void gpuComputeArticulationJacobian();

  /** Compute dense articulation Jacobians for the given articulation gpu_index values.
   *  Only the selected entries inside gpuGetArticulationJacobianCudaHandle() are updated. */
  void gpuComputeArticulationJacobian(CudaArrayHandle const &indices);

  /** Compute joint gravity compensation into gpuGetArticulationGravityCompensationCudaHandle(). */
  void gpuComputeArticulationGravityCompensation();

  /** Compute joint gravity compensation for the given articulation gpu_index values.
   *  Only the selected entries inside gpuGetArticulationGravityCompensationCudaHandle() are
   *  updated. */
  void gpuComputeArticulationGravityCompensation(CudaArrayHandle const &indices);

  /** Compute joint Coriolis and centrifugal compensation into
   *  gpuGetArticulationCoriolisAndCentrifugalCompensationCudaHandle(). */
  void gpuComputeArticulationCoriolisAndCentrifugalCompensation();

  /** Compute joint Coriolis and centrifugal compensation for the given articulation gpu_index
   *  values. Only the selected entries inside
   *  gpuGetArticulationCoriolisAndCentrifugalCompensationCudaHandle() are updated. */
  void gpuComputeArticulationCoriolisAndCentrifugalCompensation(CudaArrayHandle const &indices);
  void gpuFetchArticulationLinkIncomingJointForce();

  void gpuApplyRigidDynamicData(CudaArrayHandle const &indices);
  void gpuApplyRigidDynamicForce(CudaArrayHandle const &indices);
  void gpuApplyRigidDynamicTorque(CudaArrayHandle const &indices);
  void gpuApplyArticulationRootPose(CudaArrayHandle const &indices);
  void gpuApplyArticulationRootVel(CudaArrayHandle const &indices);
  void gpuApplyArticulationQpos(CudaArrayHandle const &indices);
  void gpuApplyArticulationQvel(CudaArrayHandle const &indices);
  void gpuApplyArticulationQf(CudaArrayHandle const &indices);
  void gpuApplyArticulationQTargetPos(CudaArrayHandle const &indices);
  void gpuApplyArticulationQTargetVel(CudaArrayHandle const &indices);
  void gpuApplyArticulationLinkForce(CudaArrayHandle const &indices);
  void gpuApplyArticulationLinkTorque(CudaArrayHandle const &indices);

  void gpuApplyRigidDynamicData();
  void gpuApplyRigidDynamicForce();
  void gpuApplyRigidDynamicTorque();
  void gpuApplyArticulationLinkForce();
  void gpuApplyArticulationLinkTorque();
  void gpuApplyArticulationRootPose();
  void gpuApplyArticulationRootVel();
  void gpuApplyArticulationQpos();
  void gpuApplyArticulationQvel();
  void gpuApplyArticulationQf();
  void gpuApplyArticulationQTargetPos();
  void gpuApplyArticulationQTargetVel();

  /** Compose one Viewer spring wrench with the exposed application wrench and apply only the
   *  selected rigid dynamic. The exposed force/torque buffers are not modified. */
  void gpuApplyViewerRigidDynamicWrench(int gpuIndex, Vec3 localAnchor, Vec3 localCenterOfMass,
                                        Vec3 target, float effectiveMass, float stiffness,
                                        float damping, float maxAcceleration);
  /** Compose and apply one Viewer spring wrench for one articulation link without modifying the
   *  exposed application wrench buffers. */
  void gpuApplyViewerArticulationLinkWrench(int articulationIndex, int linkIndex, int poseIndex,
                                            Vec3 localAnchor, Vec3 localCenterOfMass, Vec3 target,
                                            float effectiveMass, float stiffness, float damping,
                                            float maxAcceleration);
  /** Queue-safe Viewer teleport helpers that preserve current velocities by default. */
  void gpuSetViewerRigidDynamicPose(int gpuIndex, Pose pose, bool zeroVelocity = false);
  void gpuSetViewerArticulationRootPose(int articulationIndex, int rootPoseIndex, Pose pose);

  void gpuUpdateArticulationKinematics();
  void gpuUpdateArticulationKinematics(CudaArrayHandle const &indices);

  std::shared_ptr<PhysxGpuContactPairImpulseQuery> gpuCreateContactPairImpulseQuery(
      std::vector<std::pair<std::shared_ptr<PhysxRigidBaseComponent>,
                            std::shared_ptr<PhysxRigidBaseComponent>>> const &bodyPairs);
  std::shared_ptr<PhysxGpuContactBodyImpulseQuery> gpuCreateContactBodyImpulseQuery(
      std::vector<std::shared_ptr<PhysxRigidBaseComponent>> const &bodies);

  void gpuQueryContactPairImpulses(PhysxGpuContactPairImpulseQuery const &query);
  void gpuQueryContactBodyImpulses(PhysxGpuContactBodyImpulseQuery const &query);

  void syncPosesGpuToCpu();
  uint64_t getTotalSteps() const { return mTotalSteps; }
  uint64_t getSyncPosesGpuToCpuCount() const { return mSyncPosesGpuToCpuCount; }
  uint64_t getRigidDynamicFetchCount() const { return mRigidDynamicFetchCount; }
  uint64_t getArticulationLinkPoseFetchCount() const {
    return mArticulationLinkPoseFetchCount;
  }

  std::vector<float> gpuDownloadArticulationQpos(int index);
  std::vector<float> gpuDownloadArticulationQTargetPos(int index);
  std::vector<float> gpuDownloadArticulationQTargetVel(int index);
  void gpuUploadArticulationQpos(int index, Eigen::VectorXf const &q);
  void gpuUploadArticulationQTargetPos(int index, Eigen::VectorXf const &q);
  void gpuUploadArticulationQTargetVel(int index, Eigen::VectorXf const &q);

  void setSceneOffset(std::shared_ptr<Scene> scene, Vec3 offset);
  Vec3 getSceneOffset(std::shared_ptr<Scene> scene) const;

  /** Mark a scene as shared: every environment collides with whatever it holds, such as a
   *  ground plane. Only when the config declared `withSharedScene`, and only before the scene
   *  gets a PhysX body, since a body freezes its environment ID at `PxScene::addActor`. A
   *  shared scene sits outside the `numScenes` budget, so an ordinary ID already assigned to
   *  it is returned to the pool for the next scene. Idempotent on an already shared scene. */
  void markSceneShared(std::shared_ptr<Scene> scene);

  /** Set a scene's environment ID by hand. Only when the config left `numScenes` unset, since
   *  declaring it makes SAPIEN the sole assigner. Must be called before the scene gets a body,
   *  because PhysX freezes an actor's environment ID at `PxScene::addActor`.
   *
   *  `envId` is the *unshifted* ID: it is stored verbatim and read back verbatim, and the
   *  broadphase band offset is applied only on the way to `PxActor::setEnvironmentID` (see
   *  `getBroadphaseEnvironmentId`). Duplicates are allowed and meaningful -- several scenes
   *  sharing an ID is how they end up in one environment, colliding with each other. */
  void setSceneEnvironmentId(std::shared_ptr<Scene> scene, uint32_t envId);

  /** A scene's environment ID, assigning a fresh unique one when the config declared
   *  `numScenes` and the scene has none yet. Undeclared, every scene is environment 0 and
   *  nothing is recorded. */
  uint32_t getSceneEnvironmentId(std::shared_ptr<Scene> scene);

  /** A scene's environment ID, assigning a fresh unique one if it has none. Throws when the
   *  config did not declare `numScenes`, since SAPIEN then manages no IDs to hand out. */
  uint32_t getOrAssignSceneEnvironmentId(std::shared_ptr<Scene> scene);

  /** Get a scene's already assigned environment ID without assigning a new one. */
  std::optional<uint32_t> getAssignedSceneEnvironmentId(std::shared_ptr<Scene> scene) const;

  /** The PhysX broadphase environment ID for a scene, i.e. what `PxActor::setEnvironmentID`
   *  receives. Without a shared scene this is the scene's environment ID verbatim, whether
   *  SAPIEN assigned it or a caller set it. With one, the low `b` bits are placed inside the
   *  usable window and any count above it rides in the high bits, which PhysX discards when it
   *  places the box but compares exactly when it filters the pair -- so the stored ID stays
   *  unshifted and only this value carries the offset. Returns `PX_INVALID_U32` for shared
   *  scenes. Stored on first use so every body of a scene is bound to the same value. Throws
   *  when no band is usable, or past the addressable total. */
  uint32_t getBroadphaseEnvironmentId(std::shared_ptr<Scene> scene);

  std::shared_ptr<Device> getDevice() const { return mDevice; };

  ~PhysxSystemGpu();

private:
  // The components call applyCollisionGroupSceneId as they bind to a PhysX scene; it is an
  // internal registration step, not something a caller drives.
  friend class PhysxRigidStaticComponent;
  friend class PhysxRigidDynamicComponent;
  friend class PhysxArticulation;

  /** Write the scene's environment ID into the high 16 bits of every shape's fourth collision
   *  group word, where SAPIEN's filter shader reads it. Unlike the broadphase ID this is the
   *  raw environment ID with no band offset: the field is compared for equality, never encoded
   *  into bounds. Shared scenes get `0xffff`, which collides with every environment. No-op
   *  unless the config declared `withSharedScene`. */
  void applyCollisionGroupSceneId(std::shared_ptr<Scene> scene,
                                  PhysxRigidBaseComponent &component);

  std::shared_ptr<Device> mDevice;
  void ensureCudaDevice();
  uint32_t getGpuArticulationDof(int index) const;

  std::map<std::weak_ptr<Scene>, Vec3, std::owner_less<>> mSceneOffset;
  // Scene -> environment ID, always stored unshifted. In manual mode this is pure storage:
  // written only by setSceneEnvironmentId and markSceneShared, with no allocation, no free
  // list and no cap on top. A scene nobody numbered has no entry and is environment 0.
  std::map<std::weak_ptr<Scene>, uint32_t, std::owner_less<>> mSceneEnvironmentIds;
  uint32_t mNextSceneEnvironmentId{0};
  // Ordinary IDs handed back by scenes the allocator numbered that were later marked shared or
  // destroyed. Only used when `numScenes` is declared; manual IDs never enter it.
  std::vector<uint32_t> mFreeSceneEnvironmentIds;

  // The PhysX broadphase environment ID actually bound to each scene's bodies, kept so every
  // body of a scene lands in the same band. PhysX freezes an actor's environment ID at
  // PxScene::addActor, so this is decided when a scene's first body binds and never revisited.
  std::map<std::weak_ptr<Scene>, uint32_t, std::owner_less<>> mBroadphaseEnvironmentIds;
  // Declared at construction: the only point where the environment total is knowable.
  bool mManagedBroadphaseEnvIds{false};
  bool mReserveSharedBands{false};
  bool mManagedCollisionGroupSceneIds{false};

  uint32_t allocateSceneEnvironmentId();
  /** Return the slots of destroyed scenes to the free list. Called only when the counter is at
   *  the declared ceiling: a `weak_ptr` gives no expiry callback, so the sweep has to be pulled
   *  at the one point where a missing slot would otherwise be reported as exhaustion. */
  void reclaimExpiredSceneEnvironmentIds();
  bool sceneHasPhysxBodies(std::shared_ptr<Scene> scene) const;

  /** Whether any scene is registered as shared. */
  bool hasSharedEnvironmentScene() const;

  /** The range of scene environment IDs whose broadphase band still overlaps the fixed encoded
   *  interval PhysX gives objects shared by all environments. Only consulted with a shared
   *  scene -- without one no band is out of reach and no offset is applied. Depends only on the
   *  configured bit count, never on how many scenes exist: PhysX freezes an actor's environment
   *  ID at `PxScene::addActor`, so the value must be final when the first body binds. */
  BroadphaseEnvIdWindow getBroadphaseEnvIdWindow() const;

  /** How many environments land in distinct broadphase bands with the current bit configuration.
   *  Environments beyond this share bands, which costs spreading but stays correct. */
  uint32_t getBroadphaseEnvBandCount() const;

  /** Whether SAPIEN also stamps each scene's environment ID into the collision-group scene
   *  field, so its own filter shader rejects cross-environment pairs exactly. Follows
   *  `withSharedScene`; without it the collision groups are left alone for the caller. */
  bool managesCollisionGroupSceneIds() const { return mManagedCollisionGroupSceneIds; }

  std::set<std::shared_ptr<PhysxRigidDynamicComponent>, comp_cmp> mRigidDynamicComponents;
  std::set<std::shared_ptr<PhysxRigidStaticComponent>, comp_cmp> mRigidStaticComponents;
  std::set<std::shared_ptr<PhysxArticulationLinkComponent>, comp_cmp> mArticulationLinkComponents;

  uint64_t mTotalSteps{};

  bool mGpuInitialized{false};

  // cache values updated in gpuInit
  int mGpuArticulationCount{-1};
  int mGpuArticulationMaxDof{-1};
  int mGpuArticulationMaxLinkCount{-1};

  CudaEvent mCudaEventRecord;
  CudaEvent mCudaEventWait;
  CudaEvent mCudaRigidPoseFetchEvent;
  CudaEvent mCudaRigidLinearVelocityFetchEvent;
  CudaEvent mCudaRigidAngularVelocityFetchEvent;
  CudaEvent mCudaArticulationLinkPoseFetchEvent;
  cudaStream_t mCudaStream{0};

  std::optional<uint64_t> mRigidDynamicDataFetchedStep;
  std::optional<uint64_t> mArticulationLinkPoseFetchedStep;
  uint64_t mSyncPosesGpuToCpuCount{};
  uint64_t mRigidDynamicFetchCount{};
  uint64_t mArticulationLinkPoseFetchCount{};
  void invalidateRigidDynamicData();
  void invalidateArticulationLinkPose();

  CudaArray mCudaRigidDynamicScratch;
  CudaArray mCudaLinkPoseScratch;
  CudaArray mCudaLinkVelScratch;
  CudaArray mCudaRigidDynamicIndexScratch;
  CudaArray mCudaArticulationIndexScratch;

  void allocateCudaBuffers();
  void gpuComputeArticulationCompensation(
      CudaArrayHandle const &indices, CudaArrayHandle const &output,
      ::physx::PxArticulationGPUAPIComputeType::Enum computeType);

  // indx buffer for all rigid dynamic bodies
  CudaArray mCudaRigidDynamicIndexBuffer;
  CudaArray mCudaRigidDynamicOffsetBuffer;

  // dense articulation indices and dense-to-GPU-index mapping
  CudaArray mCudaArticulationIndexBuffer;
  CudaArray mCudaArticulationGpuIndexBuffer;
  CudaArray mCudaArticulationOffsetBuffer;

  CudaArray mCudaRigidBodyBuffer;
  CudaArrayHandle mCudaRigidDynamicHandle;
  CudaArrayHandle mCudaLinkHandle;

  CudaArray mCudaRigidBodyForceBuffer;
  CudaArrayHandle mCudaRigidDynamicForceHandle;
  CudaArray mCudaRigidDynamicForcePaddedScratch;
  CudaArray mCudaRigidDynamicForcePackedScratch;
  CudaArrayHandle mCudaArticulationLinkForceHandle;
  CudaArray mCudaArticulationLinkForcePaddedScratch;
  CudaArray mCudaArticulationLinkForcePackedScratch;

  CudaArray mCudaRigidBodyTorqueBuffer;
  CudaArrayHandle mCudaRigidDynamicTorqueHandle;
  CudaArray mCudaRigidDynamicTorquePaddedScratch;
  CudaArray mCudaRigidDynamicTorquePackedScratch;
  CudaArrayHandle mCudaArticulationLinkTorqueHandle;
  CudaArray mCudaArticulationLinkTorquePaddedScratch;
  CudaArray mCudaArticulationLinkTorquePackedScratch;

  CudaArray mCudaViewerForceScratch;
  CudaArray mCudaViewerTorqueScratch;
  CudaArray mCudaViewerIndexBuffer;

  CudaHostArray mCudaHostRigidBodyBuffer;

  CudaArray mCudaArticulationBuffer;
  CudaArray mCudaArticulationApplyScratch;
  CudaArrayHandle mCudaQposHandle;
  CudaArrayHandle mCudaQvelHandle;
  CudaArrayHandle mCudaQfHandle;
  CudaArrayHandle mCudaQaccHandle;
  CudaArrayHandle mCudaQTargetPosHandle;
  CudaArrayHandle mCudaQTargetVelHandle;
  CudaArrayHandle mCudaArticulationGravityCompensationHandle;
  CudaArrayHandle mCudaArticulationCoriolisAndCentrifugalCompensationHandle;
  CudaArrayHandle mCudaArticulationJacobianHandle;

  CudaArray mCudaArticulationLinkIncomingJointForceBuffer;
  CudaArray mCudaArticulationCompensationScratch;
  // Per-articulation {root_force_offset, dof_count} for joint-only compensation views.
  CudaArray mCudaArticulationCompensationMetaBuffer;
  CudaArray mCudaArticulationJacobianBuffer;
  CudaArray mCudaArticulationJacobianScratch;
  CudaArray mCudaArticulationJacobianShapeBuffer;

  CudaArray mCudaContactBuffer;
  CudaArray mCudaContactCount;

  bool mContactUpToDate{false};
  int mContactCount{0}; // current contact count, valid only when contactUpdaToDate is true
  void copyContactData();
};
#else
class PhysxSystemGpu : public PhysxSystem {
public:
  PhysxSystemGpu(std::shared_ptr<Device> device);

  void registerComponent(std::shared_ptr<PhysxRigidDynamicComponent> component) override {}
  void registerComponent(std::shared_ptr<PhysxRigidStaticComponent> component) override {}
  void registerComponent(std::shared_ptr<PhysxArticulationLinkComponent> component) override {}
  void unregisterComponent(std::shared_ptr<PhysxRigidDynamicComponent> component) override {}
  void unregisterComponent(std::shared_ptr<PhysxRigidStaticComponent> component) override {}
  void unregisterComponent(std::shared_ptr<PhysxArticulationLinkComponent> component) override {}
  std::vector<std::shared_ptr<PhysxRigidDynamicComponent>>
  getRigidDynamicComponents() const override { return {}; }
  std::vector<std::shared_ptr<PhysxRigidStaticComponent>>
  getRigidStaticComponents() const override { return {}; }
  std::vector<std::shared_ptr<PhysxArticulationLinkComponent>>
  getArticulationLinkComponents() const override { return {}; }

  void step() override {}
  bool isGpu() const override { return true; }

  ~PhysxSystemGpu() {}
};
#endif

} // namespace physx
} // namespace sapien
