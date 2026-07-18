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

  /** Set the PhysX GPU broadphase environment ID for a SAPIEN scene.
   *  envId == -1 or 0xffffffff maps to PX_INVALID_U32 internally (shared object, collides with all
   *  envs). Non-shared env IDs must be in [0, 1 << 24).
   *  Must be called before adding PhysX bodies to the scene. */
  void setSceneEnvironmentId(std::shared_ptr<Scene> scene, int64_t envId,
                             bool allowDuplicate = false);

  /** Get a scene's environment ID. If no ID was set explicitly, a new one is
   *  assigned automatically. */
  uint32_t getSceneEnvironmentId(std::shared_ptr<Scene> scene);

  /** Get a scene's already assigned environment ID without assigning a new one. */
  std::optional<uint32_t> getAssignedSceneEnvironmentId(std::shared_ptr<Scene> scene) const;

  /** Convenience: set environment IDs for multiple scenes at once. */
  void setSceneEnvironmentIds(
      std::vector<std::pair<std::shared_ptr<Scene>, int64_t>> const &mapping,
      bool allowDuplicate = false);

  std::shared_ptr<Device> getDevice() const { return mDevice; };

  ~PhysxSystemGpu();

private:
  std::shared_ptr<Device> mDevice;
  void ensureCudaDevice();
  uint32_t getGpuArticulationDof(int index) const;

  std::map<std::weak_ptr<Scene>, Vec3, std::owner_less<>> mSceneOffset;
  std::map<std::weak_ptr<Scene>, uint32_t, std::owner_less<>> mSceneEnvironmentIds;
  uint32_t mNextSceneEnvironmentId{0};

  uint32_t allocateSceneEnvironmentId();
  bool sceneHasPhysxBodies(std::shared_ptr<Scene> scene) const;
  bool isSceneEnvironmentIdUsed(uint32_t envId, std::shared_ptr<Scene> excludedScene) const;

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
