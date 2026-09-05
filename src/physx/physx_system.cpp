#include "sapien/physx/physx_system.h"
#include "../logger.h"
#include "./filter_shader.hpp"
#include "./broadphase_env_id.hpp"
#include "sapien/entity.h"
#include "sapien/math/conversion.h"
#include "sapien/physx/articulation.h"
#include "sapien/physx/articulation_link_component.h"
#include "sapien/physx/material.h"
#include "sapien/physx/physx_default.h"
#include "sapien/physx/rigid_component.h"
#include "sapien/profiler.h"
#include "sapien/utils/typestr.h"
#include <extensions/PxExtensionsAPI.h>
#include <unordered_map>

#ifdef SAPIEN_CUDA
#include "./physx_system.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudamanager/PxCudaContext.h>
#endif

using namespace physx;
namespace sapien {
namespace physx {

struct SapienBodyDataTest {
  Pose pose;
  Vec3 v;
  Vec3 w;
};

static_assert(sizeof(SapienBodyDataTest) == 52);

PhysxSystem::PhysxSystem()
    : mSceneConfig(PhysxDefault::getSceneConfig()), mEngine(PhysxEngine::Get()) {
  mEngine->registerSystem();
  mRegisteredSystem = true;
}

void PhysxSystem::internalAddScene(Scene &scene) {
  checkNotClosed();
  System::internalAddScene(scene);
}

PhysxSystemCpu::PhysxSystemCpu() {
  if (PhysxDefault::GetGPUEnabled()) {
    logger::warn(
        "A PhysX CPU system is being created while PhysX GPU is enabled. You can safely ignore "
        "this message if it is intended. To use GPU PhysX, create a sapien.physx.PhysxGpuSystem "
        "explicitly and pass it to sapien.Scene constructor.");
  }

  auto &config = mSceneConfig;
  PxSceneDesc sceneDesc(mEngine->getPxPhysics()->getTolerancesScale());
  sceneDesc.gravity = Vec3ToPxVec3(config.gravity);
  sceneDesc.filterShader = TypeAffinityIgnoreFilterShader;
  sceneDesc.solverType = config.enableTGS ? PxSolverType::eTGS : PxSolverType::ePGS;
  sceneDesc.frictionOffsetThreshold = config.frictionOffsetThreshold;
  sceneDesc.frictionCorrelationDistance = config.frictionCorrelationDistance;
  sceneDesc.bounceThresholdVelocity = config.bounceThreshold;

  PxSceneFlags sceneFlags;
  if (config.enableEnhancedDeterminism) {
    sceneFlags |= PxSceneFlag::eENABLE_ENHANCED_DETERMINISM;
  }
  if (config.enablePCM) {
    sceneFlags |= PxSceneFlag::eENABLE_PCM;
  }
  if (config.enableCCD) {
    sceneFlags |= PxSceneFlag::eENABLE_CCD;
  }
  if (config.enableFrictionEveryIteration) {
    sceneFlags |= PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION;
  }

  sceneDesc.flags = sceneFlags;

  mPxCPUDispatcher = PxDefaultCpuDispatcherCreate(config.cpuWorkers);
  if (!mPxCPUDispatcher) {
    throw std::runtime_error("PhysX system creation failed: failed to create CPU dispatcher");
  }
  sceneDesc.cpuDispatcher = mPxCPUDispatcher;
  mPxScene = mEngine->getPxPhysics()->createScene(sceneDesc);
  if (!mPxScene) {
    mPxCPUDispatcher->release();
    mPxCPUDispatcher = nullptr;
    throw std::runtime_error(
        "PhysX system creation failed: failed to create PhysX scene (see PhysX errors above)");
  }
  mPxScene->setSimulationEventCallback(&mSimulationCallback);
}

#ifdef SAPIEN_CUDA
PhysxSystemGpu::PhysxSystemGpu(std::shared_ptr<Device> device) {
  if (!PhysxDefault::GetGPUEnabled()) {
    throw std::runtime_error(
        "sapien.physx.enable_gpu() must be called before creating a PhysX GPU system.");
  }

  if (!device) {
    device = findDevice("cuda");
    if (!device) {
      throw std::runtime_error("failed to find a CUDA device for PhysX GPU");
    }
  } else if (!device->isCuda()) {
    throw std::runtime_error(
        "failed to create PhysX GPU system: device provided does not support CUDA");
  }
  mDevice = device;

  // The declaration is snapshotted here and frozen: PhysX bakes the bit count into the scene at
  // createScene and refuses setEnvironmentID once an actor is in a scene, so by the time scenes
  // are being built nothing below can still be changed.
  mReserveSharedBands = mSceneConfig.withSharedScene;
  mManagedCollisionGroupSceneIds = mSceneConfig.withSharedScene;
  mManagedBroadphaseEnvIds = mSceneConfig.numScenes.has_value();
  if (mManagedBroadphaseEnvIds) {
    const uint32_t numScenes = *mSceneConfig.numScenes;
    if (numScenes == 0) {
      throw std::runtime_error("failed to create PhysX GPU system: num_scenes must be positive");
    }
    if (mReserveSharedBands && numScenes > kMaxSharedSceneEnvCount) {
      throw std::runtime_error(
          "failed to create PhysX GPU system: " + std::to_string(numScenes) +
          " scenes exceed the " + std::to_string(kMaxSharedSceneEnvCount) +
          " a shared scene can coexist with. SAPIEN's filter shader keeps the scene ID in a "
          "16-bit collision group field and reserves 0xffff for the shared scene itself.");
    }
    const uint8_t bits = broadphaseEnvIdBitsForEnvCount(numScenes, mReserveSharedBands);
    if (bits == 0) {
      throw std::runtime_error("failed to create PhysX GPU system: " + std::to_string(numScenes) +
                               " scenes exceed the maximum of " +
                               std::to_string(maxBroadphaseEnvCount(mReserveSharedBands)) +
                               " the banding can address");
    }
    // Only the widest axis decides the band count -- every axis shifts the same ID, so the
    // narrower ones just keep a subset of its bits. Without a shared object one axis is
    // therefore enough, and leaving X and Y at zero keeps their broadphase coordinate precision.
    // With one, every banded axis must carry the same count or a narrower axis wraps the shifted
    // ID back out of the band that reaches the shared object.
    if (mReserveSharedBands) {
      mSceneConfig.setGpuBroadPhaseEnvIdBits(bits, bits, bits);
    } else {
      mSceneConfig.setGpuBroadPhaseEnvIdBits(0, 0, bits);
    }
  } else if (mReserveSharedBands) {
    // A narrower axis wraps the shifted ID back out of the band that reaches the shared object,
    // so mixed widths silently drop those contacts. Widen every banded axis to the same count,
    // which also gives the shared object the fixed interval on all three rather than its real
    // bounds on the unbanded ones.
    auto const &c = mSceneConfig;
    if (!broadphaseEnvIdBitsUniform(c.getGpuBroadPhaseNbBitsEnvIDX(),
                                    c.getGpuBroadPhaseNbBitsEnvIDY(),
                                    c.getGpuBroadPhaseNbBitsEnvIDZ())) {
      throw std::runtime_error(
          "failed to create PhysX GPU system: with_shared_scene needs every non-zero broadphase "
          "env ID bit count to be equal, got (" +
          std::to_string(c.getGpuBroadPhaseNbBitsEnvIDX()) + ", " +
          std::to_string(c.getGpuBroadPhaseNbBitsEnvIDY()) + ", " +
          std::to_string(c.getGpuBroadPhaseNbBitsEnvIDZ()) +
          "). A narrower axis keeps fewer of the shifted ID's low bits, so environments wrap out "
          "of the band that reaches the shared object and silently lose contact with it.");
    }
    const uint8_t bits =
        std::max({c.getGpuBroadPhaseNbBitsEnvIDX(), c.getGpuBroadPhaseNbBitsEnvIDY(),
                  c.getGpuBroadPhaseNbBitsEnvIDZ()});
    mSceneConfig.setGpuBroadPhaseEnvIdBits(bits, bits, bits);
  }

  if (mReserveSharedBands) {
    // Narrow bands can leave no band at all inside the fixed encoded interval PhysX gives
    // shared objects. Reject here rather than at the first body: the bit count is baked into
    // the PhysX scene at createScene, so by then nothing can be done about it.
    auto const &c = mSceneConfig;
    const auto window =
        broadphaseEnvIdWindow(c.getGpuBroadPhaseNbBitsEnvIDX(), c.getGpuBroadPhaseNbBitsEnvIDY(),
                              c.getGpuBroadPhaseNbBitsEnvIDZ());
    if (window.capacity == 0) {
      throw std::runtime_error(
          "failed to create PhysX GPU system: with_shared_scene and " +
          std::to_string(c.getGpuBroadPhaseNbBitsEnvIDZ()) +
          " broadphase env ID bits leave no band inside the fixed encoded interval PhysX gives "
          "objects shared by all environments, so every environment would silently stop "
          "colliding with the shared scene. Declare num_scenes and let SAPIEN size the bits, or "
          "raise them with PhysxSceneConfig.set_gpu_broadphase_env_id_bits().");
    }
  }

  auto &config = mSceneConfig;
  PxSceneDesc sceneDesc(mEngine->getPxPhysics()->getTolerancesScale());
  sceneDesc.gravity = Vec3ToPxVec3(config.gravity);
  sceneDesc.filterShader = TypeAffinityIgnoreFilterShaderGpu;
  sceneDesc.solverType = config.enableTGS ? PxSolverType::eTGS : PxSolverType::ePGS;
  sceneDesc.frictionOffsetThreshold = config.frictionOffsetThreshold;
  sceneDesc.frictionCorrelationDistance = config.frictionCorrelationDistance;
  sceneDesc.bounceThresholdVelocity = config.bounceThreshold;

  sceneDesc.gpuDynamicsConfig = PhysxDefault::getGpuMemoryConfig();

  // Configure GPU broadphase environment ID bits if requested. PhysX can only merge
  // environment ID bits into coordinate bits that were shifted away, so ensure each
  // shift is at least the corresponding env ID bit count.
  PxGpuBroadPhaseDesc gpuBroadPhaseDesc;
  auto validateEnvIdBits = [](uint8_t bits, char axis) {
    if (bits > 16) {
      throw std::runtime_error(std::string("failed to create PhysX GPU system: ") +
                               "gpu broadphase env ID bits on axis " + std::string(1, axis) +
                               " must be in [0, 16]");
    }
  };
  const uint8_t envIdBitsX = config.getGpuBroadPhaseNbBitsEnvIDX();
  const uint8_t envIdBitsY = config.getGpuBroadPhaseNbBitsEnvIDY();
  const uint8_t envIdBitsZ = config.getGpuBroadPhaseNbBitsEnvIDZ();
  validateEnvIdBits(envIdBitsX, 'x');
  validateEnvIdBits(envIdBitsY, 'y');
  validateEnvIdBits(envIdBitsZ, 'z');
  if (envIdBitsX || envIdBitsY || envIdBitsZ) {
    gpuBroadPhaseDesc.gpuBroadPhaseNbBitsShiftX =
        std::max<uint8_t>(gpuBroadPhaseDesc.gpuBroadPhaseNbBitsShiftX, envIdBitsX);
    gpuBroadPhaseDesc.gpuBroadPhaseNbBitsShiftY =
        std::max<uint8_t>(gpuBroadPhaseDesc.gpuBroadPhaseNbBitsShiftY, envIdBitsY);
    gpuBroadPhaseDesc.gpuBroadPhaseNbBitsShiftZ =
        std::max<uint8_t>(gpuBroadPhaseDesc.gpuBroadPhaseNbBitsShiftZ, envIdBitsZ);
    gpuBroadPhaseDesc.gpuBroadPhaseNbBitsEnvIDX = envIdBitsX;
    gpuBroadPhaseDesc.gpuBroadPhaseNbBitsEnvIDY = envIdBitsY;
    gpuBroadPhaseDesc.gpuBroadPhaseNbBitsEnvIDZ = envIdBitsZ;
    sceneDesc.gpuBroadPhaseDesc = &gpuBroadPhaseDesc;
  }

  PxSceneFlags sceneFlags;
  if (config.enableEnhancedDeterminism) {
    sceneFlags |= PxSceneFlag::eENABLE_ENHANCED_DETERMINISM;
  }
  if (config.enablePCM) {
    sceneFlags |= PxSceneFlag::eENABLE_PCM;
  }
  if (config.enableCCD) {
    sceneFlags |= PxSceneFlag::eENABLE_CCD;
  }
  if (config.enableFrictionEveryIteration) {
    sceneFlags |= PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION;
  }

  sceneFlags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
  sceneFlags |= PxSceneFlag::eENABLE_DIRECT_GPU_API;
  sceneDesc.broadPhaseType = PxBroadPhaseType::eGPU;
  mCudaContextLease = mEngine->acquireCudaContextLease(device->cudaId);
  sceneDesc.cudaContextManager = mCudaContextLease->get();
  if (!config.enablePCM) {
    logger::warn("PCM must be enabled when using GPU.");
    sceneFlags |= PxSceneFlag::eENABLE_PCM;
  }

  sceneDesc.flags = sceneFlags;

  mPxCPUDispatcher = PxDefaultCpuDispatcherCreate(config.cpuWorkers);
  if (!mPxCPUDispatcher) {
    throw std::runtime_error("PhysX system creation failed: failed to create CPU dispatcher");
  }
  sceneDesc.cpuDispatcher = mPxCPUDispatcher;
  mPxScene = mEngine->getPxPhysics()->createScene(sceneDesc);
  if (!mPxScene) {
    mPxCPUDispatcher->release();
    mPxCPUDispatcher = nullptr;
    // A failed GPU allocation during scene creation latches the shared PxCudaContext into
    // abort mode ("out-of-memory state"), which would freeze simulate() for every other GPU
    // scene on this device. Clear it so already-running systems keep stepping.
    if (auto *manager = sceneDesc.cudaContextManager) {
      if (auto *context = manager->getCudaContext(); context && context->isInAbortMode()) {
        context->setAbortMode(false);
      }
    }
    throw std::runtime_error(
        "PhysX GPU system creation failed: failed to create PhysX scene on cuda:" +
        std::to_string(device->cudaId) +
        ". This usually means the GPU is out of memory: PhysX eagerly allocates the GPU heaps "
        "configured by sapien.physx.set_gpu_memory_config() when the scene is created.");
  }
  mEngine->registerGpuSystem();
  mRegisteredGpuSystem = true;
}
#else
PhysxSystemGpu::PhysxSystemGpu(std::shared_ptr<Device> device) {
  throw std::runtime_error("Does not support PhysX GPU system.");
}
#endif

void PhysxSystemCpu::registerComponent(std::shared_ptr<PhysxRigidDynamicComponent> component) {
  mRigidDynamicComponents.insert(component);
}
void PhysxSystemCpu::registerComponent(std::shared_ptr<PhysxRigidStaticComponent> component) {
  mRigidStaticComponents.insert(component);
}
void PhysxSystemCpu::registerComponent(std::shared_ptr<PhysxArticulationLinkComponent> component) {
  mArticulationLinkComponents.insert(component);
}
void PhysxSystemCpu::unregisterComponent(std::shared_ptr<PhysxRigidDynamicComponent> component) {
  mRigidDynamicComponents.erase(component);
}
void PhysxSystemCpu::unregisterComponent(std::shared_ptr<PhysxRigidStaticComponent> component) {
  mRigidStaticComponents.erase(component);
}
void PhysxSystemCpu::unregisterComponent(
    std::shared_ptr<PhysxArticulationLinkComponent> component) {
  mArticulationLinkComponents.erase(component);
}
std::vector<std::shared_ptr<PhysxRigidDynamicComponent>>
PhysxSystemCpu::getRigidDynamicComponents() const {
  return {mRigidDynamicComponents.begin(), mRigidDynamicComponents.end()};
}
std::vector<std::shared_ptr<PhysxRigidStaticComponent>>
PhysxSystemCpu::getRigidStaticComponents() const {
  return {mRigidStaticComponents.begin(), mRigidStaticComponents.end()};
}
std::vector<std::shared_ptr<PhysxArticulationLinkComponent>>
PhysxSystemCpu::getArticulationLinkComponents() const {
  return {mArticulationLinkComponents.begin(), mArticulationLinkComponents.end()};
}

#ifdef SAPIEN_CUDA
void PhysxSystemGpu::invalidateRigidDynamicData() { mRigidDynamicDataFetchedStep.reset(); }
void PhysxSystemGpu::invalidateArticulationLinkPose() {
  mArticulationLinkPoseFetchedStep.reset();
}

void PhysxSystemGpu::registerComponent(std::shared_ptr<PhysxRigidDynamicComponent> component) {
  mRigidDynamicComponents.insert(component);
  invalidateRigidDynamicData();
  mGpuInitialized = false;
}
void PhysxSystemGpu::registerComponent(std::shared_ptr<PhysxRigidStaticComponent> component) {
  mRigidStaticComponents.insert(component);
  mGpuInitialized = false;
}
void PhysxSystemGpu::registerComponent(std::shared_ptr<PhysxArticulationLinkComponent> component) {
  mArticulationLinkComponents.insert(component);
  invalidateArticulationLinkPose();
  mGpuInitialized = false;
}
void PhysxSystemGpu::unregisterComponent(std::shared_ptr<PhysxRigidDynamicComponent> component) {
  mRigidDynamicComponents.erase(component);
  invalidateRigidDynamicData();
  mGpuInitialized = false;
}
void PhysxSystemGpu::unregisterComponent(std::shared_ptr<PhysxRigidStaticComponent> component) {
  mRigidStaticComponents.erase(component);
  mGpuInitialized = false;
}
void PhysxSystemGpu::unregisterComponent(
    std::shared_ptr<PhysxArticulationLinkComponent> component) {
  mArticulationLinkComponents.erase(component);
  invalidateArticulationLinkPose();
  mGpuInitialized = false;
}
std::vector<std::shared_ptr<PhysxRigidDynamicComponent>>
PhysxSystemGpu::getRigidDynamicComponents() const {
  return {mRigidDynamicComponents.begin(), mRigidDynamicComponents.end()};
}
std::vector<std::shared_ptr<PhysxRigidStaticComponent>>
PhysxSystemGpu::getRigidStaticComponents() const {
  return {mRigidStaticComponents.begin(), mRigidStaticComponents.end()};
}
std::vector<std::shared_ptr<PhysxArticulationLinkComponent>>
PhysxSystemGpu::getArticulationLinkComponents() const {
  return {mArticulationLinkComponents.begin(), mArticulationLinkComponents.end()};
}
#endif

std::unique_ptr<PhysxHitInfo> PhysxSystemCpu::raycast(Vec3 const &origin, Vec3 const &direction,
                                                      float distance) {
  checkNotClosed();
  PxRaycastBuffer hit;
  bool status = mPxScene->raycast(Vec3ToPxVec3(origin), Vec3ToPxVec3(direction), distance, hit);
  if (status) {
    return std::make_unique<PhysxHitInfo>(
        PxVec3ToVec3(hit.block.position), PxVec3ToVec3(hit.block.normal), hit.block.distance,
        static_cast<PhysxCollisionShape *>(hit.block.shape->userData),
        static_cast<PhysxRigidBaseComponent *>(hit.block.actor->userData));
  }
  return nullptr;
}

void PhysxSystemCpu::step() {
  checkNotClosed();
  mPxScene->simulate(mTimestep);
  mPxScene->fetchResults(true);
  for (auto c : mRigidStaticComponents) {
    c->syncPoseToEntity();
  }
  for (auto c : mRigidDynamicComponents) {
    c->syncPoseToEntity();
  }
  for (auto c : mArticulationLinkComponents) {
    c->syncPoseToEntity();
  }
}

#ifdef SAPIEN_CUDA
void PhysxSystemGpu::step() {
  if (mClosed) {
    throw std::runtime_error("failed to step: the PhysxGpuSystem is closed.");
  }
  if (!mGpuInitialized) {
    throw std::runtime_error("failed to step: gpu simulation is not initialized.");
  }
  if (mSimulateInFlight) {
    throw std::runtime_error(
        "failed to step: a GPU simulation is already in flight; call step_finish() first");
  }
  if (mContactQueryInFlight) {
    gpuWaitContactQueries();
  }

  mContactUpToDate = false;

  ++mTotalSteps;
  mPxScene->simulate(mTimestep);
  mPxScene->fetchResults(true);

  // TODO: does the GPU API require fetch results?
}

void PhysxSystemGpu::stepStart() {
  if (mClosed) {
    throw std::runtime_error("failed to step: the PhysxGpuSystem is closed.");
  }
  if (!mGpuInitialized) {
    throw std::runtime_error("failed to step: gpu simulation is not initialized.");
  }
  if (mSimulateInFlight) {
    throw std::runtime_error("failed to start step: a GPU simulation is already in flight");
  }
  if (mContactQueryInFlight) {
    gpuWaitContactQueries();
  }

  mContactUpToDate = false;

  ++mTotalSteps;
  mPxScene->simulate(mTimestep);
  mSimulateInFlight = true;
}

void PhysxSystemGpu::stepFinish() {
  if (mClosed) {
    throw std::runtime_error("failed to finish step: the PhysxGpuSystem is closed.");
  }
  if (!mSimulateInFlight) {
    throw std::runtime_error("failed to finish step: no GPU simulation is in flight");
  }
  mPxScene->fetchResults(true);
  mSimulateInFlight = false;
}

void PhysxSystemGpu::waitIdle() {
  if (mClosed || !mPxScene) {
    return;
  }
  ensureCudaDevice();
  if (mSimulateInFlight) {
    mPxScene->fetchResults(true);
    mSimulateInFlight = false;
  }
  if (mCudaStream) {
    checkCudaErrors(cudaStreamSynchronize(mCudaStream));
  }
  checkCudaErrors(cudaDeviceSynchronize());
  mContactQueryInFlight = false;
}

CudaArrayHandle PhysxSystemGpu::trackView(CudaArrayHandle handle) const {
  if (mClosed) {
    throw std::runtime_error("failed to export a CUDA array view: the PhysxGpuSystem is closed");
  }
  handle.viewGuard = std::make_shared<CudaArrayViewGuard>(mViewLifecycle);
  return handle;
}

int64_t PhysxSystemGpu::outstandingCudaViewCount() const { return mViewLifecycle->viewCount(); }
#endif

std::string PhysxSystemCpu::packState() const {
  checkNotClosed();
  std::ostringstream ss;
  for (auto &actor : mRigidDynamicComponents) {
    Pose pose = actor->getPose();
    Vec3 v = actor->getLinearVelocity();
    Vec3 w = actor->getAngularVelocity();
    ss.write(reinterpret_cast<const char *>(&pose), sizeof(Pose));
    ss.write(reinterpret_cast<const char *>(&v), sizeof(Vec3));
    ss.write(reinterpret_cast<const char *>(&w), sizeof(Vec3));
  }
  for (auto &link : mArticulationLinkComponents) {
    if (link->isRoot()) {
      auto art = link->getArticulation();

      Pose pose = art->getRootPose();
      Vec3 v = art->getRootLinearVelocity();
      Vec3 w = art->getRootAngularVelocity();
      ss.write(reinterpret_cast<const char *>(&pose), sizeof(Pose));
      ss.write(reinterpret_cast<const char *>(&v), sizeof(Vec3));
      ss.write(reinterpret_cast<const char *>(&w), sizeof(Vec3));

      auto qpos = art->getQpos();
      auto qvel = art->getQvel();

      ss.write(reinterpret_cast<const char *>(qpos.data()), qpos.size() * sizeof(float));
      ss.write(reinterpret_cast<const char *>(qvel.data()), qvel.size() * sizeof(float));

      for (auto j : art->getActiveJoints()) {
        auto pos = j->getDriveTargetPosition();
        auto vel = j->getDriveTargetVelocity();

        ss.write(reinterpret_cast<const char *>(pos.data()), pos.size() * sizeof(float));
        ss.write(reinterpret_cast<const char *>(vel.data()), vel.size() * sizeof(float));
      }
    }
  }
  return ss.str();
}

void PhysxSystemCpu::unpackState(std::string const &data) {
  checkNotClosed();
  std::istringstream ss(data);
  for (auto &actor : mRigidDynamicComponents) {
    Pose pose;
    Vec3 v, w;
    ss.read(reinterpret_cast<char *>(&pose), sizeof(Pose));
    ss.read(reinterpret_cast<char *>(&v), sizeof(Vec3));
    ss.read(reinterpret_cast<char *>(&w), sizeof(Vec3));
    actor->setPose(pose);
    if (!actor->isKinematic()) {
      actor->setLinearVelocity(v);
      actor->setAngularVelocity(w);
    }
  }
  for (auto &link : mArticulationLinkComponents) {
    if (link->isRoot()) {
      Pose pose;
      Vec3 v, w;
      ss.read(reinterpret_cast<char *>(&pose), sizeof(Pose));
      ss.read(reinterpret_cast<char *>(&v), sizeof(Vec3));
      ss.read(reinterpret_cast<char *>(&w), sizeof(Vec3));
      auto art = link->getArticulation();
      art->setRootPose(pose);
      art->setRootLinearVelocity(v);
      art->setRootAngularVelocity(w);

      Eigen::VectorXf qpos;
      Eigen::VectorXf qvel;
      qpos.resize(art->getDof());
      qvel.resize(art->getDof());
      ss.read(reinterpret_cast<char *>(qpos.data()), qpos.size() * sizeof(float));
      ss.read(reinterpret_cast<char *>(qvel.data()), qvel.size() * sizeof(float));
      art->setQpos(qpos);
      art->setQvel(qvel);

      for (auto j : art->getActiveJoints()) {
        Eigen::VectorXf pos, vel;
        pos.resize(j->getDof());
        vel.resize(j->getDof());
        ss.read(reinterpret_cast<char *>(pos.data()), pos.size() * sizeof(float));
        ss.read(reinterpret_cast<char *>(vel.data()), vel.size() * sizeof(float));
        j->setDriveTargetPosition(pos);
        j->setDriveTargetVelocity(vel);
      }
    }
  }
}

int PhysxSystem::getArticulationCount() const {
  // TODO: ensure this count matches registered articulations
  return getPxScene()->getNbArticulations();
}

int PhysxSystem::computeArticulationMaxDof() const {
  int result = 0;
  uint32_t count = getPxScene()->getNbArticulations();
  std::vector<PxArticulationReducedCoordinate *> articulations(count);
  getPxScene()->getArticulations(articulations.data(), count);
  for (auto a : articulations) {
    result = std::max(result, static_cast<int>(a->getDofs()));
  }
  return result;
}

int PhysxSystem::computeArticulationMaxLinkCount() const {
  int result = 0;
  uint32_t count = getPxScene()->getNbArticulations();
  std::vector<PxArticulationReducedCoordinate *> articulations(count);
  getPxScene()->getArticulations(articulations.data(), count);
  for (auto a : articulations) {
    result = std::max(result, static_cast<int>(a->getNbLinks()));
  }
  return result;
}

#ifdef SAPIEN_CUDA
void PhysxSystemGpu::gpuInit() {
  checkNotClosed();
  // Everything is built by now, so this is the first point where an unused reservation shows.
  if (mReserveSharedBands && !hasSharedEnvironmentScene()) {
    logger::warn(
        "PhysxGpuSystem was created with with_shared_scene=True but no scene was given "
        "environment ID -1. Bands were reserved for a shared object that does not exist, which "
        "only costs about 1.2% of them; drop with_shared_scene if nothing is shared.");
  }

  ++mTotalSteps;
  ensureCudaDevice();
  mPxScene->simulate(mTimestep);
  while (!mPxScene->fetchResults(true)) {
  }

  mGpuArticulationCount = getArticulationCount();
  mGpuArticulationMaxDof = computeArticulationMaxDof();
  mGpuArticulationMaxLinkCount = computeArticulationMaxLinkCount();

  allocateCudaBuffers();
  invalidateRigidDynamicData();
  invalidateArticulationLinkPose();

  // ensureCudaDevice();
  // mCudaEventRecord.init();
  // mCudaEventWait.init();

  mGpuInitialized = true;
}

void PhysxSystemGpu::checkGpuInitialized() const {
  if (mClosed) {
    throw std::runtime_error("failed to use GPU PhysX: the PhysxGpuSystem is closed.");
  }
  if (!isInitialized()) {
    throw std::runtime_error("GPU PhysX is not initialized.");
  }
}

void PhysxSystemGpu::gpuSetCudaStream(uintptr_t stream) {
  checkNotClosed();
  mCudaStream = (cudaStream_t)stream;
}

std::shared_ptr<PhysxGpuContactPairImpulseQuery> PhysxSystemGpu::gpuCreateContactPairImpulseQuery(
    std::vector<std::pair<std::shared_ptr<PhysxRigidBaseComponent>,
                          std::shared_ptr<PhysxRigidBaseComponent>>> const &bodyPairs) {
  if (bodyPairs.empty()) {
    throw std::runtime_error("failed to create contact query: empty body pairs");
  }
  std::vector<ActorPairQuery> pairs;
  for (uint32_t i = 0; i < bodyPairs.size(); ++i) {
    auto &[b0, b1] = bodyPairs[i];
    if (!b0 || !b1) {
      throw std::runtime_error("failed to create contact query: invalid body");
    }
    int order{0};
    ActorPair pair = makeActorPair(b0->getPxActor(), b1->getPxActor(), order);
    pairs.push_back({pair, i, order});
  }

  std::sort(pairs.begin(), pairs.end(),
            [](ActorPairQuery const &a, ActorPairQuery const &b) { return a.pair < b.pair; });

  static_assert(sizeof(ActorPairQuery) == 24);

  ensureCudaDevice();
  CudaArray query({static_cast<int>(pairs.size()), 6}, "i4");
  checkCudaErrors(cudaMemcpy(query.ptr, pairs.data(), pairs.size() * sizeof(ActorPairQuery),
                             cudaMemcpyHostToDevice));
  CudaArray buffer({static_cast<int>(pairs.size()), 3}, "f4");

  auto res = std::make_shared<PhysxGpuContactPairImpulseQuery>();
  res->query = std::move(query);
  res->buffer = std::move(buffer);
  return res;
}

std::shared_ptr<PhysxGpuContactBodyImpulseQuery> PhysxSystemGpu::gpuCreateContactBodyImpulseQuery(
    std::vector<std::shared_ptr<PhysxRigidBaseComponent>> const &bodies) {
  if (bodies.empty()) {
    throw std::runtime_error("failed to create contact query: empty body list");
  }
  std::vector<ActorQuery> actors;
  for (uint32_t i = 0; i < bodies.size(); ++i) {
    if (!bodies[i]) {
      throw std::runtime_error("failed to create contact actors: invalid body");
    }
    actors.push_back({bodies[i]->getPxActor(), i});
  }

  std::sort(actors.begin(), actors.end(),
            [](ActorQuery const &a, ActorQuery const &b) { return a.actor < b.actor; });
  static_assert(sizeof(ActorQuery) == 16);

  ensureCudaDevice();
  CudaArray query({static_cast<int>(actors.size()), 4}, "i4");
  checkCudaErrors(cudaMemcpy(query.ptr, actors.data(), actors.size() * sizeof(ActorQuery),
                             cudaMemcpyHostToDevice));
  CudaArray buffer({static_cast<int>(actors.size()), 3}, "f4");

  // TODO: use dedicated type, do not reuse contact query
  auto res = std::make_shared<PhysxGpuContactBodyImpulseQuery>();
  res->query = std::move(query);
  res->buffer = std::move(buffer);
  return res;
}
#endif

inline static int upperPowerOf2(int x) {
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  return x;
}

#ifdef SAPIEN_CUDA
namespace {
inline void *byteOffset(void *ptr, size_t offset) {
  return static_cast<void *>(static_cast<char *>(ptr) + offset);
}

inline void checkCudaIndexBuffer(CudaArrayHandle const &indices, int cudaId) {
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});
  if (typestrCode(indices.type) != 'i' || typestrBytes(indices.type) != sizeof(int)) {
    throw std::runtime_error("index buffer must be a CUDA int32 array");
  }
  if (!indices.shape.empty() && indices.shape.at(0) == 0) {
    return;
  }
  if (indices.cudaId != cudaId) {
    throw std::runtime_error("index buffer must be on the same CUDA device as the PhysX system");
  }
}

inline void *rigidDynamicPoseScratch(CudaArray &scratch) { return scratch.ptr; }

inline void *rigidDynamicLinearVelocityScratch(CudaArray &scratch, int count) {
  return byteOffset(scratch.ptr, static_cast<size_t>(count) * sizeof(PhysxPose));
}

inline void *rigidDynamicAngularVelocityScratch(CudaArray &scratch, int count) {
  return byteOffset(rigidDynamicLinearVelocityScratch(scratch, count),
                    static_cast<size_t>(count) * sizeof(Vec3));
}

inline void *articulationLinearVelocityScratch(CudaArray &scratch, int articulationCount,
                                               int maxLinkCount) {
  PX_UNUSED(articulationCount);
  PX_UNUSED(maxLinkCount);
  return scratch.ptr;
}

inline void *articulationAngularVelocityScratch(CudaArray &scratch, int articulationCount,
                                                int maxLinkCount) {
  return byteOffset(scratch.ptr, static_cast<size_t>(articulationCount) * maxLinkCount *
                                     3 * sizeof(float));
}
}

void PhysxSystemGpu::copyContactData() {
  checkGpuInitialized();
  if (mContactUpToDate) {
    return;
  }

  ensureCudaDevice();
  if (!mCudaContactCount.ptr) {
    mCudaContactCount = CudaArray({1}, "u4");
  }

  if (!mCudaContactBuffer.ptr) {
    mCudaContactBuffer = CudaArray({1024, sizeof(PxGpuContactPair)}, "u1");
  }

  auto &gpuApi = mPxScene->getDirectGPUAPI();

  if (!mCudaContactCountReadyEvent.event) {
    mCudaContactCountReadyEvent.init();
    mCudaContactDataReadyEvent.init();
  }

  // A prior asynchronous query may still read mCudaContactBuffer on mCudaStream. Make the
  // PhysX copy stream wait before reusing that buffer for the next simulation step.
  mCudaEventRecord.record(mCudaStream);
  CUevent queryConsumersDone = mCudaEventRecord.event;

  // Optimistically copy into the previous high-water-mark buffer. PhysX writes the actual
  // count even when the output is truncated, so only a contact-count spike needs a second copy.
  int capacity = mCudaContactBuffer.shape[0];
  SAPIEN_PROFILE_BLOCK_BEGIN("fetch contact data and count");
  bool success = gpuApi.copyContactData(
      mCudaContactBuffer.ptr, (PxU32 *)mCudaContactCount.ptr, capacity, queryConsumersDone,
      mCudaContactCountReadyEvent.event);
  if (!success) {
    throw std::runtime_error("failed to fetch PhysX GPU contact data");
  }
  mCudaContactCountReadyEvent.synchronize();
  checkCudaErrors(cudaMemcpy(&mContactCount, mCudaContactCount.ptr, sizeof(PxU32),
                             cudaMemcpyDeviceToHost));
  SAPIEN_PROFILE_BLOCK_END;

  if (mContactCount > capacity) {
    int size = upperPowerOf2(mContactCount);
    SAPIEN_PROFILE_BLOCK("grow and refill contact buffer");
    mCudaContactBuffer = CudaArray({size, sizeof(PxGpuContactPair)}, "u1");
    success = gpuApi.copyContactData(
        mCudaContactBuffer.ptr, (PxU32 *)mCudaContactCount.ptr, size, queryConsumersDone,
        mCudaContactDataReadyEvent.event);
    if (!success) {
      throw std::runtime_error("failed to refill grown PhysX GPU contact buffer");
    }
    mCudaContactDataReadyEvent.wait(mCudaStream);
  }
  mContactUpToDate = true;
}

void PhysxSystemGpu::gpuQueryContactPairImpulses(
    PhysxGpuContactPairImpulseQuery const &query, bool synchronize) {
  checkGpuInitialized();
  SAPIEN_PROFILE_FUNCTION;
  query.query.handle().checkShape({-1, 6});

  ensureCudaDevice();
  cudaMemsetAsync(query.buffer.ptr, 0, query.query.shape.at(0) * 3 * sizeof(float), mCudaStream);

  copyContactData();

  if (mContactCount) {
    handle_contacts((PxGpuContactPair *)mCudaContactBuffer.ptr, mContactCount,

                    (ActorPairQuery *)query.query.ptr, query.query.shape.at(0),

                    (Vec3 *)query.buffer.ptr, mCudaStream);
  }
  if (synchronize) {
    checkCudaErrors(cudaStreamSynchronize(mCudaStream));
    mContactQueryInFlight = false;
  } else {
    mContactQueryInFlight = true;
  }
}

void PhysxSystemGpu::gpuQueryContactBodyImpulses(
    PhysxGpuContactBodyImpulseQuery const &query, bool synchronize) {
  checkGpuInitialized();
  SAPIEN_PROFILE_FUNCTION;
  query.query.handle().checkShape({-1, 4});

  ensureCudaDevice();
  cudaMemsetAsync(query.buffer.ptr, 0, query.query.shape.at(0) * 3 * sizeof(float), mCudaStream);

  copyContactData();

  if (mContactCount) {
    handle_net_contact_force((PxGpuContactPair *)mCudaContactBuffer.ptr, mContactCount,
                             (ActorQuery *)query.query.ptr, query.query.shape.at(0),
                             (Vec3 *)query.buffer.ptr, mCudaStream);
  }
  if (synchronize) {
    checkCudaErrors(cudaStreamSynchronize(mCudaStream));
    mContactQueryInFlight = false;
  } else {
    mContactQueryInFlight = true;
  }
}

void PhysxSystemGpu::gpuWaitContactQueries() {
  checkGpuInitialized();
  ensureCudaDevice();
  checkCudaErrors(cudaStreamSynchronize(mCudaStream));
  mContactQueryInFlight = false;
}

void PhysxSystemGpu::gpuFetchRigidDynamicData() {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  ++mRigidDynamicFetchCount;

  if (mRigidDynamicComponents.empty()) {
    mRigidDynamicDataFetchedStep = mTotalSteps;
    return;
  }

  ensureCudaDevice();
  auto &gpuApi = mPxScene->getDirectGPUAPI();
  auto count = static_cast<PxU32>(mCudaRigidDynamicIndexBuffer.shape.at(0));
  auto poseScratch = rigidDynamicPoseScratch(mCudaRigidDynamicScratch);
  auto linearVelocityScratch = rigidDynamicLinearVelocityScratch(mCudaRigidDynamicScratch, count);
  auto angularVelocityScratch = rigidDynamicAngularVelocityScratch(mCudaRigidDynamicScratch, count);

  if (!mCudaRigidPoseFetchEvent.event) {
    mCudaRigidPoseFetchEvent.init();
    mCudaRigidLinearVelocityFetchEvent.init();
    mCudaRigidAngularVelocityFetchEvent.init();
  }
  // PhysX must not overwrite scratch until the previous SAPIEN conversion and
  // configured-stream consumers have finished reading it.
  mCudaEventRecord.record(mCudaStream);
  SAPIEN_PROFILE_BLOCK_BEGIN("PhysX rigid data fetch");
  bool success = gpuApi.getRigidDynamicData(
      poseScratch, (PxRigidDynamicGPUIndex *)mCudaRigidDynamicIndexBuffer.ptr,
      PxRigidDynamicGPUAPIReadType::eGLOBAL_POSE, count, mCudaEventRecord.event,
      mCudaRigidPoseFetchEvent.event);
  success &= gpuApi.getRigidDynamicData(
      linearVelocityScratch, (PxRigidDynamicGPUIndex *)mCudaRigidDynamicIndexBuffer.ptr,
      PxRigidDynamicGPUAPIReadType::eLINEAR_VELOCITY, count, mCudaEventRecord.event,
      mCudaRigidLinearVelocityFetchEvent.event);
  success &= gpuApi.getRigidDynamicData(
      angularVelocityScratch, (PxRigidDynamicGPUIndex *)mCudaRigidDynamicIndexBuffer.ptr,
      PxRigidDynamicGPUAPIReadType::eANGULAR_VELOCITY, count, mCudaEventRecord.event,
      mCudaRigidAngularVelocityFetchEvent.event);
  SAPIEN_PROFILE_BLOCK_END;
  if (!success) {
    throw std::runtime_error("failed to fetch PhysX GPU rigid dynamic data");
  }

  SAPIEN_PROFILE_BLOCK_BEGIN("PhysX rigid data conversion");
  mCudaRigidPoseFetchEvent.wait(mCudaStream);
  mCudaRigidLinearVelocityFetchEvent.wait(mCudaStream);
  mCudaRigidAngularVelocityFetchEvent.wait(mCudaStream);
  body_data_physx_to_sapien(mCudaRigidDynamicHandle.ptr, poseScratch, linearVelocityScratch,
                            angularVelocityScratch, mCudaRigidDynamicOffsetBuffer.ptr, count,
                            mCudaStream);
  SAPIEN_PROFILE_BLOCK_END;
  mRigidDynamicDataFetchedStep = mTotalSteps;
}

void PhysxSystemGpu::gpuFetchRigidDynamicDataIfNeeded() {
  if (!mRigidDynamicDataFetchedStep || *mRigidDynamicDataFetchedStep != mTotalSteps) {
    gpuFetchRigidDynamicData();
  }
}

void PhysxSystemGpu::gpuFetchArticulationData(
    void *data, PxArticulationGPUAPIReadType::Enum type, CudaEvent &completionEvent) {
  if (!completionEvent.event) {
    completionEvent.init();
  }
  mCudaEventRecord.record(mCudaStream);
  bool success = mPxScene->getDirectGPUAPI().getArticulationData(
      data, (PxArticulationGPUIndex *)mCudaArticulationGpuIndexBuffer.ptr, type,
      mGpuArticulationCount, mCudaEventRecord.event, completionEvent.event);
  if (!success) {
    throw std::runtime_error("failed to fetch PhysX GPU articulation data");
  }
  completionEvent.wait(mCudaStream);
}

void PhysxSystemGpu::gpuFetchArticulationLinkPose() {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  ++mArticulationLinkPoseFetchCount;

  if (mGpuArticulationCount == 0) {
    mArticulationLinkPoseFetchedStep = mTotalSteps;
    return;
  }

  ensureCudaDevice();
  auto &gpuApi = mPxScene->getDirectGPUAPI();
  auto count = static_cast<PxU32>(mGpuArticulationCount);
  if (!mCudaArticulationLinkPoseFetchEvent.event) {
    mCudaArticulationLinkPoseFetchEvent.init();
  }
  // The pose scratch is reused by later fetches and root-pose applies.
  // Order its PhysX writer after earlier configured-stream readers.
  mCudaEventRecord.record(mCudaStream);
  SAPIEN_PROFILE_BLOCK_BEGIN("PhysX articulation link pose fetch");
  bool success = gpuApi.getArticulationData(
      mCudaLinkPoseScratch.ptr,
      (PxArticulationGPUIndex *)mCudaArticulationGpuIndexBuffer.ptr,
      PxArticulationGPUAPIReadType::eLINK_GLOBAL_POSE, count, mCudaEventRecord.event,
      mCudaArticulationLinkPoseFetchEvent.event);
  SAPIEN_PROFILE_BLOCK_END;
  if (!success) {
    throw std::runtime_error("failed to fetch PhysX GPU articulation link poses");
  }
  SAPIEN_PROFILE_BLOCK_BEGIN("PhysX articulation link pose conversion");
  mCudaArticulationLinkPoseFetchEvent.wait(mCudaStream);
  link_pose_physx_to_sapien(mCudaLinkHandle.ptr, mCudaLinkPoseScratch.ptr,
                            mCudaArticulationOffsetBuffer.ptr, mGpuArticulationMaxLinkCount,
                            count * mGpuArticulationMaxLinkCount, mCudaStream);
  SAPIEN_PROFILE_BLOCK_END;
  mArticulationLinkPoseFetchedStep = mTotalSteps;
}

void PhysxSystemGpu::gpuFetchArticulationLinkPoseIfNeeded() {
  if (!mArticulationLinkPoseFetchedStep || *mArticulationLinkPoseFetchedStep != mTotalSteps) {
    gpuFetchArticulationLinkPose();
  }
}

void PhysxSystemGpu::gpuFetchArticulationLinkVel() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();
  auto count = static_cast<PxU32>(mGpuArticulationCount);
  auto linearVelocityScratch =
      articulationLinearVelocityScratch(mCudaLinkVelScratch, count, mGpuArticulationMaxLinkCount);
  auto angularVelocityScratch =
      articulationAngularVelocityScratch(mCudaLinkVelScratch, count, mGpuArticulationMaxLinkCount);

  gpuFetchArticulationData(linearVelocityScratch,
                           PxArticulationGPUAPIReadType::eLINK_LINEAR_VELOCITY,
                           mCudaArticulationLinkLinearVelocityFetchEvent);
  gpuFetchArticulationData(angularVelocityScratch,
                           PxArticulationGPUAPIReadType::eLINK_ANGULAR_VELOCITY,
                           mCudaArticulationLinkAngularVelocityFetchEvent);

  link_vel_physx_to_sapien(mCudaLinkHandle.ptr, linearVelocityScratch, angularVelocityScratch,
                           count * mGpuArticulationMaxLinkCount, mCudaStream);
}

void PhysxSystemGpu::gpuFetchArticulationQpos() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();
  gpuFetchArticulationData(mCudaQposHandle.ptr, PxArticulationGPUAPIReadType::eJOINT_POSITION,
                           mCudaArticulationQposFetchEvent);
}

void PhysxSystemGpu::gpuFetchArticulationQvel() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();
  gpuFetchArticulationData(mCudaQvelHandle.ptr, PxArticulationGPUAPIReadType::eJOINT_VELOCITY,
                           mCudaArticulationQvelFetchEvent);
}

void PhysxSystemGpu::gpuFetchArticulationQTargetPos() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();
  gpuFetchArticulationData(mCudaQTargetPosHandle.ptr,
                           PxArticulationGPUAPIReadType::eJOINT_TARGET_POSITION,
                           mCudaArticulationTargetQposFetchEvent);
}

void PhysxSystemGpu::gpuFetchArticulationQTargetVel() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();
  gpuFetchArticulationData(mCudaQTargetVelHandle.ptr,
                           PxArticulationGPUAPIReadType::eJOINT_TARGET_VELOCITY,
                           mCudaArticulationTargetQvelFetchEvent);
}

void PhysxSystemGpu::gpuComputeArticulationJacobian() {
  gpuComputeArticulationJacobian(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuComputeArticulationJacobian(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  checkCudaIndexBuffer(indices, mDevice->cudaId);

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();
  auto count = static_cast<PxU32>(indices.shape.at(0));
  if (count == 0) {
    return;
  }

  int maxRows = 6 + (mGpuArticulationMaxLinkCount - 1) * 6;
  int maxCols = 6 + mGpuArticulationMaxDof;
  void *gpuIndices = mCudaArticulationGpuIndexBuffer.ptr;
  CUevent startEvent = nullptr;
  if (indices.ptr != mCudaArticulationIndexBuffer.ptr) {
    gather_blocks(mCudaArticulationIndexScratch.ptr, mCudaArticulationGpuIndexBuffer.ptr,
                  indices.ptr, 1, count, mCudaStream);
    gpuIndices = mCudaArticulationIndexScratch.ptr;
    mCudaEventRecord.record(mCudaStream);
    startEvent = mCudaEventRecord.event;
  }

  if (!mCudaEventWait.event) {
    mCudaEventWait.init();
  }
  mPxScene->getDirectGPUAPI().computeArticulationData(
      mCudaArticulationJacobianScratch.ptr,
      (PxArticulationGPUIndex *)gpuIndices, PxArticulationGPUAPIComputeType::eDENSE_JACOBIANS,
      count, startEvent,
      mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
  scatter_articulation_jacobians(mCudaArticulationJacobianHandle.ptr,
                                 mCudaArticulationJacobianScratch.ptr, indices.ptr,
                                 mCudaArticulationJacobianShapeBuffer.ptr, maxRows, maxCols,
                                 count, mCudaStream);
}

void PhysxSystemGpu::gpuComputeArticulationGravityCompensation() {
  checkGpuInitialized();
  gpuComputeArticulationGravityCompensation(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuComputeArticulationGravityCompensation(
    CudaArrayHandle const &indices) {
  gpuComputeArticulationCompensation(indices, mCudaArticulationGravityCompensationHandle,
                                     PxArticulationGPUAPIComputeType::eGRAVITY_COMPENSATION);
}

void PhysxSystemGpu::gpuComputeArticulationCoriolisAndCentrifugalCompensation() {
  checkGpuInitialized();
  gpuComputeArticulationCoriolisAndCentrifugalCompensation(
      mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuComputeArticulationCoriolisAndCentrifugalCompensation(
    CudaArrayHandle const &indices) {
  gpuComputeArticulationCompensation(
      indices, mCudaArticulationCoriolisAndCentrifugalCompensationHandle,
      PxArticulationGPUAPIComputeType::eCORIOLIS_AND_CENTRIFUGAL_COMPENSATION);
}

void PhysxSystemGpu::gpuComputeArticulationCompensation(
    CudaArrayHandle const &indices, CudaArrayHandle const &output,
    PxArticulationGPUAPIComputeType::Enum computeType) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  checkCudaIndexBuffer(indices, mDevice->cudaId);

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();
  auto count = static_cast<PxU32>(indices.shape.at(0));
  if (count == 0) {
    return;
  }

  void *gpuIndices = mCudaArticulationGpuIndexBuffer.ptr;
  CUevent startEvent = nullptr;
  if (indices.ptr != mCudaArticulationIndexBuffer.ptr) {
    gather_blocks(mCudaArticulationIndexScratch.ptr, mCudaArticulationGpuIndexBuffer.ptr,
                  indices.ptr, 1, count, mCudaStream);
    gpuIndices = mCudaArticulationIndexScratch.ptr;
    mCudaEventRecord.record(mCudaStream);
    startEvent = mCudaEventRecord.event;
  }

  if (!mCudaEventWait.event) {
    mCudaEventWait.init();
  }
  mPxScene->getDirectGPUAPI().computeArticulationData(
      mCudaArticulationCompensationScratch.ptr, (PxArticulationGPUIndex *)gpuIndices,
      computeType, count, startEvent, mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
  scatter_articulation_joint_forces(output.ptr, mCudaArticulationCompensationScratch.ptr,
                                    indices.ptr, mCudaArticulationCompensationMetaBuffer.ptr,
                                    mGpuArticulationMaxDof, count, mCudaStream);
}

void PhysxSystemGpu::gpuFetchArticulationLinkIncomingJointForce() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();
  gpuFetchArticulationData(mCudaArticulationLinkIncomingJointForceBuffer.ptr,
                           PxArticulationGPUAPIReadType::eLINK_INCOMING_JOINT_FORCE,
                           mCudaArticulationIncomingJointForceFetchEvent);
}

void PhysxSystemGpu::gpuFetchArticulationQacc() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }
  ensureCudaDevice();
  gpuFetchArticulationData(mCudaQaccHandle.ptr,
                           PxArticulationGPUAPIReadType::eJOINT_ACCELERATION,
                           mCudaArticulationQaccFetchEvent);
}

void PhysxSystemGpu::gpuUpdateArticulationKinematics() {
  gpuUpdateArticulationKinematics(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuUpdateArticulationKinematics(CudaArrayHandle const &indices) {
  checkGpuInitialized();
  invalidateArticulationLinkPose();
  checkCudaIndexBuffer(indices, mDevice->cudaId);

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();
  auto count = static_cast<PxU32>(indices.shape.at(0));
  if (count == 0) {
    return;
  }

  void *gpuIndices = mCudaArticulationGpuIndexBuffer.ptr;
  CUevent startEvent = nullptr;
  if (indices.ptr != mCudaArticulationIndexBuffer.ptr) {
    gather_blocks(mCudaArticulationIndexScratch.ptr, mCudaArticulationGpuIndexBuffer.ptr,
                  indices.ptr, 1, count, mCudaStream);
    gpuIndices = mCudaArticulationIndexScratch.ptr;
    mCudaEventRecord.record(mCudaStream);
    startEvent = mCudaEventRecord.event;
  }

  if (!mCudaEventWait.event) {
    mCudaEventWait.init();
  }
  mPxScene->getDirectGPUAPI().computeArticulationData(
      nullptr, (PxArticulationGPUIndex *)gpuIndices,
      PxArticulationGPUAPIComputeType::eUPDATE_KINEMATIC, count, startEvent,
      mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
}

void PhysxSystemGpu::gpuApplyRigidDynamicData() {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  invalidateRigidDynamicData();

  if (mRigidDynamicComponents.empty()) {
    return;
  }

  ensureCudaDevice();
  auto &gpuApi = mPxScene->getDirectGPUAPI();
  auto count = static_cast<PxU32>(mCudaRigidDynamicIndexBuffer.shape.at(0));
  auto poseScratch = rigidDynamicPoseScratch(mCudaRigidDynamicScratch);
  auto linearVelocityScratch = rigidDynamicLinearVelocityScratch(mCudaRigidDynamicScratch, count);
  auto angularVelocityScratch = rigidDynamicAngularVelocityScratch(mCudaRigidDynamicScratch, count);

  body_data_sapien_to_physx(poseScratch, linearVelocityScratch, angularVelocityScratch,
                            mCudaRigidDynamicHandle.ptr, mCudaRigidDynamicOffsetBuffer.ptr, count,
                            mCudaStream);

  mCudaEventRecord.record(mCudaStream);
  gpuApi.setRigidDynamicData(poseScratch, (PxRigidDynamicGPUIndex *)mCudaRigidDynamicIndexBuffer.ptr,
                             PxRigidDynamicGPUAPIWriteType::eGLOBAL_POSE, count,
                             mCudaEventRecord.event);
  gpuApi.setRigidDynamicData(linearVelocityScratch,
                             (PxRigidDynamicGPUIndex *)mCudaRigidDynamicIndexBuffer.ptr,
                             PxRigidDynamicGPUAPIWriteType::eLINEAR_VELOCITY, count,
                             mCudaEventRecord.event);
  gpuApi.setRigidDynamicData(angularVelocityScratch,
                             (PxRigidDynamicGPUIndex *)mCudaRigidDynamicIndexBuffer.ptr,
                             PxRigidDynamicGPUAPIWriteType::eANGULAR_VELOCITY, count,
                             mCudaEventRecord.event);
}

void PhysxSystemGpu::gpuApplyRigidDynamicData(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  invalidateRigidDynamicData();
  checkCudaIndexBuffer(indices, mDevice->cudaId);

  if (mRigidDynamicComponents.empty()) {
    return;
  }

  ensureCudaDevice();
  auto &gpuApi = mPxScene->getDirectGPUAPI();
  auto count = static_cast<PxU32>(indices.shape.at(0));
  auto poseScratch = rigidDynamicPoseScratch(mCudaRigidDynamicScratch);
  auto linearVelocityScratch = rigidDynamicLinearVelocityScratch(mCudaRigidDynamicScratch, count);
  auto angularVelocityScratch = rigidDynamicAngularVelocityScratch(mCudaRigidDynamicScratch, count);

  body_data_sapien_to_physx(poseScratch, linearVelocityScratch, angularVelocityScratch,
                            mCudaRigidDynamicIndexScratch.ptr, mCudaRigidDynamicHandle.ptr,
                            mCudaRigidDynamicIndexBuffer.ptr, indices.ptr,
                            mCudaRigidDynamicOffsetBuffer.ptr, count, mCudaStream);
  mCudaEventRecord.record(mCudaStream);
  gpuApi.setRigidDynamicData(poseScratch, (PxRigidDynamicGPUIndex *)mCudaRigidDynamicIndexScratch.ptr,
                             PxRigidDynamicGPUAPIWriteType::eGLOBAL_POSE, count,
                             mCudaEventRecord.event);
  gpuApi.setRigidDynamicData(linearVelocityScratch,
                             (PxRigidDynamicGPUIndex *)mCudaRigidDynamicIndexScratch.ptr,
                             PxRigidDynamicGPUAPIWriteType::eLINEAR_VELOCITY, count,
                             mCudaEventRecord.event);
  gpuApi.setRigidDynamicData(angularVelocityScratch,
                             (PxRigidDynamicGPUIndex *)mCudaRigidDynamicIndexScratch.ptr,
                             PxRigidDynamicGPUAPIWriteType::eANGULAR_VELOCITY, count,
                             mCudaEventRecord.event);
}

void PhysxSystemGpu::gpuApplyRigidDynamicForce() {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  if (mRigidDynamicComponents.empty()) {
    return;
  }

  ensureCudaDevice();
  auto count = static_cast<PxU32>(mCudaRigidDynamicIndexBuffer.shape.at(0));
  pack_vec3(mCudaRigidDynamicScratch.ptr, mCudaRigidDynamicForceHandle.ptr, 4, count, mCudaStream);
  mCudaEventRecord.record(mCudaStream);
  mPxScene->getDirectGPUAPI().setRigidDynamicData(
      mCudaRigidDynamicScratch.ptr, (PxRigidDynamicGPUIndex *)mCudaRigidDynamicIndexBuffer.ptr,
      PxRigidDynamicGPUAPIWriteType::eFORCE, count, mCudaEventRecord.event);
}

void PhysxSystemGpu::gpuApplyRigidDynamicForce(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  checkCudaIndexBuffer(indices, mDevice->cudaId);
  auto count = static_cast<PxU32>(indices.shape[0]);
  if (count == 0) {
    return;
  }

  ensureCudaDevice();
  gather_blocks(mCudaRigidDynamicForcePaddedScratch.ptr, mCudaRigidDynamicForceHandle.ptr,
                indices.ptr, 4, count, mCudaStream);
  pack_vec3(mCudaRigidDynamicForcePackedScratch.ptr,
            mCudaRigidDynamicForcePaddedScratch.ptr, 4, count, mCudaStream);
  gather_blocks(mCudaRigidDynamicIndexScratch.ptr, mCudaRigidDynamicIndexBuffer.ptr,
                indices.ptr, 1, count, mCudaStream);
  mCudaEventRecord.record(mCudaStream);
  mPxScene->getDirectGPUAPI().setRigidDynamicData(
      mCudaRigidDynamicForcePackedScratch.ptr,
      static_cast<PxRigidDynamicGPUIndex *>(mCudaRigidDynamicIndexScratch.ptr),
      PxRigidDynamicGPUAPIWriteType::eFORCE, count, mCudaEventRecord.event);
}

void PhysxSystemGpu::gpuApplyRigidDynamicTorque() {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  if (mRigidDynamicComponents.empty()) {
    return;
  }

  ensureCudaDevice();
  auto count = static_cast<PxU32>(mCudaRigidDynamicIndexBuffer.shape.at(0));
  pack_vec3(mCudaRigidDynamicScratch.ptr, mCudaRigidDynamicTorqueHandle.ptr, 4, count,
            mCudaStream);
  mCudaEventRecord.record(mCudaStream);
  mPxScene->getDirectGPUAPI().setRigidDynamicData(
      mCudaRigidDynamicScratch.ptr, (PxRigidDynamicGPUIndex *)mCudaRigidDynamicIndexBuffer.ptr,
      PxRigidDynamicGPUAPIWriteType::eTORQUE, count, mCudaEventRecord.event);
}

void PhysxSystemGpu::gpuApplyRigidDynamicTorque(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  checkCudaIndexBuffer(indices, mDevice->cudaId);
  auto count = static_cast<PxU32>(indices.shape[0]);
  if (count == 0) {
    return;
  }

  ensureCudaDevice();
  gather_blocks(mCudaRigidDynamicTorquePaddedScratch.ptr, mCudaRigidDynamicTorqueHandle.ptr,
                indices.ptr, 4, count, mCudaStream);
  pack_vec3(mCudaRigidDynamicTorquePackedScratch.ptr,
            mCudaRigidDynamicTorquePaddedScratch.ptr, 4, count, mCudaStream);
  gather_blocks(mCudaRigidDynamicIndexScratch.ptr, mCudaRigidDynamicIndexBuffer.ptr,
                indices.ptr, 1, count, mCudaStream);
  mCudaEventRecord.record(mCudaStream);
  mPxScene->getDirectGPUAPI().setRigidDynamicData(
      mCudaRigidDynamicTorquePackedScratch.ptr,
      static_cast<PxRigidDynamicGPUIndex *>(mCudaRigidDynamicIndexScratch.ptr),
      PxRigidDynamicGPUAPIWriteType::eTORQUE, count, mCudaEventRecord.event);
}

void PhysxSystemGpu::gpuApplyArticulationLinkForce() {
  gpuApplyArticulationLinkForce(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuApplyArticulationLinkForce(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  checkCudaIndexBuffer(indices, mDevice->cudaId);

  if (mGpuArticulationCount == 0) {
    return;
  }
  ensureCudaDevice();

  auto count = static_cast<PxU32>(indices.shape.at(0));
  if (count == 0) {
    return;
  }

  void *paddedData = mCudaArticulationLinkForceHandle.ptr;
  void *gpuIndices = mCudaArticulationGpuIndexBuffer.ptr;
  if (indices.ptr != mCudaArticulationIndexBuffer.ptr) {
    gather_blocks(mCudaArticulationLinkForcePaddedScratch.ptr,
                  mCudaArticulationLinkForceHandle.ptr, indices.ptr,
                  mGpuArticulationMaxLinkCount * 4, count, mCudaStream);
    gather_blocks(mCudaArticulationIndexScratch.ptr, mCudaArticulationGpuIndexBuffer.ptr,
                  indices.ptr, 1, count, mCudaStream);
    paddedData = mCudaArticulationLinkForcePaddedScratch.ptr;
    gpuIndices = mCudaArticulationIndexScratch.ptr;
  }

  pack_vec3(mCudaArticulationLinkForcePackedScratch.ptr, paddedData, 4,
            count * mGpuArticulationMaxLinkCount, mCudaStream);
  mCudaEventRecord.record(mCudaStream);
  mPxScene->getDirectGPUAPI().setArticulationData(
      mCudaArticulationLinkForcePackedScratch.ptr, (PxArticulationGPUIndex *)gpuIndices,
      PxArticulationGPUAPIWriteType::eLINK_FORCE, count, mCudaEventRecord.event);
}

void PhysxSystemGpu::gpuApplyArticulationLinkTorque() {
  gpuApplyArticulationLinkTorque(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuApplyArticulationLinkTorque(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  checkCudaIndexBuffer(indices, mDevice->cudaId);

  if (mGpuArticulationCount == 0) {
    return;
  }
  ensureCudaDevice();

  auto count = static_cast<PxU32>(indices.shape.at(0));
  if (count == 0) {
    return;
  }

  void *paddedData = mCudaArticulationLinkTorqueHandle.ptr;
  void *gpuIndices = mCudaArticulationGpuIndexBuffer.ptr;
  if (indices.ptr != mCudaArticulationIndexBuffer.ptr) {
    gather_blocks(mCudaArticulationLinkTorquePaddedScratch.ptr,
                  mCudaArticulationLinkTorqueHandle.ptr, indices.ptr,
                  mGpuArticulationMaxLinkCount * 4, count, mCudaStream);
    gather_blocks(mCudaArticulationIndexScratch.ptr, mCudaArticulationGpuIndexBuffer.ptr,
                  indices.ptr, 1, count, mCudaStream);
    paddedData = mCudaArticulationLinkTorquePaddedScratch.ptr;
    gpuIndices = mCudaArticulationIndexScratch.ptr;
  }

  pack_vec3(mCudaArticulationLinkTorquePackedScratch.ptr, paddedData, 4,
            count * mGpuArticulationMaxLinkCount, mCudaStream);
  mCudaEventRecord.record(mCudaStream);
  mPxScene->getDirectGPUAPI().setArticulationData(
      mCudaArticulationLinkTorquePackedScratch.ptr, (PxArticulationGPUIndex *)gpuIndices,
      PxArticulationGPUAPIWriteType::eLINK_TORQUE, count, mCudaEventRecord.event);
}

void PhysxSystemGpu::gpuApplyArticulationRootPose() {
  gpuApplyArticulationRootPose(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuApplyArticulationRootPose(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  invalidateArticulationLinkPose();
  checkCudaIndexBuffer(indices, mDevice->cudaId);

  if (mGpuArticulationCount == 0) {
    return;
  }
  ensureCudaDevice();

  auto count = static_cast<PxU32>(indices.shape.at(0));
  void *gpuIndices = mCudaArticulationGpuIndexBuffer.ptr;
  if (indices.ptr != mCudaArticulationIndexBuffer.ptr) {
    gather_blocks(mCudaArticulationIndexScratch.ptr, mCudaArticulationGpuIndexBuffer.ptr,
                  indices.ptr, 1, count, mCudaStream);
    gpuIndices = mCudaArticulationIndexScratch.ptr;
  }

  root_pose_sapien_to_physx(mCudaLinkPoseScratch.ptr, mCudaLinkHandle.ptr, indices.ptr,
                            mCudaArticulationOffsetBuffer.ptr, mGpuArticulationMaxLinkCount,
                            count, mCudaStream);
  mCudaEventRecord.record(mCudaStream);
  mPxScene->getDirectGPUAPI().setArticulationData(
      mCudaLinkPoseScratch.ptr, (PxArticulationGPUIndex *)gpuIndices,
      PxArticulationGPUAPIWriteType::eROOT_GLOBAL_POSE, count, mCudaEventRecord.event);
}

void PhysxSystemGpu::gpuApplyArticulationRootVel() {
  gpuApplyArticulationRootVel(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuApplyArticulationRootVel(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  checkCudaIndexBuffer(indices, mDevice->cudaId);

  if (mGpuArticulationCount == 0) {
    return;
  }
  ensureCudaDevice();

  auto count = static_cast<PxU32>(indices.shape.at(0));
  void *gpuIndices = mCudaArticulationGpuIndexBuffer.ptr;
  if (indices.ptr != mCudaArticulationIndexBuffer.ptr) {
    gather_blocks(mCudaArticulationIndexScratch.ptr, mCudaArticulationGpuIndexBuffer.ptr,
                  indices.ptr, 1, count, mCudaStream);
    gpuIndices = mCudaArticulationIndexScratch.ptr;
  }

  auto linearVelocityScratch = articulationLinearVelocityScratch(mCudaLinkVelScratch, count, 1);
  auto angularVelocityScratch = articulationAngularVelocityScratch(mCudaLinkVelScratch, count, 1);
  root_vel_sapien_to_physx(linearVelocityScratch, angularVelocityScratch, mCudaLinkHandle.ptr,
                           indices.ptr, mGpuArticulationMaxLinkCount, count, mCudaStream);
  mCudaEventRecord.record(mCudaStream);
  mPxScene->getDirectGPUAPI().setArticulationData(
      linearVelocityScratch, (PxArticulationGPUIndex *)gpuIndices,
      PxArticulationGPUAPIWriteType::eROOT_LINEAR_VELOCITY, count, mCudaEventRecord.event);
  mPxScene->getDirectGPUAPI().setArticulationData(
      angularVelocityScratch, (PxArticulationGPUIndex *)gpuIndices,
      PxArticulationGPUAPIWriteType::eROOT_ANGULAR_VELOCITY, count, mCudaEventRecord.event);
}

void PhysxSystemGpu::gpuApplyArticulationQpos() {
  gpuApplyArticulationQpos(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuApplyArticulationQpos(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  invalidateArticulationLinkPose();
  checkCudaIndexBuffer(indices, mDevice->cudaId);

  if (mGpuArticulationCount == 0) {
    return;
  }
  ensureCudaDevice();

  auto count = static_cast<PxU32>(indices.shape.at(0));
  void *data = mCudaQposHandle.ptr;
  void *gpuIndices = mCudaArticulationGpuIndexBuffer.ptr;
  if (indices.ptr != mCudaArticulationIndexBuffer.ptr) {
    gather_blocks(mCudaArticulationApplyScratch.ptr, mCudaQposHandle.ptr, indices.ptr,
                  mGpuArticulationMaxDof, count, mCudaStream);
    gather_blocks(mCudaArticulationIndexScratch.ptr, mCudaArticulationGpuIndexBuffer.ptr,
                  indices.ptr, 1, count, mCudaStream);
    data = mCudaArticulationApplyScratch.ptr;
    gpuIndices = mCudaArticulationIndexScratch.ptr;
  }

  mCudaEventRecord.record(mCudaStream);
  mPxScene->getDirectGPUAPI().setArticulationData(
      data, (PxArticulationGPUIndex *)gpuIndices, PxArticulationGPUAPIWriteType::eJOINT_POSITION,
      count, mCudaEventRecord.event);
}

void PhysxSystemGpu::gpuApplyArticulationQvel() {
  gpuApplyArticulationQvel(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuApplyArticulationQvel(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  checkCudaIndexBuffer(indices, mDevice->cudaId);

  if (mGpuArticulationCount == 0) {
    return;
  }
  ensureCudaDevice();

  auto count = static_cast<PxU32>(indices.shape.at(0));
  void *data = mCudaQvelHandle.ptr;
  void *gpuIndices = mCudaArticulationGpuIndexBuffer.ptr;
  if (indices.ptr != mCudaArticulationIndexBuffer.ptr) {
    gather_blocks(mCudaArticulationApplyScratch.ptr, mCudaQvelHandle.ptr, indices.ptr,
                  mGpuArticulationMaxDof, count, mCudaStream);
    gather_blocks(mCudaArticulationIndexScratch.ptr, mCudaArticulationGpuIndexBuffer.ptr,
                  indices.ptr, 1, count, mCudaStream);
    data = mCudaArticulationApplyScratch.ptr;
    gpuIndices = mCudaArticulationIndexScratch.ptr;
  }

  mCudaEventRecord.record(mCudaStream);
  mPxScene->getDirectGPUAPI().setArticulationData(
      data, (PxArticulationGPUIndex *)gpuIndices, PxArticulationGPUAPIWriteType::eJOINT_VELOCITY,
      count, mCudaEventRecord.event);
}

void PhysxSystemGpu::gpuApplyArticulationQf() {
  gpuApplyArticulationQf(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuApplyArticulationQf(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  checkCudaIndexBuffer(indices, mDevice->cudaId);

  if (mGpuArticulationCount == 0) {
    return;
  }
  ensureCudaDevice();

  auto count = static_cast<PxU32>(indices.shape.at(0));
  void *data = mCudaQfHandle.ptr;
  void *gpuIndices = mCudaArticulationGpuIndexBuffer.ptr;
  if (indices.ptr != mCudaArticulationIndexBuffer.ptr) {
    gather_blocks(mCudaArticulationApplyScratch.ptr, mCudaQfHandle.ptr, indices.ptr,
                  mGpuArticulationMaxDof, count, mCudaStream);
    gather_blocks(mCudaArticulationIndexScratch.ptr, mCudaArticulationGpuIndexBuffer.ptr,
                  indices.ptr, 1, count, mCudaStream);
    data = mCudaArticulationApplyScratch.ptr;
    gpuIndices = mCudaArticulationIndexScratch.ptr;
  }

  mCudaEventRecord.record(mCudaStream);
  mPxScene->getDirectGPUAPI().setArticulationData(
      data, (PxArticulationGPUIndex *)gpuIndices, PxArticulationGPUAPIWriteType::eJOINT_FORCE,
      count, mCudaEventRecord.event);
}

void PhysxSystemGpu::gpuApplyArticulationQTargetPos() {
  gpuApplyArticulationQTargetPos(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuApplyArticulationQTargetPos(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  checkCudaIndexBuffer(indices, mDevice->cudaId);

  if (mGpuArticulationCount == 0) {
    return;
  }
  ensureCudaDevice();

  auto count = static_cast<PxU32>(indices.shape.at(0));
  void *data = mCudaQTargetPosHandle.ptr;
  void *gpuIndices = mCudaArticulationGpuIndexBuffer.ptr;
  if (indices.ptr != mCudaArticulationIndexBuffer.ptr) {
    gather_blocks(mCudaArticulationApplyScratch.ptr, mCudaQTargetPosHandle.ptr, indices.ptr,
                  mGpuArticulationMaxDof, count, mCudaStream);
    gather_blocks(mCudaArticulationIndexScratch.ptr, mCudaArticulationGpuIndexBuffer.ptr,
                  indices.ptr, 1, count, mCudaStream);
    data = mCudaArticulationApplyScratch.ptr;
    gpuIndices = mCudaArticulationIndexScratch.ptr;
  }

  mCudaEventRecord.record(mCudaStream);
  mPxScene->getDirectGPUAPI().setArticulationData(
      data, (PxArticulationGPUIndex *)gpuIndices,
      PxArticulationGPUAPIWriteType::eJOINT_TARGET_POSITION, count, mCudaEventRecord.event);
}

void PhysxSystemGpu::gpuApplyArticulationQTargetVel() {
  gpuApplyArticulationQTargetVel(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuApplyArticulationQTargetVel(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  checkCudaIndexBuffer(indices, mDevice->cudaId);

  if (mGpuArticulationCount == 0) {
    return;
  }
  ensureCudaDevice();

  auto count = static_cast<PxU32>(indices.shape.at(0));
  void *data = mCudaQTargetVelHandle.ptr;
  void *gpuIndices = mCudaArticulationGpuIndexBuffer.ptr;
  if (indices.ptr != mCudaArticulationIndexBuffer.ptr) {
    gather_blocks(mCudaArticulationApplyScratch.ptr, mCudaQTargetVelHandle.ptr, indices.ptr,
                  mGpuArticulationMaxDof, count, mCudaStream);
    gather_blocks(mCudaArticulationIndexScratch.ptr, mCudaArticulationGpuIndexBuffer.ptr,
                  indices.ptr, 1, count, mCudaStream);
    data = mCudaArticulationApplyScratch.ptr;
    gpuIndices = mCudaArticulationIndexScratch.ptr;
  }

  mCudaEventRecord.record(mCudaStream);
  mPxScene->getDirectGPUAPI().setArticulationData(
      data, (PxArticulationGPUIndex *)gpuIndices,
      PxArticulationGPUAPIWriteType::eJOINT_TARGET_VELOCITY, count, mCudaEventRecord.event);
}

void PhysxSystemGpu::gpuApplyViewerRigidDynamicWrench(
    int gpuIndex, Vec3 localAnchor, Vec3 localCenterOfMass, Vec3 target, float effectiveMass,
    float stiffness, float damping, float maxAcceleration) {
  checkGpuInitialized();
  if (gpuIndex < 0 || gpuIndex >= mCudaRigidDynamicHandle.shape[0]) {
    throw std::runtime_error("failed to apply Viewer wrench: invalid rigid dynamic GPU index");
  }
  if (!localAnchor.isSane() || !localCenterOfMass.isSane() || !target.isSane() ||
      !std::isfinite(effectiveMass) || !std::isfinite(stiffness) || !std::isfinite(damping) ||
      !std::isfinite(maxAcceleration) || effectiveMass <= 0.f || stiffness < 0.f ||
      damping < 0.f || maxAcceleration < 0.f) {
    throw std::runtime_error("failed to apply Viewer wrench: invalid spring parameters");
  }

  gpuFetchRigidDynamicDataIfNeeded();
  ensureCudaDevice();
  compose_viewer_wrench(
      static_cast<Vec3 *>(mCudaViewerForceScratch.ptr),
      static_cast<Vec3 *>(mCudaViewerTorqueScratch.ptr),
      static_cast<float const *>(mCudaRigidDynamicForceHandle.ptr),
      static_cast<float const *>(mCudaRigidDynamicTorqueHandle.ptr),
      static_cast<SapienBodyData const *>(mCudaRigidBodyBuffer.ptr), gpuIndex, gpuIndex,
      localAnchor, localCenterOfMass, target, effectiveMass, stiffness, damping,
      maxAcceleration, mCudaStream);
  mCudaEventRecord.record(mCudaStream);

  auto *physxIndex = static_cast<PxRigidDynamicGPUIndex *>(mCudaRigidDynamicIndexBuffer.ptr) +
                     gpuIndex;
  auto &gpuApi = mPxScene->getDirectGPUAPI();
  gpuApi.setRigidDynamicData(mCudaViewerForceScratch.ptr, physxIndex,
                             PxRigidDynamicGPUAPIWriteType::eFORCE, 1,
                             mCudaEventRecord.event);
  gpuApi.setRigidDynamicData(mCudaViewerTorqueScratch.ptr, physxIndex,
                             PxRigidDynamicGPUAPIWriteType::eTORQUE, 1,
                             mCudaEventRecord.event);
}

void PhysxSystemGpu::gpuApplyViewerArticulationLinkWrench(
    int articulationIndex, int linkIndex, int poseIndex, Vec3 localAnchor,
    Vec3 localCenterOfMass, Vec3 target, float effectiveMass, float stiffness, float damping,
    float maxAcceleration) {
  checkGpuInitialized();
  if (articulationIndex < 0 || articulationIndex >= mGpuArticulationCount || linkIndex < 0 ||
      linkIndex >= mGpuArticulationMaxLinkCount || poseIndex < 0 ||
      poseIndex >= mCudaRigidBodyBuffer.shape[0]) {
    throw std::runtime_error("failed to apply Viewer wrench: invalid articulation link index");
  }
  if (!localAnchor.isSane() || !localCenterOfMass.isSane() || !target.isSane() ||
      !std::isfinite(effectiveMass) || !std::isfinite(stiffness) || !std::isfinite(damping) ||
      !std::isfinite(maxAcceleration) || effectiveMass <= 0.f || stiffness < 0.f ||
      damping < 0.f || maxAcceleration < 0.f) {
    throw std::runtime_error("failed to apply Viewer wrench: invalid spring parameters");
  }

  gpuFetchArticulationLinkPoseIfNeeded();
  gpuFetchArticulationLinkVel();
  ensureCudaDevice();
  compose_viewer_articulation_wrench(
      static_cast<Vec3 *>(mCudaViewerForceScratch.ptr),
      static_cast<Vec3 *>(mCudaViewerTorqueScratch.ptr),
      static_cast<float const *>(mCudaArticulationLinkForceHandle.ptr),
      static_cast<float const *>(mCudaArticulationLinkTorqueHandle.ptr),
      static_cast<SapienBodyData const *>(mCudaRigidBodyBuffer.ptr), articulationIndex, linkIndex,
      mGpuArticulationMaxLinkCount, poseIndex, localAnchor, localCenterOfMass, target,
      effectiveMass, stiffness, damping, maxAcceleration, mCudaStream);
  mCudaEventRecord.record(mCudaStream);

  auto *physxIndex =
      static_cast<PxArticulationGPUIndex *>(mCudaArticulationGpuIndexBuffer.ptr) +
      articulationIndex;
  auto &gpuApi = mPxScene->getDirectGPUAPI();
  gpuApi.setArticulationData(mCudaViewerForceScratch.ptr, physxIndex,
                             PxArticulationGPUAPIWriteType::eLINK_FORCE, 1,
                             mCudaEventRecord.event);
  gpuApi.setArticulationData(mCudaViewerTorqueScratch.ptr, physxIndex,
                             PxArticulationGPUAPIWriteType::eLINK_TORQUE, 1,
                             mCudaEventRecord.event);
}

void PhysxSystemGpu::gpuSetViewerRigidDynamicPose(int gpuIndex, Pose pose,
                                                  bool zeroVelocity) {
  checkGpuInitialized();
  if (gpuIndex < 0 || gpuIndex >= mCudaRigidDynamicHandle.shape[0] || !pose.isSane()) {
    throw std::runtime_error("failed to teleport Viewer rigid dynamic: invalid pose or index");
  }
  gpuFetchRigidDynamicDataIfNeeded();
  ensureCudaDevice();
  set_viewer_body_pose(static_cast<SapienBodyData *>(mCudaRigidBodyBuffer.ptr), gpuIndex, pose,
                       zeroVelocity, mCudaStream);
  checkCudaErrors(cudaMemcpyAsync(mCudaViewerIndexBuffer.ptr, &gpuIndex, sizeof(int),
                                  cudaMemcpyHostToDevice, mCudaStream));
  gpuApplyRigidDynamicData(mCudaViewerIndexBuffer.handle());
}

void PhysxSystemGpu::gpuSetViewerArticulationRootPose(int articulationIndex, int rootPoseIndex,
                                                      Pose pose) {
  checkGpuInitialized();
  if (articulationIndex < 0 || articulationIndex >= mGpuArticulationCount || rootPoseIndex < 0 ||
      rootPoseIndex >= mCudaRigidBodyBuffer.shape[0] || !pose.isSane()) {
    throw std::runtime_error("failed to teleport Viewer articulation root: invalid pose or index");
  }
  gpuFetchArticulationLinkPoseIfNeeded();
  ensureCudaDevice();
  set_viewer_body_pose(static_cast<SapienBodyData *>(mCudaRigidBodyBuffer.ptr), rootPoseIndex,
                       pose, false, mCudaStream);
  checkCudaErrors(cudaMemcpyAsync(mCudaViewerIndexBuffer.ptr, &articulationIndex, sizeof(int),
                                  cudaMemcpyHostToDevice, mCudaStream));
  gpuApplyArticulationRootPose(mCudaViewerIndexBuffer.handle());
}

void PhysxSystemGpu::syncPosesGpuToCpu() {
  checkGpuInitialized();
  ++mSyncPosesGpuToCpuCount;
  gpuFetchRigidDynamicDataIfNeeded();
  gpuFetchArticulationLinkPoseIfNeeded();
  if (mCudaHostRigidBodyBuffer.shape != mCudaRigidBodyBuffer.shape) {
    mCudaHostRigidBodyBuffer =
        CudaHostArray(mCudaRigidBodyBuffer.shape, mCudaRigidBodyBuffer.type);
  }
  mCudaHostRigidBodyBuffer.copyFrom(mCudaRigidBodyBuffer);
  auto data = (SapienBodyData *)mCudaHostRigidBodyBuffer.ptr;

  for (auto &body : mRigidDynamicComponents) {
    assert(body->getGpuPoseIndex() >= 0);
    body->getEntity()->internalSyncPose(
        {data[body->getGpuPoseIndex()].p, data[body->getGpuPoseIndex()].q});
  }
  for (auto &body : mArticulationLinkComponents) {
    assert(body->getGpuPoseIndex() >= 0);
    body->getEntity()->internalSyncPose(
        {data[body->getGpuPoseIndex()].p, data[body->getGpuPoseIndex()].q});
  }
}

uint32_t PhysxSystemGpu::getGpuArticulationDof(int index) const {
  if (index < 0 || index >= mGpuArticulationCount) {
    throw std::runtime_error("invalid articulation GPU index");
  }
  for (auto const &link : mArticulationLinkComponents) {
    auto articulation = link->getArticulation();
    if (articulation->getGpuIndex() == index) {
      return articulation->getDof();
    }
  }
  throw std::runtime_error("articulation GPU index is not registered with this system");
}

std::vector<float> PhysxSystemGpu::gpuDownloadArticulationQpos(int index) {
  checkGpuInitialized();
  uint32_t dof = getGpuArticulationDof(index);
  ensureCudaDevice();
  gpuFetchArticulationQpos();
  checkCudaErrors(cudaStreamSynchronize(mCudaStream));

  std::vector<float> buffer(dof);
  checkCudaErrors(cudaMemcpy(
      buffer.data(), static_cast<float *>(mCudaQposHandle.ptr) + index * mGpuArticulationMaxDof,
      dof * sizeof(float), cudaMemcpyDeviceToHost));
  return buffer;
}

std::vector<float> PhysxSystemGpu::gpuDownloadArticulationQTargetPos(int index) {
  checkGpuInitialized();
  uint32_t dof = getGpuArticulationDof(index);
  ensureCudaDevice();
  gpuFetchArticulationQTargetPos();
  checkCudaErrors(cudaStreamSynchronize(mCudaStream));

  std::vector<float> buffer(dof);
  checkCudaErrors(cudaMemcpy(
      buffer.data(), static_cast<float *>(mCudaQTargetPosHandle.ptr) +
                         index * mGpuArticulationMaxDof,
      dof * sizeof(float), cudaMemcpyDeviceToHost));
  return buffer;
}

std::vector<float> PhysxSystemGpu::gpuDownloadArticulationQTargetVel(int index) {
  checkGpuInitialized();
  uint32_t dof = getGpuArticulationDof(index);
  ensureCudaDevice();
  gpuFetchArticulationQTargetVel();
  checkCudaErrors(cudaStreamSynchronize(mCudaStream));

  std::vector<float> buffer(dof);
  checkCudaErrors(cudaMemcpy(
      buffer.data(), static_cast<float *>(mCudaQTargetVelHandle.ptr) +
                         index * mGpuArticulationMaxDof,
      dof * sizeof(float), cudaMemcpyDeviceToHost));
  return buffer;
}

void PhysxSystemGpu::gpuUploadArticulationQpos(int index, Eigen::VectorXf const &q) {
  checkGpuInitialized();
  if (q.size() != getGpuArticulationDof(index)) {
    throw std::runtime_error("failed to upload articulation qpos: invalid index or shape");
  }
  ensureCudaDevice();
  checkCudaErrors(cudaStreamSynchronize(mCudaStream));
  checkCudaErrors(cudaMemcpy(
      static_cast<float *>(mCudaQposHandle.ptr) + index * mGpuArticulationMaxDof, q.data(),
      q.size() * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(mCudaViewerIndexBuffer.ptr, &index, sizeof(int),
                             cudaMemcpyHostToDevice));
  gpuApplyArticulationQpos(mCudaViewerIndexBuffer.handle());
  gpuUpdateArticulationKinematics(mCudaViewerIndexBuffer.handle());
}

void PhysxSystemGpu::gpuUploadArticulationQTargetPos(int index, Eigen::VectorXf const &q) {
  checkGpuInitialized();
  if (q.size() != getGpuArticulationDof(index)) {
    throw std::runtime_error(
        "failed to upload articulation position target: invalid index or shape");
  }
  ensureCudaDevice();
  checkCudaErrors(cudaStreamSynchronize(mCudaStream));
  checkCudaErrors(cudaMemcpy(
      static_cast<float *>(mCudaQTargetPosHandle.ptr) + index * mGpuArticulationMaxDof, q.data(),
      q.size() * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(mCudaViewerIndexBuffer.ptr, &index, sizeof(int),
                             cudaMemcpyHostToDevice));
  gpuApplyArticulationQTargetPos(mCudaViewerIndexBuffer.handle());
}

void PhysxSystemGpu::gpuUploadArticulationQTargetVel(int index, Eigen::VectorXf const &q) {
  checkGpuInitialized();
  if (q.size() != getGpuArticulationDof(index)) {
    throw std::runtime_error(
        "failed to upload articulation velocity target: invalid index or shape");
  }
  ensureCudaDevice();
  checkCudaErrors(cudaStreamSynchronize(mCudaStream));
  checkCudaErrors(cudaMemcpy(
      static_cast<float *>(mCudaQTargetVelHandle.ptr) + index * mGpuArticulationMaxDof, q.data(),
      q.size() * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(mCudaViewerIndexBuffer.ptr, &index, sizeof(int),
                             cudaMemcpyHostToDevice));
  gpuApplyArticulationQTargetVel(mCudaViewerIndexBuffer.handle());
}

void PhysxSystemGpu::setSceneOffset(std::shared_ptr<Scene> scene, Vec3 offset) {
  // clean up occasionally
  if (mSceneOffset.size() % 1024 == 0) {
    std::erase_if(mSceneOffset, [](const auto &p) { return p.first.expired(); });
  }

  mSceneOffset[scene] = offset;
}

Vec3 PhysxSystemGpu::getSceneOffset(std::shared_ptr<Scene> scene) const {
  if (mSceneOffset.contains(scene)) {
    return mSceneOffset.at(scene);
  }
  return Vec3(0.0f);
}

bool PhysxSystemGpu::sceneHasPhysxBodies(std::shared_ptr<Scene> scene) const {
  if (!scene) {
    return false;
  }
  for (auto const &entity : scene->getEntities()) {
    if (entity->getComponent<PhysxRigidBaseComponent>()) {
      return true;
    }
  }
  return false;
}

void PhysxSystemGpu::reclaimExpiredSceneEnvironmentIds() {
  // A destroyed scene took its PhysX actors with it, so its ordinary ID is genuinely free.
  // Shared entries carry the sentinel, not a slot, and are only dropped.
  for (auto it = mSceneEnvironmentIds.begin(); it != mSceneEnvironmentIds.end();) {
    if (!it->first.expired()) {
      ++it;
      continue;
    }
    if (it->second != PX_INVALID_U32) {
      mFreeSceneEnvironmentIds.push_back(it->second);
    }
    it = mSceneEnvironmentIds.erase(it);
  }
  std::erase_if(mBroadphaseEnvironmentIds, [](const auto &p) { return p.first.expired(); });
}

void PhysxSystemGpu::onSceneClosed(Scene &scene) {
  PhysxSystem::onSceneClosed(scene);

  // Weak-keyed maps cannot be looked up by the scene key without a shared_ptr, so
  // identify the entry by address; expired entries are dropped as a bonus sweep.
  auto eraseIfScene = [&scene](auto &map) {
    for (auto it = map.begin(); it != map.end();) {
      auto locked = it->first.lock();
      if (!locked || locked.get() == &scene) {
        it = map.erase(it);
        continue;
      }
      ++it;
    }
  };
  eraseIfScene(mSceneOffset);
  eraseIfScene(mBroadphaseEnvironmentIds);

  // Return a managed environment-ID slot immediately instead of waiting for the lazy
  // expiry sweep, so repeated close/recreate cycles never exhaust num_scenes.
  for (auto it = mSceneEnvironmentIds.begin(); it != mSceneEnvironmentIds.end();) {
    auto locked = it->first.lock();
    if (!locked) {
      if (it->second != PX_INVALID_U32) {
        mFreeSceneEnvironmentIds.push_back(it->second);
      }
      it = mSceneEnvironmentIds.erase(it);
      continue;
    }
    if (locked.get() == &scene) {
      if (mManagedBroadphaseEnvIds && it->second != PX_INVALID_U32) {
        mFreeSceneEnvironmentIds.push_back(it->second);
      }
      it = mSceneEnvironmentIds.erase(it);
      continue;
    }
    ++it;
  }
}

uint32_t PhysxSystemGpu::allocateSceneEnvironmentId() {
  // Slots handed back by scenes that were turned shared, or by scenes that were destroyed the
  // last time the counter ran out. Reuse before growing, so neither costs an environment.
  if (!mFreeSceneEnvironmentIds.empty()) {
    const uint32_t reused = mFreeSceneEnvironmentIds.back();
    mFreeSceneEnvironmentIds.pop_back();
    return reused;
  }

  // The counter only moves forward, so a destroyed scene's slot is invisible until now. Sweep
  // for it at the boundary rather than on every allocation: the common path stays O(1), and a
  // slot is never lost without the caller being told.
  const uint32_t limit = mReserveSharedBands ? kMaxSharedSceneEnvCount : kMaxSceneEnvironmentId;
  const uint32_t declared = mSceneConfig.numScenes.value_or(limit);
  const uint32_t ceiling = std::min(declared, limit);
  if (mNextSceneEnvironmentId >= ceiling) {
    reclaimExpiredSceneEnvironmentIds();
    if (!mFreeSceneEnvironmentIds.empty()) {
      const uint32_t reused = mFreeSceneEnvironmentIds.back();
      mFreeSceneEnvironmentIds.pop_back();
      return reused;
    }
  }

  // `numScenes` is how many ordinary scenes the simulation will hold, so it is a hard cap, not
  // a sizing hint: the bit count derived from it is baked into the PhysX scene at createScene
  // and cannot grow later. Refuse the extra scene here rather than silently under-spreading it.
  if (mSceneConfig.numScenes.has_value() &&
      mNextSceneEnvironmentId >= *mSceneConfig.numScenes) {
    throw std::runtime_error(
        "failed to assign PhysX GPU scene environment ID: the scene config declared num_scenes=" +
        std::to_string(*mSceneConfig.numScenes) +
        ", which is how many ordinary scenes this simulation holds at once. Raise num_scenes "
        "before creating the system; the broadphase bit count derived from it is baked into the "
        "PhysX scene and cannot change afterwards.");
  }

  // The declared count is checked at construction against these ceilings too, but an undeclared
  // simulation still assigns IDs on request and has to stop somewhere.
  if (mNextSceneEnvironmentId >= limit) {
    throw std::runtime_error(
        "failed to assign PhysX GPU scene environment ID: more than " + std::to_string(limit) +
        " environments" +
        (mReserveSharedBands
             ? ". With a shared scene SAPIEN keeps the scene ID in a 16-bit collision group "
               "field and reserves 0xffff for the shared scene itself."
             : "."));
  }
  return mNextSceneEnvironmentId++;
}

void PhysxSystemGpu::markSceneShared(std::shared_ptr<Scene> scene) {
  if (!scene) {
    throw std::runtime_error("failed to mark PhysX GPU scene as shared: scene is null");
  }
  if (!mReserveSharedBands) {
    throw std::runtime_error(
        "failed to mark PhysX GPU scene as shared: the scene config did not declare "
        "with_shared_scene, so no broadphase bands were reserved for it. PhysX gives shared "
        "objects a fixed encoded interval that the outermost bands fall outside of, and the "
        "bit count is baked into the PhysX scene at creation, so this cannot be arranged now. "
        "Set with_shared_scene on the config before creating the system.");
  }

  if (mSceneEnvironmentIds.size() % 1024 == 0) {
    std::erase_if(mSceneEnvironmentIds, [](const auto &p) { return p.first.expired(); });
    std::erase_if(mBroadphaseEnvironmentIds, [](const auto &p) { return p.first.expired(); });
  }

  if (mSceneEnvironmentIds.contains(scene) &&
      mSceneEnvironmentIds.at(scene) == PX_INVALID_U32) {
    return;
  }

  // Bodies freeze their environment ID at PxScene::addActor, so once one exists the scene can
  // no longer change camp -- and its ID cannot be handed to anyone else either.
  if (sceneHasPhysxBodies(scene)) {
    throw std::runtime_error(
        "failed to mark PhysX GPU scene as shared: it must be marked before adding "
        "actors/articulations to the scene");
  }

  // Only a slot the allocator handed out belongs in the free list. A manually set ID was never
  // drawn from the counter, and in manual mode the allocator never runs to drain it.
  if (mManagedBroadphaseEnvIds) {
    if (auto it = mSceneEnvironmentIds.find(scene); it != mSceneEnvironmentIds.end()) {
      mFreeSceneEnvironmentIds.push_back(it->second);
    }
  }

  mSceneEnvironmentIds[scene] = PX_INVALID_U32;
  mBroadphaseEnvironmentIds.erase(scene);
}

void PhysxSystemGpu::setSceneEnvironmentId(std::shared_ptr<Scene> scene, uint32_t envId) {
  if (!scene) {
    throw std::runtime_error("failed to set PhysX GPU scene environment ID: scene is null");
  }
  if (mManagedBroadphaseEnvIds) {
    throw std::runtime_error(
        "failed to set PhysX GPU scene environment ID: the scene config declared num_scenes, so "
        "SAPIEN assigns every environment ID itself. Drop num_scenes to drive them by hand, or "
        "use get_or_assign_environment_id() to take the one SAPIEN picked.");
  }
  if (envId >= kMaxSceneEnvironmentId) {
    throw std::runtime_error("failed to set PhysX GPU scene environment ID: " +
                             std::to_string(envId) + " is past the maximum of " +
                             std::to_string(kMaxSceneEnvironmentId - 1) + ".");
  }
  // The shared scene is not an environment; it is marked, not numbered.
  if (mReserveSharedBands && envId >= kCollisionGroupSharedId) {
    throw std::runtime_error(
        "failed to set PhysX GPU scene environment ID: " + std::to_string(envId) +
        " does not fit the collision-group scene field, which is 16 bits wide with 0xffff "
        "reserved for shared scenes. Use set_shared_environment() to mark the shared scene.");
  }
  // PhysX freezes an actor's environment ID at PxScene::addActor, so a scene that already has
  // bodies cannot be renumbered -- the existing ones would keep the old value.
  if (sceneHasPhysxBodies(scene)) {
    throw std::runtime_error(
        "failed to set PhysX GPU scene environment ID: it must be set before adding "
        "actors/articulations to the scene");
  }

  // Stored unshifted. getBroadphaseEnvironmentId applies the band offset on the way to PhysX,
  // so what callers read back here is always the ID they chose. Duplicates are deliberate in
  // manual mode: several scenes sharing an ID is how they end up in one environment.
  mSceneEnvironmentIds[scene] = envId;
  mBroadphaseEnvironmentIds.erase(scene);
}

uint32_t PhysxSystemGpu::getSceneEnvironmentId(std::shared_ptr<Scene> scene) {
  if (!scene) {
    throw std::runtime_error("failed to get PhysX GPU scene environment ID: scene is null");
  }

  // Memory hygiene only, in both modes: weak_ptr gives no expiry callback, so dead entries
  // have to be swept opportunistically. This never touches an ID a live scene still holds.
  if (mSceneEnvironmentIds.size() % 1024 == 0) {
    std::erase_if(mSceneEnvironmentIds, [](const auto &p) { return p.first.expired(); });
    std::erase_if(mBroadphaseEnvironmentIds, [](const auto &p) { return p.first.expired(); });
  }

  if (mSceneEnvironmentIds.contains(scene)) {
    return mSceneEnvironmentIds.at(scene);
  }

  // Manual mode: SAPIEN numbers nothing and records nothing. A scene the caller never numbered
  // is environment 0, the same as PhysX's own default of no separation. Managed mode: hand out
  // a fresh unique one -- that is the whole point of declaring the count.
  if (!mManagedBroadphaseEnvIds) {
    return 0;
  }

  uint32_t envId = allocateSceneEnvironmentId();
  mSceneEnvironmentIds[scene] = envId;
  return envId;
}

uint32_t PhysxSystemGpu::getOrAssignSceneEnvironmentId(std::shared_ptr<Scene> scene) {
  if (!mManagedBroadphaseEnvIds) {
    throw std::runtime_error(
        "failed to assign PhysX GPU scene environment ID: the scene config did not declare "
        "num_scenes, so SAPIEN manages no environment IDs to hand out. Set num_scenes on the "
        "config before creating the system to have them assigned, or number the scenes "
        "yourself with set_environment_id().");
  }
  return getSceneEnvironmentId(scene);
}

std::optional<uint32_t>
PhysxSystemGpu::getAssignedSceneEnvironmentId(std::shared_ptr<Scene> scene) const {
  if (!scene) {
    throw std::runtime_error("failed to get PhysX GPU scene environment ID: scene is null");
  }
  if (mSceneEnvironmentIds.contains(scene)) {
    return mSceneEnvironmentIds.at(scene);
  }
  return std::nullopt;
}

bool PhysxSystemGpu::hasSharedEnvironmentScene() const {
  for (auto const &[scene, envId] : mSceneEnvironmentIds) {
    if (envId == PX_INVALID_U32 && !scene.expired()) {
      return true;
    }
  }
  return false;
}

BroadphaseEnvIdWindow PhysxSystemGpu::getBroadphaseEnvIdWindow() const {
  // Only meaningful with a shared scene: without one there is no fixed encoded interval to
  // miss, so no band is out of reach and no offset is applied at all.
  auto const &config = getSceneConfig();
  return broadphaseEnvIdWindow(config.getGpuBroadPhaseNbBitsEnvIDX(),
                               config.getGpuBroadPhaseNbBitsEnvIDY(),
                               config.getGpuBroadPhaseNbBitsEnvIDZ());
}

uint32_t PhysxSystemGpu::getBroadphaseEnvBandCount() const {
  auto const &config = getSceneConfig();
  return broadphaseEnvBandCount(config.getGpuBroadPhaseNbBitsEnvIDX(),
                                config.getGpuBroadPhaseNbBitsEnvIDY(),
                                config.getGpuBroadPhaseNbBitsEnvIDZ());
}

uint32_t PhysxSystemGpu::getBroadphaseEnvironmentId(std::shared_ptr<Scene> scene) {
  uint32_t envId = getSceneEnvironmentId(scene);
  if (envId == PX_INVALID_U32) {
    return envId;
  }

  // The offset exists for one reason: to keep every environment inside the band that still
  // overlaps the fixed encoded interval PhysX gives shared objects. No shared scene, no
  // interval to miss, no reason to touch the ID -- PhysX gets exactly what the scene has,
  // whether SAPIEN assigned it or the caller set it.
  if (!mReserveSharedBands) {
    return envId;
  }

  // Reuse the value already bound to this scene so every one of its bodies shares a band.
  if (mBroadphaseEnvironmentIds.contains(scene)) {
    return mBroadphaseEnvironmentIds.at(scene);
  }

  auto const window = getBroadphaseEnvIdWindow();
  if (window.capacity == 0) {
    throw std::runtime_error(
        "the GPU broadphase environment-ID bit count is too small to host any environment: no "
        "band lies entirely inside the fixed encoded interval PhysX gives objects shared by all "
        "environments, so every environment would silently stop colliding with a shared ground. "
        "Declare num_scenes and let SAPIEN size the bits, or raise them with "
        "PhysxSceneConfig.set_gpu_broadphase_env_id_bits().");
  }

  // PhysX bands on the low `bits` of the ID (`envId << (32 - bits)` in 32-bit arithmetic
  // discards everything above), but `filtering()` compares the full 32-bit value. So only the
  // band has to land inside the window; the count above it rides in the high bits, where it
  // keeps environments distinct without moving them out of a reachable band.
  const uint32_t bandCount = getBroadphaseEnvBandCount();
  const uint32_t band = window.offset + envId % window.capacity;
  const uint32_t wrap = envId / window.capacity;
  const uint64_t broadphaseId = uint64_t{band} + uint64_t{wrap} * bandCount;
  if (broadphaseId >= kMaxSceneEnvironmentId) {
    throw std::runtime_error(
        "PhysX GPU broadphase environment ID " + std::to_string(envId) + " exceeds the " +
        std::to_string(maxBroadphaseEnvCount(true)) +
        " environments the banding can address. Environments past the band window wrap into the "
        "high bits of the ID, which PhysX still filters on exactly, but the result must stay "
        "below " +
        std::to_string(kMaxSceneEnvironmentId) + ".");
  }

  mBroadphaseEnvironmentIds[scene] = static_cast<uint32_t>(broadphaseId);
  return static_cast<uint32_t>(broadphaseId);
}

void PhysxSystemGpu::applyCollisionGroupSceneId(std::shared_ptr<Scene> scene,
                                                PhysxRigidBaseComponent &component) {
  if (!mManagedCollisionGroupSceneIds) {
    return;
  }

  // The raw environment ID, deliberately without the broadphase band offset: this field is only
  // ever compared for equality in SAPIEN's filter shader, never encoded into bounds, so the
  // offset that keeps bands overlapping shared objects would be meaningless noise here.
  uint32_t envId = getSceneEnvironmentId(scene);
  uint32_t sceneId = kCollisionGroupSharedId;
  if (envId != PX_INVALID_U32) {
    if (envId >= kCollisionGroupSharedId) {
      throw std::runtime_error(
          "environment ID " + std::to_string(envId) +
          " does not fit the collision-group scene field, which is 16 bits wide with 0xffff "
          "reserved for shared scenes. Construct PhysxGpuSystem without with_shared_scene to "
          "go past " +
          std::to_string(kCollisionGroupSharedId) +
          " environments; SAPIEN then leaves the collision groups alone and the broadphase "
          "alone separates them.");
    }
    sceneId = envId;
  }

  for (auto &shape : component.getCollisionShapes()) {
    auto groups = shape->getCollisionGroups();
    groups[3] = (groups[3] & kCollisionGroupIgnoreIdMask) | (sceneId << 16);
    shape->setCollisionGroups(groups);
  }
}

void PhysxSystemGpu::allocateCudaBuffers() {
  SAPIEN_PROFILE_FUNCTION;
  int rigidDynamicCount = mRigidDynamicComponents.size();
  int rigidBodyCount = rigidDynamicCount + mGpuArticulationCount * mGpuArticulationMaxLinkCount;

  ensureCudaDevice();

  // rigid body data buffer
  mCudaRigidBodyBuffer = CudaArray({rigidBodyCount, 13}, "f4");
  mCudaRigidDynamicHandle = CudaArrayHandle{.shape = {rigidDynamicCount, 13},
                                            .strides = {52, 4},
                                            .type = "f4",
                                            .cudaId = mCudaRigidBodyBuffer.cudaId,
                                            .ptr = (float *)mCudaRigidBodyBuffer.ptr};
  mCudaLinkHandle =
      CudaArrayHandle{.shape = {mGpuArticulationCount, mGpuArticulationMaxLinkCount, 13},
                      .strides = {mGpuArticulationMaxLinkCount * 52, 52, 4},
                      .type = "f4",
                      .cudaId = mCudaRigidBodyBuffer.cudaId,
                      .ptr = (float *)mCudaRigidBodyBuffer.ptr + 13 * rigidDynamicCount};

  mCudaArticulationLinkIncomingJointForceBuffer =
      CudaArray({mGpuArticulationCount, mGpuArticulationMaxLinkCount, 6}, "f4");

  // rigid body force torque buffer
  mCudaRigidBodyForceBuffer = CudaArray({rigidBodyCount, 4}, "f4");
  mCudaRigidDynamicForceHandle = CudaArrayHandle{.shape = {rigidDynamicCount, 4},
                                                 .strides = {16, 4},
                                                 .type = "f4",
                                                 .cudaId = mCudaRigidBodyForceBuffer.cudaId,
                                                 .ptr = (float *)mCudaRigidBodyForceBuffer.ptr};
  mCudaRigidDynamicForcePaddedScratch = CudaArray({rigidDynamicCount, 4}, "f4");
  mCudaRigidDynamicForcePackedScratch = CudaArray({rigidDynamicCount, 3}, "f4");
  mCudaArticulationLinkForceHandle =
      CudaArrayHandle{.shape = {mGpuArticulationCount, mGpuArticulationMaxLinkCount, 4},
                      .strides = {mGpuArticulationMaxLinkCount * 16, 16, 4},
                      .type = "f4",
                      .cudaId = mCudaRigidBodyForceBuffer.cudaId,
                      .ptr = (float *)mCudaRigidBodyForceBuffer.ptr + 4 * rigidDynamicCount};
  mCudaArticulationLinkForcePaddedScratch =
      CudaArray({mGpuArticulationCount, mGpuArticulationMaxLinkCount, 4}, "f4");
  mCudaArticulationLinkForcePackedScratch =
      CudaArray({mGpuArticulationCount, mGpuArticulationMaxLinkCount, 3}, "f4");

  mCudaRigidBodyTorqueBuffer = CudaArray({rigidBodyCount, 4}, "f4");
  mCudaRigidDynamicTorqueHandle = CudaArrayHandle{.shape = {rigidDynamicCount, 4},
                                                  .strides = {16, 4},
                                                  .type = "f4",
                                                  .cudaId = mCudaRigidBodyTorqueBuffer.cudaId,
                                                  .ptr = (float *)mCudaRigidBodyTorqueBuffer.ptr};
  mCudaRigidDynamicTorquePaddedScratch = CudaArray({rigidDynamicCount, 4}, "f4");
  mCudaRigidDynamicTorquePackedScratch = CudaArray({rigidDynamicCount, 3}, "f4");
  mCudaArticulationLinkTorqueHandle =
      CudaArrayHandle{.shape = {mGpuArticulationCount, mGpuArticulationMaxLinkCount, 4},
                      .strides = {mGpuArticulationMaxLinkCount * 16, 16, 4},
                      .type = "f4",
                      .cudaId = mCudaRigidBodyTorqueBuffer.cudaId,
                      .ptr = (float *)mCudaRigidBodyTorqueBuffer.ptr + 4 * rigidDynamicCount};
  mCudaArticulationLinkTorquePaddedScratch =
      CudaArray({mGpuArticulationCount, mGpuArticulationMaxLinkCount, 4}, "f4");
  mCudaArticulationLinkTorquePackedScratch =
      CudaArray({mGpuArticulationCount, mGpuArticulationMaxLinkCount, 3}, "f4");

  int viewerWrenchCount = std::max(1, mGpuArticulationMaxLinkCount);
  mCudaViewerForceScratch = CudaArray({viewerWrenchCount, 3}, "f4");
  mCudaViewerTorqueScratch = CudaArray({viewerWrenchCount, 3}, "f4");
  mCudaViewerIndexBuffer = CudaArray({1}, "i4");

  int articulationBufferBlockSize = mGpuArticulationCount * mGpuArticulationMaxDof;
  mCudaArticulationBuffer = CudaArray({articulationBufferBlockSize * 8}, "f4");
  mCudaArticulationApplyScratch = CudaArray({articulationBufferBlockSize}, "f4");
  mCudaArticulationCompensationScratch =
      CudaArray({mGpuArticulationCount * (mGpuArticulationMaxDof + 6)}, "f4");

  float *articulationBufferPtr = static_cast<float *>(mCudaArticulationBuffer.ptr);
  auto articulationBufferHandle = [&](int block) {
    return CudaArrayHandle{
        .shape = {mGpuArticulationCount, mGpuArticulationMaxDof},
        .strides = {mGpuArticulationMaxDof * 4, 4},
        .type = "f4",
        .cudaId = mCudaArticulationBuffer.cudaId,
        .ptr = articulationBufferPtr ? articulationBufferPtr + articulationBufferBlockSize * block
                                     : nullptr,
    };
  };

  mCudaQposHandle = articulationBufferHandle(0);
  mCudaQvelHandle = articulationBufferHandle(1);
  mCudaQfHandle = articulationBufferHandle(2);
  mCudaQaccHandle = articulationBufferHandle(3);
  mCudaQTargetPosHandle = articulationBufferHandle(4);
  mCudaQTargetVelHandle = articulationBufferHandle(5);
  mCudaArticulationGravityCompensationHandle = articulationBufferHandle(6);
  mCudaArticulationCoriolisAndCentrifugalCompensationHandle = articulationBufferHandle(7);

  {
    int jacobianMaxRows = 6 + (mGpuArticulationMaxLinkCount - 1) * 6;
    int jacobianMaxCols = 6 + mGpuArticulationMaxDof;
    mCudaArticulationJacobianBuffer =
        CudaArray({mGpuArticulationCount, jacobianMaxRows, jacobianMaxCols}, "f4");
    mCudaArticulationJacobianHandle = mCudaArticulationJacobianBuffer.handle();
    mCudaArticulationJacobianScratch =
        CudaArray({mGpuArticulationCount * jacobianMaxRows * jacobianMaxCols}, "f4");
    mCudaArticulationJacobianShapeBuffer = CudaArray({mGpuArticulationCount, 2}, "u4");
  }

  {
    mCudaRigidDynamicIndexBuffer = CudaArray({rigidDynamicCount}, "u4");
    mCudaRigidDynamicIndexScratch = CudaArray({rigidDynamicCount}, "u4");
    mCudaRigidDynamicOffsetBuffer = CudaArray({rigidDynamicCount, 3}, "f4");

    std::vector<std::array<float, 3>> host_offset;
    std::vector<PxRigidDynamicGPUIndex> host_index;
    auto bodies = getRigidDynamicComponents();
    for (uint32_t i = 0; i < bodies.size(); ++i) {
      // body set internal gpu id
      Vec3 offset = getSceneOffset(bodies[i]->getScene());
      host_offset.push_back({offset.x, offset.y, offset.z});
      host_index.push_back(bodies[i]->getPxActor()->getGPUIndex());
      bodies[i]->internalSetGpuIndex(i);
    }

    checkCudaErrors(cudaMemcpy(mCudaRigidDynamicIndexBuffer.ptr, host_index.data(),
                               host_index.size() * sizeof(PxRigidDynamicGPUIndex),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(mCudaRigidDynamicOffsetBuffer.ptr, host_offset.data(),
                               host_offset.size() * sizeof(float) * 3, cudaMemcpyHostToDevice));
  }

  {
    mCudaArticulationOffsetBuffer = CudaArray({mGpuArticulationCount, 3}, "f4");
    mCudaArticulationIndexBuffer = CudaArray({mGpuArticulationCount}, "i4");
    mCudaArticulationGpuIndexBuffer = CudaArray({mGpuArticulationCount}, "u4");
    mCudaArticulationIndexScratch = CudaArray({mGpuArticulationCount}, "u4");
    mCudaArticulationCompensationMetaBuffer = CudaArray({mGpuArticulationCount, 2}, "u4");

    std::vector<std::array<float, 3>> host_offset(mGpuArticulationCount, {0.f, 0.f, 0.f});
    std::vector<std::array<uint32_t, 2>> host_jacobian_shape(mGpuArticulationCount, {0u, 0u});
    std::vector<std::array<uint32_t, 2>> host_compensation_meta(mGpuArticulationCount, {0u, 0u});
    std::vector<int> host_dense_index;
    std::vector<PxArticulationGPUIndex> host_gpu_index;
    std::unordered_map<PxArticulationReducedCoordinate *, int> articulation_to_dense_index;

    std::vector<PxArticulationReducedCoordinate *> articulations(mGpuArticulationCount);
    mPxScene->getArticulations(articulations.data(), mGpuArticulationCount);
    for (int i = 0; i < mGpuArticulationCount; ++i) {
      host_dense_index.push_back(i);
      host_gpu_index.push_back(articulations[i]->getGPUIndex());
      articulation_to_dense_index[articulations[i]] = i;
    }

    for (auto a : getArticulationLinkComponents()) {
      int art_idx = articulation_to_dense_index.at(a->getArticulation()->getPxArticulation());
      a->getArticulation()->internalSetGpuIndex(art_idx);
      if (a->isRoot()) {
        Vec3 offset = getSceneOffset(a->getScene());
        host_offset.at(art_idx) = {offset.x, offset.y, offset.z};
        auto pxArticulation = a->getArticulation()->getPxArticulation();
        bool fixedBase =
            pxArticulation->getArticulationFlags().isSet(PxArticulationFlag::eFIX_BASE);
        uint32_t linkCount = pxArticulation->getNbLinks();
        uint32_t dofCount = pxArticulation->getDofs();
        uint32_t rootForceOffset = fixedBase ? 0u : 6u;
        host_jacobian_shape.at(art_idx) = {
            rootForceOffset + (linkCount - 1) * 6u,
            rootForceOffset + dofCount,
        };
        host_compensation_meta.at(art_idx) = {rootForceOffset, dofCount};
      }
      a->internalSetGpuPoseIndex(rigidDynamicCount + art_idx * mGpuArticulationMaxLinkCount +
                                 a->getIndex());
    }
    checkCudaErrors(cudaMemcpy(mCudaArticulationOffsetBuffer.ptr, host_offset.data(),
                               host_offset.size() * sizeof(float) * 3, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(mCudaArticulationIndexBuffer.ptr, host_dense_index.data(),
                               host_dense_index.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(mCudaArticulationGpuIndexBuffer.ptr, host_gpu_index.data(),
                               host_gpu_index.size() * sizeof(PxArticulationGPUIndex),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(mCudaArticulationJacobianShapeBuffer.ptr, host_jacobian_shape.data(),
                               host_jacobian_shape.size() * sizeof(uint32_t) * 2,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(mCudaArticulationCompensationMetaBuffer.ptr,
                               host_compensation_meta.data(),
                               host_compensation_meta.size() * sizeof(uint32_t) * 2,
                               cudaMemcpyHostToDevice));
  }

  int bodySize = rigidDynamicCount * 16 * sizeof(float);
  mCudaRigidDynamicScratch = CudaArray({bodySize}, "u1");

  int linkPoseSize = mGpuArticulationCount * mGpuArticulationMaxLinkCount * 7 * sizeof(float);
  mCudaLinkPoseScratch = CudaArray({linkPoseSize}, "u1");

  int linkVelSize = mGpuArticulationCount * mGpuArticulationMaxLinkCount * 6 * sizeof(float);
  mCudaLinkVelScratch = CudaArray({linkVelSize}, "u1");
}

void PhysxSystemGpu::ensureCudaDevice() { checkCudaErrors(cudaSetDevice(mDevice->cudaId)); }
#endif

PhysxSystem::~PhysxSystem() {
  // The leaf destructor already called close(); this catches systems whose leaf
  // constructor threw before close() could run, so the generic counter never leaks.
  if (mRegisteredSystem && mEngine) {
    mEngine->unregisterSystem();
    mRegisteredSystem = false;
  }
  logger::info("Deleting PhysxSystem");
}

void PhysxSystem::close() {
  if (mClosed) {
    return;
  }
  if (!mScenes.empty()) {
    throw std::runtime_error("failed to close PhysX system: " + std::to_string(mScenes.size()) +
                             " scenes are still attached; close every scene first");
  }
  closeImpl();
  mClosed = true;
  if (mRegisteredSystem) {
    mEngine->unregisterSystem();
    mRegisteredSystem = false;
  }
}

void PhysxSystem::closeNoThrow() {
  try {
    close();
    return;
  } catch (std::exception const &e) {
    logger::error("failed to close PhysX system cleanly during destruction: {}", e.what());
  } catch (...) {
    logger::error("failed to close PhysX system cleanly during destruction");
  }
  // Last resort: match the historical destructor behavior (release the scene and the
  // dispatcher) instead of leaking; components left dangling are the caller's bug.
  try {
    if (mPxScene) {
      mPxScene->release();
      mPxScene = nullptr;
    }
    if (mPxCPUDispatcher) {
      mPxCPUDispatcher->release();
      mPxCPUDispatcher = nullptr;
    }
    mClosed = true;
    if (mRegisteredSystem) {
      mEngine->unregisterSystem();
      mRegisteredSystem = false;
    }
  } catch (...) {
    logger::error("PhysX system destructor failed to release the PhysX scene");
  }
}

void PhysxSystem::checkNotClosed() const {
  if (mClosed) {
    throw std::runtime_error("failed to use PhysX system: the system is closed");
  }
}

PhysxSystemCpu::~PhysxSystemCpu() { closeNoThrow(); }

void PhysxSystemCpu::closeImpl() {
  size_t registered = mRigidDynamicComponents.size() + mRigidStaticComponents.size() +
                      mArticulationLinkComponents.size();
  if (registered != 0) {
    throw std::runtime_error(
        "failed to close PhysxCpuSystem: " + std::to_string(registered) +
        " components are still registered; close every scene owned by this system first");
  }
  if (mPxScene) {
    mPxScene->release();
    mPxScene = nullptr;
  }
  if (mPxCPUDispatcher) {
    mPxCPUDispatcher->release();
    mPxCPUDispatcher = nullptr;
  }
}

#ifdef SAPIEN_CUDA
PhysxSystemGpu::~PhysxSystemGpu() {
  closeNoThrow();
  // If the strict close failed, the base fallback released the PxScene; the lease
  // member releases the CUDA context manager here, never before the scene.
  if (mRegisteredGpuSystem) {
    mEngine->unregisterGpuSystem();
    mRegisteredGpuSystem = false;
  }
}

void PhysxSystemGpu::closeImpl() {
  size_t registered = mRigidDynamicComponents.size() + mRigidStaticComponents.size() +
                      mArticulationLinkComponents.size();
  if (registered != 0) {
    throw std::runtime_error(
        "failed to close PhysxGpuSystem: " + std::to_string(registered) +
        " components are still registered; close every scene owned by this system first");
  }
  int64_t views = mViewLifecycle->viewCount();
  if (views != 0) {
    throw std::runtime_error(
        "failed to close PhysxGpuSystem: " + std::to_string(views) +
        " external CUDA array views of this system are still alive; drop Torch/JAX/CuPy "
        "views exported from its cuda_* buffers first");
  }

  waitIdle();

  if (mPxScene) {
    mPxScene->release();
    mPxScene = nullptr;
  }
  if (mPxCPUDispatcher) {
    mPxCPUDispatcher->release();
    mPxCPUDispatcher = nullptr;
  }
  // PhysX requires every scene using a context manager to be released before the
  // manager. Releasing the lease after the scene satisfies that; when this was the
  // last lease for the device, the manager is released now.
  mCudaContextLease.reset();
  mGpuInitialized = false;
  if (mRegisteredGpuSystem) {
    mEngine->unregisterGpuSystem();
    mRegisteredGpuSystem = false;
  }
}
#endif
} // namespace physx
} // namespace sapien
