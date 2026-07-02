#include "sapien/physx/physx_system.h"
#include "../logger.h"
#include "./filter_shader.hpp"
#include "sapien/entity.h"
#include "sapien/math/conversion.h"
#include "sapien/physx/articulation.h"
#include "sapien/physx/articulation_link_component.h"
#include "sapien/physx/material.h"
#include "sapien/physx/physx_default.h"
#include "sapien/physx/rigid_component.h"
#include "sapien/profiler.h"
#include <extensions/PxExtensionsAPI.h>
#include <unordered_map>

#ifdef SAPIEN_CUDA
#include "./physx_system.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
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
    : mSceneConfig(PhysxDefault::getSceneConfig()), mEngine(PhysxEngine::Get()) {}

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
  validateEnvIdBits(config.gpuBroadPhaseNbBitsEnvIDX, 'x');
  validateEnvIdBits(config.gpuBroadPhaseNbBitsEnvIDY, 'y');
  validateEnvIdBits(config.gpuBroadPhaseNbBitsEnvIDZ, 'z');
  if (config.gpuBroadPhaseNbBitsEnvIDX || config.gpuBroadPhaseNbBitsEnvIDY ||
      config.gpuBroadPhaseNbBitsEnvIDZ) {
    gpuBroadPhaseDesc.gpuBroadPhaseNbBitsShiftX =
        std::max<uint8_t>(gpuBroadPhaseDesc.gpuBroadPhaseNbBitsShiftX,
                          config.gpuBroadPhaseNbBitsEnvIDX);
    gpuBroadPhaseDesc.gpuBroadPhaseNbBitsShiftY =
        std::max<uint8_t>(gpuBroadPhaseDesc.gpuBroadPhaseNbBitsShiftY,
                          config.gpuBroadPhaseNbBitsEnvIDY);
    gpuBroadPhaseDesc.gpuBroadPhaseNbBitsShiftZ =
        std::max<uint8_t>(gpuBroadPhaseDesc.gpuBroadPhaseNbBitsShiftZ,
                          config.gpuBroadPhaseNbBitsEnvIDZ);
    gpuBroadPhaseDesc.gpuBroadPhaseNbBitsEnvIDX = config.gpuBroadPhaseNbBitsEnvIDX;
    gpuBroadPhaseDesc.gpuBroadPhaseNbBitsEnvIDY = config.gpuBroadPhaseNbBitsEnvIDY;
    gpuBroadPhaseDesc.gpuBroadPhaseNbBitsEnvIDZ = config.gpuBroadPhaseNbBitsEnvIDZ;
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
  sceneDesc.cudaContextManager = mEngine->getCudaContextManager(device->cudaId);
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
}
#else
PhysxSystemGpu::PhysxSystemGpu(std::shared_ptr<Device> device) {
  throw std::runtime_error(
        "Does not support PhysX GPU system.");
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
void PhysxSystemGpu::registerComponent(std::shared_ptr<PhysxRigidDynamicComponent> component) {
  mRigidDynamicComponents.insert(component);
  mGpuInitialized = false;
}
void PhysxSystemGpu::registerComponent(std::shared_ptr<PhysxRigidStaticComponent> component) {
  mRigidStaticComponents.insert(component);
  mGpuInitialized = false;
}
void PhysxSystemGpu::registerComponent(std::shared_ptr<PhysxArticulationLinkComponent> component) {
  mArticulationLinkComponents.insert(component);
  mGpuInitialized = false;
}
void PhysxSystemGpu::unregisterComponent(std::shared_ptr<PhysxRigidDynamicComponent> component) {
  mRigidDynamicComponents.erase(component);
  mGpuInitialized = false;
}
void PhysxSystemGpu::unregisterComponent(std::shared_ptr<PhysxRigidStaticComponent> component) {
  mRigidStaticComponents.erase(component);
  mGpuInitialized = false;
}
void PhysxSystemGpu::unregisterComponent(
    std::shared_ptr<PhysxArticulationLinkComponent> component) {
  mArticulationLinkComponents.erase(component);
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
  if (!mGpuInitialized) {
    throw std::runtime_error("failed to step: gpu simulation is not initialized.");
  }

  mContactUpToDate = false;

  ++mTotalSteps;
  mPxScene->simulate(mTimestep);
  mPxScene->fetchResults(true);

  // TODO: does the GPU API require fetch results?
}

void PhysxSystemGpu::stepStart() {
  if (!mGpuInitialized) {
    throw std::runtime_error("failed to step: gpu simulation is not initialized.");
  }

  mContactUpToDate = false;

  ++mTotalSteps;
  mPxScene->simulate(mTimestep);
}

void PhysxSystemGpu::stepFinish() { mPxScene->fetchResults(true); }
#endif

std::string PhysxSystemCpu::packState() const {
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
  ++mTotalSteps;
  ensureCudaDevice();
  mPxScene->simulate(mTimestep);
  while (!mPxScene->fetchResults(true)) {
  }

  mGpuArticulationCount = getArticulationCount();
  mGpuArticulationMaxDof = computeArticulationMaxDof();
  mGpuArticulationMaxLinkCount = computeArticulationMaxLinkCount();

  allocateCudaBuffers();

  // ensureCudaDevice();
  // mCudaEventRecord.init();
  // mCudaEventWait.init();

  mGpuInitialized = true;
}

void PhysxSystemGpu::checkGpuInitialized() const {
  if (!isInitialized()) {
    throw std::runtime_error("GPU PhysX is not initialized.");
  }
}

void PhysxSystemGpu::gpuSetCudaStream(uintptr_t stream) { mCudaStream = (cudaStream_t)stream; }

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

  SAPIEN_PROFILE_BLOCK_BEGIN("fetch contact count");
  gpuApi.copyContactData(mCudaContactBuffer.ptr, (PxU32 *)mCudaContactCount.ptr, 0);
  cudaMemcpy(&mContactCount, mCudaContactCount.ptr, sizeof(PxU32), cudaMemcpyDeviceToHost);
  SAPIEN_PROFILE_BLOCK_END;

  int size = upperPowerOf2(mContactCount);
  if (mCudaContactBuffer.shape[0] < size) {
    SAPIEN_PROFILE_BLOCK("re-allocate contact buffer");
    mCudaContactBuffer = CudaArray({size, sizeof(PxGpuContactPair)}, "u1");
  }

  gpuApi.copyContactData(mCudaContactBuffer.ptr, (PxU32 *)mCudaContactCount.ptr, size);

  mContactUpToDate = true;
}

void PhysxSystemGpu::gpuQueryContactPairImpulses(PhysxGpuContactPairImpulseQuery const &query) {
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
  cudaStreamSynchronize(mCudaStream);
}

void PhysxSystemGpu::gpuQueryContactBodyImpulses(PhysxGpuContactBodyImpulseQuery const &query) {
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
  cudaStreamSynchronize(mCudaStream);
}

void PhysxSystemGpu::gpuFetchRigidDynamicData() {
  checkGpuInitialized();

  if (mRigidDynamicComponents.empty()) {
    return;
  }

  ensureCudaDevice();
  auto &gpuApi = mPxScene->getDirectGPUAPI();
  auto count = static_cast<PxU32>(mCudaRigidDynamicIndexBuffer.shape.at(0));
  auto poseScratch = rigidDynamicPoseScratch(mCudaRigidDynamicScratch);
  auto linearVelocityScratch = rigidDynamicLinearVelocityScratch(mCudaRigidDynamicScratch, count);
  auto angularVelocityScratch = rigidDynamicAngularVelocityScratch(mCudaRigidDynamicScratch, count);

  gpuApi.getRigidDynamicData(poseScratch, (PxRigidDynamicGPUIndex *)mCudaRigidDynamicIndexBuffer.ptr,
                             PxRigidDynamicGPUAPIReadType::eGLOBAL_POSE, count);
  gpuApi.getRigidDynamicData(linearVelocityScratch,
                             (PxRigidDynamicGPUIndex *)mCudaRigidDynamicIndexBuffer.ptr,
                             PxRigidDynamicGPUAPIReadType::eLINEAR_VELOCITY, count);
  gpuApi.getRigidDynamicData(angularVelocityScratch,
                             (PxRigidDynamicGPUIndex *)mCudaRigidDynamicIndexBuffer.ptr,
                             PxRigidDynamicGPUAPIReadType::eANGULAR_VELOCITY, count);

  body_data_physx_to_sapien(mCudaRigidDynamicHandle.ptr, poseScratch, linearVelocityScratch,
                            angularVelocityScratch, mCudaRigidDynamicOffsetBuffer.ptr, count,
                            mCudaStream);
}

void PhysxSystemGpu::gpuFetchArticulationLinkPose() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();
  auto &gpuApi = mPxScene->getDirectGPUAPI();
  auto count = static_cast<PxU32>(mGpuArticulationCount);
  gpuApi.getArticulationData(mCudaLinkPoseScratch.ptr,
                             (PxArticulationGPUIndex *)mCudaArticulationGpuIndexBuffer.ptr,
                             PxArticulationGPUAPIReadType::eLINK_GLOBAL_POSE, count);
  link_pose_physx_to_sapien(mCudaLinkHandle.ptr, mCudaLinkPoseScratch.ptr,
                            mCudaArticulationOffsetBuffer.ptr, mGpuArticulationMaxLinkCount,
                            count * mGpuArticulationMaxLinkCount, mCudaStream);
}

void PhysxSystemGpu::gpuFetchArticulationLinkVel() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();
  auto &gpuApi = mPxScene->getDirectGPUAPI();
  auto count = static_cast<PxU32>(mGpuArticulationCount);
  auto linearVelocityScratch =
      articulationLinearVelocityScratch(mCudaLinkVelScratch, count, mGpuArticulationMaxLinkCount);
  auto angularVelocityScratch =
      articulationAngularVelocityScratch(mCudaLinkVelScratch, count, mGpuArticulationMaxLinkCount);

  gpuApi.getArticulationData(linearVelocityScratch,
                             (PxArticulationGPUIndex *)mCudaArticulationGpuIndexBuffer.ptr,
                             PxArticulationGPUAPIReadType::eLINK_LINEAR_VELOCITY, count);
  gpuApi.getArticulationData(angularVelocityScratch,
                             (PxArticulationGPUIndex *)mCudaArticulationGpuIndexBuffer.ptr,
                             PxArticulationGPUAPIReadType::eLINK_ANGULAR_VELOCITY, count);

  link_vel_physx_to_sapien(mCudaLinkHandle.ptr, linearVelocityScratch, angularVelocityScratch,
                           count * mGpuArticulationMaxLinkCount, mCudaStream);
}

void PhysxSystemGpu::gpuFetchArticulationQpos() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();
  mPxScene->getDirectGPUAPI().getArticulationData(
      mCudaQposHandle.ptr, (PxArticulationGPUIndex *)mCudaArticulationGpuIndexBuffer.ptr,
      PxArticulationGPUAPIReadType::eJOINT_POSITION, mGpuArticulationCount);
}

void PhysxSystemGpu::gpuFetchArticulationQvel() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();
  mPxScene->getDirectGPUAPI().getArticulationData(
      mCudaQvelHandle.ptr, (PxArticulationGPUIndex *)mCudaArticulationGpuIndexBuffer.ptr,
      PxArticulationGPUAPIReadType::eJOINT_VELOCITY, mGpuArticulationCount);
}

void PhysxSystemGpu::gpuFetchArticulationQTargetPos() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();
  mPxScene->getDirectGPUAPI().getArticulationData(
      mCudaQTargetPosHandle.ptr, (PxArticulationGPUIndex *)mCudaArticulationGpuIndexBuffer.ptr,
      PxArticulationGPUAPIReadType::eJOINT_TARGET_POSITION, mGpuArticulationCount);
}

void PhysxSystemGpu::gpuFetchArticulationQTargetVel() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();
  mPxScene->getDirectGPUAPI().getArticulationData(
      mCudaQTargetVelHandle.ptr, (PxArticulationGPUIndex *)mCudaArticulationGpuIndexBuffer.ptr,
      PxArticulationGPUAPIReadType::eJOINT_TARGET_VELOCITY, mGpuArticulationCount);
}

void PhysxSystemGpu::gpuComputeArticulationJacobian() {
  gpuComputeArticulationJacobian(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuComputeArticulationJacobian(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});

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
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});

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
  mPxScene->getDirectGPUAPI().getArticulationData(
      mCudaArticulationLinkIncomingJointForceBuffer.ptr,
      (PxArticulationGPUIndex *)mCudaArticulationGpuIndexBuffer.ptr,
      PxArticulationGPUAPIReadType::eLINK_INCOMING_JOINT_FORCE, mGpuArticulationCount);
}

void PhysxSystemGpu::gpuFetchArticulationQacc() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }
  ensureCudaDevice();

  mPxScene->getDirectGPUAPI().getArticulationData(
      mCudaQaccHandle.ptr, (PxArticulationGPUIndex *)mCudaArticulationGpuIndexBuffer.ptr,
      PxArticulationGPUAPIReadType::eJOINT_ACCELERATION, mGpuArticulationCount);
}

void PhysxSystemGpu::gpuUpdateArticulationKinematics() {
  gpuUpdateArticulationKinematics(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuUpdateArticulationKinematics(CudaArrayHandle const &indices) {
  checkGpuInitialized();
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});

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
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});

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

void PhysxSystemGpu::gpuApplyArticulationLinkForce() {
  gpuApplyArticulationLinkForce(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuApplyArticulationLinkForce(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});

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
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});

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
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});

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
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});

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
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});

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
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});

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
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});

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
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});

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
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});

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

void PhysxSystemGpu::syncPosesGpuToCpu() {
  checkGpuInitialized();
  gpuFetchRigidDynamicData();
  gpuFetchArticulationLinkPose();
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

std::vector<float> PhysxSystemGpu::gpuDownloadArticulationQpos(int index) {
  ensureCudaDevice();
  gpuFetchArticulationQpos();
  cudaStreamSynchronize(mCudaStream);

  if (index < 0 || index >= mGpuArticulationCount) {
    throw std::runtime_error("failed to download articulation qpos: invalid index");
  }

  std::vector<float> buffer(mGpuArticulationMaxDof);

  cudaMemcpy(buffer.data(), &((float *)mCudaQposHandle.ptr)[index * mGpuArticulationMaxDof],
             mGpuArticulationMaxDof * sizeof(float), cudaMemcpyDeviceToHost);
  return buffer;
}

void PhysxSystemGpu::gpuUploadArticulationQpos(int index, Eigen::VectorXf const &q) {
  ensureCudaDevice();
  cudaStreamSynchronize(mCudaStream);
  if (index < 0 || index >= mGpuArticulationCount) {
    throw std::runtime_error("failed to download articulation qpos: invalid index");
  }

  cudaMemcpy(&((float *)mCudaQposHandle.ptr)[index * mGpuArticulationMaxDof], q.data(),
             q.size() * sizeof(float), cudaMemcpyHostToDevice);
  CudaArray cudaIndex({1}, "i4");
  checkCudaErrors(cudaMemcpy(cudaIndex.ptr, &index, sizeof(int), cudaMemcpyHostToDevice));
  gpuApplyArticulationQpos(cudaIndex.handle());
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

namespace {
constexpr uint32_t kMaxSceneEnvironmentId = 1u << 24;

uint32_t normalizeSceneEnvironmentId(int64_t envId) {
  if (envId == -1 || envId == static_cast<int64_t>(PX_INVALID_U32)) {
    return PX_INVALID_U32;
  }
  if (envId < 0 || envId >= static_cast<int64_t>(kMaxSceneEnvironmentId)) {
    throw std::runtime_error(
        "failed to set PhysX GPU scene environment ID: env_id must be -1, "
        "0xffffffff, or in [0, 1 << 24)");
  }
  return static_cast<uint32_t>(envId);
}
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

bool PhysxSystemGpu::isSceneEnvironmentIdUsed(uint32_t envId,
                                               std::shared_ptr<Scene> excludedScene) const {
  if (envId == PX_INVALID_U32) {
    return false;
  }

  for (auto const &[scene, assignedEnvId] : mSceneEnvironmentIds) {
    if (assignedEnvId != envId) {
      continue;
    }
    auto lockedScene = scene.lock();
    if (!lockedScene) {
      continue;
    }
    if (!excludedScene || lockedScene != excludedScene) {
      return true;
    }
  }
  return false;
}

uint32_t PhysxSystemGpu::allocateSceneEnvironmentId() {
  while (mNextSceneEnvironmentId < kMaxSceneEnvironmentId) {
    uint32_t envId = mNextSceneEnvironmentId++;
    if (!isSceneEnvironmentIdUsed(envId, nullptr)) {
      return envId;
    }
  }
  throw std::runtime_error(
      "failed to auto-assign PhysX GPU scene environment ID: too many environments");
}

void PhysxSystemGpu::setSceneEnvironmentId(std::shared_ptr<Scene> scene, int64_t envIdValue,
                                           bool allowDuplicate) {
  if (!scene) {
    throw std::runtime_error("failed to set PhysX GPU scene environment ID: scene is null");
  }

  uint32_t envId = normalizeSceneEnvironmentId(envIdValue);

  if (mSceneEnvironmentIds.size() % 1024 == 0) {
    std::erase_if(mSceneEnvironmentIds, [](const auto &p) { return p.first.expired(); });
  }

  if (mSceneEnvironmentIds.contains(scene)) {
    uint32_t current = mSceneEnvironmentIds.at(scene);
    if (current == envId) {
      return;
    }
    if (sceneHasPhysxBodies(scene)) {
      throw std::runtime_error(
          "failed to change PhysX GPU scene environment ID: env_id must be set before "
          "adding actors/articulations to the scene");
    }
  } else if (sceneHasPhysxBodies(scene)) {
    throw std::runtime_error(
        "failed to set PhysX GPU scene environment ID: env_id must be set before adding "
        "actors/articulations to the scene");
  }

  if (!allowDuplicate && isSceneEnvironmentIdUsed(envId, scene)) {
    throw std::runtime_error(
        "failed to set PhysX GPU scene environment ID: env_id is already used by another "
        "scene. Pass allow_duplicate=True to intentionally share a non-shared env_id");
  }

  mSceneEnvironmentIds[scene] = envId;
}

uint32_t PhysxSystemGpu::getSceneEnvironmentId(std::shared_ptr<Scene> scene) {
  if (!scene) {
    throw std::runtime_error("failed to get PhysX GPU scene environment ID: scene is null");
  }

  if (mSceneEnvironmentIds.size() % 1024 == 0) {
    std::erase_if(mSceneEnvironmentIds, [](const auto &p) { return p.first.expired(); });
  }

  if (mSceneEnvironmentIds.contains(scene)) {
    return mSceneEnvironmentIds.at(scene);
  }

  uint32_t envId = allocateSceneEnvironmentId();
  mSceneEnvironmentIds[scene] = envId;
  return envId;
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

void PhysxSystemGpu::setSceneEnvironmentIds(
    std::vector<std::pair<std::shared_ptr<Scene>, int64_t>> const &mapping,
    bool allowDuplicate) {
  for (auto &[scene, envId] : mapping) {
    setSceneEnvironmentId(scene, envId, allowDuplicate);
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

PhysxSystem::~PhysxSystem() { logger::info("Deleting PhysxSystem"); }

PhysxSystemCpu::~PhysxSystemCpu() {
  if (mPxScene) {
    mPxScene->release();
  }
  if (mPxCPUDispatcher) {
    mPxCPUDispatcher->release();
  }
}

#ifdef SAPIEN_CUDA
PhysxSystemGpu::~PhysxSystemGpu() {
  if (mPxScene) {
    mPxScene->release();
  }
  if (mPxCPUDispatcher) {
    mPxCPUDispatcher->release();
  }
}
#endif
} // namespace physx
} // namespace sapien
