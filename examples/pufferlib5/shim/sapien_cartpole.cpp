// libsapien_cartpole.so: the C interface of shim/sapien_cartpole.h over SAPIEN's C++ API.
//
// Built against the installed SAPIEN package with only its include and library directories
// (build.sh): the package's headers carry their compile options and its libraries their RUNPATHs.
//
// Each cartpole is one 3-link articulation without collision shapes, so all of them share one scene:
//   base (fixed root) -> cart (prismatic along world x) -> pole (revolute about world y,
//   center of mass 0.5 m above the pivot, upright at angle 0).
// All joints are frictionless and undamped with zero armature and no drive: the environment moves
// the cart with joint forces only.
#include "sapien_cartpole.h"

#include <sapien/device.h>
#include <sapien/entity.h>
#include <sapien/math/pose.h>
#include <sapien/physx/articulation_link_component.h>
#include <sapien/physx/physx_default.h>
#include <sapien/physx/physx_system.h>
#include <sapien/scene.h>

#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sapien;
using namespace sapien::physx;

struct SapienCartpole {
  std::shared_ptr<PhysxSystemGpu> system;
  std::shared_ptr<Scene> scene;
  int maxDofs{};
  float *qpos{}, *qvel{}, *qf{};
  std::string error;
};

namespace {

constexpr float kTimestep = 0.02f;

std::shared_ptr<PhysxArticulationLinkComponent>
addLink(Scene &scene, std::shared_ptr<PhysxArticulationLinkComponent> parent, float mass,
        Vec3 inertia, Pose cmass, ::physx::PxArticulationJointType::Enum jointType,
        Pose jointPose) {
  auto link = PhysxArticulationLinkComponent::Create(parent);
  link->setMass(mass);
  link->setInertia(inertia);
  link->setCMassLocalPose(cmass);

  auto joint = link->getJoint();
  joint->setType(jointType);
  if (jointType != ::physx::PxArticulationJointType::eFIX) {
    // The joint acts along / about the x axis of its frame, the same frame in parent and child.
    joint->setAnchorPoseInParent(jointPose);
    joint->setAnchorPoseInChild(jointPose);
    Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor> limit(1, 2);
    limit << -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity();
    joint->setLimit(limit);
    joint->setFriction(0.f); // the SAPIEN default is 0.05
    joint->setDriveProperties(0.f, 0.f, std::numeric_limits<float>::max(),
                              ::physx::PxArticulationDriveType::eFORCE);
    joint->setArmature(Eigen::VectorXf::Zero(1));
  }

  auto entity = std::make_shared<Entity>();
  entity->addComponent(link);
  scene.addEntity(entity);
  return link;
}

void buildCartpole(Scene &scene) {
  Vec3 const boxInertia(0.01f, 0.01f, 0.01f);
  auto base = addLink(scene, nullptr, 1.f, boxInertia, Pose(),
                      ::physx::PxArticulationJointType::eFIX, Pose());
  auto cart = addLink(scene, base, 1.f, boxInertia, Pose(),
                      ::physx::PxArticulationJointType::ePRISMATIC, Pose());
  // turn the joint frame's x axis onto world y
  Pose yAxis(Quat(Vec3(0.f, 0.f, 1.f), std::numbers::pi_v<float> / 2.f));
  // a 1 m, 0.1 kg rod: I = m L^2 / 12 about its center
  addLink(scene, cart, 0.1f, Vec3(0.1f / 12.f, 0.1f / 12.f, 1e-4f), Pose(Vec3(0.f, 0.f, 0.5f)),
          ::physx::PxArticulationJointType::eREVOLUTE_UNWRAPPED, yAxis);
}

int fail(SapienCartpole *h, char const *where, std::exception const &e) {
  h->error = std::string(where) + ": " + e.what();
  return -1;
}

} // namespace

extern "C" {

SapienCartpole *sapien_cartpole_create(int num_envs, int cuda_id, char *err, int err_len) {
  try {
    if (!PhysxDefault::GetGPUEnabled()) {
      PhysxDefault::EnableGPU();
    }
    auto h = std::make_unique<SapienCartpole>();
    h->system = std::make_shared<PhysxSystemGpu>(findDevice("cuda:" + std::to_string(cuda_id)));
    h->system->setTimestep(kTimestep);
    h->scene = std::make_shared<Scene>(std::vector<std::shared_ptr<System>>{h->system});
    for (int i = 0; i < num_envs; ++i) {
      buildCartpole(*h->scene);
    }
    h->system->gpuInit();

    auto qpos = h->system->gpuGetArticulationQposCudaHandle();
    auto qvel = h->system->gpuGetArticulationQvelCudaHandle();
    auto qf = h->system->gpuGetArticulationQfCudaHandle();
    if (qpos.shape != std::vector<int>{num_envs, 2} || qvel.shape != qpos.shape ||
        qf.shape != qpos.shape) {
      throw std::runtime_error("unexpected articulation buffer shape");
    }
    h->maxDofs = qpos.shape[1];
    h->qpos = static_cast<float *>(qpos.ptr);
    h->qvel = static_cast<float *>(qvel.ptr);
    h->qf = static_cast<float *>(qf.ptr);
    return h.release();
  } catch (std::exception const &e) {
    if (err && err_len > 0) {
      std::snprintf(err, err_len, "%s", e.what());
    }
    return nullptr;
  }
}

void sapien_cartpole_destroy(SapienCartpole *h) {
  if (!h) {
    return;
  }
  try {
    h->scene->close();
    h->system->close();
  } catch (std::exception const &e) {
    std::fprintf(stderr, "sapien_cartpole_destroy: %s\n", e.what());
  }
  delete h;
}

const char *sapien_cartpole_error(const SapienCartpole *h) { return h->error.c_str(); }
int sapien_cartpole_max_dofs(const SapienCartpole *h) { return h->maxDofs; }
float *sapien_cartpole_qpos_ptr(const SapienCartpole *h) { return h->qpos; }
float *sapien_cartpole_qvel_ptr(const SapienCartpole *h) { return h->qvel; }
float *sapien_cartpole_qf_ptr(const SapienCartpole *h) { return h->qf; }

int sapien_cartpole_set_stream(SapienCartpole *h, void *stream) {
  try {
    h->system->gpuSetCudaStream(reinterpret_cast<uintptr_t>(stream));
    return 0;
  } catch (std::exception const &e) {
    return fail(h, "sapien_cartpole_set_stream", e);
  }
}

int sapien_cartpole_apply_qf(SapienCartpole *h) {
  try {
    h->system->gpuApplyArticulationQf();
    return 0;
  } catch (std::exception const &e) {
    return fail(h, "sapien_cartpole_apply_qf", e);
  }
}

int sapien_cartpole_step(SapienCartpole *h) {
  try {
    h->system->step();
    return 0;
  } catch (std::exception const &e) {
    return fail(h, "sapien_cartpole_step", e);
  }
}

int sapien_cartpole_fetch_state(SapienCartpole *h) {
  try {
    h->system->gpuFetchArticulationQpos();
    h->system->gpuFetchArticulationQvel();
    return 0;
  } catch (std::exception const &e) {
    return fail(h, "sapien_cartpole_fetch_state", e);
  }
}

int sapien_cartpole_apply_state(SapienCartpole *h) {
  try {
    h->system->gpuApplyArticulationQpos();
    h->system->gpuApplyArticulationQvel();
    h->system->gpuUpdateArticulationKinematics();
    return 0;
  } catch (std::exception const &e) {
    return fail(h, "sapien_cartpole_apply_state", e);
  }
}

} // extern "C"
