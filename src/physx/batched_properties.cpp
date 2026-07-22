#include "sapien/physx/batched_properties.h"
#include "sapien/physx/articulation_link_component.h"
#include "sapien/physx/material.h"
#include "sapien/physx/rigid_component.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace sapien {
namespace physx {

// All functions validate every entry before mutating anything so that a failed call never
// leaves the batch partially applied.

static void checkSize(char const *what, size_t expected, Eigen::Index actual) {
  if (static_cast<size_t>(actual) != expected) {
    throw std::runtime_error(std::string(what) +
                             " must have the same length as the component list (expected " +
                             std::to_string(expected) + ", got " + std::to_string(actual) + ")");
  }
}

void batchSetBodyMasses(std::vector<std::shared_ptr<PhysxRigidBodyComponent>> const &bodies,
                        Eigen::VectorXf const &masses, bool scaleInertia) {
  checkSize("masses", bodies.size(), masses.size());
  for (size_t i = 0; i < bodies.size(); ++i) {
    if (!bodies[i]) {
      throw std::runtime_error("bodies[" + std::to_string(i) + "] is None");
    }
    float mass = masses(static_cast<Eigen::Index>(i));
    if (!std::isfinite(mass) || mass <= 0.f) {
      throw std::runtime_error("masses[" + std::to_string(i) + "] must be finite and positive");
    }
    if (scaleInertia && bodies[i]->getMass() <= 0.f) {
      throw std::runtime_error("bodies[" + std::to_string(i) +
                               "] has non-positive mass; cannot scale inertia");
    }
  }
  for (size_t i = 0; i < bodies.size(); ++i) {
    auto &body = bodies[i];
    float mass = masses(static_cast<Eigen::Index>(i));
    if (scaleInertia) {
      float ratio = mass / body->getMass();
      Vec3 inertia = body->getInertia();
      body->setMass(mass);
      body->setInertia({inertia.x * ratio, inertia.y * ratio, inertia.z * ratio});
    } else {
      body->setMass(mass);
    }
  }
}

static void checkDrivableJoint(std::shared_ptr<PhysxArticulationJoint> const &joint, size_t i) {
  if (!joint) {
    throw std::runtime_error("joints[" + std::to_string(i) + "] is None");
  }
  if (joint->getDof() == 0) {
    throw std::runtime_error("joints[" + std::to_string(i) +
                             "] has 0 DOF; only joints with at least 1 DOF are supported");
  }
}

void batchSetJointFrictions(std::vector<std::shared_ptr<PhysxArticulationJoint>> const &joints,
                            Eigen::VectorXf const &frictions) {
  checkSize("frictions", joints.size(), frictions.size());
  for (size_t i = 0; i < joints.size(); ++i) {
    checkDrivableJoint(joints[i], i);
    float friction = frictions(static_cast<Eigen::Index>(i));
    if (!std::isfinite(friction) || friction < 0.f) {
      throw std::runtime_error("frictions[" + std::to_string(i) +
                               "] must be finite and non-negative");
    }
  }
  for (size_t i = 0; i < joints.size(); ++i) {
    joints[i]->setFriction(frictions(static_cast<Eigen::Index>(i)));
  }
}

void batchSetJointDriveProperties(
    std::vector<std::shared_ptr<PhysxArticulationJoint>> const &joints,
    std::optional<Eigen::VectorXf> const &stiffnesses,
    std::optional<Eigen::VectorXf> const &dampings,
    std::optional<Eigen::VectorXf> const &forceLimits) {
  if (!stiffnesses && !dampings && !forceLimits) {
    throw std::runtime_error(
        "at least one of stiffness, damping, and force_limit must be provided");
  }
  if (stiffnesses) {
    checkSize("stiffness", joints.size(), stiffnesses->size());
  }
  if (dampings) {
    checkSize("damping", joints.size(), dampings->size());
  }
  if (forceLimits) {
    checkSize("force_limit", joints.size(), forceLimits->size());
  }

  auto checkGain = [](char const *what, float value, size_t i) {
    if (!std::isfinite(value) || value < 0.f) {
      throw std::runtime_error(std::string(what) + "[" + std::to_string(i) +
                               "] must be finite and non-negative");
    }
  };

  for (size_t i = 0; i < joints.size(); ++i) {
    checkDrivableJoint(joints[i], i);
    auto idx = static_cast<Eigen::Index>(i);
    if (stiffnesses) {
      checkGain("stiffness", (*stiffnesses)(idx), i);
    }
    if (dampings) {
      checkGain("damping", (*dampings)(idx), i);
    }
    if (forceLimits && (std::isnan((*forceLimits)(idx)) || (*forceLimits)(idx) < 0.f)) {
      throw std::runtime_error("force_limit[" + std::to_string(i) +
                               "] must be non-negative (may be inf)");
    }
  }

  for (size_t i = 0; i < joints.size(); ++i) {
    auto &joint = joints[i];
    auto idx = static_cast<Eigen::Index>(i);
    float stiffness = stiffnesses ? (*stiffnesses)(idx) : joint->getDriveStiffness();
    float damping = dampings ? (*dampings)(idx) : joint->getDriveDamping();
    float forceLimit = forceLimits ? (*forceLimits)(idx) : joint->getDriveForceLimit();
    joint->setDriveProperties(stiffness, damping, forceLimit, joint->getDriveType());
  }
}

void batchSetBodyInertias(
    std::vector<std::shared_ptr<PhysxRigidBodyComponent>> const &bodies,
    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> const &inertias) {
  checkSize("inertias", bodies.size(), inertias.rows());
  for (size_t i = 0; i < bodies.size(); ++i) {
    if (!bodies[i]) {
      throw std::runtime_error("bodies[" + std::to_string(i) + "] is None");
    }
    auto idx = static_cast<Eigen::Index>(i);
    for (Eigen::Index c = 0; c < 3; ++c) {
      if (!std::isfinite(inertias(idx, c)) || inertias(idx, c) <= 0.f) {
        throw std::runtime_error("inertias[" + std::to_string(i) +
                                 "] must be finite and positive");
      }
    }
  }
  for (size_t i = 0; i < bodies.size(); ++i) {
    auto idx = static_cast<Eigen::Index>(i);
    bodies[i]->setInertia({inertias(idx, 0), inertias(idx, 1), inertias(idx, 2)});
  }
}

void batchSetBodyCMassLocalPoses(
    std::vector<std::shared_ptr<PhysxRigidBodyComponent>> const &bodies,
    Eigen::Matrix<float, Eigen::Dynamic, 7, Eigen::RowMajor> const &poses) {
  checkSize("poses", bodies.size(), poses.rows());
  for (size_t i = 0; i < bodies.size(); ++i) {
    if (!bodies[i]) {
      throw std::runtime_error("bodies[" + std::to_string(i) + "] is None");
    }
    auto idx = static_cast<Eigen::Index>(i);
    for (Eigen::Index c = 0; c < 7; ++c) {
      if (!std::isfinite(poses(idx, c))) {
        throw std::runtime_error("poses[" + std::to_string(i) + "] must be finite");
      }
    }
    float norm = poses.row(idx).tail<4>().norm();
    if (norm < 1e-6f) {
      throw std::runtime_error("poses[" + std::to_string(i) +
                               "] quaternion [qw, qx, qy, qz] must have non-zero norm");
    }
  }
  for (size_t i = 0; i < bodies.size(); ++i) {
    auto idx = static_cast<Eigen::Index>(i);
    float norm = poses.row(idx).tail<4>().norm();
    Quat q{poses(idx, 3) / norm, poses(idx, 4) / norm, poses(idx, 5) / norm,
           poses(idx, 6) / norm};
    bodies[i]->setCMassLocalPose(Pose({poses(idx, 0), poses(idx, 1), poses(idx, 2)}, q));
  }
}

void batchSetJointArmatures(std::vector<std::shared_ptr<PhysxArticulationJoint>> const &joints,
                            Eigen::VectorXf const &armatures) {
  checkSize("armatures", joints.size(), armatures.size());
  for (size_t i = 0; i < joints.size(); ++i) {
    checkDrivableJoint(joints[i], i);
    float armature = armatures(static_cast<Eigen::Index>(i));
    if (!std::isfinite(armature) || armature < 0.f) {
      throw std::runtime_error("armatures[" + std::to_string(i) +
                               "] must be finite and non-negative");
    }
  }
  for (size_t i = 0; i < joints.size(); ++i) {
    auto &joint = joints[i];
    joint->setArmature(Eigen::VectorXf::Constant(joint->getDof(),
                                                 armatures(static_cast<Eigen::Index>(i))));
  }
}

void batchSetMaterialProperties(std::vector<std::shared_ptr<PhysxMaterial>> const &materials,
                                std::optional<Eigen::VectorXf> const &staticFrictions,
                                std::optional<Eigen::VectorXf> const &dynamicFrictions,
                                std::optional<Eigen::VectorXf> const &restitutions) {
  if (!staticFrictions && !dynamicFrictions && !restitutions) {
    throw std::runtime_error("at least one of static_friction, dynamic_friction, and "
                             "restitution must be provided");
  }
  if (staticFrictions) {
    checkSize("static_friction", materials.size(), staticFrictions->size());
  }
  if (dynamicFrictions) {
    checkSize("dynamic_friction", materials.size(), dynamicFrictions->size());
  }
  if (restitutions) {
    checkSize("restitution", materials.size(), restitutions->size());
  }

  for (size_t i = 0; i < materials.size(); ++i) {
    if (!materials[i]) {
      throw std::runtime_error("materials[" + std::to_string(i) + "] is None");
    }
    auto idx = static_cast<Eigen::Index>(i);
    if (staticFrictions &&
        (!std::isfinite((*staticFrictions)(idx)) || (*staticFrictions)(idx) < 0.f)) {
      throw std::runtime_error("static_friction[" + std::to_string(i) +
                               "] must be finite and non-negative");
    }
    if (dynamicFrictions &&
        (!std::isfinite((*dynamicFrictions)(idx)) || (*dynamicFrictions)(idx) < 0.f)) {
      throw std::runtime_error("dynamic_friction[" + std::to_string(i) +
                               "] must be finite and non-negative");
    }
    if (restitutions && (!std::isfinite((*restitutions)(idx)) || (*restitutions)(idx) < 0.f ||
                         (*restitutions)(idx) > 1.f)) {
      throw std::runtime_error("restitution[" + std::to_string(i) + "] must be in [0, 1]");
    }
  }

  for (size_t i = 0; i < materials.size(); ++i) {
    auto idx = static_cast<Eigen::Index>(i);
    if (staticFrictions) {
      materials[i]->setStaticFriction((*staticFrictions)(idx));
    }
    if (dynamicFrictions) {
      materials[i]->setDynamicFriction((*dynamicFrictions)(idx));
    }
    if (restitutions) {
      materials[i]->setRestitution((*restitutions)(idx));
    }
  }
}

} // namespace physx
} // namespace sapien
