#pragma once
#include <Eigen/Eigen>
#include <memory>
#include <optional>
#include <vector>

namespace sapien {
namespace physx {
class PhysxRigidBodyComponent;
class PhysxArticulationJoint;
class PhysxMaterial;

/** Batched physical-property setters, primarily for reset-time domain randomization.
 *
 *  These functions write through the PhysX CPU API and are valid on both CPU and GPU
 *  systems. On PhysxSystemGpu they may be called any time after gpu_init() as long as no
 *  simulation step is in flight (never between stepStart() and stepFinish()); PhysX uploads
 *  the dirty mass/joint/material data to the GPU during the next simulation step without
 *  disturbing GPU-side pose, velocity, or joint state. */

/** Set the mass of each body (rigid dynamic body or articulation link).
 *  When scaleInertia is true, each body's diagonal inertia is multiplied by
 *  newMass / oldMass, keeping the center-of-mass pose; this requires a positive old mass. */
void batchSetBodyMasses(std::vector<std::shared_ptr<PhysxRigidBodyComponent>> const &bodies,
                        Eigen::VectorXf const &masses, bool scaleInertia);

/** Set the diagonal inertia (in the center-of-mass frame) of each body (rigid dynamic
 *  body or articulation link). Each row of inertias is a positive [ix, iy, iz]. */
void batchSetBodyInertias(std::vector<std::shared_ptr<PhysxRigidBodyComponent>> const &bodies,
                          Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> const &inertias);

/** Set the center-of-mass local pose of each body (rigid dynamic body or articulation
 *  link). Each row of poses is [x, y, z, qw, qx, qy, qz]; quaternions are normalized
 *  before being applied. */
void batchSetBodyCMassLocalPoses(
    std::vector<std::shared_ptr<PhysxRigidBodyComponent>> const &bodies,
    Eigen::Matrix<float, Eigen::Dynamic, 7, Eigen::RowMajor> const &poses);

/** Set the friction coefficient of each articulation joint. Every joint must have at least
 *  1 DOF. */
void batchSetJointFrictions(std::vector<std::shared_ptr<PhysxArticulationJoint>> const &joints,
                            Eigen::VectorXf const &frictions);

/** Set drive stiffness, damping, and force limit of each articulation joint.
 *  Passing std::nullopt keeps the current value of that field; the drive type is always
 *  preserved. Every joint must have at least 1 DOF. */
void batchSetJointDriveProperties(
    std::vector<std::shared_ptr<PhysxArticulationJoint>> const &joints,
    std::optional<Eigen::VectorXf> const &stiffnesses,
    std::optional<Eigen::VectorXf> const &dampings,
    std::optional<Eigen::VectorXf> const &forceLimits);

/** Set the armature of each articulation joint; the value of a joint applies to all of
 *  its DOFs. Every joint must have at least 1 DOF. */
void batchSetJointArmatures(std::vector<std::shared_ptr<PhysxArticulationJoint>> const &joints,
                            Eigen::VectorXf const &armatures);

/** Set static friction, dynamic friction, and restitution of each physical material.
 *  Passing std::nullopt keeps the current value of that field. Restitution must be in
 *  [0, 1]. */
void batchSetMaterialProperties(std::vector<std::shared_ptr<PhysxMaterial>> const &materials,
                                std::optional<Eigen::VectorXf> const &staticFrictions,
                                std::optional<Eigen::VectorXf> const &dynamicFrictions,
                                std::optional<Eigen::VectorXf> const &restitutions);

} // namespace physx
} // namespace sapien
