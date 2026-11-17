// libsapien_robot_arm.so: the C interface of shim/sapien_robot_arm.h over SAPIEN's C++ API.
//
// Built against SAPIEN's C++ SDK with only its include and library directories (build.sh), like
// libsapien_cartpole.so.
//
// The world of PufferLib's robot_arm task (ocean/robot_arm/robot_arm.h at ea7f989): a fixed-base
// Franka Panda (base, links 1-7 on revolute joints, a hand fixed to link 7, two prismatic fingers)
// and a 7 cm cube on a table. The tables below copy robot_arm.h: joint frames (ra_fk), limits
// (ra_jmin, ra_jmax), masses and inertias (ra_dmass, ra_dcom, ra_dinert), collision boxes
// (ra_linkb, ra_gripb) and the gains of its joint motors (ra_prep). robot_arm's world is y-up with
// the arm base rotated by -90 degrees about x; SAPIEN's world here is that base frame, z-up.
#include "sapien_robot_arm.h"

#include <sapien/array.h>
#include <sapien/device.h>
#include <sapien/entity.h>
#include <sapien/math/pose.h>
#include <sapien/physx/articulation.h>
#include <sapien/physx/articulation_link_component.h>
#include <sapien/physx/collision_shape.h>
#include <sapien/physx/material.h>
#include <sapien/physx/physx_default.h>
#include <sapien/physx/physx_system.h>
#include <sapien/physx/rigid_component.h>
#include <sapien/scene.h>
#include <sapien/utils/cuda.h>

#include <Eigen/Eigenvalues>

#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sapien;
using namespace sapien::physx;
using ::physx::PxArticulationDriveType;
using ::physx::PxArticulationJointType;

struct SapienRobotArm {
  std::shared_ptr<PhysxSystemGpu> system;
  std::shared_ptr<Scene> tableScene;
  std::vector<std::shared_ptr<Scene>> scenes;
  std::shared_ptr<PhysxGpuContactPairImpulseQuery> padQuery;
  std::shared_ptr<PhysxGpuContactBodyImpulseQuery> handQuery;
  CudaArray articulationRows;
  CudaArray cubeRows;
  SapienRobotArmBuffers buffers{};
  std::string error;
};

namespace {

constexpr int kArmDofs = 7;
constexpr int kDofs = kArmDofs + 2;
constexpr float kPi = std::numbers::pi_v<float>;
constexpr float kHalfSqrt2 = 0.70710678118f;

// Collision filtering. SAPIEN's filter shader lets two shapes collide when the word0 of one meets
// the word1 of the other: word0 is the shape's kind, word1 the kinds it collides with. As in
// robot_arm, the arm links touch only the cube, the hand and fingers touch the cube and the
// table, and the robot never collides with itself.
constexpr uint32_t kTable = 1, kCube = 2, kArm = 4, kGripper = 8;
constexpr std::array<uint32_t, 4> kTableGroups{kTable, kCube | kGripper, 0, 0};
constexpr std::array<uint32_t, 4> kCubeGroups{kCube, kTable | kArm | kGripper, 0, 0};
constexpr std::array<uint32_t, 4> kArmGroups{kArm, kCube, 0, 0};
constexpr std::array<uint32_t, 4> kGripperGroups{kGripper, kCube | kTable, 0, 0};

// ra_fk: origin of joint k in the frame of link k, and the rotation before its angle
// (the joint turns about the z axis of that rotated frame).
const Vec3 kJointOffset[kArmDofs] = {{0.f, 0.f, 0.333f},   {0.f, 0.f, 0.f},
                                     {0.f, -0.316f, 0.f},  {0.0825f, 0.f, 0.f},
                                     {-0.0825f, 0.384f, 0.f}, {0.f, 0.f, 0.f},
                                     {0.088f, 0.f, 0.f}};
const Quat kJointStatic[kArmDofs] = {
    {1.f, 0.f, 0.f, 0.f},
    {kHalfSqrt2, -kHalfSqrt2, 0.f, 0.f},
    {kHalfSqrt2, kHalfSqrt2, 0.f, 0.f},
    {kHalfSqrt2, kHalfSqrt2, 0.f, 0.f},
    {kHalfSqrt2, -kHalfSqrt2, 0.f, 0.f},
    {kHalfSqrt2, kHalfSqrt2, 0.f, 0.f},
    {kHalfSqrt2, kHalfSqrt2, 0.f, 0.f},
};
const float kJointMin[kArmDofs] = {-2.8973f, -1.7628f, -2.8973f, -3.0718f,
                                   -2.8973f, -0.0175f, -2.8973f};
const float kJointMax[kArmDofs] = {2.8973f, 1.7628f, 2.8973f, -0.0698f,
                                   2.8973f, 3.7525f, 2.8973f};
// ra_prep's PD motor per joint, and its viscous term (-qd), folded into the drive damping
const float kStiffness[kArmDofs] = {4500.f, 4500.f, 3500.f, 3500.f, 2000.f, 2000.f, 2000.f};
const float kDamping[kArmDofs] = {451.f, 451.f, 351.f, 351.f, 201.f, 201.f, 201.f};
const float kTorqueLimit[kArmDofs] = {87.f, 87.f, 87.f, 87.f, 12.f, 12.f, 12.f};
constexpr float kMaxJointVelocity = 12.f; // ra_prep clamps qd to [-12, 12]

// The gripper motor (stiffness 1500 N/m, damping 18.57 N s/m, at most 100 N on the width) acts on
// each finger, whose position is half the width.
constexpr float kFingerStiffness = 2.f * 1500.f;
constexpr float kFingerDamping = 2.f * 18.57f;
constexpr float kFingerForceLimit = 100.f;
constexpr float kFingerMin = 0.002f, kFingerMax = 0.040f; // width in [0.004, 0.080]

// ra_dmass, ra_dcom, ra_dinert (xx, yy, zz, xy, xz, yz) of links 1-7, the hand and a finger
struct MassProperties {
  float mass;
  Vec3 com;
  float inertia[6];
};
const MassProperties kLinkMass[kArmDofs] = {
    {4.970684f, {0.003875f, 0.002081f, -0.047620f},
     {0.703370f, 0.706610f, 0.009117f, -0.000139f, 0.006772f, 0.019169f}},
    {0.646926f, {-0.003141f, -0.028720f, 0.003495f},
     {0.007962f, 0.028110f, 0.025995f, -0.003925f, 0.010254f, 0.000704f}},
    {3.228604f, {0.027518f, 0.039252f, -0.066502f},
     {0.037242f, 0.036155f, 0.010830f, -0.004761f, -0.011396f, -0.012805f}},
    {3.587895f, {-0.053170f, 0.104419f, 0.027454f},
     {0.025853f, 0.019552f, 0.028323f, 0.007796f, -0.001332f, 0.008641f}},
    {1.225946f, {-0.011953f, 0.041065f, -0.038437f},
     {0.035549f, 0.029474f, 0.008627f, -0.002117f, -0.004037f, 0.000229f}},
    {1.666555f, {0.060149f, -0.014117f, -0.010517f},
     {0.001964f, 0.004354f, 0.005433f, 0.000109f, -0.001158f, 0.000341f}},
    {0.735522f, {0.010517f, -0.004252f, 0.061597f},
     {0.012516f, 0.010027f, 0.004815f, -0.000428f, -0.001196f, -0.000741f}},
};
const MassProperties kHandMass{
    0.73f, {-0.01f, 0.f, 0.03f}, {0.001f, 0.0025f, 0.0017f, 0.f, 0.f, 0.f}};
const MassProperties kFingerMass{
    0.015f, {0.f, 0.f, 0.f}, {0.000002375f, 0.000002375f, 0.00000075f, 0.f, 0.f, 0.f}};

struct Box {
  Vec3 center;
  Vec3 half;
};
// ra_linkb: one box per link in the frame of links 1-7
const Box kLinkBox[kArmDofs] = {
    {{-0.00001f, -0.03719f, -0.06850f}, {0.05501f, 0.09220f, 0.12350f}},
    {{-0.00001f, -0.06949f, 0.03720f}, {0.05502f, 0.12451f, 0.09220f}},
    {{0.04124f, 0.02803f, -0.03300f}, {0.09626f, 0.08303f, 0.08800f}},
    {{-0.04126f, 0.03450f, 0.02803f}, {0.09625f, 0.08950f, 0.08303f}},
    {{-0.00001f, 0.03747f, -0.10340f}, {0.05500f, 0.09246f, 0.15560f}},
    {{0.04206f, 0.01523f, 0.00613f}, {0.08996f, 0.06643f, 0.05012f}},
    {{0.01864f, 0.01863f, 0.07940f}, {0.06267f, 0.06265f, 0.02740f}},
};
// ra_gripb boxes 0-2 in the hand frame; 3 (proximal) and 4 (distal, the pad) in a finger frame,
// whose +y points away from the other finger
const Box kHandBox[3] = {
    {{0.f, 0.f, -0.0055f}, {0.0320f, 0.1040f, 0.0205f}},
    {{0.f, 0.f, 0.0250f}, {0.0240f, 0.1020f, 0.0100f}},
    {{0.f, 0.f, 0.0505f}, {0.0220f, 0.1010f, 0.0155f}},
};
const Box kFingerBox[2] = {
    {{0.f, 0.0144f, 0.0150f}, {0.0105f, 0.0120f, 0.0150f}},
    {{0.f, 0.0080f, 0.0420f}, {0.0095f, 0.0080f, 0.0120f}},
};

constexpr float kCubeHalf = 0.035f;
constexpr float kCubeMass = 0.10f;

// rotates a joint frame's x axis (SAPIEN's joint axis) onto z (robot_arm's joint axis)
const Quat kXToZ(Vec3(0.f, 1.f, 0.f), -kPi / 2.f);

/** Mass, center of mass and inertia tensor (in the link frame) as PhysX principal axes. */
void setMassProperties(PhysxRigidBodyComponent &body, MassProperties const &m) {
  Eigen::Matrix3f tensor;
  tensor << m.inertia[0], m.inertia[3], m.inertia[4], //
      m.inertia[3], m.inertia[1], m.inertia[5],       //
      m.inertia[4], m.inertia[5], m.inertia[2];
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(tensor);
  Eigen::Matrix3f axes = solver.eigenvectors();
  if (axes.determinant() < 0.f) {
    axes.col(2) *= -1.f;
  }
  Eigen::Quaternionf frame(axes);
  Eigen::Vector3f moments = solver.eigenvalues();
  body.setMass(m.mass);
  body.setInertia(Vec3(moments.x(), moments.y(), moments.z()));
  body.setCMassLocalPose(Pose(m.com, Quat(frame.w(), frame.x(), frame.y(), frame.z())));
}

void attachBox(PhysxRigidBodyComponent &body, Box const &box,
               std::shared_ptr<PhysxMaterial> const &material,
               std::array<uint32_t, 4> const &groups) {
  auto shape = std::make_shared<PhysxCollisionShapeBox>(box.half, material);
  shape->setLocalPose(Pose(box.center));
  shape->setCollisionGroups(groups);
  body.attachCollision(shape);
}

// PhysX rejects joint motion changes (types, limits) once the articulation is in a scene, so
// links are configured completely before buildRobot adds them.
std::shared_ptr<PhysxArticulationLinkComponent>
newLink(std::shared_ptr<PhysxArticulationLinkComponent> parent,
        PxArticulationJointType::Enum type, Pose anchorInParent, Pose anchorInChild) {
  auto link = PhysxArticulationLinkComponent::Create(parent);
  auto joint = link->getJoint();
  joint->setType(type);
  joint->setAnchorPoseInParent(anchorInParent);
  joint->setAnchorPoseInChild(anchorInChild);
  return link;
}

void setDrive(PhysxArticulationLinkComponent &link, float low, float high, float stiffness,
              float damping, float forceLimit) {
  auto joint = link.getJoint();
  Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor> limit(1, 2);
  limit << low, high;
  joint->setLimit(limit);
  joint->setFriction(0.f); // the SAPIEN default is 0.05
  joint->setArmature(Eigen::VectorXf::Zero(1));
  joint->setDriveProperties(stiffness, damping, forceLimit, PxArticulationDriveType::eFORCE);
}

struct Robot {
  std::shared_ptr<PhysxArticulationLinkComponent> root, hand, left, right;
  std::vector<std::shared_ptr<PhysxArticulationJoint>> dofOrder;
};

struct Materials {
  std::shared_ptr<PhysxMaterial> robot, finger, cube, table;
};

Robot buildRobot(Scene &scene, Materials const &materials) {
  Robot robot;
  robot.root = newLink(nullptr, PxArticulationJointType::eFIX, Pose(), Pose());
  robot.root->setMass(1.f); // fixed base without collision
  robot.root->setInertia(Vec3(0.01f));
  std::vector<std::shared_ptr<PhysxArticulationLinkComponent>> links{robot.root};

  auto parent = robot.root;
  for (int k = 0; k < kArmDofs; ++k) {
    auto link = newLink(parent, PxArticulationJointType::eREVOLUTE,
                        Pose(kJointOffset[k], kJointStatic[k] * kXToZ), Pose(kXToZ));
    attachBox(*link, kLinkBox[k], materials.robot, kArmGroups);
    setMassProperties(*link, kLinkMass[k]);
    setDrive(*link, kJointMin[k], kJointMax[k], kStiffness[k], kDamping[k], kTorqueLimit[k]);
    link->getJoint()->setMaxJointVelocity(kMaxJointVelocity);
    robot.dofOrder.push_back(link->getJoint());
    links.push_back(link);
    parent = link;
  }

  // ra_fk: the hand sits 0.107 m along link 7's z, turned -45 degrees about it
  robot.hand = newLink(parent, PxArticulationJointType::eFIX,
                       Pose(Vec3(0.f, 0.f, 0.107f), Quat(Vec3(0.f, 0.f, 1.f), -kPi / 4.f)),
                       Pose());
  for (Box const &box : kHandBox) {
    attachBox(*robot.hand, box, materials.robot, kGripperGroups);
  }
  setMassProperties(*robot.hand, kHandMass);
  links.push_back(robot.hand);

  // Fingers 0.0584 m along the hand's z. The left one moves along the hand's +y in the hand's
  // orientation, the right one along -y turned 180 degrees about z (ra_fk).
  Quat toY(Vec3(0.f, 0.f, 1.f), kPi / 2.f);
  Quat toMinusY(Vec3(0.f, 0.f, 1.f), -kPi / 2.f);
  Vec3 fingerBase(0.f, 0.f, 0.0584f);
  robot.left = newLink(robot.hand, PxArticulationJointType::ePRISMATIC, Pose(fingerBase, toY),
                       Pose(toY));
  robot.right = newLink(robot.hand, PxArticulationJointType::ePRISMATIC,
                        Pose(fingerBase, toMinusY), Pose(toY));
  for (auto const &finger : {robot.left, robot.right}) {
    attachBox(*finger, kFingerBox[0], materials.robot, kGripperGroups);
    attachBox(*finger, kFingerBox[1], materials.finger, kGripperGroups);
    setMassProperties(*finger, kFingerMass);
    setDrive(*finger, kFingerMin, kFingerMax, kFingerStiffness, kFingerDamping,
             kFingerForceLimit);
    robot.dofOrder.push_back(finger->getJoint());
    links.push_back(finger);
  }
  for (auto const &link : links) {
    auto entity = std::make_shared<Entity>();
    entity->addComponent(link);
    scene.addEntity(entity);
  }
  return robot;
}

std::shared_ptr<PhysxRigidDynamicComponent> buildCube(Scene &scene, Materials const &materials) {
  auto cube = std::make_shared<PhysxRigidDynamicComponent>();
  attachBox(*cube, {Vec3(0.f), Vec3(kCubeHalf)}, materials.cube, kCubeGroups);
  float inertia = kCubeMass * (2.f * kCubeHalf) * (2.f * kCubeHalf) / 6.f;
  cube->setMass(kCubeMass);
  cube->setInertia(Vec3(inertia));
  auto entity = std::make_shared<Entity>();
  entity->addComponent(cube);
  entity->setPose(Pose(Vec3(0.5f, 0.f, kCubeHalf)));
  scene.addEntity(entity);
  return cube;
}

void addTable(Scene &scene, Materials const &materials) {
  auto table = std::make_shared<PhysxRigidStaticComponent>();
  auto plane = std::make_shared<PhysxCollisionShapePlane>(materials.table);
  // a PhysX plane faces +x; turn it to face +z at the table top (z = 0)
  plane->setLocalPose(Pose(Quat(Vec3(0.f, 1.f, 0.f), -kPi / 2.f)));
  plane->setCollisionGroups(kTableGroups);
  table->attachCollision(plane);
  auto entity = std::make_shared<Entity>();
  entity->addComponent(table);
  scene.addEntity(entity);
}

int fail(SapienRobotArm *h, char const *where, std::exception const &e) {
  h->error = std::string(where) + ": " + e.what();
  return -1;
}

} // namespace

extern "C" {

SapienRobotArm *sapien_robot_arm_create(int num_envs, int cuda_id, float physics_dt,
                                        int solver_position_iterations,
                                        int solver_velocity_iterations, char *err, int err_len) {
  try {
    if (num_envs <= 0 || physics_dt <= 0.f || solver_position_iterations <= 0 ||
        solver_velocity_iterations < 0) {
      throw std::invalid_argument("invalid environment count, time step or solver iterations");
    }
    if (!PhysxDefault::GetGPUEnabled()) {
      PhysxDefault::EnableGPU();
    }
    // Every environment is its own scene; one shared scene holds the table.
    PhysxSceneConfig sceneConfig = PhysxDefault::getSceneConfig();
    sceneConfig.numScenes = static_cast<uint32_t>(num_envs);
    sceneConfig.withSharedScene = true;
    PhysxDefault::setSceneConfig(sceneConfig);
    PhysxDefault::setBodyConfig(PhysxBodyConfig{
        .solverPositionIterations = static_cast<uint32_t>(solver_position_iterations),
        .solverVelocityIterations = static_cast<uint32_t>(solver_velocity_iterations),
    });
    // Contact and pair capacities for thousands of arms grasping cubes (PhysX's defaults are
    // sized for a single scene).
    PhysxDefault::setGpuMemoryConfig(1u << 24, 1u << 20, 1u << 19, 1u << 26, 1u << 22, 1u << 10,
                                     1u << 10, 1u << 26);

    auto h = std::make_unique<SapienRobotArm>();
    h->system = std::make_shared<PhysxSystemGpu>(findDevice("cuda:" + std::to_string(cuda_id)));
    h->system->setTimestep(physics_dt);

    Materials materials{
        .robot = std::make_shared<PhysxMaterial>(0.80f, 0.80f, 0.f),  // RA_HAND_COLLISION_FRICTION
        .finger = std::make_shared<PhysxMaterial>(0.80f, 0.80f, 0.f), // RA_FINGER_FRICTION
        .cube = std::make_shared<PhysxMaterial>(0.72f, 0.72f, 0.12f), // RA_CUBE_*
        .table = std::make_shared<PhysxMaterial>(0.72f, 0.72f, 0.12f),
    };
    h->tableScene = std::make_shared<Scene>(std::vector<std::shared_ptr<System>>{h->system});
    h->system->markSceneShared(h->tableScene);
    addTable(*h->tableScene, materials);

    std::vector<Robot> robots;
    std::vector<std::shared_ptr<PhysxRigidDynamicComponent>> cubes;
    robots.reserve(num_envs);
    cubes.reserve(num_envs);
    h->scenes.reserve(num_envs);
    for (int i = 0; i < num_envs; ++i) {
      auto scene = std::make_shared<Scene>(std::vector<std::shared_ptr<System>>{h->system});
      robots.push_back(buildRobot(*scene, materials));
      cubes.push_back(buildCube(*scene, materials));
      h->scenes.push_back(scene);
    }
    h->system->gpuInit();

    // The kernels of the environment index joint rows by dof order
    // [joint 1..7, left finger, right finger].
    if (robots[0].root->getArticulation()->getActiveJoints() != robots[0].dofOrder) {
      throw std::runtime_error("unexpected articulation dof order");
    }
    auto qpos = h->system->gpuGetArticulationQposCudaHandle();
    auto rigid = h->system->gpuGetRigidDynamicCudaHandle();
    if (qpos.shape != std::vector<int>{num_envs, kDofs} ||
        rigid.shape != std::vector<int>{num_envs, 13}) {
      throw std::runtime_error(
          "unexpected GPU buffer shape: qpos [" + std::to_string(qpos.shape.at(0)) + ", " +
          std::to_string(qpos.shape.at(1)) + "], rigid dynamics [" +
          std::to_string(rigid.shape.at(0)) + ", " + std::to_string(rigid.shape.at(1)) + "]");
    }

    std::vector<int> articulationRows, cubeRows;
    std::vector<std::pair<std::shared_ptr<PhysxRigidBaseComponent>,
                          std::shared_ptr<PhysxRigidBaseComponent>>>
        padPairs;
    std::vector<std::shared_ptr<PhysxRigidBaseComponent>> handBodies;
    for (int i = 0; i < num_envs; ++i) {
      articulationRows.push_back(robots[i].root->getArticulation()->getGpuIndex());
      cubeRows.push_back(cubes[i]->getGpuIndex());
      padPairs.push_back({robots[i].left, cubes[i]});
      padPairs.push_back({robots[i].right, cubes[i]});
      handBodies.insert(handBodies.end(), {robots[i].hand, robots[i].left, robots[i].right});
    }
    h->padQuery = h->system->gpuCreateContactPairImpulseQuery(padPairs);
    h->handQuery = h->system->gpuCreateContactBodyImpulseQuery(handBodies);
    {
      CudaDeviceScope scope(cuda_id);
      h->articulationRows = CudaArray::FromData(articulationRows);
      h->cubeRows = CudaArray::FromData(cubeRows);
    }

    h->buffers = SapienRobotArmBuffers{
        .dofs = kDofs,
        .qpos = static_cast<float *>(qpos.ptr),
        .qvel = static_cast<float *>(h->system->gpuGetArticulationQvelCudaHandle().ptr),
        .qtarget = static_cast<float *>(h->system->gpuGetArticulationQTargetPosCudaHandle().ptr),
        .rigid_dynamic = static_cast<float *>(rigid.ptr),
        .articulation = static_cast<int const *>(h->articulationRows.ptr),
        .cube = static_cast<int const *>(h->cubeRows.ptr),
        .pad_impulse = static_cast<float const *>(h->padQuery->buffer.ptr),
        .hand_impulse = static_cast<float const *>(h->handQuery->buffer.ptr),
    };
    return h.release();
  } catch (std::exception const &e) {
    if (err && err_len > 0) {
      std::snprintf(err, err_len, "%s", e.what());
    }
    return nullptr;
  }
}

void sapien_robot_arm_destroy(SapienRobotArm *h) {
  if (!h) {
    return;
  }
  try {
    h->padQuery.reset();
    h->handQuery.reset();
    for (auto &scene : h->scenes) {
      scene->close();
    }
    h->tableScene->close();
    h->system->close();
  } catch (std::exception const &e) {
    std::fprintf(stderr, "sapien_robot_arm_destroy: %s\n", e.what());
  }
  delete h;
}

const char *sapien_robot_arm_error(const SapienRobotArm *h) { return h->error.c_str(); }

SapienRobotArmBuffers sapien_robot_arm_buffers(const SapienRobotArm *h) { return h->buffers; }

int sapien_robot_arm_set_stream(SapienRobotArm *h, void *stream) {
  try {
    h->system->gpuSetCudaStream(reinterpret_cast<uintptr_t>(stream));
    return 0;
  } catch (std::exception const &e) {
    return fail(h, "sapien_robot_arm_set_stream", e);
  }
}

int sapien_robot_arm_apply_targets(SapienRobotArm *h) {
  try {
    h->system->gpuApplyArticulationQTargetPos();
    return 0;
  } catch (std::exception const &e) {
    return fail(h, "sapien_robot_arm_apply_targets", e);
  }
}

int sapien_robot_arm_simulate(SapienRobotArm *h, int steps) {
  try {
    for (int i = 0; i < steps; ++i) {
      h->system->step();
    }
    return 0;
  } catch (std::exception const &e) {
    return fail(h, "sapien_robot_arm_simulate", e);
  }
}

int sapien_robot_arm_fetch(SapienRobotArm *h) {
  try {
    h->system->gpuFetchArticulationQpos();
    h->system->gpuFetchArticulationQvel();
    h->system->gpuFetchRigidDynamicData();
    // ordered on the system's stream; the environment's kernels read them there
    h->system->gpuQueryContactPairImpulses(*h->padQuery, false);
    h->system->gpuQueryContactBodyImpulses(*h->handQuery, false);
    return 0;
  } catch (std::exception const &e) {
    return fail(h, "sapien_robot_arm_fetch", e);
  }
}

int sapien_robot_arm_apply_state(SapienRobotArm *h) {
  try {
    h->system->gpuApplyArticulationQpos();
    h->system->gpuApplyArticulationQvel();
    h->system->gpuUpdateArticulationKinematics();
    h->system->gpuApplyRigidDynamicData();
    return 0;
  } catch (std::exception const &e) {
    return fail(h, "sapien_robot_arm_apply_state", e);
  }
}

} // extern "C"
