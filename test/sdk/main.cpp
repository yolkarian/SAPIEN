// A program outside this repository that uses SAPIEN's C++ SDK; scripts/test_sdk.sh builds it
// against an unpacked SDK archive.
//
//   sdk_test cpu   CPU PhysX: EnableGPU() rejects a missing runtime, then a box falls
//   sdk_test gpu   GPU PhysX through the runtime EnableGPU() finds, then a box falls
//
// Exits with a non-zero status when a check fails.
#include <sapien/entity.h>
#include <sapien/physx/collision_shape.h>
#include <sapien/physx/physx_default.h>
#include <sapien/physx/physx_system.h>
#include <sapien/physx/rigid_component.h>
#include <sapien/sapien_renderer/sapien_renderer_system.h>
#include <sapien/scene.h>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

// Without these two agreeing, the program would see other class layouts than the libraries.
#if defined(SAPIEN_CUDA) != defined(SVULKAN2_CUDA_INTEROP)
#error "sapien/config.h and svulkan2/build_options.h disagree about CUDA support"
#endif

using namespace sapien;
using namespace sapien::physx;

// Drops a 0.2 m box from z = 1 for 100 steps of 0.01 s and returns its height.
template <class System, class Steps>
static float dropBox(std::shared_ptr<System> system, Steps steps) {
  auto scene = std::make_shared<Scene>(std::vector<std::shared_ptr<sapien::System>>{system});
  auto body = std::make_shared<PhysxRigidDynamicComponent>();
  body->attachCollision(std::make_shared<PhysxCollisionShapeBox>(Vec3(0.1f, 0.1f, 0.1f)));
  auto entity = std::make_shared<Entity>();
  entity->addComponent(body);
  entity->setPose(Pose(Vec3(0.f, 0.f, 1.f)));
  scene->addEntity(entity);
  system->setTimestep(0.01f);
  steps(*system);
  float z = entity->getPose().p.z;
  scene->close();
  system->close();
  return z;
}

int main(int argc, char **argv) {
  if (argc != 2 || (std::strcmp(argv[1], "cpu") != 0 && std::strcmp(argv[1], "gpu") != 0)) {
    std::fprintf(stderr, "usage: %s cpu|gpu\n", argv[0]);
    return 2;
  }
  bool gpu = std::strcmp(argv[1], "gpu") == 0;
  try {
    float z;
    if (!gpu) {
      try {
        PhysxDefault::EnableGPU("/nonexistent/libPhysXGpu_64.so");
        std::fputs("EnableGPU() accepted a missing PhysX GPU runtime\n", stderr);
        return 1;
      } catch (std::runtime_error const &) {
      }
      z = dropBox(std::make_shared<PhysxSystemCpu>(), [](PhysxSystemCpu &system) {
        for (int i = 0; i < 100; ++i) {
          system.step();
        }
      });
    } else {
      PhysxDefault::EnableGPU();
      z = dropBox(std::make_shared<PhysxSystemGpu>(nullptr), [](PhysxSystemGpu &system) {
        system.gpuInit();
        for (int i = 0; i < 100; ++i) {
          system.step();
        }
        system.syncPosesGpuToCpu();
      });
    }
    std::printf("%s PhysX: box at z = %.3f\n", argv[1], z);
    // Free fall for about 1 s: z = 1 - g t^2 / 2 is about -3.9 (semi-implicit steps and the extra
    // step of gpuInit() move it by about 0.1).
    if (!(-4.2f < z && z < -3.7f)) {
      std::fputs("the box did not fall as expected\n", stderr);
      return 1;
    }
  } catch (std::exception const &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return 0;
}
