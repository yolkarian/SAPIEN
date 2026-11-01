"""C++ programs built against the installed package, with only its include and library directories.

The wheel serves C++ programs that use SAPIEN without Python at runtime, such as native PufferLib 5.0
environments. They must compile without -D flags or extra header paths (generated config headers,
shipped Eigen) and load libsapien without extra loader paths (the shipped libraries' RUNPATHs).
"""

import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import sapien

PACKAGE = Path(sapien.__file__).resolve().parent
LIBRARIES = PACKAGE.parent / "sapien.libs"

PROGRAM = r"""
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

#if defined(SAPIEN_CUDA) != defined(SVULKAN2_CUDA_INTEROP)
#error "sapien/config.h and svulkan2/build_options.h disagree about CUDA support"
#endif

using namespace sapien;
using namespace sapien::physx;

// Drops a 0.2 m box from z = 1 for 100 steps of 0.01 s and prints its height.
template <class System, class Steps> static void dropBox(std::shared_ptr<System> system, Steps steps) {
  auto scene = std::make_shared<Scene>(std::vector<std::shared_ptr<sapien::System>>{system});
  auto body = std::make_shared<PhysxRigidDynamicComponent>();
  body->attachCollision(std::make_shared<PhysxCollisionShapeBox>(Vec3(0.1f, 0.1f, 0.1f)));
  auto entity = std::make_shared<Entity>();
  entity->addComponent(body);
  entity->setPose(Pose(Vec3(0.f, 0.f, 1.f)));
  scene->addEntity(entity);
  system->setTimestep(0.01f);
  steps(*system);
  std::printf("z %.6f\n", entity->getPose().p.z);
  scene->close();
  system->close();
}

int main(int argc, char **argv) {
  if (argc != 2) {
    return 2;
  }
  try {
    if (std::strcmp(argv[1], "cpu") == 0) {
      try {
        PhysxDefault::EnableGPU("/nonexistent/libPhysXGpu_64.so");
        std::puts("accepted a missing runtime");
      } catch (std::runtime_error const &e) {
        std::printf("rejected a missing runtime: %s\n", e.what());
      }
      dropBox(std::make_shared<PhysxSystemCpu>(), [](PhysxSystemCpu &system) {
        for (int i = 0; i < 100; ++i) {
          system.step();
        }
      });
    } else {
      // the runtime sapien.physx.enable_gpu() downloaded
      PhysxDefault::EnableGPU();
      dropBox(std::make_shared<PhysxSystemGpu>(nullptr), [](PhysxSystemGpu &system) {
        system.gpuInit();
        for (int i = 0; i < 100; ++i) {
          system.step();
        }
        system.syncPosesGpuToCpu();
      });
    }
  } catch (std::exception const &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return 0;
}
"""

# Free fall from z = 1 for about 1 s: z = 1 - g t^2 / 2 is about -3.9 (semi-implicit steps and the
# extra step of gpu_init() move it by about 0.1).
FALL_RANGE = (-4.2, -3.7)


def _cuda_device_available() -> bool:
    try:
        return ctypes.CDLL("libcuda.so.1").cuInit(0) == 0
    except OSError:
        return False


@unittest.skipUnless(platform.system() == "Linux", "the wheel's library layout is Linux-specific")
@unittest.skipUnless(shutil.which("g++"), "requires g++")
@unittest.skipUnless((LIBRARIES / "libsapien.so").is_file(), "requires an installed wheel (sapien.libs)")
class TestCppPackage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        source = Path(cls.directory.name) / "program.cpp"
        source.write_text(PROGRAM)
        cls.binary = Path(cls.directory.name) / "program"
        include = PACKAGE / "include"
        # Only the package's own directories: no -D flags, no Eigen, OIDN or loader paths.
        cls.build = subprocess.run(
            [
                "g++", "-std=c++20", "-O1",
                "-I", str(include), "-I", str(include / "physx" / "include"),
                str(source), "-o", str(cls.binary),
                "-L", str(LIBRARIES), "-lsapien", f"-Wl,-rpath,{LIBRARIES}",
            ],
            capture_output=True,
            text=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def _run(self, mode: str) -> str:
        self.assertEqual(self.build.returncode, 0, self.build.stderr[-6000:])
        # Without LD_LIBRARY_PATH, libsapien's dependencies resolve through RUNPATHs only.
        env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
        result = subprocess.run(
            [str(self.binary), mode], capture_output=True, text=True, timeout=600, env=env
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def _height(self, output: str) -> float:
        match = re.search(r"^z (\S+)$", output, re.MULTILINE)
        self.assertIsNotNone(match, output)
        return float(match.group(1))

    def test_cpu_program_builds_and_simulates(self) -> None:
        output = self._run("cpu")
        self.assertIn("rejected a missing runtime: ", output)
        self.assertIn("/nonexistent/libPhysXGpu_64.so", output)
        z = self._height(output)
        self.assertTrue(FALL_RANGE[0] < z < FALL_RANGE[1], output)

    @unittest.skipUnless(_cuda_device_available(), "requires a CUDA device")
    def test_gpu_program_enables_gpu_physx_without_python(self) -> None:
        # Install the runtime into the cache PhysxDefault::EnableGPU() searches. Another process
        # does it, since enabling GPU PhysX here would bind this test process.
        subprocess.run(
            [sys.executable, "-c", "import sapien; sapien.physx.enable_gpu()"],
            check=True,
            timeout=1800,
        )
        z = self._height(self._run("gpu"))
        self.assertTrue(FALL_RANGE[0] < z < FALL_RANGE[1], z)


if __name__ == "__main__":
    unittest.main()
