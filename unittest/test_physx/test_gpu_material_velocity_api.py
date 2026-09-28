"""Run material/velocity behavior on GPU without changing the parent's PhysX mode."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


class TestGpuMaterialVelocityAPI(unittest.TestCase):
    def test_gpu_material_and_velocity_behavior(self) -> None:
        environment = dict(os.environ, SAPIEN_TEST_GPU="1")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(Path(__file__).resolve().parents[1]),
                "-p",
                "test_material_velocity_api.py",
                "-v",
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        if "GPU PhysX unavailable:" in output:
            self.skipTest(output.strip())
        self.assertIn("OK", output, output)


if __name__ == "__main__":
    unittest.main()
