import tempfile
import unittest
import warnings
from pathlib import Path

from sapien.wrapper.urdf_loader import URDFLoader


class TestURDFLoaderSDF(unittest.TestCase):
    def _parse_urdf(
        self,
        urdf_text: str,
        *,
        multiple_convex: bool = False,
        load_visuals: bool = True,
        load_collisions: bool = True,
    ):
        with tempfile.TemporaryDirectory() as tempdir:
            urdf_path = Path(tempdir) / "robot.urdf"
            urdf_path.write_text(urdf_text)

            loader = URDFLoader()
            loader.load_multiple_collisions_from_file = multiple_convex
            loader.load_visuals = load_visuals
            loader.load_collisions = load_collisions
            return loader.parse(str(urdf_path))

    def test_collision_sdf_auto_enables_nonconvex(self):
        urdf = """<?xml version="1.0"?>
<robot name="sdf_mesh_test">
  <link name="base">
    <collision name="sdf_mesh">
      <geometry>
        <mesh filename="mesh.obj" scale="1 1 1"/>
      </geometry>
      <sdf resolution="512" bits_per_subgrid_pixel="32" narrow_band_thickness="0.02"/>
    </collision>
    <collision name="plain_mesh">
      <geometry>
        <mesh filename="mesh.obj" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>
"""

        articulations, actors, cameras = self._parse_urdf(urdf)

        self.assertEqual(len(articulations), 0)
        self.assertEqual(len(cameras), 0)
        self.assertEqual(len(actors), 1)

        records = actors[0].collision_records
        self.assertEqual(records[0].type, "nonconvex_mesh")
        self.assertIsNotNone(records[0].sdf_config)
        self.assertEqual(records[0].sdf_config.resolution, 512)
        self.assertEqual(records[0].sdf_config.bits_per_subgrid_pixel, 32)
        self.assertAlmostEqual(records[0].sdf_config.narrow_band_thickness, 0.02)

        self.assertEqual(records[1].type, "convex_mesh")
        self.assertIsNone(records[1].sdf_config)

    def test_collision_sdf_overrides_multiple_convex_and_warns_on_primitive(self):
        urdf = """<?xml version="1.0"?>
<robot name="sdf_override_test">
  <link name="base">
    <collision name="box_collision">
      <geometry>
        <box size="1 1 1"/>
      </geometry>
      <sdf resolution="64"/>
    </collision>
    <collision name="mesh_collision">
      <geometry>
        <mesh filename="mesh.obj" scale="1 1 1"/>
      </geometry>
      <sdf resolution="128" enable_remeshing="true"/>
    </collision>
  </link>
</robot>
"""

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            articulations, actors, cameras = self._parse_urdf(urdf, multiple_convex=True)

        self.assertEqual(len(articulations), 0)
        self.assertEqual(len(cameras), 0)
        self.assertEqual(len(actors), 1)
        self.assertEqual(len(caught), 1)
        self.assertIn("ignoring <sdf> on non-mesh collision", str(caught[0].message))

        records = actors[0].collision_records
        self.assertEqual(records[0].type, "box")
        self.assertIsNone(records[0].sdf_config)

        self.assertEqual(records[1].type, "nonconvex_mesh")
        self.assertIsNotNone(records[1].sdf_config)
        self.assertEqual(records[1].sdf_config.resolution, 128)
        self.assertTrue(records[1].sdf_config.enable_remeshing)

    def test_physics_only_skips_visual_and_collision_records(self):
        urdf = """<?xml version="1.0"?>
<robot name="physics_only_test">
  <link name="base">
    <visual name="visual_box">
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </visual>
    <collision name="collision_box">
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>
"""

        articulations, actors, cameras = self._parse_urdf(
            urdf,
            load_visuals=False,
            load_collisions=False,
        )

        self.assertEqual(len(articulations), 0)
        self.assertEqual(len(cameras), 0)
        self.assertEqual(len(actors), 1)
        self.assertEqual(len(actors[0].visual_records), 0)
        self.assertEqual(len(actors[0].collision_records), 0)


if __name__ == "__main__":
    unittest.main()
