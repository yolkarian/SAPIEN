"""Point-cloud edits made after a capture must show up in the next capture."""

import unittest

import numpy as np
import sapien

SIZE = 32


class TestPointCloud(unittest.TestCase):
    def test_vertices_set_after_capture_are_rendered(self) -> None:
        with sapien.Scene() as scene:
            camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0)
            camera.entity.pose = sapien.Pose([-3.0, 0.0, 0.0])

            cloud = sapien.render.RenderPointCloudComponent(1)
            cloud.set_vertices(np.array([[0.0, 0.0, 10.0]], np.float32))  # out of view
            cloud.set_attribute("color", np.array([[1.0, 0.0, 0.0, 1.0]], np.float32))
            entity = sapien.Entity()
            entity.add_component(cloud)
            scene.add_entity(entity)

            def center_color() -> np.ndarray:
                scene.update_render()
                camera.take_picture()
                return camera.get_picture("Color")[SIZE // 2, SIZE // 2, :3].copy()

            hidden = center_color()
            # the uploaded vertex buffer is already bound by recorded draws
            cloud.set_vertices(np.array([[0.0, 0.0, 0.0]], np.float32))  # view center
            shown = center_color()

            self.assertGreater(shown[0], hidden[0] + 0.3)
            self.assertGreater(shown[0], shown[1] + 0.3)


if __name__ == "__main__":
    unittest.main()
