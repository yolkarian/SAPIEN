"""Rasterization packs read their lights from uniform buffers sized by the light counts: any
number of each kind, and a light added after the first frame is drawn."""

import unittest

import numpy as np
import sapien

SIZE = 32
# a camera two metres above the ground, looking straight down
LOOK_DOWN = sapien.Pose([0.0, 0.0, 2.0], [0.7071068, 0.0, 0.7071068, 0.0])


def ground_camera(scene: sapien.Scene, pack: str) -> sapien.render.RenderCameraComponent:
    """A grey ground without ambient light, seen from above by a camera of `pack`."""
    scene.set_ambient_light([0.0, 0.0, 0.0])
    scene.add_ground(0.0)
    config = sapien.render.RenderConfig(pack)
    camera = scene.add_camera("camera", SIZE, SIZE, 1.0, 0.1, 20.0, render_config=config)
    camera.entity.pose = LOOK_DOWN
    return camera


def add_lights(scene: sapien.Scene, kind: str, count: int) -> None:
    """`count` identical dim lights of `kind` one metre above the origin, shining down."""
    color = [0.05, 0.05, 0.05]
    for _ in range(count):
        if kind == "point":
            scene.add_point_light([0.0, 0.0, 1.0], color)
        elif kind == "directional":
            scene.add_directional_light([0.0, 0.0, -1.0], color)
        else:
            scene.add_spot_light([0.0, 0.0, 1.0], [0.0, 0.0, -1.0], 2.0, 2.5, color)


def radiance(scene: sapien.Scene, camera, pack: str) -> np.ndarray:
    """Linear radiance of the ground: `Lighting` of default, gamma-decoded Color of minimal."""
    scene.update_render()
    camera.take_picture()
    if pack == "minimal":
        return camera.get_picture("Color")[..., :3].astype(np.float64) ** 2.2
    return camera.get_picture("Lighting")[..., :3].astype(np.float64)


def direct_light(pack: str, kind: str, count: int) -> float:
    """Mean radiance `count` lights of `kind` add to the ground."""
    with sapien.Scene() as scene:
        camera = ground_camera(scene, pack)
        unlit = radiance(scene, camera, pack)
        add_lights(scene, kind, count)
        return float((radiance(scene, camera, pack) - unlit).mean())


class RasterLights(unittest.TestCase):
    def test_every_kind_beyond_the_former_limits(self) -> None:
        """The default pack held 10 point, 3 directional and 10 spot lights: more of each now
        add up exactly."""
        for kind, count in (("point", 24), ("directional", 6), ("spot", 14)):
            with self.subTest(kind=kind):
                one = direct_light("default", kind, 1)
                self.assertGreater(one, 1e-3)
                self.assertAlmostEqual(
                    direct_light("default", kind, count) / one, count, delta=1e-3 * count
                )

    def test_forward_pack_beyond_the_former_limit(self) -> None:
        """`minimal` lights in its forward pass and held 10 point lights."""
        one = direct_light("minimal", "point", 1)
        self.assertGreater(one, 1e-4)
        self.assertAlmostEqual(direct_light("minimal", "point", 12) / one, 12, delta=0.12)

    def test_lights_added_after_the_first_frame_are_drawn(self) -> None:
        with sapien.Scene() as scene:
            camera = ground_camera(scene, "default")
            unlit = radiance(scene, camera, "default")
            add_lights(scene, "point", 1)
            one = float((radiance(scene, camera, "default") - unlit).mean())
            add_lights(scene, "point", 15)  # grows the light buffer after recording
            sixteen = float((radiance(scene, camera, "default") - unlit).mean())
        self.assertGreater(one, 1e-3)
        self.assertAlmostEqual(sixteen / one, 16, delta=0.016)


if __name__ == "__main__":
    unittest.main()
