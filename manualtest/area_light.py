import numpy as np
import sapien
import trimesh
from sapien.utils import Viewer

engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)

render_config = sapien.render.RenderConfig("../vulkan_shader/rt")
render_config["spp"] = 2
render_config.denoiser = "oidn"
render_config["aperture"] = 0.001
render_config["focusPlane"] = 2.0
sapien.render.set_default_render_config("viewer", render_config)

scene = engine.create_scene()
scene.add_ground(0)
# builder = scene.create_actor_builder().add_visual_from_file(
#     "../assets/models/suzanne.dae", scale=[0.1, 0.1, 0.1]
# )
# model = builder.build_kinematic()
# model.set_pose(sapien.Pose([0, 0, 0.1]))


def add_area_light():
    light = scene.add_area_light_for_ray_tracing(
        sapien.Pose([0, 0, 0.01], [0.7071068, 0, 0.7071068, 0]), [1, 1, 1], 1, 0.1
    )


def add_emission_plane():
    mesh = renderer.create_mesh(
        [[-0.05, -0.05, 0], [-0.05, 0.05, 0], [0.05, 0.05, 0], [0.05, -0.05, 0]],
        [[0, 1, 2], [0, 2, 3]],
    )
    mat = renderer.create_material()
    mat.set_base_color([0, 0, 0, 1])
    mat.set_emission([1, 1, 1, 1])
    light = (
        scene.create_actor_builder()
        .add_visual_from_mesh(mesh, material=mat)
        .build_kinematic()
    )
    light.set_pose(sapien.Pose([0, 0, 0.3]))


add_area_light()

viewer = Viewer(renderer)
viewer.set_scene(scene)

viewer.set_camera_xyz(-1, 0, 0.5)

scene.step()
while not viewer.closed:
    scene.update_render()
    viewer.render()
