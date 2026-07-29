"""Regenerate tutorial images with the installed SAPIEN package.

Run this script from any directory after installing the current repository wheel:

    python docs/generate_tutorial_assets.py

The script needs a Vulkan-capable device. Ray-tracing images additionally need
Vulkan ray-tracing support.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sapien


DOCS_SOURCE = Path(__file__).resolve().parent / "source" / "tutorial"
BASIC_ASSETS = DOCS_SOURCE / "basic" / "assets"
RENDERING_ASSETS = DOCS_SOURCE / "rendering" / "assets"


def look_at_pose(position: np.ndarray, target: np.ndarray) -> sapien.Pose:
    """Return a camera pose whose local +x axis points at ``target``."""
    forward = target - position
    forward /= np.linalg.norm(forward)
    left = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float32), forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.column_stack([forward, left, up])
    transform[:3, 3] = position
    return sapien.Pose(transform)


def add_camera(
    scene: sapien.Scene,
    position: list[float],
    target: list[float],
    width: int,
    height: int,
) -> sapien.render.RenderCameraComponent:
    """Add a free camera configured for tutorial asset rendering."""
    camera = scene.add_camera(
        "tutorial_camera",
        width,
        height,
        np.deg2rad(50.0),
        0.01,
        100.0,
    )
    camera.local_pose = look_at_pose(
        np.asarray(position, dtype=np.float32),
        np.asarray(target, dtype=np.float32),
    )
    return camera


def add_lighting(scene: sapien.Scene) -> None:
    """Add the standard tutorial fill and shadow-casting key light."""
    scene.set_ambient_light([0.12, 0.12, 0.12])
    scene.add_directional_light(
        [1.0, 1.0, -1.0],
        [2.0, 1.9, 1.8],
        shadow=True,
        shadow_map_size=4096,
    )


def picture_to_image(camera: sapien.render.RenderCameraComponent) -> Image.Image:
    """Convert the current floating-point Color target to an RGB image."""
    rgba = camera.get_picture("Color")
    rgb = (rgba[..., :3].clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def capture(scene: sapien.Scene, camera: sapien.render.RenderCameraComponent) -> Image.Image:
    """Submit CPU scene poses and capture one camera image."""
    scene.update_render()
    camera.take_picture()
    return picture_to_image(camera)


def save_gif(frames: list[Image.Image], path: Path, duration_ms: int = 50) -> None:
    """Save RGB frames as a compact looping GIF."""
    quantized = [frame.quantize(colors=192) for frame in frames]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def build_red_cube_scene() -> tuple[sapien.Scene, sapien.Entity]:
    """Create the scene shared by the hello-world and smoke-test images."""
    scene = sapien.Scene()
    scene.add_ground(0.0)

    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[0.5, 0.5, 0.5])
    builder.add_box_visual(
        half_size=[0.5, 0.5, 0.5],
        material=[1.0, 0.0, 0.0],
    )
    box = builder.build(name="box")
    box.pose = sapien.Pose([0.0, 0.0, 0.5])
    add_lighting(scene)
    return scene, box


def generate_basic_assets() -> None:
    """Regenerate every image referenced by the basic tutorials."""
    BASIC_ASSETS.mkdir(parents=True, exist_ok=True)

    scene, _ = build_red_cube_scene()
    camera = add_camera(scene, [-3.0, -2.2, 2.0], [0.0, 0.0, 0.5], 960, 540)
    capture(scene, camera).save(BASIC_ASSETS / "hello_world.png")

    scene, _ = build_red_cube_scene()
    mount = scene.create_actor_builder().build_kinematic(name="camera_mount")
    mount.pose = sapien.Pose([-3.0, 0.0, 0.5])
    camera = scene.add_mounted_camera(
        "offscreen_camera",
        mount,
        sapien.Pose(),
        128,
        128,
        1.0,
        0.01,
        10.0,
    )
    capture(scene, camera).save(BASIC_ASSETS / "example.offscreen.png")

    scene = sapien.Scene()
    scene.add_ground(0.0)
    add_lighting(scene)

    table_builder = scene.create_actor_builder()
    table_builder.add_box_collision(half_size=[0.6, 0.4, 0.05])
    table_builder.add_box_visual(
        half_size=[0.6, 0.4, 0.05], material=[0.48, 0.27, 0.12]
    )
    for x in (-0.45, 0.45):
        for y in (-0.3, 0.3):
            leg_pose = sapien.Pose([x, y, -0.35])
            table_builder.add_box_collision(
                pose=leg_pose, half_size=[0.05, 0.05, 0.35]
            )
            table_builder.add_box_visual(
                pose=leg_pose,
                half_size=[0.05, 0.05, 0.35],
                material=[0.48, 0.27, 0.12],
            )
    table = table_builder.build_static(name="table")
    table.pose = sapien.Pose([0.0, 0.0, 0.75])

    primitive_specs = (
        ("box", [-0.3, -0.05, 0.95], [0.75, 0.12, 0.08]),
        ("sphere", [0.0, 0.0, 0.98], [0.10, 0.35, 0.85]),
        ("capsule", [0.3, 0.05, 0.94], [0.15, 0.65, 0.20]),
    )
    for shape, position, color in primitive_specs:
        builder = scene.create_actor_builder()
        if shape == "box":
            builder.add_box_visual(half_size=[0.13, 0.13, 0.13], material=color)
        elif shape == "sphere":
            builder.add_sphere_visual(radius=0.16, material=color)
        else:
            builder.add_capsule_visual(radius=0.09, half_length=0.16, material=color)
        entity = builder.build_static(name=shape)
        entity.pose = sapien.Pose(position)

    camera = add_camera(scene, [-2.0, -2.5, 1.8], [0.0, 0.0, 0.75], 960, 540)
    capture(scene, camera).save(BASIC_ASSETS / "create_actors.png")

    scene = sapien.Scene()
    scene.add_ground(0.0)
    add_lighting(scene)
    builder = scene.create_articulation_builder()
    root = builder.create_link_builder()
    root.set_name("base")
    root.add_box_collision(half_size=[0.2, 0.15, 0.1])
    root.add_box_visual(half_size=[0.2, 0.15, 0.1], material=[0.75, 0.12, 0.08])
    arm = builder.create_link_builder(root)
    arm.set_name("arm")
    arm.set_joint_name("hinge")
    arm_pose = sapien.Pose([0.0, 0.4, 0.0])
    arm.add_box_collision(pose=arm_pose, half_size=[0.05, 0.4, 0.05])
    arm.add_box_visual(
        pose=arm_pose,
        half_size=[0.05, 0.4, 0.05],
        material=[0.10, 0.30, 0.85],
    )
    arm.set_joint_properties(
        "revolute",
        limits=[[-1.2, 1.2]],
        pose_in_parent=sapien.Pose([0.0, 0.15, 0.0]),
        pose_in_child=sapien.Pose(),
    )
    builder.set_initial_pose(sapien.Pose([0.0, 0.0, 1.0]))
    articulation = builder.build(fix_root_link=True)
    camera = add_camera(scene, [-2.2, -1.3, 1.7], [0.0, 0.35, 1.0], 640, 360)

    frames: list[Image.Image] = []
    for phase in np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False):
        articulation.qpos = np.array([0.95 * np.sin(phase)], dtype=np.float32)
        frames.append(capture(scene, camera))
    save_gif(frames, BASIC_ASSETS / "create_articulations.gif")

    scene = sapien.Scene()
    scene.set_timestep(1.0 / 240.0)
    scene.add_ground(0.0)
    add_lighting(scene)

    ramp_builder = scene.create_actor_builder()
    ramp_builder.add_box_collision(half_size=[1.2, 0.5, 0.05])
    ramp_builder.add_box_visual(
        half_size=[1.2, 0.5, 0.05], material=[0.32, 0.38, 0.46]
    )
    ramp = ramp_builder.build_static(name="ramp")
    angle = 0.24
    ramp.pose = sapien.Pose(
        [0.0, 0.0, 0.65],
        [np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0],
    )

    ball_builder = scene.create_actor_builder()
    ball_builder.add_sphere_collision(radius=0.16)
    ball_builder.add_sphere_visual(radius=0.16, material=[0.08, 0.28, 0.90])
    ball = ball_builder.build(name="ball")
    ball.pose = sapien.Pose([-0.82, 0.0, 1.05])

    camera = add_camera(scene, [-2.3, -2.8, 1.8], [0.1, 0.0, 0.55], 640, 360)
    frames = []
    for _ in range(60):
        for _ in range(4):
            scene.step()
        frames.append(capture(scene, camera))
    save_gif(frames, BASIC_ASSETS / "physics.gif")


def build_material_scene(
    shader_dir: str,
) -> tuple[sapien.Scene, sapien.render.RenderCameraComponent]:
    """Create the material scene used for raster/RT comparison images."""
    sapien.render.set_camera_shader_dir(shader_dir)
    if shader_dir == "rt":
        sapien.render.set_ray_tracing_samples_per_pixel(64)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")

    scene = sapien.Scene()
    floor_material = sapien.render.RenderMaterial(
        base_color=[0.28, 0.31, 0.36, 1.0], roughness=0.42
    )
    scene.add_ground(0.0, render_material=floor_material)
    scene.set_ambient_light([0.025, 0.025, 0.025])
    scene.add_directional_light(
        [1.0, 0.8, -1.3], [2.5, 2.35, 2.2], shadow=True, shadow_map_size=4096
    )
    scene.add_area_light_for_ray_tracing(
        pose=sapien.Pose([0.0, -0.5, 3.5]),
        color=[8.0, 7.5, 7.0],
        half_width=1.0,
        half_height=0.7,
    )

    materials = (
        sapien.render.RenderMaterial(
            base_color=[0.72, 0.08, 0.05, 1.0], roughness=0.32
        ),
        sapien.render.RenderMaterial(
            base_color=[0.95, 0.72, 0.25, 1.0], roughness=0.13, metallic=1.0
        ),
        sapien.render.RenderMaterial(
            base_color=[0.72, 0.90, 1.0, 0.28],
            roughness=0.03,
            transmission=0.92,
            ior=1.45,
        ),
    )
    positions = ([-0.72, 0.0, 0.38], [0.0, 0.0, 0.38], [0.72, 0.0, 0.38])
    for index, (material, position) in enumerate(zip(materials, positions)):
        builder = scene.create_actor_builder()
        if index == 0:
            builder.add_box_visual(half_size=[0.32, 0.32, 0.32], material=material)
        else:
            builder.add_sphere_visual(radius=0.36, material=material)
        entity = builder.build_static(name=f"material_{index}")
        entity.pose = sapien.Pose(position)

    camera = add_camera(scene, [-2.6, -3.2, 1.9], [0.0, 0.0, 0.45], 800, 450)
    return scene, camera


def generate_rendering_assets() -> None:
    """Regenerate the current raster/RT comparison and RT material image."""
    RENDERING_ASSETS.mkdir(parents=True, exist_ok=True)

    raster_scene, raster_camera = build_material_scene("default")
    raster = capture(raster_scene, raster_camera)

    rt_scene, rt_camera = build_material_scene("rt")
    ray_traced = capture(rt_scene, rt_camera)
    ray_traced.save(RENDERING_ASSETS / "mat_rt.png")

    label_height = 40
    comparison = Image.new(
        "RGB",
        (raster.width + ray_traced.width, raster.height + label_height),
        "white",
    )
    comparison.paste(raster, (0, label_height))
    comparison.paste(ray_traced, (raster.width, label_height))
    labels = ImageDraw.Draw(comparison)
    label_font = ImageFont.load_default(size=20)
    labels.text((12, 8), "Rasterization", fill="black", font=label_font)
    labels.text(
        (raster.width + 12, 8),
        "Ray tracing",
        fill="black",
        font=label_font,
    )
    comparison.save(RENDERING_ASSETS / "rst_vs_rt.png")

    sapien.render.set_camera_shader_dir("default")


def main() -> None:
    """Parse command-line arguments and regenerate the selected asset group."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "group",
        choices=("all", "basic", "rendering"),
        default="all",
        nargs="?",
        help="asset group to regenerate (default: all)",
    )
    args = parser.parse_args()

    sapien.render.set_log_level("warn")
    if args.group in ("all", "basic"):
        generate_basic_assets()
    if args.group in ("all", "rendering"):
        generate_rendering_assets()


if __name__ == "__main__":
    main()
