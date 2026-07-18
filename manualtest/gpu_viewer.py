"""Benchmark legacy and direct PhysX GPU render pose transports.

Run on a display-capable CUDA/Vulkan machine, for example:

    python manualtest/gpu_viewer.py --transport render-system-group --selection default
    python manualtest/gpu_viewer.py --transport direct --selection explicit --scenes 16
    python manualtest/gpu_viewer.py --transport cpu-debug --shader rt --articulations
    python manualtest/gpu_viewer.py --transport cpu-debug --render-device pci:0000:00:02.0

Use Nsight Systems to inspect the NVTX ranges emitted by the PhysX fetch,
conversion, transform, synchronization, RT update, Viewer update, and draw paths.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Literal

import numpy as np
import sapien
from sapien.utils import Viewer

Shader = Literal["default", "rt"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        choices=("direct", "cpu-debug", "render-system-group"),
        default="direct",
    )
    parser.add_argument("--shader", choices=("default", "rt"), default="default")
    parser.add_argument("--scenes", type=int, default=16)
    parser.add_argument("--selection", choices=("default", "explicit"), default="explicit")
    parser.add_argument(
        "--render-device",
        default="cuda",
        help="Vulkan device alias, for example cuda or pci:0000:00:02.0",
    )
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--articulations", action="store_true")
    return parser.parse_args()


def _build_benchmark(
    scene_count: int,
    shader: Shader,
    articulations: bool,
    render_device_alias: str,
) -> tuple[
    sapien.physx.PhysxGpuSystem,
    list[sapien.Scene],
    sapien.Scene,
    list[sapien.render.RenderCameraComponent],
    sapien.Entity,
]:
    sapien.physx.enable_gpu()
    sapien.render.set_viewer_shader_dir(shader)
    sapien.render.set_camera_shader_dir(shader)
    if shader == "rt":
        sapien.render.set_ray_tracing_samples_per_pixel(2)
        sapien.render.set_ray_tracing_path_depth(2)
        sapien.render.set_ray_tracing_denoiser("none")

    compute_device = sapien.Device("cuda")
    render_device = sapien.Device(render_device_alias)
    physx = sapien.physx.PhysxGpuSystem(compute_device)
    scenes: list[sapien.Scene] = []
    for environment_id in range(scene_count):
        scene = sapien.Scene([physx, sapien.render.RenderSystem(render_device)])
        scene.set_environment_id(environment_id)
        scene.set_ambient_light([0.25, 0.25, 0.25])
        scenes.append(scene)

    shared_scene = sapien.Scene([physx, sapien.render.RenderSystem(render_device)])
    shared_scene.set_environment_id(-1)
    shared_scene.add_ground(0.0, render_half_size=[8.0, 8.0])
    shared_scene.add_directional_light(
        [1.0, 1.0, -1.0], [1.5, 1.5, 1.5], shadow=True
    )

    first_dynamic_entity: sapien.Entity | None = None
    if articulations:
        urdf_path = Path(__file__).resolve().parents[1] / "assets/robot/panda/panda.urdf"
        loader = scenes[0].create_urdf_loader()
        articulation_builder = loader.load_file_as_articulation_builder(str(urdf_path))
        for scene in scenes:
            articulation_builder.set_scene(scene)
            articulation_builder.initial_pose = sapien.Pose([0.0, 0.0, 0.5])
            articulation = articulation_builder.build()
            if first_dynamic_entity is None:
                first_dynamic_entity = articulation.links[0].entity
    else:
        builder = scenes[0].create_actor_builder()
        builder.add_sphere_collision(radius=0.15)
        builder.add_sphere_visual(radius=0.15, material=[0.8, 0.1, 0.1])
        for scene in scenes:
            builder.set_scene(scene)
            builder.initial_pose = sapien.Pose([0.0, 0.0, 2.0])
            entity = builder.build()
            if first_dynamic_entity is None:
                first_dynamic_entity = entity

    cameras = []
    for index, scene in enumerate(scenes):
        camera = scene.add_camera(
            f"benchmark_camera_{index}", 320, 240, 0.8, 0.05, 100.0
        )
        camera.pose = sapien.Pose([-4.0, 0.0, 2.0])
        cameras.append(camera)

    assert first_dynamic_entity is not None
    physx.gpu_init()
    return physx, scenes, shared_scene, cameras, first_dynamic_entity


def _mean_ms(samples: list[float]) -> float:
    return statistics.fmean(samples) * 1000.0 if samples else 0.0


def main() -> None:
    args = _parse_args()
    if args.transport == "render-system-group" and args.articulations:
        raise ValueError(
            "the legacy RenderSystemGroup baseline only accepts the rigid-body pose buffer"
        )

    physx, scenes, shared_scene, cameras, tracked_entity = _build_benchmark(
        args.scenes,
        args.shader,
        args.articulations,
        args.render_device,
    )
    initial_cpu_pose = tracked_entity.pose

    viewer: Viewer | None = None
    render_group: sapien.render.RenderSystemGroup | None = None
    camera_group: sapien.render.RenderCameraGroup | None = None
    selected_cameras = [cameras[0]]
    if args.selection == "explicit":
        cameras[0].set_scenes(scenes)

    if args.transport == "render-system-group":
        render_group = sapien.render.RenderSystemGroup(
            [scene.render_system for scene in scenes] + [shared_scene.render_system]
        )
        camera_group = render_group.create_camera_group(selected_cameras, ["Color"])
        render_group.set_cuda_poses(physx.cuda_rigid_body_data)
        physx.gpu_fetch_rigid_dynamic_data()
        physx.gpu_fetch_articulation_link_pose()
        render_group.update_render()
        camera_group.take_picture()
    else:
        viewer = Viewer(resolutions=(1280, 720))
        viewer.configure_physx_gpu_rendering(physx, args.transport)
        if args.selection == "explicit":
            viewer.set_scenes(scenes)
        else:
            viewer.set_scene(scenes[0])
        viewer.set_camera_xyz(-4.0, 0.0, 2.0)
        viewer.update_render()
        viewer.render()

    initial_sync_count = physx._sync_poses_gpu_to_cpu_count
    initial_rigid_fetch_count = physx._gpu_fetch_rigid_dynamic_data_count
    initial_link_fetch_count = physx._gpu_fetch_articulation_link_pose_count

    if viewer is not None:
        viewer.render()
    else:
        assert camera_group is not None
        camera_group.take_picture()
    assert physx._sync_poses_gpu_to_cpu_count == initial_sync_count
    assert physx._gpu_fetch_rigid_dynamic_data_count == initial_rigid_fetch_count
    assert physx._gpu_fetch_articulation_link_pose_count == initial_link_fetch_count

    fetch_times: list[float] = []
    update_times: list[float] = []
    draw_times: list[float] = []
    completed_frames = 0
    for _ in range(args.frames):
        if viewer is not None and viewer.closed:
            break
        for _ in range(args.substeps):
            physx.step()

        if args.transport in ("direct", "render-system-group"):
            start = time.perf_counter()
            physx.gpu_fetch_rigid_dynamic_data()
            physx.gpu_fetch_articulation_link_pose()
            fetch_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        if render_group is not None:
            render_group.update_render()
        else:
            assert viewer is not None
            viewer.update_render()
        update_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        if camera_group is not None:
            camera_group.take_picture()
        else:
            assert viewer is not None
            viewer.render()
        draw_times.append(time.perf_counter() - start)
        completed_frames += 1

    sync_calls = physx._sync_poses_gpu_to_cpu_count - initial_sync_count
    rigid_fetches = (
        physx._gpu_fetch_rigid_dynamic_data_count - initial_rigid_fetch_count
    )
    link_fetches = (
        physx._gpu_fetch_articulation_link_pose_count - initial_link_fetch_count
    )
    # cuda_rigid_body_data is the complete unified buffer; the articulation-link array is a view
    # into the same storage and must not be counted a second time.
    pose_buffer = physx.cuda_rigid_body_data
    full_pose_bytes = np.prod(pose_buffer.shape) * 4
    d2h_bytes = sync_calls * int(full_pose_bytes)
    active_transport = (
        viewer.pose_transport if viewer is not None else "render-system-group-direct"
    )

    print(f"transport: {active_transport}")
    print(f"shader: {args.shader}")
    print(f"render device: {args.render_device}")
    print(f"selection: {args.selection}")
    print(f"scenes: {args.scenes}")
    print(f"frames: {completed_frames}")
    print(f"PhysX fetch enqueue mean: {_mean_ms(fetch_times):.3f} ms")
    print(f"render update mean: {_mean_ms(update_times):.3f} ms")
    print(f"draw/capture mean: {_mean_ms(draw_times):.3f} ms")
    print(f"rigid dynamic fetches: {rigid_fetches}")
    print(f"articulation link pose fetches: {link_fetches}")
    print(f"sync_poses_gpu_to_cpu calls: {sync_calls}")
    print(f"full-pose D2H bytes: {d2h_bytes}")
    print("SAPIEN cudaDeviceSynchronize calls: 0")
    print("Detailed GPU stages: capture the emitted NVTX ranges with Nsight Systems.")

    if args.transport in ("direct", "render-system-group"):
        assert sync_calls == 0, "direct path performed a full CPU pose synchronization"
        assert rigid_fetches == completed_frames, "render path duplicated rigid-body fetches"
        assert link_fetches == completed_frames, "render path duplicated articulation fetches"
        assert np.allclose(tracked_entity.pose.p, initial_cpu_pose.p), (
            "direct path unexpectedly updated Entity.pose"
        )


if __name__ == "__main__":
    main()
