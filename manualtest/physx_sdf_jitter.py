from __future__ import annotations

"""Headless benchmark for dynamic non-convex SDF jitter.

This script builds a dynamic torus triangle mesh with SDF collision against a
static plane. It is meant to quantify contact stability under different PhysX
defaults rather than provide a visual demo.

Examples:
    python manualtest/physx_sdf_jitter.py --system cpu --measure-steps 600
    python manualtest/physx_sdf_jitter.py --system cpu --dt 0.002 --solver-position-iterations 32
    python manualtest/physx_sdf_jitter.py --system cpu --contact-offset 0.002 --sdf-spacing 0.002
"""

import argparse
import json
from typing import Any

import numpy as np
import sapien
from sapien.pysapien import Pose
from sapien.wrapper.scene import Scene

from _physx_precision_bench_utils import (
    apply_default_physx_config,
    change_frequency,
    detrended_rms,
    gpu_impulses_to_numpy,
    jsonable,
    make_physx_system,
    make_plane,
    maybe_gpu_init,
    rms,
    summarize_contacts_cpu,
    torus_mesh,
    triangle_mesh_component,
    vector_norm_rms,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--gpu-device", default="cuda")
    parser.add_argument("--dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--warmup-steps", type=int, default=240)
    parser.add_argument("--measure-steps", type=int, default=480)
    parser.add_argument("--drop-height", type=float, default=0.14)
    parser.add_argument("--tilt-deg", type=float, default=52.0)
    parser.add_argument("--geometry-scale", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=850.0)
    parser.add_argument("--static-friction", type=float, default=1.0)
    parser.add_argument("--dynamic-friction", type=float, default=0.9)
    parser.add_argument("--restitution", type=float, default=0.0)
    parser.add_argument("--patch-radius", type=float, default=0.0)
    parser.add_argument("--min-patch-radius", type=float, default=0.0)
    parser.add_argument("--solver-position-iterations", type=int, default=10)
    parser.add_argument("--solver-velocity-iterations", type=int, default=1)
    parser.add_argument("--sleep-threshold", type=float, default=0.0)
    parser.add_argument("--contact-offset", type=float, default=0.01)
    parser.add_argument("--rest-offset", type=float, default=0.0)
    parser.add_argument("--sdf-spacing", type=float, default=0.01)
    parser.add_argument("--sdf-subgrid-size", type=int, default=6)
    parser.add_argument("--sdf-threads", type=int, default=4)
    parser.add_argument("--enable-ccd", action="store_true")
    parser.add_argument("--enable-enhanced-determinism", action="store_true")
    parser.add_argument("--disable-gyroscopic-forces", action="store_true")
    parser.add_argument("--friction-offset-threshold", type=float, default=0.04)
    parser.add_argument("--friction-correlation-distance", type=float, default=0.025)
    parser.add_argument("--cpu-workers", type=int, default=0)
    parser.add_argument("--major-radius", type=float, default=0.045)
    parser.add_argument("--minor-radius", type=float, default=0.018)
    parser.add_argument("--major-segments", type=int, default=48)
    parser.add_argument("--minor-segments", type=int, default=24)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    material = sapien.physx.PhysxMaterial(
        args.static_friction,
        args.dynamic_friction,
        args.restitution,
    )
    config = apply_default_physx_config(
        gravity=(0.0, 0.0, -9.81),
        bounce_threshold=2.0,
        enable_pcm=True,
        enable_tgs=True,
        enable_ccd=args.enable_ccd,
        enable_enhanced_determinism=args.enable_enhanced_determinism,
        enable_friction_every_iteration=True,
        friction_offset_threshold=args.friction_offset_threshold,
        friction_correlation_distance=args.friction_correlation_distance,
        cpu_workers=args.cpu_workers,
        solver_position_iterations=args.solver_position_iterations,
        solver_velocity_iterations=args.solver_velocity_iterations,
        sleep_threshold=args.sleep_threshold,
        contact_offset=args.contact_offset,
        rest_offset=args.rest_offset,
        sdf_spacing=args.sdf_spacing,
        sdf_subgrid_size=args.sdf_subgrid_size,
        sdf_threads=args.sdf_threads,
    )

    system = make_physx_system(args.system, args.gpu_device)
    system.timestep = args.dt
    scene = Scene([system])

    _, ground_component = make_plane(scene, material)

    vertices, triangles, mesh_metadata = torus_mesh(
        major_radius=args.major_radius * args.geometry_scale,
        minor_radius=args.minor_radius * args.geometry_scale,
        major_segments=args.major_segments,
        minor_segments=args.minor_segments,
    )
    entity, body, shape = triangle_mesh_component(
        vertices,
        triangles,
        material,
        sdf=True,
        dynamic=True,
        density=args.density,
        patch_radius=args.patch_radius,
        min_patch_radius=args.min_patch_radius,
    )
    body.solver_position_iterations = args.solver_position_iterations
    body.solver_velocity_iterations = args.solver_velocity_iterations
    body.sleep_threshold = args.sleep_threshold
    body.gyroscopic_forces = not args.disable_gyroscopic_forces
    entity.name = "torus"
    tilt_rad = np.deg2rad(args.tilt_deg)
    entity.pose = Pose(
        p=[0.0, 0.0, args.drop_height],
        q=[
            float(np.cos(tilt_rad / 2.0)),
            0.0,
            float(np.sin(tilt_rad / 2.0)),
            0.0,
        ],
    )
    scene.add_entity(entity)

    gpu_query = None
    if args.system == "gpu":
        maybe_gpu_init(system)
        gpu_query = system.gpu_create_contact_pair_impulse_query([(body, ground_component)])

    positions: list[np.ndarray] = []
    linear_velocities: list[np.ndarray] = []
    angular_velocities: list[np.ndarray] = []
    contact_counts: list[float] = []
    point_counts: list[float] = []
    impulse_sums: list[float] = []
    min_separations: list[float] = []
    max_separations: list[float] = []
    sleeping: list[float] = []

    total_steps = args.warmup_steps + args.measure_steps
    for step in range(total_steps):
        scene.step()
        if step < args.warmup_steps:
            continue

        positions.append(np.asarray(entity.pose.p, dtype=np.float64))
        linear_velocities.append(np.asarray(body.linear_velocity, dtype=np.float64))
        angular_velocities.append(np.asarray(body.angular_velocity, dtype=np.float64))
        sleeping.append(1.0 if body.is_sleeping else 0.0)

        if args.system == "cpu":
            stats = summarize_contacts_cpu(system.get_contacts(), [body, ground_component])
            contact_counts.append(stats["contact_count"] or 0.0)
            point_counts.append(stats["point_count"] or 0.0)
            impulse_sums.append(stats["impulse_norm_sum"] or 0.0)
            if stats["min_separation"] is not None:
                min_separations.append(float(stats["min_separation"]))
            if stats["max_separation"] is not None:
                max_separations.append(float(stats["max_separation"]))
        else:
            assert gpu_query is not None
            system.gpu_query_contact_pair_impulses(gpu_query)
            impulse = gpu_impulses_to_numpy(gpu_query)[0]
            contact_counts.append(float(np.linalg.norm(impulse) > 0.0))
            point_counts.append(0.0)
            impulse_sums.append(float(np.linalg.norm(impulse)))

    positions_array = np.asarray(positions, dtype=np.float64)
    position_centered = positions_array - np.mean(positions_array, axis=0, keepdims=True)
    result: dict[str, Any] = {
        "benchmark": "physx_sdf_jitter",
        "system": args.system,
        "timestep": args.dt,
        "steps": {
            "warmup": args.warmup_steps,
            "measure": args.measure_steps,
        },
        "physx_defaults": config,
        "shape_overrides": {
            "patch_radius": args.patch_radius,
            "min_patch_radius": args.min_patch_radius,
            "gyroscopic_forces": body.gyroscopic_forces,
        },
        "asset": {
            **mesh_metadata,
            "unit_scale": args.geometry_scale,
            "world_major_radius": args.major_radius * args.geometry_scale,
            "world_minor_radius": args.minor_radius * args.geometry_scale,
            "world_units": "meters",
        },
        "metrics": {
            "position_rms": vector_norm_rms(position_centered),
            "position_rms_xyz": [
                rms(position_centered[:, 0]),
                rms(position_centered[:, 1]),
                rms(position_centered[:, 2]),
            ],
            "height_detrended_rms": detrended_rms(positions_array[:, 2]),
            "linear_velocity_rms": vector_norm_rms(np.asarray(linear_velocities, dtype=np.float64)),
            "angular_velocity_rms": vector_norm_rms(np.asarray(angular_velocities, dtype=np.float64)),
            "sleep_fraction": float(np.mean(sleeping)) if sleeping else 0.0,
            "contact_count_change_frequency_hz": change_frequency(contact_counts, args.dt),
            "contact_count_mean": float(np.mean(contact_counts)) if contact_counts else 0.0,
            "contact_point_mean": float(np.mean(point_counts)) if point_counts else 0.0,
            "contact_impulse_mean": float(np.mean(impulse_sums)) if impulse_sums else 0.0,
            "contact_impulse_std": float(np.std(impulse_sums)) if impulse_sums else 0.0,
            "min_separation": float(min(min_separations)) if min_separations else None,
            "max_separation": float(max(max_separations)) if max_separations else None,
        },
        "limitations": {
            "gpu_contact_points_available": args.system == "cpu",
            "gpu_separation_available": False,
            "speculative_ccd_exposed": False,
        },
    }
    print(json.dumps(jsonable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
