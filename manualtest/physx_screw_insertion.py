from __future__ import annotations

"""Headless benchmark for helical screw insertion contact stability.

This benchmark creates a static threaded nut and a driven threaded screw. The
screw is inserted along the x-axis while being rotated either with the matching
helical feed law or with an intentionally mismatched control mode.

Examples:
    python manualtest/physx_screw_insertion.py --system cpu --control-mode helix
    python manualtest/physx_screw_insertion.py --system cpu --control-mode push_only
    python manualtest/physx_screw_insertion.py --system cpu --articulation-position-iterations 32
"""

import argparse
import json
import math
from typing import Any, cast

import numpy as np
import sapien
from sapien.pysapien import Entity, Pose
from sapien.wrapper.scene import Scene

from _physx_precision_bench_utils import (
    apply_default_physx_config,
    change_frequency,
    detrended_rms,
    gpu_impulses_to_numpy,
    jsonable,
    make_physx_system,
    maybe_gpu_init,
    summarize_contacts_cpu,
    threaded_nut_mesh,
    threaded_rod_mesh,
    triangle_mesh_component,
    vector_norm_rms,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--gpu-device", default="cuda")
    parser.add_argument("--dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--control-mode", choices=["helix", "push_only", "reverse_helix"], default="helix")
    parser.add_argument("--unit-scale", type=float, default=1.0)
    parser.add_argument("--pitch", type=float, default=0.012)
    parser.add_argument("--feed-rate", type=float, default=0.012)
    parser.add_argument("--target-depth", type=float, default=0.022)
    parser.add_argument("--initial-gap", type=float, default=0.008)
    parser.add_argument("--screw-length", type=float, default=0.05)
    parser.add_argument("--screw-core-radius", type=float, default=0.0045)
    parser.add_argument("--thread-height", type=float, default=0.0016)
    parser.add_argument("--nut-length", type=float, default=0.03)
    parser.add_argument("--nut-outer-radius", type=float, default=0.0095)
    parser.add_argument("--radial-clearance", type=float, default=0.0005)
    parser.add_argument("--radial-segments", type=int, default=48)
    parser.add_argument("--axial-segments", type=int, default=96)
    parser.add_argument("--density", type=float, default=1100.0)
    parser.add_argument("--static-friction", type=float, default=1.1)
    parser.add_argument("--dynamic-friction", type=float, default=1.0)
    parser.add_argument("--restitution", type=float, default=0.0)
    parser.add_argument("--patch-radius", type=float, default=0.0)
    parser.add_argument("--min-patch-radius", type=float, default=0.0)
    parser.add_argument("--solver-position-iterations", type=int, default=10)
    parser.add_argument("--solver-velocity-iterations", type=int, default=1)
    parser.add_argument("--articulation-position-iterations", type=int, default=10)
    parser.add_argument("--articulation-velocity-iterations", type=int, default=1)
    parser.add_argument("--sleep-threshold", type=float, default=0.0)
    parser.add_argument("--contact-offset", type=float, default=0.01)
    parser.add_argument("--rest-offset", type=float, default=0.0)
    parser.add_argument("--sdf-spacing", type=float, default=0.01)
    parser.add_argument("--sdf-subgrid-size", type=int, default=6)
    parser.add_argument("--sdf-threads", type=int, default=4)
    parser.add_argument("--enable-ccd", action="store_true")
    parser.add_argument("--enable-enhanced-determinism", action="store_true")
    parser.add_argument("--joint-damping", type=float, default=1e4)
    parser.add_argument("--joint-force-limit", type=float, default=5e4)
    parser.add_argument("--friction-offset-threshold", type=float, default=0.04)
    parser.add_argument("--friction-correlation-distance", type=float, default=0.025)
    parser.add_argument("--cpu-workers", type=int, default=0)
    return parser


def create_screw_articulation(
    scene: Scene,
    material: sapien.physx.PhysxMaterial,
    *,
    vertices: np.ndarray,
    triangles: np.ndarray,
    patch_radius: float,
    min_patch_radius: float,
    density: float,
    initial_gap: float,
    screw_length: float,
    nut_length: float,
    solver_position_iterations: int,
    solver_velocity_iterations: int,
    sleep_threshold: float,
    joint_damping: float,
    joint_force_limit: float,
) -> tuple[
    sapien.physx.PhysxArticulation,
    sapien.physx.PhysxArticulationLinkComponent,
    sapien.physx.PhysxArticulationJoint,
    sapien.physx.PhysxArticulationJoint,
]:
    root = sapien.physx.PhysxArticulationLinkComponent()
    root.joint.type = "fixed"
    root.disable_gravity = True
    root.mass = 1.0
    root.inertia = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    root_entity = Entity()
    root_entity.name = "screw_root"
    root_x = -0.5 * (screw_length + nut_length) - initial_gap
    root_entity.pose = Pose([root_x, 0.0, 0.0])
    root_entity.add_component(root)

    feed = sapien.physx.PhysxArticulationLinkComponent(root)
    feed.disable_gravity = True
    feed.mass = 0.05
    feed.inertia = np.asarray([1e-4, 1e-4, 1e-4], dtype=np.float32)
    feed_entity = Entity()
    feed_entity.name = "feed"
    feed_entity.add_component(feed)

    screw = sapien.physx.PhysxArticulationLinkComponent(feed)
    screw.disable_gravity = True
    screw_entity = Entity()
    screw_entity.name = "screw"
    screw_entity.add_component(screw)

    feed_joint = feed.joint
    feed_joint.name = "feed"
    feed_joint.type = "prismatic"
    feed_joint.limit = cast(
        np.ndarray,
        np.asarray([[0.0, 1.0]], dtype=np.float32).reshape(1, 2),
    )
    feed_joint.set_drive_properties(0.0, joint_damping, joint_force_limit, "force")
    feed_joint.set_drive_velocity_target(0.0)

    spin_joint = screw.joint
    spin_joint.name = "spin"
    spin_joint.type = "revolute_unwrapped"
    spin_joint.limit = cast(
        np.ndarray,
        np.asarray([[-np.inf, np.inf]], dtype=np.float32).reshape(1, 2),
    )
    spin_joint.set_drive_properties(0.0, joint_damping, joint_force_limit, "force")
    spin_joint.set_drive_velocity_target(0.0)

    screw_shape = sapien.physx.PhysxCollisionShapeTriangleMesh(
        vertices.astype(np.float32),
        triangles.astype(np.uint32),
        [1.0, 1.0, 1.0],
        material,
        sdf=True,
    )
    screw_shape.patch_radius = patch_radius
    screw_shape.min_patch_radius = min_patch_radius
    screw_shape.density = density
    screw.attach(screw_shape)

    for entity in (root_entity, feed_entity, screw_entity):
        scene.add_entity(entity)

    articulation = root.articulation
    articulation.solver_position_iterations = solver_position_iterations
    articulation.solver_velocity_iterations = solver_velocity_iterations
    articulation.sleep_threshold = sleep_threshold
    return articulation, screw, feed_joint, spin_joint


def classify_result(final_depth: float, target_depth: float, impulse_std: float) -> tuple[bool, str]:
    if final_depth >= target_depth * 0.95:
        return True, "success"
    if final_depth < target_depth * 0.25:
        return False, "jammed"
    if impulse_std > 5.0:
        return False, "unstable_contact"
    return False, "partial_insertion"


def main() -> None:
    args = build_parser().parse_args()

    scale = args.unit_scale
    pitch = args.pitch * scale
    target_depth = args.target_depth * scale
    initial_gap = args.initial_gap * scale
    screw_length = args.screw_length * scale
    nut_length = args.nut_length * scale
    screw_core_radius = args.screw_core_radius * scale
    thread_height = args.thread_height * scale
    nut_outer_radius = args.nut_outer_radius * scale
    radial_clearance = args.radial_clearance * scale

    material = sapien.physx.PhysxMaterial(
        args.static_friction,
        args.dynamic_friction,
        args.restitution,
    )
    config = apply_default_physx_config(
        gravity=(0.0, 0.0, 0.0),
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

    screw_vertices, screw_triangles, screw_mesh = threaded_rod_mesh(
        length=screw_length,
        core_radius=screw_core_radius,
        thread_height=thread_height,
        pitch=pitch,
        radial_segments=args.radial_segments,
        axial_segments=args.axial_segments,
    )
    nut_vertices, nut_triangles, nut_mesh = threaded_nut_mesh(
        length=nut_length,
        outer_radius=nut_outer_radius,
        hole_radius=screw_core_radius + radial_clearance,
        thread_height=thread_height,
        pitch=pitch,
        radial_segments=args.radial_segments,
        axial_segments=max(args.axial_segments // 2, 16),
    )

    nut_entity, nut_component, nut_shape = triangle_mesh_component(
        nut_vertices,
        nut_triangles,
        material,
        sdf=False,
        dynamic=False,
        density=args.density,
        patch_radius=args.patch_radius,
        min_patch_radius=args.min_patch_radius,
    )
    nut_entity.name = "nut"
    scene.add_entity(nut_entity)

    articulation, screw_link, feed_joint, spin_joint = create_screw_articulation(
        scene,
        material,
        vertices=screw_vertices,
        triangles=screw_triangles,
        patch_radius=args.patch_radius,
        min_patch_radius=args.min_patch_radius,
        density=args.density,
        initial_gap=initial_gap,
        screw_length=screw_length,
        nut_length=nut_length,
        solver_position_iterations=args.articulation_position_iterations,
        solver_velocity_iterations=args.articulation_velocity_iterations,
        sleep_threshold=args.sleep_threshold,
        joint_damping=args.joint_damping,
        joint_force_limit=args.joint_force_limit,
    )

    gpu_query = None
    if args.system == "gpu":
        maybe_gpu_init(system)
        gpu_query = system.gpu_create_contact_pair_impulse_query([(screw_link, nut_component)])

    commanded_feed = 0.0
    qpos_history: list[np.ndarray] = []
    qvel_history: list[np.ndarray] = []
    screw_positions: list[np.ndarray] = []
    impulse_sums: list[float] = []
    contact_counts: list[float] = []
    point_counts: list[float] = []
    min_separations: list[float] = []
    max_separations: list[float] = []

    for _ in range(args.steps):
        feed_velocity = args.feed_rate * scale if commanded_feed < target_depth else 0.0
        spin_velocity = 0.0
        if args.control_mode == "helix":
            spin_velocity = 2.0 * math.pi * feed_velocity / pitch
        elif args.control_mode == "reverse_helix":
            spin_velocity = -2.0 * math.pi * feed_velocity / pitch

        feed_joint.set_drive_velocity_target(feed_velocity)
        spin_joint.set_drive_velocity_target(spin_velocity)
        scene.step()

        commanded_feed = min(commanded_feed + feed_velocity * args.dt, target_depth)
        qpos = np.asarray(articulation.qpos, dtype=np.float64)
        qvel = np.asarray(articulation.qvel, dtype=np.float64)
        qpos_history.append(qpos)
        qvel_history.append(qvel)
        screw_positions.append(np.asarray(screw_link.entity.pose.p, dtype=np.float64))

        if args.system == "cpu":
            stats = summarize_contacts_cpu(system.get_contacts(), [screw_link, nut_component])
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

    qpos_array = np.asarray(qpos_history, dtype=np.float64)
    qvel_array = np.asarray(qvel_history, dtype=np.float64)
    screw_positions_array = np.asarray(screw_positions, dtype=np.float64)

    final_depth = float(qpos_array[-1, 0]) if qpos_array.size else 0.0
    success, failure_mode = classify_result(
        final_depth,
        target_depth,
        float(np.std(impulse_sums)) if impulse_sums else 0.0,
    )

    result: dict[str, Any] = {
        "benchmark": "physx_screw_insertion",
        "system": args.system,
        "control_mode": args.control_mode,
        "timestep": args.dt,
        "steps": args.steps,
        "physx_defaults": config,
        "shape_overrides": {
            "patch_radius": args.patch_radius,
            "min_patch_radius": args.min_patch_radius,
            "enable_ccd": args.enable_ccd,
        },
        "asset": {
            "unit_scale": scale,
            "thread_pitch_world": pitch,
            "target_depth_world": target_depth,
            "radial_clearance_world": radial_clearance,
            "screw": screw_mesh,
            "nut": nut_mesh,
        },
        "metrics": {
            "success": success,
            "failure_mode": failure_mode,
            "final_depth": final_depth,
            "target_depth": target_depth,
            "depth_ratio": final_depth / target_depth if target_depth > 0 else 0.0,
            "feed_tracking_rms": detrended_rms(qpos_array[:, 0] - np.linspace(0.0, commanded_feed, len(qpos_array))),
            "spin_velocity_rms": vector_norm_rms(qvel_array[:, 1:2]) if qvel_array.shape[1] > 1 else 0.0,
            "feed_velocity_rms": vector_norm_rms(qvel_array[:, 0:1]) if qvel_array.shape[1] > 0 else 0.0,
            "screw_radial_jitter_rms": vector_norm_rms(screw_positions_array[:, 1:3]),
            "screw_axial_detrended_rms": detrended_rms(screw_positions_array[:, 0]),
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
