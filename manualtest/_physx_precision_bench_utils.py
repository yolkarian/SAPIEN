from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import sapien
from sapien.pysapien import Entity, Pose
from sapien.wrapper.scene import Scene


def make_physx_system(system: str, device: str) -> sapien.physx.PhysxSystem:
    if system == "gpu":
        sapien.physx.enable_gpu()
        return sapien.physx.PhysxGpuSystem(device)
    if system == "cpu":
        return sapien.physx.PhysxCpuSystem()
    raise ValueError(f"unsupported system: {system}")


def apply_default_physx_config(
    *,
    gravity: Sequence[float],
    bounce_threshold: float,
    enable_pcm: bool,
    enable_tgs: bool,
    enable_ccd: bool,
    enable_enhanced_determinism: bool,
    enable_friction_every_iteration: bool,
    friction_offset_threshold: float,
    friction_correlation_distance: float,
    cpu_workers: int,
    solver_position_iterations: int,
    solver_velocity_iterations: int,
    sleep_threshold: float,
    contact_offset: float,
    rest_offset: float,
    sdf_spacing: float,
    sdf_subgrid_size: int,
    sdf_threads: int,
) -> dict[str, Any]:
    sapien.physx.set_scene_config(
        gravity=np.asarray(gravity, dtype=np.float32),
        bounce_threshold=bounce_threshold,
        enable_pcm=enable_pcm,
        enable_tgs=enable_tgs,
        enable_ccd=enable_ccd,
        enable_enhanced_determinism=enable_enhanced_determinism,
        enable_friction_every_iteration=enable_friction_every_iteration,
        friction_offset_threshold=friction_offset_threshold,
        friction_correlation_distance=friction_correlation_distance,
        cpu_workers=cpu_workers,
    )
    sapien.physx.set_body_config(
        solver_position_iterations=solver_position_iterations,
        solver_velocity_iterations=solver_velocity_iterations,
        sleep_threshold=sleep_threshold,
    )
    sapien.physx.set_shape_config(
        contact_offset=contact_offset,
        rest_offset=rest_offset,
    )
    sapien.physx.set_sdf_config(
        spacing=sdf_spacing,
        subgrid_size=sdf_subgrid_size,
        num_threads_for_construction=sdf_threads,
    )
    return get_default_physx_config_dict()


def get_default_physx_config_dict() -> dict[str, Any]:
    scene = sapien.physx.get_scene_config()
    shape = sapien.physx.get_shape_config()
    body = sapien.physx.get_body_config()
    sdf = sapien.physx.get_sdf_config()
    return {
        "scene": {
            "gravity": [float(v) for v in scene.gravity],
            "bounce_threshold": float(scene.bounce_threshold),
            "enable_pcm": bool(scene.enable_pcm),
            "enable_tgs": bool(scene.enable_tgs),
            "enable_ccd": bool(scene.enable_ccd),
            "enable_enhanced_determinism": bool(scene.enable_enhanced_determinism),
            "enable_friction_every_iteration": bool(scene.enable_friction_every_iteration),
            "friction_offset_threshold": float(scene.friction_offset_threshold),
            "friction_correlation_distance": float(scene.friction_correlation_distance),
            "cpu_workers": int(scene.cpu_workers),
        },
        "body": {
            "sleep_threshold": float(body.sleep_threshold),
            "solver_position_iterations": int(body.solver_position_iterations),
            "solver_velocity_iterations": int(body.solver_velocity_iterations),
        },
        "shape": {
            "contact_offset": float(shape.contact_offset),
            "rest_offset": float(shape.rest_offset),
        },
        "sdf": {
            "spacing": float(sdf.spacing),
            "subgrid_size": int(getattr(sdf, "subgrid_size", sdf.subgridSize)),
            "num_threads_for_construction": int(sdf.num_threads_for_construction),
        },
    }


def maybe_gpu_init(system: sapien.physx.PhysxSystem) -> None:
    if hasattr(system, "gpu_init"):
        system.gpu_init()


def make_plane(
    scene: Scene,
    material: sapien.physx.PhysxMaterial,
    altitude: float = 0.0,
    name: str = "ground",
) -> tuple[Entity, sapien.physx.PhysxRigidStaticComponent]:
    component = sapien.physx.PhysxRigidStaticComponent()
    shape = sapien.physx.PhysxCollisionShapePlane(material)
    shape.local_pose = Pose(
        p=[0.0, 0.0, altitude],
        q=[0.7071068, 0.0, -0.7071068, 0.0],
    )
    component.attach(shape)
    entity = Entity()
    entity.name = name
    entity.add_component(component)
    scene.add_entity(entity)
    return entity, component


def triangle_mesh_component(
    vertices: np.ndarray,
    triangles: np.ndarray,
    material: sapien.physx.PhysxMaterial,
    *,
    sdf: bool,
    dynamic: bool,
    density: float,
    patch_radius: float,
    min_patch_radius: float,
) -> tuple[Entity, sapien.physx.PhysxRigidBaseComponent, sapien.physx.PhysxCollisionShapeTriangleMesh]:
    shape = sapien.physx.PhysxCollisionShapeTriangleMesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(triangles, dtype=np.uint32),
        [1.0, 1.0, 1.0],
        material,
        sdf=sdf,
    )
    shape.patch_radius = patch_radius
    shape.min_patch_radius = min_patch_radius
    shape.density = density

    if dynamic:
        component = sapien.physx.PhysxRigidDynamicComponent()
    else:
        component = sapien.physx.PhysxRigidStaticComponent()
    component.attach(shape)
    entity = Entity()
    entity.add_component(component)
    return entity, component, shape


def matches_bodies(contact: sapien.physx.PhysxContact, bodies: Sequence[object]) -> bool:
    if len(bodies) == 1:
        body = bodies[0]
        return any(candidate is body for candidate in contact.bodies)
    if len(bodies) == 2:
        a, b = bodies
        return (contact.bodies[0] is a and contact.bodies[1] is b) or (
            contact.bodies[0] is b and contact.bodies[1] is a
        )
    raise ValueError("bodies must contain one or two tracked bodies")


def summarize_contacts_cpu(
    contacts: Sequence[sapien.physx.PhysxContact],
    tracked_bodies: Sequence[object],
) -> dict[str, Optional[float]]:
    contact_count = 0
    point_count = 0
    total_impulse = 0.0
    min_separation = math.inf
    max_separation = -math.inf

    for contact in contacts:
        if not matches_bodies(contact, tracked_bodies):
            continue
        contact_count += 1
        point_count += len(contact.points)
        for point in contact.points:
            total_impulse += float(np.linalg.norm(point.impulse))
            min_separation = min(min_separation, float(point.separation))
            max_separation = max(max_separation, float(point.separation))

    if point_count == 0:
        min_separation_value = None
        max_separation_value = None
    else:
        min_separation_value = min_separation
        max_separation_value = max_separation

    return {
        "contact_count": float(contact_count),
        "point_count": float(point_count),
        "impulse_norm_sum": total_impulse,
        "min_separation": min_separation_value,
        "max_separation": max_separation_value,
    }


def gpu_impulses_to_numpy(query: Any) -> np.ndarray:
    import torch

    return query.cuda_impulses.torch().cpu().numpy()


def rms(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(array))))


def vector_norm_rms(values: Sequence[Sequence[float]] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    norms = np.linalg.norm(array, axis=-1)
    return rms(norms)


def detrended_rms(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return 0.0
    x = np.arange(array.shape[0], dtype=np.float64)
    coeffs = np.polyfit(x, array, deg=1)
    trend = coeffs[0] * x + coeffs[1]
    return rms(array - trend)


def change_frequency(values: Sequence[float] | np.ndarray, timestep: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return 0.0
    changes = np.count_nonzero(array[1:] != array[:-1])
    return float(changes / ((array.size - 1) * timestep))


def is_watertight(triangles: np.ndarray) -> bool:
    tri = np.asarray(triangles, dtype=np.int64)
    if tri.ndim != 2 or tri.shape[1] != 3 or tri.shape[0] == 0:
        return False
    edges = np.concatenate(
        [tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def torus_mesh(
    *,
    major_radius: float,
    minor_radius: float,
    major_segments: int,
    minor_segments: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    theta = np.linspace(0.0, 2.0 * math.pi, major_segments, endpoint=False)
    phi = np.linspace(0.0, 2.0 * math.pi, minor_segments, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    radius = major_radius + minor_radius * np.cos(phi_grid)
    x = radius * np.cos(theta_grid)
    y = radius * np.sin(theta_grid)
    z = minor_radius * np.sin(phi_grid)
    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)

    faces: list[list[int]] = []
    for i in range(major_segments):
        ni = (i + 1) % major_segments
        for j in range(minor_segments):
            nj = (j + 1) % minor_segments
            a = i * minor_segments + j
            b = i * minor_segments + nj
            c = ni * minor_segments + nj
            d = ni * minor_segments + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    triangles = np.asarray(faces, dtype=np.uint32)

    metadata = {
        "kind": "torus",
        "major_radius": float(major_radius),
        "minor_radius": float(minor_radius),
        "major_segments": int(major_segments),
        "minor_segments": int(minor_segments),
        "vertex_count": int(vertices.shape[0]),
        "triangle_count": int(triangles.shape[0]),
        "watertight": is_watertight(triangles),
        "self_intersection_known": False,
    }
    return vertices, triangles, metadata


def _thread_radius(
    theta: np.ndarray,
    axial: np.ndarray,
    *,
    base_radius: float,
    thread_height: float,
    pitch: float,
    starts: int,
    handedness: int,
    phase: float,
) -> np.ndarray:
    helix_phase = starts * theta - handedness * (2.0 * math.pi * axial / pitch) + phase
    return base_radius + thread_height * (0.5 + 0.5 * np.cos(helix_phase))


def threaded_rod_mesh(
    *,
    length: float,
    core_radius: float,
    thread_height: float,
    pitch: float,
    radial_segments: int,
    axial_segments: int,
    starts: int = 1,
    handedness: int = 1,
    phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x_values = np.linspace(-0.5 * length, 0.5 * length, axial_segments + 1)
    theta_values = np.linspace(0.0, 2.0 * math.pi, radial_segments, endpoint=False)

    vertices: list[list[float]] = []
    for axial in x_values:
        theta = theta_values
        radii = _thread_radius(
            theta,
            np.full_like(theta, axial),
            base_radius=core_radius,
            thread_height=thread_height,
            pitch=pitch,
            starts=starts,
            handedness=handedness,
            phase=phase,
        )
        y = radii * np.cos(theta)
        z = radii * np.sin(theta)
        vertices.extend(np.stack([np.full_like(theta, axial), y, z], axis=-1).tolist())

    start_center = len(vertices)
    vertices.append([-0.5 * length, 0.0, 0.0])
    end_center = len(vertices)
    vertices.append([0.5 * length, 0.0, 0.0])

    faces: list[list[int]] = []
    for i in range(axial_segments):
        for j in range(radial_segments):
            nj = (j + 1) % radial_segments
            a = i * radial_segments + j
            b = i * radial_segments + nj
            c = (i + 1) * radial_segments + nj
            d = (i + 1) * radial_segments + j
            faces.append([a, b, c])
            faces.append([a, c, d])

    for j in range(radial_segments):
        nj = (j + 1) % radial_segments
        faces.append([start_center, nj, j])

    end_offset = axial_segments * radial_segments
    for j in range(radial_segments):
        nj = (j + 1) % radial_segments
        faces.append([end_center, end_offset + j, end_offset + nj])

    vertices_array = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(faces, dtype=np.uint32)
    metadata = {
        "kind": "threaded_rod",
        "length": float(length),
        "core_radius": float(core_radius),
        "thread_height": float(thread_height),
        "pitch": float(pitch),
        "starts": int(starts),
        "handedness": int(handedness),
        "vertex_count": int(vertices_array.shape[0]),
        "triangle_count": int(triangles.shape[0]),
        "watertight": is_watertight(triangles),
        "self_intersection_known": False,
    }
    return vertices_array, triangles, metadata


def threaded_nut_mesh(
    *,
    length: float,
    outer_radius: float,
    hole_radius: float,
    thread_height: float,
    pitch: float,
    radial_segments: int,
    axial_segments: int,
    starts: int = 1,
    handedness: int = 1,
    phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x_values = np.linspace(-0.5 * length, 0.5 * length, axial_segments + 1)
    theta_values = np.linspace(0.0, 2.0 * math.pi, radial_segments, endpoint=False)

    outer_vertices: list[list[float]] = []
    inner_vertices: list[list[float]] = []

    for axial in x_values:
        theta = theta_values
        outer_y = outer_radius * np.cos(theta)
        outer_z = outer_radius * np.sin(theta)
        outer_vertices.extend(
            np.stack([np.full_like(theta, axial), outer_y, outer_z], axis=-1).tolist()
        )

        inner_radii = _thread_radius(
            theta,
            np.full_like(theta, axial),
            base_radius=hole_radius,
            thread_height=thread_height,
            pitch=pitch,
            starts=starts,
            handedness=handedness,
            phase=phase,
        )
        inner_y = inner_radii * np.cos(theta)
        inner_z = inner_radii * np.sin(theta)
        inner_vertices.extend(
            np.stack([np.full_like(theta, axial), inner_y, inner_z], axis=-1).tolist()
        )

    vertices = np.asarray(outer_vertices + inner_vertices, dtype=np.float32)
    outer_base = 0
    inner_base = (axial_segments + 1) * radial_segments

    faces: list[list[int]] = []

    for i in range(axial_segments):
        for j in range(radial_segments):
            nj = (j + 1) % radial_segments
            a = outer_base + i * radial_segments + j
            b = outer_base + i * radial_segments + nj
            c = outer_base + (i + 1) * radial_segments + nj
            d = outer_base + (i + 1) * radial_segments + j
            faces.append([a, b, c])
            faces.append([a, c, d])

    for i in range(axial_segments):
        for j in range(radial_segments):
            nj = (j + 1) % radial_segments
            a = inner_base + i * radial_segments + j
            b = inner_base + i * radial_segments + nj
            c = inner_base + (i + 1) * radial_segments + nj
            d = inner_base + (i + 1) * radial_segments + j
            faces.append([a, c, b])
            faces.append([a, d, c])

    for j in range(radial_segments):
        nj = (j + 1) % radial_segments
        o0 = outer_base + j
        o1 = outer_base + nj
        i0 = inner_base + j
        i1 = inner_base + nj
        faces.append([o0, i0, i1])
        faces.append([o0, i1, o1])

    end_outer = outer_base + axial_segments * radial_segments
    end_inner = inner_base + axial_segments * radial_segments
    for j in range(radial_segments):
        nj = (j + 1) % radial_segments
        o0 = end_outer + j
        o1 = end_outer + nj
        i0 = end_inner + j
        i1 = end_inner + nj
        faces.append([o0, i1, i0])
        faces.append([o0, o1, i1])

    triangles = np.asarray(faces, dtype=np.uint32)
    metadata = {
        "kind": "threaded_nut",
        "length": float(length),
        "outer_radius": float(outer_radius),
        "hole_radius": float(hole_radius),
        "thread_height": float(thread_height),
        "pitch": float(pitch),
        "starts": int(starts),
        "handedness": int(handedness),
        "vertex_count": int(vertices.shape[0]),
        "triangle_count": int(triangles.shape[0]),
        "watertight": is_watertight(triangles),
        "self_intersection_known": False,
    }
    return vertices, triangles, metadata


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value
