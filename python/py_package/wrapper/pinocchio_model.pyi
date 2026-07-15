from __future__ import annotations

import numpy
import sapien.pysapien
import typing

__all__ = ["PinocchioModel"]


class PinocchioModel:
    def __init__(
        self,
        urdf: str,
        gravity: numpy.ndarray[typing.Literal[3], numpy.dtype[numpy.float64]]
        | list[float]
        | tuple,
    ) -> None: ...
    def compute_coriolis_matrix(
        self,
        qpos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]
        | list[float]
        | tuple,
        qvel: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]
        | list[float]
        | tuple,
    ) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: ...
    def compute_forward_dynamics(
        self,
        qpos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]
        | list[float]
        | tuple,
        qvel: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]
        | list[float]
        | tuple,
        qf: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]
        | list[float]
        | tuple,
    ) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: ...
    def compute_forward_kinematics(
        self,
        qpos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]
        | list[float]
        | tuple,
    ) -> None:
        """Compute and cache forward kinematics for subsequent link-pose queries."""
    def compute_full_jacobian(
        self,
        qpos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]
        | list[float]
        | tuple,
    ) -> None:
        """Compute and cache the Jacobians for all links."""
    def compute_generalized_mass_matrix(
        self,
        qpos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]
        | list[float]
        | tuple,
    ) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: ...
    def compute_inverse_dynamics(
        self,
        qpos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]
        | list[float]
        | tuple,
        qvel: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]
        | list[float]
        | tuple,
        qacc: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]
        | list[float]
        | tuple,
    ) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: ...
    def compute_inverse_kinematics(
        self,
        link_index: int,
        pose: sapien.pysapien.Pose,
        initial_qpos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]
        | list[float]
        | tuple = ...,
        active_qmask: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]]
        | list[int]
        | tuple = ...,
        eps: float = 0.0001,
        max_iterations: int = 1000,
        dt: float = 0.1,
        damp: float = 1e-06,
    ) -> tuple[
        numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]],
        bool,
        numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]],
    ]:
        """Compute inverse kinematics with the CLIK algorithm."""
    def compute_single_link_local_jacobian(
        self,
        qpos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]
        | list[float]
        | tuple,
        link_index: int,
    ) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
        """Compute one link's body-frame Jacobian without caching every link."""
    def get_link_jacobian(
        self, link_index: int, local: bool = False
    ) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
        """Return a cached link Jacobian in the world or body frame."""
    def get_link_pose(self, link_index: int) -> sapien.pysapien.Pose:
        """Return a cached link pose in the articulation base frame."""
    def set_joint_order(self, names: list[str]) -> None: ...
    def set_link_order(self, names: list[str]) -> None: ...
