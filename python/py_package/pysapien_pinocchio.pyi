from __future__ import annotations
import numpy
import sapien.pysapien
import typing
__all__ = ['PinocchioModel']
class PinocchioModel:
    def __init__(self, urdf: str, gravity: numpy.ndarray[typing.Literal[3], numpy.dtype[numpy.float64]] | list[float] | tuple) -> None:
        ...
    def compute_coriolis_matrix(self, qpos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] | list[float] | tuple, qvel: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] | list[float] | tuple) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
        ...
    def compute_forward_dynamics(self, qpos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] | list[float] | tuple, qvel: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] | list[float] | tuple, qf: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] | list[float] | tuple) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
        ...
    def compute_forward_kinematics(self, qpos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] | list[float] | tuple) -> None:
        """
        Compute and cache forward kinematics. After computation, use get_link_pose to retrieve the computed pose for a specific link.
        """
    def compute_full_jacobian(self, qpos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] | list[float] | tuple) -> None:
        """
        Compute and cache Jacobian for all links
        """
    def compute_generalized_mass_matrix(self, qpos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] | list[float] | tuple) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
        ...
    def compute_inverse_dynamics(self, qpos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] | list[float] | tuple, qvel: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] | list[float] | tuple, qacc: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] | list[float] | tuple) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
        ...
    def compute_inverse_kinematics(self, link_index: int, pose: sapien.pysapien.Pose, initial_qpos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] | list[float] | tuple = ..., active_qmask: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]] | list[int] | tuple = ..., eps: float = 0.0001, max_iterations: int = 1000, dt: float = 0.1, damp: float = 1e-06) -> tuple[numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], bool, numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]]:
        """
        Compute inverse kinematics with CLIK algorithm.
        Details see https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b-examples_i-inverse-kinematics.html
        Args:
            link_index: index of the link
            pose: target pose of the link in articulation base frame
            initial_qpos: initial qpos to start CLIK
            active_qmask: dof sized integer array, 1 to indicate active joints and 0 for inactive joints, default to all 1s
            max_iterations: number of iterations steps
            dt: iteration step "speed"
            damp: iteration step "damping"
        Returns:
            result: qpos from IK
            success: whether IK is successful
            error: SE(3) error vector for the best result
        """
    def compute_single_link_local_jacobian(self, qpos: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] | list[float] | tuple, link_index: int) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
        """
        Compute the link(body) Jacobian for a single link. It is faster than compute_full_jacobian followed by get_link_jacobian
        """
    def get_link_jacobian(self, link_index: int, local: bool = False) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
        """
        Given link index, get the Jacobian. Must be called after compute_full_jacobian.
        """
    def get_link_pose(self, link_index: int) -> sapien.pysapien.Pose:
        """
        Given link index, get link pose (in articulation base frame) from forward kinematics. Must be called after compute_forward_kinematics.
        """
    def set_joint_order(self, names: list[str]) -> None:
        ...
    def set_link_order(self, names: list[str]) -> None:
        ...
