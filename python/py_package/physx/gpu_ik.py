from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..pysapien import CudaArray, Pose

if TYPE_CHECKING:
    import torch

    from ..pysapien.physx import (
        PhysxArticulation,
        PhysxArticulationLinkComponent,
        PhysxGpuSystem,
    )


def _require_torch() -> Any:
    try:
        import torch as torch_module
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "GPU inverse kinematics requires PyTorch. Install a CUDA-enabled "
            "torch build or use the low-level PhysX Jacobian buffers directly."
        ) from exc

    if not torch_module.cuda.is_available():
        raise RuntimeError("GPU inverse kinematics requires torch.cuda to be available")
    globals()["torch"] = torch_module
    return torch_module


def _pose_to_numpy(pose: Pose | Sequence[float]) -> np.ndarray:
    if isinstance(pose, Pose):
        return np.concatenate(
            [np.asarray(pose.p, dtype=np.float32), np.asarray(pose.q, dtype=np.float32)]
        )
    return np.asarray(pose, dtype=np.float32)


def _normalize_quaternion(quat: torch.Tensor) -> torch.Tensor:
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _quat_inverse(quat: torch.Tensor) -> torch.Tensor:
    result = quat.clone()
    result[..., 1:] = -result[..., 1:]
    return result / (quat * quat).sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _quat_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            lhs[..., 0] * rhs[..., 0]
            - lhs[..., 1] * rhs[..., 1]
            - lhs[..., 2] * rhs[..., 2]
            - lhs[..., 3] * rhs[..., 3],
            lhs[..., 0] * rhs[..., 1]
            + lhs[..., 1] * rhs[..., 0]
            + lhs[..., 2] * rhs[..., 3]
            - lhs[..., 3] * rhs[..., 2],
            lhs[..., 0] * rhs[..., 2]
            - lhs[..., 1] * rhs[..., 3]
            + lhs[..., 2] * rhs[..., 0]
            + lhs[..., 3] * rhs[..., 1],
            lhs[..., 0] * rhs[..., 3]
            + lhs[..., 1] * rhs[..., 2]
            - lhs[..., 2] * rhs[..., 1]
            + lhs[..., 3] * rhs[..., 0],
        ],
        dim=-1,
    )


def _quat_rotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quat = _normalize_quaternion(quat)
    q_vector = quat[..., 1:]
    uv = torch.cross(q_vector, vector, dim=-1)
    uuv = torch.cross(q_vector, uv, dim=-1)
    return vector + 2.0 * (quat[..., :1] * uv + uuv)


def _quat_to_rotation_vector(quat: torch.Tensor) -> torch.Tensor:
    quat = _normalize_quaternion(quat)
    quat = torch.where(quat[..., :1] < 0.0, -quat, quat)

    vector = quat[..., 1:]
    vector_norm = vector.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, quat[..., :1])
    small = vector_norm < 1e-12
    return torch.where(small, 2.0 * vector, vector / vector_norm.clamp_min(1e-12) * angle)


def _pose_error(target_pose: torch.Tensor, current_pose: torch.Tensor) -> torch.Tensor:
    target_position = target_pose[:, :3]
    current_position = current_pose[:, :3]
    target_quat = target_pose[:, 3:7]
    current_quat = current_pose[:, 3:7]
    rotation_error = _quat_to_rotation_vector(
        _quat_multiply(target_quat, _quat_inverse(current_quat))
    )
    return torch.cat([target_position - current_position, rotation_error], dim=1)


class GpuInverseKinematicsSolver:
    """Batched GPU inverse kinematics using PhysX dense articulation Jacobians.

    The solver operates on articulations that are already registered in a
    :class:`sapien.physx.PhysxGpuSystem` and initialized with ``gpu_init()``. It
    keeps all iterative IK math on CUDA tensors and uses PhysX GPU Jacobians, so
    it does not depend on Pinocchio.

    Each batch item solves one target pose for one articulation/link pair. The
    returned ``qpos`` tensor is padded to the PhysX GPU system's maximum DOF.
    Only valid and active joint columns are updated.
    """

    def __init__(
        self,
        system: PhysxGpuSystem,
        articulations: Sequence[PhysxArticulation],
        link_indices: Sequence[int | PhysxArticulationLinkComponent],
        *,
        active_qmask: Any | None = None,
        qlimits: Any | None = None,
        set_cuda_stream: bool = True,
    ) -> None:
        """Create a batched GPU IK solver.

        Args:
            system: Initialized ``PhysxGpuSystem`` that owns the articulations.
            articulations: Batch of articulations to solve. Duplicate
                articulations are not supported.
            link_indices: Target link index, or link component, for each
                articulation.
            active_qmask: Optional mask of active DOFs with shape ``(batch,
                max_dof)`` or ``(max_dof,)``. Defaults to all valid DOFs.
            qlimits: Optional joint limits with shape ``(batch, max_dof, 2)``.
                Defaults to ``articulation.get_qlimits()`` padded to
                ``[-inf, inf]``.
            set_cuda_stream: If true, set the PhysX GPU stream to PyTorch's
                current CUDA stream. Leave false only if the caller manages CUDA
                stream synchronization explicitly.
        """
        if len(articulations) == 0:
            raise ValueError("at least one articulation is required")
        if len(articulations) != len(link_indices):
            raise ValueError("articulations and link_indices must have the same length")

        torch = _require_torch()
        self._torch = torch
        self.system = system
        self.articulations = list(articulations)
        self.batch_size = len(self.articulations)

        self.qpos_buffer = system.cuda_articulation_qpos.torch()
        self.link_data_buffer = system.cuda_articulation_link_data.torch()
        self.jacobian_buffer = system.cuda_articulation_jacobian.torch()
        self.device = self.qpos_buffer.device
        self.dtype = self.qpos_buffer.dtype
        self.max_dof = int(self.qpos_buffer.shape[1])
        self.max_jacobian_cols = int(self.jacobian_buffer.shape[2])

        if set_cuda_stream:
            stream = torch.cuda.current_stream(self.device)
            system.gpu_set_cuda_stream(int(stream.cuda_stream))

        articulation_indices = [int(articulation.gpu_index) for articulation in self.articulations]
        if any(index < 0 for index in articulation_indices):
            raise RuntimeError(
                "all articulations must be initialized with PhysxGpuSystem.gpu_init()"
            )
        if len(set(articulation_indices)) != len(articulation_indices):
            raise ValueError("duplicate articulations are not supported by this solver")

        self.articulation_indices = torch.tensor(
            articulation_indices, device=self.device, dtype=torch.long
        )
        self._index_tensor = self.articulation_indices.to(dtype=torch.int32).contiguous()
        self.index_buffer = CudaArray(self._index_tensor)

        dof_counts: list[int] = []
        root_column_counts: list[int] = []
        row_starts: list[int] = []
        link_row_masks: list[bool] = []
        resolved_link_indices: list[int] = []
        cmass_local_offsets: list[np.ndarray] = []
        qlimits_array = np.full((self.batch_size, self.max_dof, 2), [-np.inf, np.inf], np.float32)

        for batch_index, (articulation, link_or_index) in enumerate(
            zip(self.articulations, link_indices, strict=True)
        ):
            links = articulation.get_links()
            link_index = self._resolve_link_index(articulation, links, link_or_index)
            resolved_link_indices.append(link_index)

            dof = int(articulation.dof)
            rows, cols = map(int, articulation.get_jacobian_shape())
            root_columns = cols - dof
            if root_columns not in (0, 6):
                raise RuntimeError(
                    f"unsupported Jacobian shape for articulation {articulation.name!r}: "
                    f"rows={rows}, cols={cols}, dof={dof}"
                )
            if dof > self.max_dof or root_columns + dof > self.max_jacobian_cols:
                raise RuntimeError("articulation DOF exceeds PhysX GPU buffer shape")

            dof_counts.append(dof)
            root_column_counts.append(root_columns)
            if root_columns == 0:
                row_starts.append(max(link_index - 1, 0) * 6)
                link_row_masks.append(link_index != 0)
            else:
                row_starts.append(link_index * 6)
                link_row_masks.append(True)

            cmass_local_offsets.append(
                np.asarray(links[link_index].cmass_local_pose.p, dtype=np.float32)
            )
            limits = np.asarray(articulation.get_qlimits(), dtype=np.float32).reshape(-1, 2)
            qlimits_array[batch_index, : min(dof, len(limits))] = limits[:dof]

        self.link_indices = torch.tensor(
            resolved_link_indices, device=self.device, dtype=torch.long
        )
        self.dof_counts = torch.tensor(dof_counts, device=self.device, dtype=torch.long)
        self.root_column_counts = torch.tensor(
            root_column_counts, device=self.device, dtype=torch.long
        )
        self.row_starts = torch.tensor(row_starts, device=self.device, dtype=torch.long)
        self.link_row_mask = torch.tensor(
            link_row_masks, device=self.device, dtype=torch.bool
        ).view(self.batch_size, 1, 1)
        self.cmass_local_offsets = torch.as_tensor(
            np.stack(cmass_local_offsets), device=self.device, dtype=self.dtype
        )

        dof_range = torch.arange(self.max_dof, device=self.device, dtype=torch.long)
        row_range = torch.arange(6, device=self.device, dtype=torch.long)
        self.valid_dof_mask = dof_range.unsqueeze(0) < self.dof_counts.unsqueeze(1)
        self.joint_column_indices = dof_range.unsqueeze(0) + self.root_column_counts.unsqueeze(1)
        self.link_row_indices = self.row_starts.unsqueeze(1) + row_range.unsqueeze(0)

        self.qlimits = self._prepare_qlimits(qlimits, qlimits_array)
        self.active_qmask = self._prepare_active_qmask(active_qmask) & self.valid_dof_mask

    @staticmethod
    def _resolve_link_index(
        articulation: PhysxArticulation,
        links: Sequence[PhysxArticulationLinkComponent],
        link_or_index: int | PhysxArticulationLinkComponent,
    ) -> int:
        if isinstance(link_or_index, int):
            link_index = link_or_index
        else:
            if int(link_or_index.articulation.gpu_index) != int(articulation.gpu_index):
                raise ValueError("target link does not belong to the corresponding articulation")
            link_index = int(link_or_index.index)

        if link_index < 0 or link_index >= len(links):
            raise ValueError(f"link index {link_index} is out of range")
        return link_index

    def _as_batch_tensor(
        self, value: Any, trailing_shape: tuple[int, ...], name: str
    ) -> torch.Tensor:
        torch = self._torch
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.device, dtype=self.dtype)
        else:
            tensor = torch.as_tensor(value, device=self.device, dtype=self.dtype)

        if tensor.shape == trailing_shape:
            tensor = tensor.unsqueeze(0).expand(self.batch_size, *trailing_shape)
        expected_shape = (self.batch_size, *trailing_shape)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape} or {trailing_shape}")
        return tensor

    def _prepare_target_poses(self, target_poses: Any) -> torch.Tensor:
        torch = self._torch
        if isinstance(target_poses, Pose):
            target_poses = _pose_to_numpy(target_poses)
        elif (
            isinstance(target_poses, Sequence)
            and len(target_poses) > 0
            and isinstance(target_poses[0], Pose)
        ):
            target_poses = np.stack([_pose_to_numpy(pose) for pose in target_poses])

        target = self._as_batch_tensor(target_poses, (7,), "target_poses")
        target = target.clone()
        target[:, 3:7] = _normalize_quaternion(target[:, 3:7])
        return target

    def _prepare_qpos(self, initial_qpos: Any | None) -> torch.Tensor:
        if initial_qpos is None:
            return self.qpos_buffer[self.articulation_indices, : self.max_dof].clone()
        return self._as_batch_tensor(initial_qpos, (self.max_dof,), "initial_qpos").clone()

    def _prepare_qlimits(self, qlimits: Any | None, default_limits: np.ndarray) -> torch.Tensor:
        if qlimits is None:
            limits = self._torch.as_tensor(default_limits, device=self.device, dtype=self.dtype)
        else:
            limits = self._as_batch_tensor(qlimits, (self.max_dof, 2), "qlimits")
        return limits

    def _prepare_active_qmask(self, active_qmask: Any | None) -> torch.Tensor:
        torch = self._torch
        if active_qmask is None:
            return torch.ones(
                (self.batch_size, self.max_dof), device=self.device, dtype=torch.bool
            )
        mask = self._as_batch_tensor(active_qmask, (self.max_dof,), "active_qmask")
        return mask.to(dtype=torch.bool)

    def _clip_qpos_to_limits(self, qpos: torch.Tensor) -> torch.Tensor:
        lower = self.qlimits[:, :, 0]
        upper = self.qlimits[:, :, 1]
        has_lower = torch.isfinite(lower) & (lower > -1e10) & self.valid_dof_mask
        has_upper = torch.isfinite(upper) & (upper < 1e10) & self.valid_dof_mask
        qpos = torch.where(has_lower, torch.maximum(qpos, lower), qpos)
        return torch.where(has_upper, torch.minimum(qpos, upper), qpos)

    def _task_weights(
        self, position_weight: float | Sequence[float], rotation_weight: float | Sequence[float]
    ) -> torch.Tensor:
        torch = self._torch
        position = torch.as_tensor(position_weight, device=self.device, dtype=self.dtype)
        rotation = torch.as_tensor(rotation_weight, device=self.device, dtype=self.dtype)
        if position.ndim == 0:
            position = position.repeat(3)
        if rotation.ndim == 0:
            rotation = rotation.repeat(3)
        if tuple(position.shape) != (3,) or tuple(rotation.shape) != (3,):
            raise ValueError(
                "position_weight and rotation_weight must be scalars or length-3 sequences"
            )
        return torch.cat([position, rotation]).view(1, 6)

    def _selected_joint_jacobian(self, current_pose: torch.Tensor) -> torch.Tensor:
        batch_jacobian = self.jacobian_buffer[self.articulation_indices]
        row_indices = self.link_row_indices[:, :, None].expand(
            self.batch_size, 6, self.max_jacobian_cols
        )
        link_jacobian = batch_jacobian.gather(1, row_indices) * self.link_row_mask
        joint_indices = self.joint_column_indices[:, None, :].expand(
            self.batch_size, 6, self.max_dof
        )
        joint_jacobian = link_jacobian.gather(2, joint_indices).clone()

        # PhysX reports linear velocity at the link center of mass. Shift it to
        # the link frame origin so the Jacobian matches the target pose error.
        origin_minus_com = -_quat_rotate(current_pose[:, 3:7], self.cmass_local_offsets)
        angular_jacobian = joint_jacobian[:, 3:6, :].transpose(1, 2)
        correction = -torch.cross(
            origin_minus_com[:, None, :].expand_as(angular_jacobian),
            angular_jacobian,
            dim=2,
        ).transpose(1, 2)
        joint_jacobian[:, :3, :] += correction
        return joint_jacobian

    def solve(
        self,
        target_poses: Any,
        *,
        initial_qpos: Any | None = None,
        max_iterations: int = 100,
        eps: float = 1e-4,
        damping: float = 1e-5,
        step_size: float = 0.5,
        position_weight: float | Sequence[float] = 1.0,
        rotation_weight: float | Sequence[float] = 1.0,
        return_best: bool = True,
        apply_result: bool = True,
        fetch_initial_qpos: bool = True,
        early_stop: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve batched IK on the GPU.

        Args:
            target_poses: Target link poses with shape ``(batch, 7)`` in
                ``xyz + wxyz`` order. A single ``(7,)`` pose is broadcast to the
                batch. ``sapien.Pose`` and lists of ``sapien.Pose`` are accepted.
            initial_qpos: Optional padded initial qpos tensor with shape
                ``(batch, max_dof)``. If omitted, the current PhysX GPU qpos
                buffer is used.
            max_iterations: Number of damped least-squares iterations.
            eps: Success threshold on the weighted 6D pose-error norm.
            damping: Damping added to ``J J^T``.
            step_size: Multiplicative step size for each qpos update.
            position_weight: Scalar or length-3 weights for position rows.
            rotation_weight: Scalar or length-3 weights for rotation rows. Use
                ``0.0`` for position-only IK.
            return_best: Return the lowest-error qpos seen during the solve.
            apply_result: Apply the returned qpos to the PhysX GPU system. If
                false, the original qpos buffer is restored before returning.
            fetch_initial_qpos: Fetch the current PhysX qpos buffer before
                reading it when ``initial_qpos`` is omitted, or before saving
                the state restored by ``apply_result=False``.
            early_stop: If true, stop once every batch item satisfies ``eps``.
                This checks a CUDA scalar on the CPU each iteration; leave false
                for fully fixed-iteration batched GPU workloads.

        Returns:
            ``(qpos, success, error_norm)`` as CUDA torch tensors. ``qpos`` has
            shape ``(batch, max_dof)``, ``success`` has shape ``(batch,)``, and
            ``error_norm`` stores the best weighted error norm when
            ``return_best`` is true.
        """
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if damping <= 0.0:
            raise ValueError("damping must be positive")

        torch = self._torch
        target = self._prepare_target_poses(target_poses)
        if initial_qpos is not None:
            # Clone explicit seeds before fetching, because the fetch overwrites
            # the shared CUDA qpos buffer that callers may have sliced from.
            qpos = self._prepare_qpos(initial_qpos)
        if fetch_initial_qpos and (initial_qpos is None or not apply_result):
            self.system.gpu_fetch_articulation_qpos()
        original_qpos = self.qpos_buffer[self.articulation_indices, : self.max_dof].clone()
        if initial_qpos is None:
            qpos = self._prepare_qpos(None)
        qpos = torch.where(self.valid_dof_mask, qpos, torch.zeros_like(qpos))
        qpos = self._clip_qpos_to_limits(qpos)

        task_weights = self._task_weights(position_weight, rotation_weight)
        eye = torch.eye(6, device=self.device, dtype=self.dtype).expand(self.batch_size, 6, 6)
        best_qpos = qpos.clone()
        best_error_norm = torch.full(
            (self.batch_size,), torch.inf, device=self.device, dtype=self.dtype
        )
        success = torch.zeros((self.batch_size,), device=self.device, dtype=torch.bool)
        last_error_norm = best_error_norm.clone()

        for _ in range(max_iterations):
            self.qpos_buffer[self.articulation_indices, : self.max_dof] = qpos
            self.system.gpu_apply_articulation_qpos(self.index_buffer)
            self.system.gpu_update_articulation_kinematics(self.index_buffer)
            self.system.gpu_fetch_articulation_link_pose()
            self.system.gpu_compute_articulation_jacobian(self.index_buffer)

            current_pose = self.link_data_buffer[self.articulation_indices, self.link_indices, :7]
            error = _pose_error(target, current_pose)
            weighted_error = error * task_weights
            error_norm = torch.linalg.vector_norm(weighted_error, dim=1)
            last_error_norm = error_norm

            improved = error_norm < best_error_norm
            best_error_norm = torch.where(improved, error_norm, best_error_norm)
            best_qpos = torch.where(improved[:, None], qpos, best_qpos)
            success = success | (error_norm < eps)
            if early_stop and bool(torch.all(success)):
                break

            joint_jacobian = self._selected_joint_jacobian(current_pose)
            active_mask = self.active_qmask & (~success[:, None])
            weighted_jacobian = joint_jacobian * task_weights[:, :, None]
            weighted_jacobian = weighted_jacobian * active_mask[:, None, :].to(self.dtype)
            lhs = weighted_jacobian @ weighted_jacobian.transpose(1, 2) + damping * eye
            step = weighted_jacobian.transpose(1, 2) @ torch.linalg.solve(
                lhs, weighted_error.unsqueeze(2)
            )
            step = step.squeeze(2) * active_mask.to(self.dtype)
            qpos = self._clip_qpos_to_limits(qpos + step_size * step)

        result = best_qpos if return_best else qpos
        result_error_norm = best_error_norm if return_best else last_error_norm
        final_qpos = result if apply_result else original_qpos
        self.qpos_buffer[self.articulation_indices, : self.max_dof] = final_qpos
        self.system.gpu_apply_articulation_qpos(self.index_buffer)
        self.system.gpu_update_articulation_kinematics(self.index_buffer)
        self.system.gpu_fetch_articulation_link_pose()
        return result, success, result_error_norm


def gpu_inverse_kinematics(
    system: PhysxGpuSystem,
    articulations: Sequence[PhysxArticulation],
    link_indices: Sequence[int | PhysxArticulationLinkComponent],
    target_poses: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-shot batched GPU inverse kinematics helper.

    For repeated solves, construct :class:`GpuInverseKinematicsSolver` once and
    reuse it so CUDA buffer views and index buffers are cached. Keyword
    arguments ``active_qmask``, ``qlimits``, and ``set_cuda_stream`` are passed
    to the solver constructor; all other keywords are passed to ``solve``.
    """
    solver_kwargs = {
        "active_qmask": kwargs.pop("active_qmask", None),
        "qlimits": kwargs.pop("qlimits", None),
        "set_cuda_stream": kwargs.pop("set_cuda_stream", True),
    }
    solver = GpuInverseKinematicsSolver(system, articulations, link_indices, **solver_kwargs)
    return solver.solve(target_poses, **kwargs)
