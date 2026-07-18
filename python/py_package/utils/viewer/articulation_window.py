from __future__ import annotations

import numpy as np
import sapien
from sapien import internal_renderer as R

from .plugin import Plugin, copy_to_clipboard


class ArticulationWindow(Plugin):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.articulation: sapien.physx.PhysxArticulation | None = None
        self.ui_window = None
        self.ui_dirty = True
        self.joint_details: list[bool] = []
        self.joint_entries: list[tuple[sapien.physx.PhysxArticulationJoint, int, int]] = []
        self.qpos = np.zeros(0, dtype=np.float32)
        self.target_qpos = np.zeros(0, dtype=np.float32)
        self.target_qvel = np.zeros(0, dtype=np.float32)
        self.qpos_widgets = []
        self.target_qpos_widgets: dict[int, object] = {}
        self.target_qvel_widgets: dict[int, object] = {}

    def close(self) -> None:
        self.reset()

    @property
    def selected_entity(self) -> sapien.Entity | None:
        return self.viewer.selected_entity

    @property
    def gpu_mode(self) -> bool:
        return (
            self.articulation is not None
            and self.viewer._entity_uses_physx_gpu(self.articulation.root.entity)
        )

    def notify_selected_entity_change(self) -> None:
        articulation = None
        if self.selected_entity:
            articulation = self.selected_entity.find_component_by_type(
                sapien.physx.PhysxArticulationLinkComponent
            )
            articulation = articulation.articulation if articulation is not None else None

        self.articulation = articulation
        self.joint_entries = []
        if articulation is not None:
            qpos_index = 0
            for joint in articulation.joints:
                for local_index in range(joint.dof):
                    self.joint_entries.append((joint, local_index, qpos_index))
                    qpos_index += 1
        self.joint_details = [False] * len(self.joint_entries)
        self.ui_dirty = True

    def set_joint_details(self, index: int, value: bool) -> None:
        self.joint_details[index] = value
        self.ui_dirty = True

    def _cpu_targets(self, velocity: bool) -> np.ndarray:
        values: list[float] = []
        assert self.articulation is not None
        for joint in self.articulation.joints:
            if joint.dof == 0:
                continue
            target = (
                joint.drive_velocity_target if velocity else joint.drive_target
            )
            values.extend(np.asarray(target, dtype=np.float32).reshape(-1).tolist())
        return np.asarray(values, dtype=np.float32)

    def _refresh_state(self) -> None:
        assert self.articulation is not None
        if self.gpu_mode:
            self.qpos, self.target_qpos, self.target_qvel = (
                self.viewer._get_gpu_articulation_state(self.articulation)
            )
        else:
            self.qpos = np.asarray(self.articulation.get_qpos(), dtype=np.float32)
            self.target_qpos = self._cpu_targets(False)
            self.target_qvel = self._cpu_targets(True)

    def _set_qpos(self, index: int, value: float) -> None:
        assert self.articulation is not None
        self.qpos[index] = value
        if self.gpu_mode:
            self.viewer._queue_gpu_articulation_qpos(self.articulation, self.qpos)
        else:
            self.articulation.set_qpos(self.qpos)

    def _set_target(self, index: int, value: float, velocity: bool) -> None:
        assert self.articulation is not None
        values = self.target_qvel if velocity else self.target_qpos
        values[index] = value
        if self.gpu_mode:
            if velocity:
                self.viewer._queue_gpu_articulation_target_qvel(
                    self.articulation, values
                )
            else:
                self.viewer._queue_gpu_articulation_target_qpos(
                    self.articulation, values
                )
            return

        joint, local_index, _ = self.joint_entries[index]
        target = np.asarray(
            joint.drive_velocity_target if velocity else joint.drive_target,
            dtype=np.float32,
        ).reshape(-1)
        target[local_index] = value
        if velocity:
            joint.set_drive_velocity_target(target)
        else:
            joint.set_drive_target(target)

    def _copy_qpos(self, _: object) -> None:
        copy_to_clipboard(f"[{', '.join(str(value) for value in self.qpos)}]")

    def _show_collision(self, show: bool) -> None:
        assert self.articulation is not None
        for plugin in self.viewer.plugins:
            if plugin.__class__.__name__ != "EntityWindow":
                continue
            for link in self.articulation.links:
                if show:
                    plugin.enable_collision_visual(link.entity)
                else:
                    plugin.disable_collision_visual(link.entity)

    def _rebuild_ui(self) -> None:
        self.ui_window = R.UIWindow().Label("Articulation")
        self.qpos_widgets = []
        self.target_qpos_widgets = {}
        self.target_qvel_widgets = {}

        articulation = self.articulation
        if articulation is None:
            self.ui_window.append(R.UIDisplayText().Text("No articulation selected."))
            self.ui_dirty = False
            return

        self.ui_window.append(
            R.UIDisplayText().Text(
                f"Name: {articulation.name if articulation.name else '(no name)'}"
            ),
            R.UIDisplayText().Text(
                f"Base Link Entity Id: {articulation.root.entity.per_scene_id}"
            ),
        )
        joints_section = R.UISection().Label("Joints")
        self.ui_window.append(joints_section)

        for entry_index, (joint, local_index, qpos_index) in enumerate(
            self.joint_entries
        ):
            limits = joint.limit[local_index]
            qpos_widget = (
                R.UISliderFloat()
                .WidthRatio(0.5)
                .Id(f"joint_{qpos_index}")
                .Min(max(limits[0], -20))
                .Max(min(limits[1], 20))
                .Value(self.qpos[qpos_index])
                .Callback(
                    lambda slider, index=qpos_index: self._set_qpos(
                        index, slider.value
                    )
                )
            )
            self.qpos_widgets.append(qpos_widget)
            line = R.UISameLine().append(qpos_widget)
            joints_section.append(line)

            label = joint.name if joint.dof == 1 else f"{joint.name}[{local_index}]"
            if self.joint_details[entry_index]:
                line.append(
                    R.UIButton()
                    .Label("-")
                    .Id(str(entry_index))
                    .Width(40)
                    .Callback(
                        lambda _, index=entry_index: self.set_joint_details(
                            index, False
                        )
                    ),
                    R.UIDisplayText().Text(label),
                )
                target_qpos_widget = (
                    R.UISliderFloat()
                    .Label("Position Target")
                    .Id(str(qpos_index))
                    .WidthRatio(0.5)
                    .Min(max(limits[0], -20))
                    .Max(min(limits[1], 20))
                    .Value(self.target_qpos[qpos_index])
                    .Callback(
                        lambda slider, index=qpos_index: self._set_target(
                            index, slider.value, False
                        )
                    )
                )
                target_qvel_widget = (
                    R.UISliderFloat()
                    .Label("Velocity Target")
                    .Id(str(qpos_index))
                    .WidthRatio(0.5)
                    .Min(-1)
                    .Max(1)
                    .Value(self.target_qvel[qpos_index])
                    .Callback(
                        lambda slider, index=qpos_index: self._set_target(
                            index, slider.value, True
                        )
                    )
                )
                self.target_qpos_widgets[qpos_index] = target_qpos_widget
                self.target_qvel_widgets[qpos_index] = target_qvel_widget
                joints_section.append(
                    target_qpos_widget,
                    target_qvel_widget,
                    R.UIInputFloat()
                    .Label("Damping")
                    .Id(str(qpos_index))
                    .WidthRatio(0.5)
                    .Value(joint.damping)
                    .Callback(
                        lambda value, active_joint=joint: active_joint.set_drive_property(
                            active_joint.stiffness,
                            value.value,
                            active_joint.force_limit,
                            active_joint.drive_mode,
                        )
                    ),
                    R.UIInputFloat()
                    .Label("Stiffness")
                    .Id(str(qpos_index))
                    .WidthRatio(0.5)
                    .Value(joint.stiffness)
                    .Callback(
                        lambda value, active_joint=joint: active_joint.set_drive_property(
                            value.value,
                            active_joint.damping,
                            active_joint.force_limit,
                            active_joint.drive_mode,
                        )
                    ),
                    R.UIInputFloat()
                    .Label("Force Limit")
                    .Id(str(qpos_index))
                    .WidthRatio(0.5)
                    .Value(joint.force_limit)
                    .Callback(
                        lambda value, active_joint=joint: active_joint.set_drive_property(
                            active_joint.stiffness,
                            active_joint.damping,
                            value.value,
                            active_joint.drive_mode,
                        )
                    ),
                    R.UIInputFloat()
                    .Label("Friction")
                    .Id(str(qpos_index))
                    .WidthRatio(0.5)
                    .Value(joint.friction)
                    .Callback(
                        lambda value, active_joint=joint: active_joint.set_friction(
                            value.value
                        )
                    ),
                    R.UICheckbox()
                    .Label("Acceleration")
                    .Id(str(qpos_index))
                    .Checked(joint.drive_mode == "acceleration")
                    .Callback(
                        lambda value, active_joint=joint: active_joint.set_drive_property(
                            active_joint.stiffness,
                            active_joint.damping,
                            active_joint.force_limit,
                            "acceleration" if value.checked else "force",
                        )
                    ),
                    R.UIDummy().Height(20),
                )
            else:
                line.append(
                    R.UIButton()
                    .Label("+")
                    .Id(str(entry_index))
                    .Width(40)
                    .Callback(
                        lambda _, index=entry_index: self.set_joint_details(
                            index, True
                        )
                    ),
                    R.UIDisplayText().Text(label),
                )

        self.ui_window.append(
            R.UIButton().Label("Copy Joint Positions").Callback(self._copy_qpos),
            R.UISameLine().append(
                R.UIButton().Label("Show").Callback(lambda _: self._show_collision(True)),
                R.UIButton().Label("Hide").Callback(lambda _: self._show_collision(False)),
                R.UIDisplayText().Text("Collision"),
            ),
        )
        self.ui_dirty = False

    def build(self) -> None:
        if self.viewer.render_scene is None:
            self.ui_window = None
            return
        if self.ui_window is not None and not self.ui_window.expanded:
            return

        if self.articulation is not None:
            self._refresh_state()
        if self.ui_window is None or self.ui_dirty:
            self._rebuild_ui()
            return

        for widget, value in zip(self.qpos_widgets, self.qpos):
            widget.Value(value)
        for index, widget in self.target_qpos_widgets.items():
            widget.Value(self.target_qpos[index])
        for index, widget in self.target_qvel_widgets.items():
            widget.Value(self.target_qvel[index])

    def get_ui_windows(self) -> list[R.UIWindow]:
        self.build()
        return [self.ui_window] if self.ui_window else []
