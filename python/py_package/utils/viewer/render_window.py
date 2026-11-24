from importlib.resources import files
from pathlib import Path

import numpy as np
import sapien
from sapien import internal_renderer as R

from .plugin import Plugin


def build_config_for_pack(pack: str, current: "sapien.render.RenderConfig"):
    """A `RenderConfig(pack)` carrying over `current`'s explicit assignments.

    Only parameters the new pack declares with a compatible type survive: the value is copied
    when the new pack.json declares it with the same type, an int also for a float parameter
    (stored as a float). Everything else falls back to the new pack's default.
    """
    config = sapien.render.RenderConfig(pack)
    parameters = config.parameters
    for name, value in current.properties.items():
        declared = parameters.get(name)
        if declared is None:
            continue
        try:
            config[name] = value
        except RuntimeError:
            # the new pack declares the name with another type or range: keep its default
            continue
    return config


class _ParameterBinding:
    """Getter/setter pair for one pack.json parameter, so a UI widget binds by name.

    The widgets read the value back every frame and write through `set_parameter`, which
    converts to the declared type, so the panel always shows the renderer's live value.
    """

    def __init__(self, panel: "RenderOptionsWindow", name: str):
        self._panel = panel
        self._name = name

    @property
    def value(self):
        value = self._panel.get_parameter(self._name)
        if isinstance(value, str):
            # a variant parameter never binds to a scalar widget
            return 0
        return value

    @value.setter
    def value(self, value):
        self._panel.set_parameter(self._name, value)


class RenderOptionsWindow(Plugin):
    def __init__(self):
        self.reset()

    def init(self, viewer):
        super().init(viewer)
        self._setup_shader_dir()

    @property
    def window(self):
        return self.viewer.window

    @property
    def shader_name_list(self):
        return [d.name for d in self.shader_list]

    def set_shader(self, index):
        """Switch the window to the pack at `index`.

        The new config is built from scratch, so every parameter starts at the default its
        pack.json declares; only explicit assignments of the current config that the new pack
        declares with a compatible type carry over.
        """
        self.shader_dir = str(self.shader_list[index])
        self.shader_type = str(self.shader_types[index])
        config = build_config_for_pack(self.shader_dir, self.window.render_config)
        self.window.set_render_config(config)
        self.viewer.render_config = self.window.render_config
        self.viewer.render_target = "Color"
        self.shader_index = index

    def reset(self):
        self.ui_window = None
        self.shader_dir = None
        self.shader_index = None
        self.shader_list = None
        self.shader_type = None
        self.shader_types = None

    @property
    def is_rt(self):
        return getattr(self, "shader_type", None) == "rt"

    @property
    def denoiser_index(self):
        if self.viewer.window.denoiser == "oidn":
            return 1
        if self.viewer.window.denoiser == "optix":
            return 2
        return 0

    @denoiser_index.setter
    def denoiser_index(self, index):
        if index == 1:
            self.viewer.window.denoiser = "oidn"
        elif index == 2:
            self.viewer.window.denoiser = "optix"
        else:
            self.viewer.window.denoiser = "none"

    def get_parameter(self, name):
        """Effective value of a parameter of the current pack: its runtime value, else the
        pack.json default."""
        return self.window.render_config[name]

    def set_parameter(self, name, value):
        """Set one uniform parameter of the current pack on the window renderer.

        The value is converted to the type pack.json declares and range-checked by the renderer,
        so a widget can never push a value the pack rejects.
        """
        declared = self.window.render_config.parameters[name]
        kind = declared["type"]
        if kind == "bool":
            self.window.set_camera_property(name, 1 if value else 0)
        elif kind == "int":
            self.window.set_camera_property(name, int(value))
        elif kind == "float":
            self.window.set_camera_property(name, float(value))
        else:
            raise RuntimeError(f"parameter '{name}' is a variant: reopen the pack to change it")

    def _parameter_widget(self, name):
        """A widget bound to parameter `name`, chosen from the type and range pack.json
        declares."""
        declared = self.window.render_config.parameters[name]
        binding = _ParameterBinding(self, name)
        if declared["type"] == "bool":
            return R.UICheckbox().Label(name).Bind(binding, "value")
        if declared["type"] == "int":
            return R.UIInputInt().Label(name).Bind(binding, "value")
        minimum = declared.get("minimum", 0.0)
        maximum = declared.get("maximum")
        if maximum is None:
            return R.UIInputFloat().Label(name).Bind(binding, "value")
        return R.UISliderFloat().Label(name).Min(minimum).Max(maximum).Bind(binding, "value")

    def _setup_shader_dir(self):
        default_shader_dir = Path(self.viewer.render_config.shader_pack)

        default_type = "rast"
        for x in default_shader_dir.iterdir():
            if str(x).endswith("camera.rgen"):
                default_type = "rt"
                break

        self.shader_list = []
        self.shader_types = []

        try:
            all_shader_dir = Path(str(files("sapien").joinpath("vulkan_shader")))

            for f in all_shader_dir.iterdir():
                if f.is_dir():
                    # a pack that cannot be a window renderer (a batched pack, which renders only
                    # through a RenderCameraGroup, or a directory without pack.json) is not listed
                    if not (f / "pack.json").is_file():
                        continue
                    if any("gbuffer.frag" in x.name for x in f.iterdir()):
                        self.shader_list.append(f)
                        self.shader_types.append("rast")
                    if any("camera.rgen" in x.name for x in f.iterdir()):
                        self.shader_list.append(f)
                        self.shader_types.append("rt")

        except Exception:
            pass

        self.shader_type = default_type
        self.shader_dir = str(default_shader_dir)
        if default_shader_dir not in self.shader_list:
            self.shader_list = [default_shader_dir] + self.shader_list
            self.shader_types = [default_type] + self.shader_types

        self.shader_index = self.shader_list.index(default_shader_dir)

    def build(self):
        if self.viewer.render_scene is None:
            self.ui_window = None
            return

        if self.ui_window is None:
            self.ui_window = R.UIWindow().Label("Render").Pos(10, 10).Size(400, 400)

            # one row per uniform parameter of the current pack, with the widget its declared
            # type and range call for. Variant parameters select shader source and cannot change
            # on a live renderer, so they are not editable here.
            widgets = [
                R.UIOptions()
                .Label("Shader")
                .Style("select")
                .BindItems(self, "shader_name_list")
                .BindIndex(self, "shader_index")
                .Callback(lambda p: self.set_shader(p.index)),
                R.UIConditional()
                .Bind(self, "is_rt")
                .append(
                    R.UIOptions()
                    .Style("select")
                    .Label("Denoiser")
                    .Items(["none", "oidn", "optix"])
                    .BindIndex(self, "denoiser_index"),
                ),
            ]
            for name, declared in self.window.render_config.parameters.items():
                if declared["kind"] != "uniform":
                    continue
                widgets.append(self._parameter_widget(name))
            self.ui_window.append(*widgets)

    def get_ui_windows(self):
        self.build()
        if self.ui_window:
            return [self.ui_window]
        return []

    def close(self):
        self.reset()
