from warnings import warn
from importlib.resources import files
import os
from pathlib import Path
from .version import __version__

os.environ["SAPIEN_PACKAGE_PATH"] = os.path.dirname(__file__)
from . import _oidn_tricks

from . import pysapien

from .pysapien import Entity, Component, System, CudaArray, Pose, Device
from .pysapien import profile
from .pysapien import set_log_level
from .pysapien import math

from .pysapien import simsense

from . import physx
from . import render


def get_live_resources() -> dict[str, dict[str, object]]:
    """Snapshot of render and PhysX resources that block job-scope shutdown."""
    return {
        "render": render.get_live_resources(),
        "physx": physx.get_live_resources(),
    }


def can_shutdown() -> bool:
    """Whether :func:`shutdown` can complete without invalidating caller-owned objects."""
    return render.can_shutdown() and physx.can_shutdown()


def shutdown() -> None:
    """Terminally shut down SAPIEN-owned render then PhysX state for one job.

    The preflight is side-effect-free; if either subsystem still has caller-owned
    resources, nothing is shut down. CUDA primary context state shared with Torch/JAX
    is intentionally preserved.
    """
    resources = get_live_resources()
    if not can_shutdown():
        raise RuntimeError(f"SAPIEN resources are still alive: {resources}")
    render.shutdown()
    physx.shutdown()

from . import _vulkan_tricks

from .wrapper.scene import Scene, SceneConfig, Widget
from .wrapper.engine import Engine
from .wrapper.renderer import SapienRenderer
from .wrapper.actor_builder import ActorBuilder
from .wrapper.articulation_builder import ArticulationBuilder
from .wrapper.pinocchio_model import PinocchioModel

try:
    render.set_imgui_ini_filename(str(Path.home() / ".sapien" / "imgui.ini"))
    pysapien.render._internal_set_shader_search_path(
        str(files("sapien").joinpath("vulkan_shader"))
    )
    render.set_viewer_shader_dir("default")
    render.set_camera_shader_dir("default")
except RuntimeError:
    pass

from . import utils
from . import asset
