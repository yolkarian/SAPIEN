"""Generate curated Sphinx autodoc pages for the supported Python API."""

from __future__ import annotations

from pathlib import Path


API_DIR = Path(__file__).resolve().parent / "source" / "apidoc"
PAGES = {
    "modules.rst": """Public Python API
=================

The pages below document supported user-facing modules. Compatibility aliases
and implementation modules such as ``sapien.core``, ``sapien.pysapien``,
``sapien.wrapper``, and ``sapien.internal_renderer`` are intentionally omitted;
new code should use ``sapien``, ``sapien.physx``, and ``sapien.render``.

.. toctree::
   :maxdepth: 2

   sapien
   sapien.physx
   sapien.render
   sapien.math
   sapien.sensor
   sapien.utils
   sapien.asset
""",
    "sapien.rst": """Core ``sapien`` API
===================

.. automodule:: sapien
   :members: Scene, Entity, Component, System, Pose, Device, CudaArray, ActorBuilder, ArticulationBuilder, PinocchioModel, profile, set_log_level
   :imported-members:
   :show-inheritance:
   :undoc-members:
""",
    "sapien.physx.rst": """``sapien.physx``
=================

.. automodule:: sapien.physx
   :members:
   :imported-members:
   :exclude-members: Path, PhysxEngine, ZipFile, platform, requests, set_locked_motion_axes
   :show-inheritance:
   :undoc-members:

Additional native method
------------------------

.. py:method:: PhysxRigidDynamicComponent.set_locked_motion_axes(axes)

   Lock selected rigid-body degrees of freedom.

   :param axes: Six Boolean values in linear X, Y, Z, then angular X, Y, Z
      order. ``True`` locks the corresponding axis.
""",
    "sapien.render.rst": """``sapien.render``
=================

.. automodule:: sapien.render
   :members:
   :imported-members:
   :exclude-members: SapienRenderer
   :show-inheritance:
   :undoc-members:
""",
    "sapien.math.rst": """``sapien.math``
=================

The math helpers are exposed publicly as ``sapien.math``. Their implementation
module name is shown below because they are native bindings.

.. automodule:: sapien.pysapien.math
   :members:
   :show-inheritance:
   :undoc-members:

Coordinate-frame constants
--------------------------

.. py:data:: sapien.math.pose_gl_to_ros

   Fixed pose that converts the renderer/OpenGL camera convention to the
   robotics camera convention.

.. py:data:: sapien.math.pose_ros_to_gl

   Inverse fixed pose that converts the robotics camera convention to the
   renderer/OpenGL camera convention.
""",
    "sapien.sensor.rst": """``sapien.sensor``
=================

.. automodule:: sapien.sensor
   :members:
   :imported-members:
   :show-inheritance:
   :undoc-members:
""",
    "sapien.utils.rst": """``sapien.utils``
================

.. automodule:: sapien.utils
   :members:
   :imported-members:
   :show-inheritance:
   :undoc-members:
""",
    "sapien.asset.rst": """``sapien.asset``
================

.. automodule:: sapien.asset
   :members:
   :show-inheritance:
   :undoc-members:
""",
}


def main() -> None:
    """Replace generated API pages with the supported public module set."""
    API_DIR.mkdir(parents=True, exist_ok=True)
    for path in API_DIR.glob("*.rst"):
        path.unlink()
    for filename, content in PAGES.items():
        (API_DIR / filename).write_text(content)


if __name__ == "__main__":
    main()
