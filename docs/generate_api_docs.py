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
   sapien.sensor
   sapien.utils
   sapien.asset
""",
    "sapien.rst": """Core ``sapien`` API
===================

.. automodule:: sapien
   :members: Scene, Entity, Component, System, Pose, Device, CudaArray, ActorBuilder, ArticulationBuilder, profile, set_log_level
   :show-inheritance:
   :undoc-members:
""",
    "sapien.physx.rst": """``sapien.physx``
=================

.. automodule:: sapien.physx
   :members:
   :exclude-members: PhysxEngine
   :show-inheritance:
   :undoc-members:
""",
    "sapien.render.rst": """``sapien.render``
=================

.. automodule:: sapien.render
   :members:
   :exclude-members: SapienRenderer
   :show-inheritance:
   :undoc-members:
""",
    "sapien.sensor.rst": """``sapien.sensor``
=================

.. automodule:: sapien.sensor
   :members:
   :show-inheritance:
   :undoc-members:
""",
    "sapien.utils.rst": """``sapien.utils``
================

.. automodule:: sapien.utils
   :members:
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
