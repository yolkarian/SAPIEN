from ..pysapien.render import *
from ..pysapien import render as _native_render


def _default_ground_texture_cache_size() -> int:
    # Lazy import avoids a package initialization cycle: sapien.render is imported
    # before the high-level Scene wrapper.
    from ..wrapper.scene import _default_ground_texture_cache_size

    return _default_ground_texture_cache_size()


def _clear_default_ground_texture_cache() -> None:
    from ..wrapper.scene import _clear_default_ground_texture_cache

    _clear_default_ground_texture_cache()


def get_live_resources() -> dict[str, object]:
    resources = dict(_native_render.get_live_resources())
    cache_size = _default_ground_texture_cache_size()
    resources["default_ground_textures"] = cache_size
    resources["external_engine_owners_excluding_default_ground_textures"] = max(
        int(resources["external_engine_owners"]) - cache_size,
        0,
    )
    return resources


def can_shutdown() -> bool:
    resources = get_live_resources()
    return resources["external_engine_owners_excluding_default_ground_textures"] == 0


def shutdown() -> None:
    resources = get_live_resources()
    if resources["external_engine_owners_excluding_default_ground_textures"] != 0:
        raise RuntimeError(f"SAPIEN render resources are still alive: {resources}")
    _clear_default_ground_texture_cache()
    _native_render.shutdown()
