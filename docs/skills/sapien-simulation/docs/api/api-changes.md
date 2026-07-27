# API Changes By Release

Removals and behavior changes only, newest first. Signatures and parameter meanings are discoverable at runtime with `help(...)` or `__doc__`; this file records what you would otherwise call and find missing.

## 3.0.0+fork.14

Removed, with replacement:

| Removed | Use instead |
|---|---|
| `PhysxGpuSystem.set_scene_environment_id(s)`, bulk setter, `allow_duplicate` | `Scene.set_environment_id()`, or declare `num_scenes` |
| `Scene.environment_id` setter | `Scene.set_environment_id()`; property is read-only |
| `PhysxSceneConfig.gpu_broadphase_env_id_bits`, writable `gpu_broadphase_nb_bits_env_id_x/y/z` | `set_gpu_broadphase_env_id_bits(x, y, z)` |
| `set_gpu_broadphase_env_count`, `gpu_broadphase_env_band_count`, `gpu_broadphase_max_env_count` | declare `num_scenes` |
| module-level `broadphase_env_id_bits_for_env_count`, `max_broadphase_env_count`, `broadphase_env_id_window` | — |
| `PhysxGpuSystem.has_shared_environment_scene`, `has_managed_broadphase_env_ids`, `manages_collision_group_scene_ids`, `broadphase_env_id_window`, `broadphase_env_band_count` | — |
| `RenderMaterial.diffuse_texture` and its getter/setter | `base_color_texture` |

Behavior that no docstring will warn you about:

- Environment IDs have two disjoint modes. `num_scenes = N` makes SAPIEN the sole assigner and `Scene.set_environment_id()` raises; leaving it unset makes `Scene.get_or_assign_environment_id()` raise instead. In manual mode duplicate IDs are legal and put those scenes in one environment.
- `num_scenes` caps scenes held *at once*, not over the run; destroyed or shared scenes return their slot.
- `Scene.environment_id` is the raw ID; `PhysxGpuSystem.get_broadphase_environment_id()` is what PhysX receives. They differ only when a shared scene exists.
- `set_light_directions` rejects spot and textured lights.
- Misconfigurations raise at `PhysxGpuSystem()` construction, not at first body.

Packaging:

- PhysX Linux binaries moved to the `<version>-Linux` release tag, matching `-windows`. Affects any Docker pre-bake step; see [`../gpu-workflows.md`](../gpu-workflows.md).
- Wheels before fork.14 claimed `manylinux_2_28` but needed glibc 2.38 and failed to import on 2.28–2.37.
