(migration_index)=

# Migration notes

This page summarizes the major API differences between older SAPIEN 1.x/2.x
examples and the current SAPIEN 3-style Python API in this repository.

## Scene creation

Old examples often create an engine, renderer, and scene separately:

```python
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
scene = engine.create_scene()
```

Current code should create scenes directly:

```python
scene = sapien.Scene()
```

`Engine` and `SapienRenderer` remain as compatibility wrappers, but both are
deprecated for normal Python usage. Configure PhysX through `sapien.physx` and
rendering through `sapien.render` before creating systems/cameras when needed.

## Entity/component model

SAPIEN 3 exposes entities and components directly. `ActorBuilder` still exists,
but `builder.build(...)` returns a `sapien.Entity` with a PhysX component and
optional render component attached.

```python
entity = scene.create_actor_builder().build(name="box")
body = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
render_body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
```

Velocity, mass, damping, collision shapes, and forces are properties/methods of
PhysX components, not of the entity itself. The entity stores the world pose.

## Modules and names

Prefer:

```python
import sapien
import sapien.physx
import sapien.render
```

`sapien.core` is kept as a compatibility alias, but new examples should use
`import sapien` and the explicit `sapien.physx` / `sapien.render`
namespaces. Physical materials are `sapien.physx.PhysxMaterial`. Render
materials are `sapien.render.RenderMaterial`.

## Cameras

Cameras are `sapien.render.RenderCameraComponent` objects attached to entities.
Use `scene.add_camera` for a free camera or `scene.add_mounted_camera` to add
a camera component to an existing entity.

```python
camera = scene.add_camera("cam", 640, 480, 1.0, 0.01, 100.0)
camera.local_pose = sapien.Pose([-3, 0, 1])

mount = scene.create_actor_builder().build_kinematic(name="mount")
mounted = scene.add_mounted_camera("mounted", mount, sapien.Pose(), 640, 480, 1.0, 0.01, 100.0)
```

The old `CameraEntity` name, `camera.set_parent` workflow, and
`get_float_texture` / `get_uint32_texture` / `get_color_rgba` accessors are
obsolete. Use `camera.get_picture(name)` and `camera.get_picture_cuda(name)`.

## Rendering configuration

Rendering configuration is per consumer through `sapien.render.RenderConfig`;
the process-wide setters of earlier versions are gone. A camera, window or
Viewer copies the config it is given.

| Removed | Use |
|---|---|
| `set_camera_shader_dir` / `set_viewer_shader_dir` | `RenderConfig(pack)`; `set_default_render_config("camera" / "viewer", cfg)` |
| `RenderCameraComponent(w, h, shader_dir)`, `RenderWindow(w, h, shader_dir)`, `Viewer(shader_dir=...)` | the same constructors with `render_config=cfg` |
| `RenderWindow.set_shader_dir` | `RenderWindow.set_render_config(cfg)` |
| `set_ray_tracing_samples_per_pixel` / `..._path_depth` / `..._denoiser` / `..._dof_aperture` / `..._dof_plane` | `cfg["spp"]`, `cfg["maxDepth"]`, `cfg.denoiser`, `cfg["aperture"]`, `cfg["focusPlane"]` |
| `set_picture_format(name, format)` | `cfg.picture_formats = {name: format}` |
| `set_msaa` / `get_msaa` | `cfg.msaa` |
| `cfg.defines = {...}` | parameters of the pack's `pack.json`; `cfg["shadows"] = True`, `cfg["shading"] = "pbr"` |
| `camera.set_property(name, value)` / `camera.get_property(name)` | unchanged; both now read the pack's `pack.json` |

```python
config = sapien.render.RenderConfig("rt")
config["spp"] = 32
config["maxDepth"] = 8
config.denoiser = "oidn"
config.picture_formats = {"Color": "r16g16b16a16Sfloat"}  # rasterization only
camera = scene.add_camera("camera", 640, 480, 1.0, 0.01, 100.0, render_config=config)

sapien.render.set_default_render_config("viewer", sapien.render.RenderConfig("rt"))
```

### `pack.json` and the removed `defines`

Every shader pack, built-in or not, now needs a `pack.json` next to its entry
point. It declares the parameters a config may assign (their kind, type, default,
range and description) and optional presets. A pack without one raises
`RuntimeError` naming the directory. `RenderConfig` validates every assignment
against it immediately, so a typo, a wrong type or an out-of-range value fails
where it is written; `cfg[name]` returns the effective value (assignment or
default), `del cfg[name]` reverts to the default, and `cfg.parameters` /
`cfg.presets` expose the manifest.

`cfg.defines` is gone: shader preprocessor definitions now come from variant
parameters. `cfg.defines = {"SHADING_PBR": "", "SHADOWS": ""}` becomes
`cfg["shading"] = "pbr"; cfg["shadows"] = True`, and the `rt_batch` `SPP` define
is gone too — `cfg["spp"]` is the only sampler control.

```python
config = sapien.render.RenderConfig("rt_batch", preset="pbr")
config["shadows"] = True
config["spp"] = 4
```

Defaults also moved: `spp=32`, `maxDepth=8`, `aperture=0.0`, `focusPlane=1.0`,
`russianRoulette=False`, `russianRouletteMinBounces=2`, `exposure=1.0` and
`toneMapper=2` are now the `pack.json` defaults of the `rt` pack rather than
hard-coded SAPIEN values, and the `rt_batch` `toneMapper` default is 0. A pack
that declares a different default renders that default.

`camera.set_property` accepts uniform parameters only; a variant parameter raises
telling you to set it on the `RenderConfig` before creating the camera, and
`camera.get_property` returns the runtime override or the config's effective value.

`sapien.render.clear_cache(shaders=True)` also drops the parsed manifests, so an
edited `pack.json` is read again.

Old examples may use `sapien.render_config`; that module does not exist.

## Viewer rendering

Viewer state submission is now explicit. Older loops often update the scene and
rely on `viewer.render()` to push any pending Viewer state:

```python
scene.update_render()
viewer.render()
```

Current Viewer loops submit through the Viewer itself. `viewer.render()` only
draws the most recently submitted state and UI:

```python
scene.step()
viewer.update_render()
viewer.render()
```

`Viewer.set_scenes(scenes, offsets=...)`, its implicit grid layout, and
`Viewer.scene_offset` have been removed. Use `Viewer.set_scenes(scenes)` or
`RenderCameraComponent.set_scenes(scenes)` to select base render scenes; shared
render scenes are included once and no render-layer transform is applied. The
low-level `RenderWindow.set_scenes()` API follows the same no-offset signature.
Place entities explicitly when scenes should appear spatially separated.

The old `Viewer.notify_render_update()`, `Viewer.reset_notifications()`, and
`Viewer.render_updated` notification gate have also been removed. Call
`Viewer.update_render()` at the exact state-submission boundary instead.

For PhysX GPU visualization, initialize GPU PhysX before Viewer submission and
use `viewer.configure_physx_gpu_rendering(physx_system, transport="auto")` when
an explicit choice is needed. Direct and staged transports leave CPU entity
poses stale. Call `viewer.apply_interactions()` immediately before every physics
substep when GPU dragging, queued property edits, or queued teleports are in use.
Reserve `sync_poses_gpu_to_cpu()` for explicit CPU debugging or the
`"cpu-debug"` Viewer transport.

## Lights

Light helper methods live on `Scene`:

```python
scene.set_ambient_light([0.5, 0.5, 0.5])
light = scene.add_directional_light([0, 1, -1], [1, 1, 1], shadow=True)
scene.remove_light(light)
```

The old `scene.renderer_scene.add_xxx_light` path is obsolete.

## Contacts

Contacts expose PhysX components and shapes:

```python
for contact in scene.get_contacts():
   body0, body1 = contact.bodies
   entity0 = body0.entity
   entity1 = body1.entity
   for point in contact.points:
      print(point.position, point.normal, point.impulse, point.separation)
```

The old `contact.actor0` / `contact.actor1` fields and the misspelled
`seperation` spelling should not be used.

## Robot drives

Set drive properties and targets on individual active joints:

```python
for joint in robot.active_joints:
   joint.set_drive_properties(stiffness=1000, damping=100, force_limit=1000)
   joint.set_drive_target(0.0)
   joint.set_drive_velocity_target(0.0)
```

The current API does not provide an articulation-level `robot.set_drive_target`
helper.

## GPU simulation

For PhysX GPU simulation, call `sapien.physx.enable_gpu()` before creating a
`PhysxGpuSystem`. Configure scene defaults before system creation. In batched
multi-scene GPU workflows, prefer scene environment IDs for isolation and set the
ID before adding bodies:

```python
sapien.physx.enable_gpu()

config = sapien.physx.PhysxSceneConfig()
config.num_scenes = 8  # SAPIEN derives the broadphase bits; see the RL tutorial
sapien.physx.set_scene_config(config)

physx_system = sapien.physx.PhysxGpuSystem()
scene = sapien.Scene([physx_system])
scene.get_or_assign_environment_id()

# add actors/articulations, then initialize GPU buffers
physx_system.gpu_init()
```

## Raster shader packs (breaking)

Rasterization shader packs no longer use per-object and per-material descriptor sets.
Port a custom pack by including `common/scene.glsl` once per pass (after
`#define SET_NUM n`) and reading the drawn object's `instance` and `material`, plus
`transforms[]`, `materials[]` and `textures[]`; see the interface section of
{ref}`customize_shaders`.
Packs that include `object_set.glsl` / `material_set.glsl` fail to load; there is no
compatibility layer. Per-object custom textures (`render_body.set_texture`) are now
`textures[]` slots in `int` fields and must be 2D. Rendering requires a Vulkan device with
descriptor-indexing features.
