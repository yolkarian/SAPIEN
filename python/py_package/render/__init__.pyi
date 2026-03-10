from __future__ import annotations
import numpy
import pybind11_stubgen.typing_ext
import sapien.pysapien
import sapien.pysapien.internal_renderer
import typing

# Re-exported from sapien.pysapien.render
from sapien.pysapien.render import RenderBodyComponent as RenderBodyComponent
from sapien.pysapien.render import RenderCameraComponent as RenderCameraComponent
from sapien.pysapien.render import RenderCameraGroup as RenderCameraGroup
from sapien.pysapien.render import RenderCubemap as RenderCubemap
from sapien.pysapien.render import RenderCudaMeshComponent as RenderCudaMeshComponent
from sapien.pysapien.render import RenderDirectionalLightComponent as RenderDirectionalLightComponent
from sapien.pysapien.render import RenderLightComponent as RenderLightComponent
from sapien.pysapien.render import RenderMaterial as RenderMaterial
from sapien.pysapien.render import RenderParallelogramLightComponent as RenderParallelogramLightComponent
from sapien.pysapien.render import RenderPointCloudComponent as RenderPointCloudComponent
from sapien.pysapien.render import RenderPointLightComponent as RenderPointLightComponent
from sapien.pysapien.render import RenderSceneLoaderNode as RenderSceneLoaderNode
from sapien.pysapien.render import RenderShape as RenderShape
from sapien.pysapien.render import RenderShapeBox as RenderShapeBox
from sapien.pysapien.render import RenderShapeCapsule as RenderShapeCapsule
from sapien.pysapien.render import RenderShapeCylinder as RenderShapeCylinder
from sapien.pysapien.render import RenderShapePlane as RenderShapePlane
from sapien.pysapien.render import RenderShapePrimitive as RenderShapePrimitive
from sapien.pysapien.render import RenderShapeSphere as RenderShapeSphere
from sapien.pysapien.render import RenderShapeTriangleMesh as RenderShapeTriangleMesh
from sapien.pysapien.render import RenderShapeTriangleMeshPart as RenderShapeTriangleMeshPart
from sapien.pysapien.render import RenderSpotLightComponent as RenderSpotLightComponent
from sapien.pysapien.render import RenderSystem as RenderSystem
from sapien.pysapien.render import RenderSystemGroup as RenderSystemGroup
from sapien.pysapien.render import RenderTexture as RenderTexture
from sapien.pysapien.render import RenderTexture2D as RenderTexture2D
from sapien.pysapien.render import RenderTexturedLightComponent as RenderTexturedLightComponent
from sapien.pysapien.render import RenderVRDisplay as RenderVRDisplay
from sapien.pysapien.render import RenderWindow as RenderWindow
from sapien.pysapien.render import SapienRenderer as SapienRenderer
from sapien.pysapien.render import clear_cache as clear_cache
from sapien.pysapien.render import enable_vr as enable_vr
from sapien.pysapien.render import get_camera_shader_dir as get_camera_shader_dir
from sapien.pysapien.render import get_device_summary as get_device_summary
from sapien.pysapien.render import get_imgui_ini_filename as get_imgui_ini_filename
from sapien.pysapien.render import get_msaa as get_msaa
from sapien.pysapien.render import get_ray_tracing_denoiser as get_ray_tracing_denoiser
from sapien.pysapien.render import get_ray_tracing_dof_aperture as get_ray_tracing_dof_aperture
from sapien.pysapien.render import get_ray_tracing_dof_plane as get_ray_tracing_dof_plane
from sapien.pysapien.render import get_ray_tracing_path_depth as get_ray_tracing_path_depth
from sapien.pysapien.render import get_ray_tracing_samples_per_pixel as get_ray_tracing_samples_per_pixel
from sapien.pysapien.render import get_viewer_shader_dir as get_viewer_shader_dir
from sapien.pysapien.render import get_vr_action_manifest_filename as get_vr_action_manifest_filename
from sapien.pysapien.render import get_vr_enabled as get_vr_enabled
from sapien.pysapien.render import load_scene as load_scene
from sapien.pysapien.render import set_camera_shader_dir as set_camera_shader_dir
from sapien.pysapien.render import set_global_config as set_global_config
from sapien.pysapien.render import set_imgui_ini_filename as set_imgui_ini_filename
from sapien.pysapien.render import set_log_level as set_log_level
from sapien.pysapien.render import set_msaa as set_msaa
from sapien.pysapien.render import set_picture_format as set_picture_format
from sapien.pysapien.render import set_ray_tracing_denoiser as set_ray_tracing_denoiser
from sapien.pysapien.render import set_ray_tracing_dof_aperture as set_ray_tracing_dof_aperture
from sapien.pysapien.render import set_ray_tracing_dof_plane as set_ray_tracing_dof_plane
from sapien.pysapien.render import set_ray_tracing_path_depth as set_ray_tracing_path_depth
from sapien.pysapien.render import set_ray_tracing_samples_per_pixel as set_ray_tracing_samples_per_pixel
from sapien.pysapien.render import set_viewer_shader_dir as set_viewer_shader_dir
from sapien.pysapien.render import set_vr_action_manifest_filename as set_vr_action_manifest_filename
from sapien.pysapien.render import _internal_set_shader_search_path as _internal_set_shader_search_path
from sapien.pysapien.render import _force_vr_shutdown as _force_vr_shutdown
