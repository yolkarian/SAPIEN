from __future__ import annotations
import numpy
import pybind11_stubgen.typing_ext
import sapien.pysapien
import sapien.pysapien_pinocchio
import typing

import platform

M = typing.TypeVar("M", bound=int)

# Re-exported from sapien.pysapien.physx
from sapien.pysapien.physx import PhysxArticulation as PhysxArticulation
from sapien.pysapien.physx import PhysxArticulationJoint as PhysxArticulationJoint
from sapien.pysapien.physx import PhysxArticulationLinkComponent as PhysxArticulationLinkComponent
from sapien.pysapien.physx import PhysxBaseComponent as PhysxBaseComponent
from sapien.pysapien.physx import PhysxBodyConfig as PhysxBodyConfig
from sapien.pysapien.physx import PhysxCollisionShape as PhysxCollisionShape
from sapien.pysapien.physx import PhysxCollisionShapeBox as PhysxCollisionShapeBox
from sapien.pysapien.physx import PhysxCollisionShapeCapsule as PhysxCollisionShapeCapsule
from sapien.pysapien.physx import PhysxCollisionShapeConvexMesh as PhysxCollisionShapeConvexMesh
from sapien.pysapien.physx import PhysxCollisionShapeCylinder as PhysxCollisionShapeCylinder
from sapien.pysapien.physx import PhysxCollisionShapePlane as PhysxCollisionShapePlane
from sapien.pysapien.physx import PhysxCollisionShapeSphere as PhysxCollisionShapeSphere
from sapien.pysapien.physx import PhysxCollisionShapeTriangleMesh as PhysxCollisionShapeTriangleMesh
from sapien.pysapien.physx import PhysxContact as PhysxContact
from sapien.pysapien.physx import PhysxContactPoint as PhysxContactPoint
from sapien.pysapien.physx import PhysxCpuSystem as PhysxCpuSystem
from sapien.pysapien.physx import PhysxDistanceJointComponent as PhysxDistanceJointComponent
from sapien.pysapien.physx import PhysxDriveComponent as PhysxDriveComponent
from sapien.pysapien.physx import PhysxEngine as PhysxEngine
from sapien.pysapien.physx import PhysxGearComponent as PhysxGearComponent
from sapien.pysapien.physx import PhysxJointComponent as PhysxJointComponent
from sapien.pysapien.physx import PhysxMaterial as PhysxMaterial
from sapien.pysapien.physx import PhysxRayHit as PhysxRayHit
from sapien.pysapien.physx import PhysxRigidBaseComponent as PhysxRigidBaseComponent
from sapien.pysapien.physx import PhysxRigidBodyComponent as PhysxRigidBodyComponent
from sapien.pysapien.physx import PhysxRigidDynamicComponent as PhysxRigidDynamicComponent
from sapien.pysapien.physx import PhysxRigidStaticComponent as PhysxRigidStaticComponent
from sapien.pysapien.physx import PhysxSDFConfig as PhysxSDFConfig
from sapien.pysapien.physx import PhysxSceneConfig as PhysxSceneConfig
from sapien.pysapien.physx import PhysxShapeConfig as PhysxShapeConfig
from sapien.pysapien.physx import PhysxSystem as PhysxSystem
from sapien.pysapien.physx import _enable_gpu as _enable_gpu
from sapien.pysapien.physx import get_body_config as get_body_config
from sapien.pysapien.physx import get_default_material as get_default_material
from sapien.pysapien.physx import get_scene_config as get_scene_config
from sapien.pysapien.physx import get_sdf_config as get_sdf_config
from sapien.pysapien.physx import get_shape_config as get_shape_config
from sapien.pysapien.physx import is_gpu_enabled as is_gpu_enabled
from sapien.pysapien.physx import set_body_config as set_body_config
from sapien.pysapien.physx import set_default_material as set_default_material
from sapien.pysapien.physx import set_gpu_memory_config as set_gpu_memory_config
from sapien.pysapien.physx import set_scene_config as set_scene_config
from sapien.pysapien.physx import set_sdf_config as set_sdf_config
from sapien.pysapien.physx import set_shape_config as set_shape_config
from sapien.pysapien.physx import version as version

if platform.system() != "Darwin":
    from sapien.pysapien.physx import PhysxGpuContactBodyImpulseQuery as PhysxGpuContactBodyImpulseQuery
    from sapien.pysapien.physx import PhysxGpuContactPairImpulseQuery as PhysxGpuContactPairImpulseQuery
    from sapien.pysapien.physx import PhysxGpuSystem as PhysxGpuSystem

# Additional function defined in sapien.physx (not from pysapien.physx)
def enable_gpu() -> None: ...
