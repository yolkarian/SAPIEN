#ifndef SAPIEN_RT_GEOMETRY_GLSL
#define SAPIEN_RT_GEOMETRY_GLSL

// One per shape in the TLAS order: its mesh and its record in materials[].
struct GeometryInstance {
  uint geometryIndex;
  uint materialIndex;
  int padding0;
  int padding1;
};

#include "./material.glsl"

#endif
