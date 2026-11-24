#version 460
#extension GL_EXT_ray_tracing : require

#include "ray.glsl"
#include "push_constant.glsl"

layout(location = 0) rayPayloadInEXT Ray ray;

void main() {
  // the shadow ray got through: transparent occluders already scaled its transmittance
  ray.isShadow = 0;
}
