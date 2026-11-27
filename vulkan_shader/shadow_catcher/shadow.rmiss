#version 460
#extension GL_EXT_ray_tracing : require

#include "ray.glsl"
#include "push_constant.glsl"

layout(location = 0) rayPayloadInEXT Ray ray;

void main() {
  ray.isShadow = 0;
}
