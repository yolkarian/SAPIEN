#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : enable

hitAttributeEXT vec3 attribs;

#include "random.glsl"
#include "ray.glsl"
#include "geometry.glsl"
#include "light.glsl"
#include "push_constant.glsl"

layout(location = 0) rayPayloadInEXT Ray ray;

#define SET_NUM 1
#include "./scene_set.glsl"
#undef SET_NUM

#include "./shading.glsl"

void main() {
  vec3 worldPosition = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
  vec3 objectPos = gl_ObjectRayOriginEXT + gl_ObjectRayDirectionEXT * gl_HitTEXT;
  int id = pointInstances[nonuniformEXT(gl_InstanceCustomIndexEXT)];
  Point p = points[nonuniformEXT(id)].p[gl_PrimitiveID];
  vec3 worldShadingNormal = normalize(vec3((objectPos - p.position) * gl_WorldToObjectEXT));
  vec3 baseColor = p.color.xyz;
  // a rough dielectric bead whose alpha becomes transmission, tinted by its colour
  OpenPBR m = openpbrDielectric(baseColor, 0.1, 1.2, 1.0 - p.color.a);
  m.transmissionTint = m.base;

  vec3 V = normalize(-ray.direction);
  vec3 wx = vec3(1, 0, 0);
  if (abs(dot(worldShadingNormal, vec3(1, 0, 0))) > 0.95) wx = vec3(0, 1, 0);
  vec3 wy = normalize(cross(worldShadingNormal, wx));
  wx = cross(wy, worldShadingNormal);
  // a sphere's normal is its own geometric normal
  Surface s = makeSurface(worldPosition, wx, wy, worldShadingNormal, worldShadingNormal, V, m);

  ray.radiance = estimateDirect(s);
  sampleBsdf(s);
  ray.normal = worldShadingNormal;
  ray.albedo = baseColor;
  ray.segmentation = uvec4(0);
}
