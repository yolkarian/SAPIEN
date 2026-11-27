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

void unpackVertex(uint index, uint geometryIndex, out vec3 position, out vec3 normal, out vec2 uv, out vec3 tangent) {
  Vertex v = vertices[nonuniformEXT(geometryIndex)].v[index];
  position = vec3(v.x, v.y, v.z);
  normal = vec3(v.nx, v.ny, v.nz);
  uv = vec2(v.u, v.v);
  tangent = vec3(v.tx, v.ty, v.tz);
}

#include "./shading.glsl"

// Light-sampling pdf of the emissive triangle this ray hit, seen from the vertex the ray left;
// 0 when the triangle is not in the emissive table. w0, w1, w2 are its world vertices.
float hitEmissiveLightPdf(int instanceId, vec3 w0, vec3 w1, vec3 w2, vec3 worldPosition) {
  if (emissiveCount <= 0 || emissiveBase.b[instanceId] < 0) {
    return 0.0;
  }
  vec3 cr = cross(w1 - w0, w2 - w0);
  float area2 = length(cr);
  float cosL = abs(dot(cr, ray.direction)) / max(area2, 1e-30);
  if (area2 <= 0.0 || cosL <= 1e-6) {
    return 0.0;
  }
  vec3 d = worldPosition - ray.scatterOrigin;
  return emissiveLightPdf(uint(emissiveBase.b[instanceId] + gl_PrimitiveID), dot(d, d), cosL,
                          area2);
}

void main() {
  Object obj = objects.o[gl_InstanceID];
  int instanceId = gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT;
  uint geometryIndex = geometryInstances.i[instanceId].geometryIndex;
  uint materialIndex = geometryInstances.i[instanceId].materialIndex;
  uint v0 = indices[nonuniformEXT(geometryIndex)].i[3 * gl_PrimitiveID + 0];
  uint v1 = indices[nonuniformEXT(geometryIndex)].i[3 * gl_PrimitiveID + 1];
  uint v2 = indices[nonuniformEXT(geometryIndex)].i[3 * gl_PrimitiveID + 2];
  vec3 p0, p1, p2, n0, n1, n2, t0, t1, t2;
  vec2 uv0, uv1, uv2;
  unpackVertex(v0, geometryIndex, p0, n0, uv0, t0);
  unpackVertex(v1, geometryIndex, p1, n1, uv1, t1);
  unpackVertex(v2, geometryIndex, p2, n2, uv2, t2);
  float b0 = 1.0 - attribs.x - attribs.y;
  float b1 = attribs.x;
  float b2 = attribs.y;
  vec3 geometricNormal = cross(p1 - p0, p2 - p0);
  vec3 shadingNormal = obj.shadeFlat != 0 ? geometricNormal : (n0 * b0 + n1 * b1 + n2 * b2);
  if (dot(shadingNormal, shadingNormal) < 1e-3) shadingNormal = geometricNormal;
  vec2 uv = uv0 * b0 + uv1 * b1 + uv2 * b2;
  vec3 tangent = t0 * b0 + t1 * b1 + t2 * b2;
  if (length(tangent) < 0.01) tangent = normalize(p1 - p0);

  vec3 worldGeometricNormal = normalize(vec3(geometricNormal * gl_WorldToObjectEXT));
  vec3 worldShadingNormal = normalize(vec3(shadingNormal * gl_WorldToObjectEXT));
  vec3 worldPosition = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
  vec3 worldTangent = normalize(vec3(gl_ObjectToWorldEXT * vec4(tangent, 0.0)));
  // geometricNormal * gl_WorldToObjectEXT always points outward even under mirrored transforms.
  bool isInside = dot(worldGeometricNormal, gl_WorldRayDirectionEXT) > 0.0;
  if (isInside) {
    worldGeometricNormal = -worldGeometricNormal;
    worldShadingNormal = -worldShadingNormal;
  }
  if (dot(worldShadingNormal, worldGeometricNormal) < 0.0)
    worldShadingNormal = -worldShadingNormal;

  Material mat = materials.m[materialIndex];
  vec4 base = mat.baseColor;
  if (mat.baseColorTexture >= 0)
    base *= texture(textures[nonuniformEXT(mat.baseColorTexture)],
                    materialUV(mat.textureTransforms[0], uv));
  float metallic = mat.metallic;
  if (mat.metallicTexture >= 0)
    metallic = texture(textures[nonuniformEXT(mat.metallicTexture)],
                       materialUV(mat.textureTransforms[3], uv)).x;
  float roughness = mat.roughness;
  if (mat.roughnessTexture >= 0)
    roughness = texture(textures[nonuniformEXT(mat.roughnessTexture)],
                        materialUV(mat.textureTransforms[1], uv)).x;
  vec3 texNormal = vec3(0.0, 0.0, 1.0);
  if (mat.normalTexture >= 0)
    texNormal = normalize(texture(textures[nonuniformEXT(mat.normalTexture)],
                                  materialUV(mat.textureTransforms[2], uv)).xyz * 2.0 - 1.0);
  vec3 emission = mat.emissionColor * mat.emissionStrength;
  if (mat.emissionTexture >= 0)
    emission *= texture(textures[nonuniformEXT(mat.emissionTexture)],
                        materialUV(mat.textureTransforms[4], uv)).rgb;
  float transmission = mat.transmissionWeight;
  if (mat.transmissionTexture >= 0)
    transmission = texture(textures[nonuniformEXT(mat.transmissionTexture)],
                           materialUV(mat.textureTransforms[5], uv)).x;

  vec3 localPos = p0 * b0 + p1 * b1 + p2 * b2;
  vec3 prevWorldPos = instancePoint(prevTlasInstances.i[gl_InstanceID], localPos);

  // A segment inside a translucent body crossed its absorbing medium (transmission_depth).
  vec3 absorbed = isInside ? exp(-openpbrAbsorption(mat) * gl_HitTEXT) : vec3(1.0);

  // geometry_opacity and object transparency: the uncovered part lets the path ray go straight
  // on. The MIS state still belongs to the vertex the ray left.
  float coverage = clamp(base.a, 0.0, 1.0) * (1.0 - clamp(obj.transparency, 0.0, 1.0));
  if (rnd(ray.seed) >= coverage) {
    ray.origin = worldPosition;
    ray.prevWorldPos = prevWorldPos;
    ray.attenuation = absorbed;
    ray.diffuseAttenuation = vec3(0.0);
    ray.normal = worldShadingNormal;
    ray.albedo = base.rgb;
    ray.segmentation = obj.segmentation;
    ray.emission = vec3(0.0);
    ray.directDiffuse = vec3(0.0);
    ray.directSpecular = vec3(0.0);
    // the path goes straight on through the uncovered part: not a scattering event, so the AOV
    // class of the vertex the ray left carries on
    ray.scattered = 0;
    return;
  }

  vec3 V = normalize(-ray.direction);
  worldTangent -= worldShadingNormal * dot(worldShadingNormal, worldTangent);
  if (dot(worldTangent, worldTangent) < 1e-6) {
    vec3 axis = abs(worldShadingNormal.x) < 0.95 ? vec3(1, 0, 0) : vec3(0, 1, 0);
    worldTangent = cross(axis, worldShadingNormal);
  }
  worldTangent = normalize(worldTangent);
  vec3 worldShadingBitangent = normalize(cross(worldShadingNormal, worldTangent));
  mat3 tbn = mat3(worldTangent, worldShadingBitangent, worldShadingNormal);
  worldShadingNormal = normalize(tbn * texNormal);
  if (dot(worldShadingNormal, worldGeometricNormal) <= 0.0)
    worldShadingNormal = worldGeometricNormal;
  worldTangent -= worldShadingNormal * dot(worldShadingNormal, worldTangent);
  if (dot(worldTangent, worldTangent) < 1e-6) {
    vec3 axis = abs(worldShadingNormal.x) < 0.95 ? vec3(1, 0, 0) : vec3(0, 1, 0);
    worldTangent = cross(axis, worldShadingNormal);
  }
  worldTangent = normalize(worldTangent);
  worldShadingBitangent = normalize(cross(worldShadingNormal, worldTangent));

  OpenPBR m = openpbrSurface(mat, base.rgb, metallic, roughness, transmission, isInside);
  Surface s = makeSurface(worldPosition, worldTangent, worldShadingBitangent, worldShadingNormal,
                          worldGeometricNormal, V, m);

  // emission found by this ray, weighted against light-sampling it from the previous vertex
  float emissionWeight = bsdfHitWeight(
      ray.bsdfPdf, hitEmissiveLightPdf(instanceId, gl_ObjectToWorldEXT * vec4(p0, 1.0),
                                       gl_ObjectToWorldEXT * vec4(p1, 1.0),
                                       gl_ObjectToWorldEXT * vec4(p2, 1.0), worldPosition));
  ray.emission = emission * openpbrEmissionFactor(m, s.wo.z) * emissionWeight * absorbed;
  DirectLighting dl = estimateDirectSplit(s);
  ray.directDiffuse = dl.diffuse * absorbed;
  ray.directSpecular = dl.specular * absorbed;
  sampleBsdf(s);
  ray.attenuation *= absorbed;
  ray.diffuseAttenuation *= absorbed;
  ray.prevWorldPos = prevWorldPos;
  ray.normal = worldShadingNormal;
  ray.albedo = base.rgb;
  ray.segmentation = obj.segmentation;
  ray.scattered = 1;
}
