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
  bool isInside = gl_HitKindEXT == gl_HitKindBackFacingTriangleEXT;
  if (isInside) {
    worldGeometricNormal = -worldGeometricNormal;
    worldShadingNormal = -worldShadingNormal;
  }
  if (dot(worldShadingNormal, worldGeometricNormal) < 0.0)
    worldShadingNormal = -worldShadingNormal;

  Material mat = materials.m[materialIndex];
  vec3 baseColor = mat.baseColor.rgb;
  float alpha = mat.baseColor.a;
  if (mat.baseColorTexture >= 0) {
    vec4 c = texture(textures[nonuniformEXT(mat.baseColorTexture)],
                      uv * mat.textureTransforms[0].zw + mat.textureTransforms[0].xy);
    baseColor *= c.rgb;
    alpha *= c.a;
  }
  float metallic = mat.metallic;
  if (mat.metallicTexture >= 0)
    metallic = texture(textures[nonuniformEXT(mat.metallicTexture)],
                       uv * mat.textureTransforms[3].zw + mat.textureTransforms[3].xy).x;
  float roughness = mat.roughness;
  if (mat.roughnessTexture >= 0)
    roughness = texture(textures[nonuniformEXT(mat.roughnessTexture)],
                        uv * mat.textureTransforms[1].zw + mat.textureTransforms[1].xy).x;
  roughness = clamp(roughness, 0.045, 1.0);
  vec3 texNormal = vec3(0.0, 0.0, 1.0);
  if (mat.normalTexture >= 0)
    texNormal = normalize(texture(textures[nonuniformEXT(mat.normalTexture)],
                                  uv * mat.textureTransforms[2].zw + mat.textureTransforms[2].xy).xyz * 2.0 - 1.0);
  vec3 emission = mat.emission.rgb;
  float strength = mat.emission.a;
  if (mat.emissionTexture >= 0)
    emission *= texture(textures[nonuniformEXT(mat.emissionTexture)],
                        uv * mat.textureTransforms[4].zw + mat.textureTransforms[4].xy).rgb;
  emission *= strength;
  float transmission = mat.transmission;
  if (mat.transmissionTexture >= 0)
    transmission = texture(textures[nonuniformEXT(mat.transmissionTexture)],
                           uv * mat.textureTransforms[5].zw + mat.textureTransforms[5].xy).x;

  // Stochastic transparency of path rays: go on in the same direction. The MIS state still
  // belongs to the vertex the ray left.
  if (rnd(ray.seed) < obj.transparency) {
    ray.origin = worldPosition;
    ray.attenuation = vec3(1.0);
    ray.normal = worldShadingNormal;
    ray.albedo = baseColor;
    ray.segmentation = obj.segmentation;
    ray.radiance = vec3(0.0);
    return;
  }

  float specular = max(mat.fresnel, 0.0);
  transmission = clamp(transmission, 0.0, 1.0);
  transmission = 1.0 - clamp(alpha, 0.0, 1.0) * (1.0 - transmission);
  metallic = clamp(metallic, 0.0, 1.0);
  float ior = clamp(mat.ior, 1.01, 4.0);
  float diffuseWeight = (1.0 - metallic) * (1.0 - transmission);
  transmission = (1.0 - metallic) * transmission;
  vec3 specularColor = (specular * 0.08) * (1.0 - metallic) + baseColor * metallic;
  vec3 diffuseColor = baseColor * diffuseWeight;

  float diffuseProb = maxVec3(diffuseColor);
  float specularProb = maxVec3(specularColor);
  float transmissionProb = transmission;
  float total = diffuseProb + specularProb + transmissionProb;
  if (total > 1e-6) {
    diffuseProb /= total; specularProb /= total; transmissionProb /= total;
  } else {
    diffuseProb = specularProb = transmissionProb = 0.0;
  }
  if (total > 1e-6) {
    if (diffuseProb > 1e-6) diffuseProb += 0.1;
    if (specularProb > 1e-6) specularProb += 0.1;
    if (transmissionProb > 1e-6) transmissionProb += 0.1;
    total = diffuseProb + specularProb + transmissionProb;
    diffuseProb /= total; specularProb /= total; transmissionProb /= total;
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
  worldTangent = normalize(worldTangent);
  worldShadingBitangent = normalize(cross(worldShadingNormal, worldTangent));
  tbn = mat3(worldTangent, worldShadingBitangent, worldShadingNormal);
  float eta = isInside ? ior : (1.0 / ior);

  Surface s;
  s.pos = worldPosition;
  s.N = worldShadingNormal;
  s.Ng = worldGeometricNormal;
  s.V = V;
  s.baseColor = baseColor;
  s.diffuseColor = diffuseColor;
  s.specularColor = specularColor;
  s.roughness = roughness;
  s.transmissionWeight = transmission;
  s.eta = eta;
  s.diffuseProb = diffuseProb;
  s.specularProb = specularProb;
  s.transmissionProb = transmissionProb;

  // emission found by this ray, weighted against light-sampling it from the previous vertex
  float emissionWeight = bsdfHitWeight(
      ray.bsdfPdf, hitEmissiveLightPdf(instanceId, gl_ObjectToWorldEXT * vec4(p0, 1.0),
                                       gl_ObjectToWorldEXT * vec4(p1, 1.0),
                                       gl_ObjectToWorldEXT * vec4(p2, 1.0), worldPosition));
  ray.radiance = emission * emissionWeight + estimateDirect(s);

  float rand = rnd(ray.seed);
  vec3 attenuation = vec3(0.0);
  vec3 L = worldShadingNormal;
  if (diffuseProb > 1e-5 && rand <= diffuseProb) {
    sampleDiffuse(ray.seed, tbn, V, diffuseColor, L, attenuation);
    attenuation /= diffuseProb;
  } else if (specularProb > 1e-5 && rand <= diffuseProb + specularProb) {
    sampleGGXReflection(ray.seed, tbn, V, roughness, specularColor, L, attenuation);
    attenuation /= specularProb;
  } else if (transmissionProb > 1e-5) {
    sampleGGXTransmission(ray.seed, tbn, V, eta, roughness, L, attenuation);
    attenuation *= transmission / transmissionProb * baseColor;
  }
  setContinuation(s, L);
  ray.attenuation = attenuation;
  ray.normal = worldShadingNormal;
  ray.albedo = baseColor;
  ray.segmentation = obj.segmentation;
}
