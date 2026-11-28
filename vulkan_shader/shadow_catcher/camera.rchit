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
#include "../common/openpbr.glsl"

layout(location = 0) rayPayloadInEXT Ray ray;

layout(set = 1, binding = 0) uniform accelerationStructureEXT tlas;

layout(set = 1, binding = 1) readonly buffer GeometryInstances
{
  GeometryInstance i[];
} geometryInstances;

// one record per material (common/material.glsl); binding 3 is unused
layout(set = 1, binding = 2) readonly buffer Materials
{
  Material m[];
} materials;

layout(set = 1, binding = 4) uniform sampler2D textures[];

layout(set = 1, binding = 5) readonly buffer PointLights
{
  PointLight l[];
} pointLights;

layout(set = 1, binding = 6) readonly buffer DirectionalLights
{
  DirectionalLight l[];
} directionalLights;

layout(set = 1, binding = 7) readonly buffer SpotLights
{
  SpotLight l[];
} spotLights;

layout(set = 1, binding = 10) uniform samplerCube samplerEnvironment;  // TODO: check this

layout(set = 1, binding = 12) readonly buffer ParallelogramLights
{
  ParallelogramLight l[];
} parallelogramLights;

struct Vertex {
  float x, y, z;
  float nx, ny, nz;
  float u, v;

  float tx, ty, tz;
  float bx, by, bz;
};

layout(std430, set = 1, binding = 8) readonly buffer Vertices {
  Vertex v[];
} vertices[];

layout(set = 1, binding = 9) readonly buffer Indices {
  uint i[];
} indices[];

struct Object {
  uvec4 segmentation;
  float transparency;
  int shadeFlat;
  int padding0;
  int padding1;
};

layout(set = 1, binding = 11) readonly buffer Objects {
  Object o[];
} objects;


void unpackVertex(uint index, uint geometryIndex, out vec3 position, out vec3 normal, out vec2 uv, out vec3 tangent) {
  Vertex v = vertices[nonuniformEXT(geometryIndex)].v[index];
  position = vec3(v.x, v.y, v.z);
  normal = vec3(v.nx, v.ny, v.nz);
  uv = vec2(v.u, v.v);
  tangent = vec3(v.tx, v.ty, v.tz);
}

float maxVec3(vec3 v) {
  return max(v.x, max(v.y, v.z));
}

// Shadow rays share payload 0 so camera.rahit handles transparent occluders and back-face culling.
vec3 traceShadow(vec3 origin, vec3 L, float tMax, uint mask) {
  if (tMax <= 0.001) {
    return vec3(1.0);
  }
  ray.isShadow = 1;
  ray.transmittance = vec3(1.0);
  traceRayEXT(tlas, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT, mask,
              0, 0, 1, origin, 0.001, L, tMax, 0);
  vec3 transmittance = ray.isShadow == 0 ? ray.transmittance : vec3(0.0);
  ray.isShadow = 0;
  return transmittance;
}

// the BSDF times cos toward world direction L at a surface with shading frame `frame`
vec3 catcherBsdf(OpenPBR m, mat3 frame, vec3 wo, vec3 L) {
  return openpbrEvalReflection(m, wo, L * frame);
}


vec3 traceDirectionalLights(vec3 pos, OpenPBR m, mat3 frame, vec3 wo, out vec3 resultNoShadow) {
  vec3 result = vec3(0.0);
  resultNoShadow = vec3(0.0);

  for (uint i = 0; i < directionalLightCount; ++i) {
    DirectionalLight light = directionalLights.l[i];
    vec3 emission = light.rgb;
    vec3 L = normalize(-light.direction);
    if (light.softness > 0.0) {
      vec3 tangent = abs(L.z) < 0.999 ? normalize(cross(L, vec3(0, 0, 1)))
                                     : vec3(1, 0, 0);
      vec3 bitangent = cross(L, tangent);
      float radius = sqrt(rnd(ray.seed)) * light.softness;
      float angle = 2.0 * M_PI * rnd(ray.seed);
      L = normalize(L + radius * (cos(angle) * tangent + sin(angle) * bitangent));
    }

    vec3 intensity = catcherBsdf(m, frame, wo, L) * emission;
    resultNoShadow += intensity;
    if (maxVec3(intensity) > 0.0) {
      result += intensity * traceShadow(pos, L, 1.0e6, 0x02u);
    }
  }
  return result;
}


vec3 traceParallelogramLights(vec3 pos, OpenPBR m, mat3 frame, vec3 wo, out vec3 resultNoShadow) {
  vec3 result = vec3(0.0);
  resultNoShadow = vec3(0.0);
  for (uint i = 0; i < parallelogramLightCount; ++i) {
    ParallelogramLight light = parallelogramLights.l[i];
    vec3 emission = light.rgb;
    vec3 d = light.position + rnd(ray.seed) * light.edge0 + rnd(ray.seed) * light.edge1 - pos;
    float d2 = max(dot(d, d), 1e-6);
    float dist = sqrt(d2);
    vec3 L = d / dist;

    vec3 lnormal = cross(light.edge0, light.edge1);
    float area = length(lnormal);
    if (area <= 1e-12) {
      continue;
    }
    lnormal /= area;
    float dotlnld = dot(lnormal, L); // light normal dot light direction

    if (dotlnld > 0.0) {
      emission = emission * dotlnld * area / d2;
      vec3 intensity = catcherBsdf(m, frame, wo, L) * emission;
      resultNoShadow += intensity;
      if (maxVec3(intensity) > 0.0) {
        result += intensity * traceShadow(pos, L, dist - 0.001, 0x02u);
      }
    }
  }
  return result;
}

vec3 tracePointLights(vec3 pos, OpenPBR m, mat3 frame, vec3 wo, out vec3 resultNoShadow) {
  vec3 result = vec3(0.0);
  resultNoShadow = vec3(0.0);

  for (uint i = 0; i < pointLightCount; ++i) {
    PointLight light = pointLights.l[i];
    vec3 emission = light.rgb;
    vec3 target = light.position;
    vec3 toCenter = target - pos;
    if (light.radius > 0.0 && dot(toCenter, toCenter) > 1e-12) {
      vec3 dir = normalize(toCenter);
      vec3 tangent = abs(dir.z) < 0.999 ? normalize(cross(dir, vec3(0, 0, 1))) : vec3(1, 0, 0);
      vec3 bitangent = cross(dir, tangent);
      float r = sqrt(rnd(ray.seed)) * light.radius;
      float angle = 2.0 * M_PI * rnd(ray.seed);
      target += r * (cos(angle) * tangent + sin(angle) * bitangent);
    }
    vec3 d = target - pos;
    float d2 = max(dot(d, d), 1e-6);
    float dist = sqrt(d2);
    vec3 L = d / dist;

    vec3 intensity = catcherBsdf(m, frame, wo, L) * emission / d2;
    resultNoShadow += intensity;
    if (maxVec3(intensity) > 0.0) {
      result += intensity * traceShadow(pos, L, dist - 0.001, 0x02u);
    }
  }
  return result;
}

vec3 traceSpotLights(vec3 pos, OpenPBR m, mat3 frame, vec3 wo, out vec3 resultNoShadow) {
  vec3 result = vec3(0.0);
  resultNoShadow = vec3(0.0);

  for (uint i = 0; i < spotLightCount; ++i) {
    SpotLight light = spotLights.l[i];

    vec3 emission = light.rgb;
    vec3 centerDir = vec3(0.0, 0.0, -1.0) * mat3(light.viewMat);  // model rotation * forward
    vec3 target = light.position;
    vec3 toCenter = target - pos;
    if (light.radius > 0.0 && dot(toCenter, toCenter) > 1e-12) {
      vec3 dir = normalize(toCenter);
      vec3 tangent = abs(dir.z) < 0.999 ? normalize(cross(dir, vec3(0, 0, 1))) : vec3(1, 0, 0);
      vec3 bitangent = cross(dir, tangent);
      float r = sqrt(rnd(ray.seed)) * light.radius;
      float angle = 2.0 * M_PI * rnd(ray.seed);
      target += r * (cos(angle) * tangent + sin(angle) * bitangent);
    }
    vec3 d = target - pos;
    float d2 = max(dot(d, d), 1e-6);
    float dist = sqrt(d2);
    vec3 L = d / dist;

    float cf1 = cos(light.fovInner / 2) + 1e-5;
    float cf2 = cos(light.fovOuter / 2);

    float visibility = clamp((dot(-L, centerDir) - cf2) / (cf1 - cf2), 0.0, 1.0);

    int texId = light.textureId;

    vec3 texColor = vec3(1.0);
    if (texId >= 0) {
      vec4 texcoord = light.projMat * (light.viewMat * vec4(pos, 1));
      texcoord /= texcoord.w;
      vec2 uv = texcoord.xy * 0.5 + 0.5;
      texColor = texture(textures[nonuniformEXT(texId)], uv).rgb;
    }

    float maxTexColor = maxVec3(texColor);

    if (visibility > 1e-5 && maxTexColor > 1e-5) {
      vec3 intensity = catcherBsdf(m, frame, wo, L) * emission / d2 * visibility * texColor;
      resultNoShadow += intensity;
      if (maxVec3(intensity) > 0.0) {
        result += intensity * traceShadow(pos, L, dist - 0.001, 0x02u);
      }
    }
  }
  return result;
}


void main() {
  float inDepth = ray.depth;
  // a hit, whatever the surface's alpha: the catcher below reports a shadowness of 0 on an
  // unshadowed pixel, and the ray generator still has to know the camera ray reached it
  ray.hit = 1;

  Object obj = objects.o[gl_InstanceID];

  int instanceId = gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT;
  uint geometryIndex = geometryInstances.i[instanceId].geometryIndex;
  uint materialIndex = geometryInstances.i[instanceId].materialIndex;

  uint v0 = indices[nonuniformEXT(geometryIndex)].i[3 * gl_PrimitiveID + 0];
  uint v1 = indices[nonuniformEXT(geometryIndex)].i[3 * gl_PrimitiveID + 1];
  uint v2 = indices[nonuniformEXT(geometryIndex)].i[3 * gl_PrimitiveID + 2];

  vec3 p0, p1, p2;
  vec3 n0, n1, n2;
  vec2 uv0, uv1, uv2;
  vec3 t0, t1, t2;
  unpackVertex(v0, geometryIndex, p0, n0, uv0, t0);
  unpackVertex(v1, geometryIndex, p1, n1, uv1, t1);
  unpackVertex(v2, geometryIndex, p2, n2, uv2, t2);

  float b0 = 1.0 - attribs.x - attribs.y;
  float b1 = attribs.x;
  float b2 = attribs.y;

  vec3 geometricNormal = cross(p1 - p0, p2 - p0);
  vec3 shadingNormal = obj.shadeFlat != 0 ? geometricNormal : (n0 * b0 + n1 * b1 + n2 * b2);
  if (dot(shadingNormal, shadingNormal) < 1e-3) {
    shadingNormal = geometricNormal;
  }

  vec2 uv = uv0 * b0 + uv1 * b1 + uv2 * b2;
  vec3 tangent = t0 * b0 + t1 * b1 + t2 * b2;

  if (length(tangent) < 0.01) {
    tangent = normalize(p1 - p0);
  }

  vec3 worldGeometricNormal = normalize(vec3(geometricNormal * gl_WorldToObjectEXT));
  vec3 worldShadingNormal = normalize(vec3(shadingNormal * gl_WorldToObjectEXT));
  vec3 worldPosition = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
  vec3 worldTangent = normalize(vec3(gl_ObjectToWorldEXT * vec4(tangent, 0.0)));

  bool isInside = dot(worldGeometricNormal, gl_WorldRayDirectionEXT) > 0.0;
  if (isInside) {
    worldGeometricNormal = -worldGeometricNormal;
    worldShadingNormal = -worldShadingNormal;
  }
  if (dot(worldShadingNormal, worldGeometricNormal) < 0.0) {
    worldShadingNormal = -worldShadingNormal;
  }

  Material mat = materials.m[materialIndex];
  vec4 base = mat.baseColor;
  float texelAlpha = 1.0;
  if (mat.baseColorTexture >= 0) {
    vec4 texel = texture(textures[nonuniformEXT(mat.baseColorTexture)],
                         materialUV(mat.textureTransforms[0], uv));
    base *= texel;
    texelAlpha = texel.a;
  }
  vec3 baseColor = base.rgb;

  float metallic = mat.metallic;
  if (mat.metallicTexture >= 0) {
    metallic = texture(textures[nonuniformEXT(mat.metallicTexture)],
                       materialUV(mat.textureTransforms[3], uv)).x;
  }

  float roughness = mat.roughness;
  if (mat.roughnessTexture >= 0) {
    roughness = texture(textures[nonuniformEXT(mat.roughnessTexture)],
                        materialUV(mat.textureTransforms[1], uv)).x;
  }

  vec3 texNormal = vec3(0.0, 0.0, 1.0);
  if (mat.normalTexture >= 0) {
    texNormal = normalize(texture(textures[nonuniformEXT(mat.normalTexture)],
                                  materialUV(mat.textureTransforms[2], uv)).xyz * 2.0 - 1.0);
  }

  vec3 emission = mat.emissionColor * mat.emissionStrength;
  if (mat.emissionTexture >= 0) {
    emission *= texture(textures[nonuniformEXT(mat.emissionTexture)],
                        materialUV(mat.textureTransforms[4], uv)).rgb;
  }

  float transmission = mat.transmissionWeight;
  if (mat.transmissionTexture >= 0) {
    transmission = texture(textures[nonuniformEXT(mat.transmissionTexture)],
                           materialUV(mat.textureTransforms[5], uv)).x;
  }

  vec3 absorbed = isInside ? exp(-openpbrAbsorption(mat) * gl_HitTEXT) : vec3(1.0);

  // geometry_opacity and object transparency: the uncovered part lets the ray go straight on
  float coverage =
      materialCoverage(mat, texelAlpha) * (1.0 - clamp(obj.transparency, 0.0, 1.0));
  if (rnd(ray.seed) >= coverage) {
    ray.origin = worldPosition;
    ray.attenuation = absorbed;
    ray.normal = worldShadingNormal;
    ray.albedo = baseColor;
    ray.segmentation = obj.segmentation;
    ray.radiance = vec3(0.0);
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

  // Keep normal maps in the same hemisphere as the geometric surface.
  worldShadingNormal = normalize(tbn * texNormal);
  if (dot(worldShadingNormal, worldGeometricNormal) <= 0.0) {
    worldShadingNormal = worldGeometricNormal;
  }
  worldTangent -= worldShadingNormal * dot(worldShadingNormal, worldTangent);
  if (dot(worldTangent, worldTangent) < 1e-6) {
    vec3 axis = abs(worldShadingNormal.x) < 0.95 ? vec3(1, 0, 0) : vec3(0, 1, 0);
    worldTangent = cross(axis, worldShadingNormal);
  }
  worldTangent = normalize(worldTangent);
  worldShadingBitangent = normalize(cross(worldShadingNormal, worldTangent));
  tbn = mat3(worldTangent, worldShadingBitangent, worldShadingNormal);

  OpenPBR m = openpbrSurface(mat, baseColor, metallic, roughness, transmission, isInside);
  vec3 wo = V * tbn;
  wo = normalize(vec3(wo.xy, max(wo.z, 1e-4))); // a normal map can tilt N away from V
  OpenPBRSample bsdf = openpbrSample(m, wo, vec3(rnd(ray.seed), rnd(ray.seed), rnd(ray.seed)));
  vec3 L = tbn * bsdf.wi;
  bool validSample = dot(bsdf.wi, bsdf.wi) > 1e-6 &&
                     ((dot(worldGeometricNormal, L) > 0.0) != bsdf.transmitted);
  vec3 attenuation = validSample ? bsdf.weight : vec3(0.0);

  vec3 plNoShadow, dlNoShadow, slNoShadow, alNoShadow;
  vec3 elNoShadow = vec3(0.0);

  ray.radiance = (emission * openpbrEmissionFactor(m, wo.z)
    + tracePointLights(worldPosition, m, tbn, wo, plNoShadow)
    + traceDirectionalLights(worldPosition, m, tbn, wo, dlNoShadow)
    + traceSpotLights(worldPosition, m, tbn, wo, slNoShadow)
    + traceParallelogramLights(worldPosition, m, tbn, wo, alNoShadow)) * absorbed;

  ray.origin = worldPosition;
  ray.direction = validSample ? L : worldGeometricNormal;
  ray.attenuation = attenuation * absorbed;
  ray.normal = worldShadingNormal;
  ray.albedo = baseColor;
  ray.segmentation = obj.segmentation;
  ray.scattered = 1;

  if (obj.segmentation.x == 1 && inDepth == 0) {
    if (validSample && maxVec3(attenuation) > 0.0) {
      vec3 env = envmap != 0
                     ? clamp(texture(samplerEnvironment, L.xzy).rgb, vec3(0.0), vec3(100.0))
                     : ambientLight;
      elNoShadow = attenuation * env;
      // Only the primary catcher hit ignores cast_shadow=False objects (mask 0x02).
      ray.radiance += elNoShadow * traceShadow(worldPosition, L, 1e4, 0x02u) * absorbed;
    }
    vec3 radianceNoShadow = (plNoShadow + dlNoShadow + slNoShadow + alNoShadow + elNoShadow) * absorbed;
    float shadowness = 1.0 - (ray.radiance.x + ray.radiance.y + ray.radiance.z + 1e-6) /
                             (radianceNoShadow.x + radianceNoShadow.y + radianceNoShadow.z + 1e-6);
    ray.radiance = vec3(0.0);
    ray.alpha = clamp(shadowness, 0.0, 1.0);
    ray.depth = maxDepth + 1;
  } else if (!validSample || maxVec3(attenuation) <= 0.0) {
    ray.attenuation = vec3(0.0);
    ray.depth = maxDepth + 1;
  } else if (uint(inDepth) >= maxDepth) {
    vec3 env = envmap != 0
                   ? clamp(texture(samplerEnvironment, L.xzy).rgb, vec3(0.0), vec3(100.0))
                   : ambientLight;
    ray.radiance += attenuation * env * traceShadow(worldPosition, L, 1e4, 0x01u) * absorbed;
    ray.depth = maxDepth + 1;
  }
}
