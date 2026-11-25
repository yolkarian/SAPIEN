#version 450

layout (constant_id = 0) const int NUM_DIRECTIONAL_LIGHTS = 0;
layout (constant_id = 1) const int NUM_POINT_LIGHTS = 0;
layout (constant_id = 2) const int NUM_DIRECTIONAL_LIGHT_SHADOWS = 1;
layout (constant_id = 3) const int NUM_POINT_LIGHT_SHADOWS = 3;
layout (constant_id = 4) const int NUM_TEXTURED_LIGHT_SHADOWS = 1;
layout (constant_id = 5) const int NUM_SPOT_LIGHT_SHADOWS = 10;
layout (constant_id = 6) const int NUM_SPOT_LIGHTS = 0;
layout (constant_id = 7) const float ambientOcclusionStrength = 0.65;
layout (constant_id = 8) const float ambientOcclusionRadius = 0.3;

#define SET_NUM 0
#include "./scene_set.glsl"
#undef SET_NUM

#define SET_NUM 1
#include "./camera_set.glsl"
#undef SET_NUM

layout(set = 2, binding = 0) uniform sampler2D samplerAlbedo;
layout(set = 2, binding = 1) uniform sampler2D samplerPositionRaw;
layout(set = 2, binding = 2) uniform sampler2D samplerSpecular;
layout(set = 2, binding = 3) uniform sampler2D samplerNormal;
layout(set = 2, binding = 4) uniform sampler2D samplerEmission;
layout(set = 2, binding = 5) uniform sampler2D samplerGbufferDepth;
layout(set = 2, binding = 6) uniform sampler2D samplerCustom;
layout(set = 2, binding = 7) uniform usampler2D samplerMaterial;

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outLighting;

#include "../common/environment.glsl"
#include "../common/raster_lighting.glsl"

const vec2 AMBIENT_OCCLUSION_SAMPLES[12] = {
  vec2(0.2500, 0.0000), vec2(-0.3190, 0.2922), vec2(0.0488, -0.5565),
  vec2(0.4028, 0.5254), vec2(-0.7385, -0.1306), vec2(0.6996, -0.4450),
  vec2(-0.2331, 0.8673), vec2(-0.4463, -0.8526), vec2(0.9341, 0.3411),
  vec2(-0.9650, 0.3983), vec2(0.4664, -0.9990), vec2(0.3430, 1.0682)
};

float computeAmbientOcclusion(vec3 position, vec3 normal) {
  float radius = max(ambientOcclusionRadius, 0.0);
  float strength = clamp(ambientOcclusionStrength, 0.0, 1.0);
  if (radius <= 1e-5 || strength <= 1e-5) {
    return 1.0;
  }

  float viewDepth = max(abs(position.z), radius);
  vec2 projectedRadius = 0.5 *
      vec2(abs(cameraBuffer.projectionMatrix[0][0]),
           abs(cameraBuffer.projectionMatrix[1][1])) * radius / viewDepth;

  float occlusion = 0.0;
  for (int i = 0; i < 12; ++i) {
    vec2 sampleUV = inUV + AMBIENT_OCCLUSION_SAMPLES[i] * projectedRadius;
    if (any(lessThanEqual(sampleUV, vec2(0.0))) ||
        any(greaterThanEqual(sampleUV, vec2(1.0)))) {
      continue;
    }

    vec4 neighbor = texture(samplerPositionRaw, sampleUV);
    if (neighbor.w < 0.5) {
      continue;
    }

    vec3 offset = neighbor.xyz - position;
    float distanceToNeighbor = length(offset);
    if (distanceToNeighbor <= 1e-5 || distanceToNeighbor >= radius) {
      continue;
    }

    float horizon = max(dot(normal, offset / distanceToNeighbor) - 0.08, 0.0);
    float distanceWeight = 1.0 - smoothstep(0.05 * radius, radius, distanceToNeighbor);
    occlusion += horizon * distanceWeight;
  }

  occlusion = clamp(occlusion * (2.5 / 12.0), 0.0, 1.0);
  return 1.0 - strength * occlusion;
}

void main() {
  vec3 albedo = texture(samplerAlbedo, inUV).xyz;
  vec4 frm = texture(samplerSpecular, inUV); // F0, roughness, metallic, material occlusion
  float specular = max(frm.x, 0.0);
  float roughness = clamp(frm.y, 0.045, 1.0);
  float metallic = clamp(frm.z, 0.0, 1.0);

  vec3 normal = normalize(texture(samplerNormal, inUV).xyz);
  float depth = texture(samplerGbufferDepth, inUV).x;

  vec4 csPosition = cameraBuffer.projectionMatrixInverse * (vec4(inUV * 2 - 1, depth, 1));
  csPosition /= csPosition.w;

  if (depth >= 1.0) {
    outLighting = vec4(
        sapienBackgroundColor((cameraBuffer.viewMatrixInverse * csPosition).xyz), 0.0);
    return;
  }

  vec3 camDir = -normalize(csPosition.xyz);

  OpenPBR m = sapienUnpackMaterial(texture(samplerMaterial, inUV), albedo, metallic, roughness);
  bool layered = openpbrHasLayers(m);
  vec3 diffuseAlbedo = albedo * (1 - metallic);
  vec3 fresnel = sapienBaseFresnel(m, albedo, specular, metallic);

  vec4 emission = texture(samplerEmission, inUV);
  vec3 color = emission.rgb * emission.a;
  if (layered) {
    color *= openpbrEmissionFactor(m, clamp(dot(normal, camDir), 1e-6, 1.0));
  }

  sapienAddDirectLighting(color, csPosition.xyz, normal, camDir, diffuseAlbedo, roughness,
                          fresnel, m, true);

  // environmental light
  vec3 wnormal = mat3(cameraBuffer.viewMatrixInverse) * normal;
  vec3 worldCamDir = mat3(cameraBuffer.viewMatrixInverse) * camDir;
  float dotNV = clamp(dot(wnormal, worldCamDir), 0.0, 1.0);
  vec3 environmentFresnel =
      sapienFresnelSchlickRoughness(fresnel, roughness, dotNV);
  // screen-space ambient occlusion times the material's occlusion texture
  float ambientOcclusion = computeAmbientOcclusion(csPosition.xyz, normal) * frm.w;
  vec3 diffuseIndirect = ambientOcclusion *
                         sapienDiffuseIBL((1.0 - environmentFresnel) * diffuseAlbedo, wnormal);
  vec3 specularIndirect = mix(1.0, ambientOcclusion, roughness) *
                          sapienSpecularIBL(fresnel, roughness, wnormal, worldCamDir);
  vec3 ambientIndirect = ambientOcclusion * sceneBuffer.ambientLight.rgb *
                         (1.0 - environmentFresnel) * diffuseAlbedo;
  if (layered) {
    float coatRoughness = sqrt(m.coatAlpha);
    vec3 irradiance = ambientOcclusion * (sapienDiffuseIBL(vec3(1.0), wnormal) +
                                          sceneBuffer.ambientLight.rgb);
    vec3 coatReflection = mix(1.0, ambientOcclusion, coatRoughness) *
                          sapienSpecularIBL(vec3(sapienCoatF0(m)), coatRoughness, wnormal,
                                            worldCamDir);
    color += sapienLayeredIndirect(diffuseIndirect + specularIndirect + ambientIndirect, m,
                                   max(dotNV, 1e-6), irradiance, coatReflection);
  } else {
    color += diffuseIndirect;
    color += specularIndirect;
    color += ambientIndirect;
  }

  outLighting = vec4(color, 1.0);
}
