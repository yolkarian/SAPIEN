#version 450

layout (constant_id = 0) const int NUM_DIRECTIONAL_LIGHTS = 3;
layout (constant_id = 1) const int NUM_POINT_LIGHTS = 10;
layout (constant_id = 2) const int NUM_DIRECTIONAL_LIGHT_SHADOWS = 1;
layout (constant_id = 3) const int NUM_POINT_LIGHT_SHADOWS = 3;
layout (constant_id = 4) const int NUM_TEXTURED_LIGHT_SHADOWS = 1;
layout (constant_id = 5) const int NUM_SPOT_LIGHT_SHADOWS = 10;
layout (constant_id = 6) const int NUM_SPOT_LIGHTS = 10;
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

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outLighting;

#include "../common/environment.glsl"

const vec2 AMBIENT_OCCLUSION_SAMPLES[12] = {
  vec2(0.2500, 0.0000), vec2(-0.3190, 0.2922), vec2(0.0488, -0.5565),
  vec2(0.4028, 0.5254), vec2(-0.7385, -0.1306), vec2(0.6996, -0.4450),
  vec2(-0.2331, 0.8673), vec2(-0.4463, -0.8526), vec2(0.9341, 0.3411),
  vec2(-0.9650, 0.3983), vec2(0.4664, -0.9990), vec2(0.3430, 1.0682)
};

vec4 world2camera(vec4 pos) {
  return cameraBuffer.viewMatrix * pos;
}

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
  vec3 frm = texture(samplerSpecular, inUV).xyz;
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

  vec3 diffuseAlbedo = albedo * (1 - metallic);
  vec3 fresnel = specular * (1 - metallic) + albedo * metallic;

  vec4 emission = texture(samplerEmission, inUV);
  vec3 color = emission.rgb * emission.a;

  // point light
  for (int i = 0; i < NUM_POINT_LIGHT_SHADOWS; ++i) {
    vec3 pos = world2camera(vec4(sceneBuffer.pointLights[i].position.xyz, 1.f)).xyz;
    mat4 shadowProj = shadowBuffer.pointLightBuffers[6 * i].projectionMatrix;

    vec3 l = pos - csPosition.xyz;
    vec3 wsl = vec3(cameraBuffer.viewMatrixInverse * vec4(l, 0));

    vec3 v = abs(wsl);
    vec4 p = shadowProj * vec4(0, 0, -max(max(v.x, v.y), v.z), 1);
    float pixelDepth = p.z / p.w;
    float shadowDepth = texture(samplerPointLightDepths[i], wsl).x;

    float visibility = step(pixelDepth - shadowDepth, 0);
    color += visibility * computePointLight(
        sceneBuffer.pointLights[i].emission.rgb,
        l, normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  for (int i = NUM_POINT_LIGHT_SHADOWS; i < NUM_POINT_LIGHTS; i++) {
    vec3 pos = world2camera(vec4(sceneBuffer.pointLights[i].position.xyz, 1.f)).xyz;
    vec3 l = pos - csPosition.xyz;
    color += computePointLight(
        sceneBuffer.pointLights[i].emission.rgb,
        l, normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  // directional light
  for (int i = 0; i < NUM_DIRECTIONAL_LIGHT_SHADOWS; ++i) {
    mat4 shadowView = shadowBuffer.directionalLightBuffers[i].viewMatrix;
    mat4 shadowProj = shadowBuffer.directionalLightBuffers[i].projectionMatrix;

    vec3 lightDir = mat3(cameraBuffer.viewMatrix) * sceneBuffer.directionalLights[i].direction.xyz;

    vec4 ssPosition = shadowView * cameraBuffer.viewMatrixInverse * vec4((csPosition.xyz), 1);
    vec4 shadowMapCoord = shadowProj * ssPosition;
    shadowMapCoord /= shadowMapCoord.w;
    shadowMapCoord.xy = shadowMapCoord.xy * 0.5 + 0.5;

    float resolution = textureSize(samplerDirectionalLightDepths[i], 0).x;
    float visibility = ShadowMapPCF(
        samplerDirectionalLightDepths[i], shadowMapCoord.xyz, resolution, 1 / resolution, 1);

    color += visibility * computeDirectionalLight(
        lightDir,
        sceneBuffer.directionalLights[i].emission.rgb,
        normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  for (int i = NUM_DIRECTIONAL_LIGHT_SHADOWS; i < NUM_DIRECTIONAL_LIGHTS; ++i) {
    color += computeDirectionalLight(
        mat3(cameraBuffer.viewMatrix) * sceneBuffer.directionalLights[i].direction.xyz,
        sceneBuffer.directionalLights[i].emission.rgb,
        normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  // spot light
  for (int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; ++i) {
    mat4 shadowView = shadowBuffer.spotLightBuffers[i].viewMatrix;
    mat4 shadowProj = shadowBuffer.spotLightBuffers[i].projectionMatrix;

    vec3 pos = world2camera(vec4(sceneBuffer.spotLights[i].position.xyz, 1.f)).xyz;
    vec3 centerDir = mat3(cameraBuffer.viewMatrix) * sceneBuffer.spotLights[i].direction.xyz;
    vec3 l = pos - csPosition.xyz;

    vec4 ssPosition = shadowView * cameraBuffer.viewMatrixInverse * vec4((csPosition.xyz), 1);
    vec4 shadowMapCoord = shadowProj * ssPosition;
    shadowMapCoord /= shadowMapCoord.w;
    shadowMapCoord.xy = shadowMapCoord.xy * 0.5 + 0.5;

    float resolution = textureSize(samplerSpotLightDepths[i], 0).x;
    float visibility = ShadowMapPCF(
        samplerSpotLightDepths[i], shadowMapCoord.xyz, resolution, 1 / resolution, 1);

    color += visibility * computeSpotLight(
        sceneBuffer.spotLights[i].emission.a,
        sceneBuffer.spotLights[i].direction.a,
        centerDir,
        sceneBuffer.spotLights[i].emission.rgb,
        l, normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  for (int i = NUM_SPOT_LIGHT_SHADOWS; i < NUM_SPOT_LIGHTS; ++i) {
    vec3 pos = world2camera(vec4(sceneBuffer.spotLights[i].position.xyz, 1.f)).xyz;
    vec3 l = pos - csPosition.xyz;
    vec3 centerDir = mat3(cameraBuffer.viewMatrix) * sceneBuffer.spotLights[i].direction.xyz;
    color += computeSpotLight(
        sceneBuffer.spotLights[i].emission.a,
        sceneBuffer.spotLights[i].direction.a,
        centerDir,
        sceneBuffer.spotLights[i].emission.rgb,
        l, normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  // textured light
  for (int i = 0; i < NUM_TEXTURED_LIGHT_SHADOWS; ++i) {
    mat4 shadowView = shadowBuffer.texturedLightBuffers[i].viewMatrix;
    mat4 shadowProj = shadowBuffer.texturedLightBuffers[i].projectionMatrix;

    vec3 pos = world2camera(vec4(sceneBuffer.texturedLights[i].position.xyz, 1.f)).xyz;
    vec3 centerDir = mat3(cameraBuffer.viewMatrix) * sceneBuffer.texturedLights[i].direction.xyz;
    vec3 l = pos - csPosition.xyz;

    vec4 ssPosition = shadowView * cameraBuffer.viewMatrixInverse * vec4((csPosition.xyz), 1);
    vec4 shadowMapCoord = shadowProj * ssPosition;
    shadowMapCoord /= shadowMapCoord.w;
    shadowMapCoord.xy = shadowMapCoord.xy * 0.5 + 0.5;

    float resolution = textureSize(samplerTexturedLightDepths[i], 0).x;
    float visibility = ShadowMapPCF(
        samplerTexturedLightDepths[i], shadowMapCoord.xyz, resolution, 1 / resolution, 1);
    visibility *= texture(samplerTexturedLightTextures[i], shadowMapCoord.xy).x;

    color += visibility * computeSpotLight(
        sceneBuffer.texturedLights[i].emission.a,
        sceneBuffer.texturedLights[i].direction.a,
        centerDir,
        sceneBuffer.texturedLights[i].emission.rgb,
        l, normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  // environmental light
  vec3 wnormal = mat3(cameraBuffer.viewMatrixInverse) * normal;
  vec3 worldCamDir = mat3(cameraBuffer.viewMatrixInverse) * camDir;
  float dotNV = clamp(dot(wnormal, worldCamDir), 0.0, 1.0);
  vec3 environmentFresnel =
      sapienFresnelSchlickRoughness(fresnel, roughness, dotNV);
  float ambientOcclusion = computeAmbientOcclusion(csPosition.xyz, normal);
  color += ambientOcclusion *
           sapienDiffuseIBL((1.0 - environmentFresnel) * diffuseAlbedo, wnormal);
  color += mix(1.0, ambientOcclusion, roughness) *
           sapienSpecularIBL(fresnel, roughness, wnormal, worldCamDir);

  color += ambientOcclusion * sceneBuffer.ambientLight.rgb *
           (1.0 - environmentFresnel) * diffuseAlbedo;

  outLighting = vec4(color, 1.0);
}
