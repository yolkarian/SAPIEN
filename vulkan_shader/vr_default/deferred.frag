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
layout(set = 2, binding = 1) uniform sampler2D samplerSpecular;
layout(set = 2, binding = 2) uniform sampler2D samplerNormal;
layout(set = 2, binding = 3) uniform sampler2D samplerEmission;
layout(set = 2, binding = 4) uniform sampler2D samplerGbufferDepth;
layout(set = 2, binding = 5) uniform usampler2D samplerMaterial;
layout(set = 2, binding = 6) uniform sampler2D samplerAmbientOcclusion;

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outLighting;

#include "../common/environment.glsl"
#include "../common/gtao.glsl"
#include "../common/raster_lighting.glsl"

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
  // half-resolution GTAO with depth-aware bilateral upsample, times the material's occlusion map
  float ambientOcclusion =
      sapienSampleBilateralAO(samplerAmbientOcclusion, samplerGbufferDepth, inUV, csPosition.z,
                              cameraBuffer.projectionMatrixInverse, ambientOcclusionStrength,
                              ambientOcclusionRadius) *
      frm.w;
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
