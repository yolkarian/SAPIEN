#version 450

layout (constant_id = 0) const int NUM_DIRECTIONAL_LIGHTS = 0;
layout (constant_id = 1) const int NUM_POINT_LIGHTS = 0;
layout (constant_id = 2) const int NUM_DIRECTIONAL_LIGHT_SHADOWS = 1;
layout (constant_id = 3) const int NUM_POINT_LIGHT_SHADOWS = 3;
layout (constant_id = 4) const int NUM_TEXTURED_LIGHT_SHADOWS = 1;
layout (constant_id = 5) const int NUM_SPOT_LIGHT_SHADOWS = 10;
layout (constant_id = 6) const int NUM_SPOT_LIGHTS = 0;

#define SET_NUM 0
#include "./scene_set.glsl"
#undef SET_NUM

#define SET_NUM 1
#include "./camera_set.glsl"
#undef SET_NUM

layout(set = 2, binding = 0) uniform sampler2D samplerAlbedo;
layout(set = 2, binding = 1) uniform sampler2D samplerPosition;
layout(set = 2, binding = 2) uniform sampler2D samplerSpecular;
layout(set = 2, binding = 3) uniform sampler2D samplerNormal;
layout(set = 2, binding = 4) uniform sampler2D samplerEmission;
layout(set = 2, binding = 5) uniform sampler2D samplerGbufferDepth;
layout(set = 2, binding = 6) uniform sampler2D samplerCustom;
layout(set = 2, binding = 7) uniform usampler2D samplerMaterial;

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outLighting;

#include "../common/raster_lighting.glsl"

vec3 getBackgroundColor(vec3 texcoord) {
  texcoord = texcoord.xzy;
  return textureLod(samplerEnvironment, texcoord, 0).rgb + sceneBuffer.ambientLight.rgb;
}

vec3 diffuseIBL(vec3 albedo, vec3 N) {
  N = N.xzy;
  vec3 color = textureLod(samplerEnvironment, N, 5).rgb;
  return color * albedo;
}

vec3 specularIBL(vec3 fresnel, float roughness, vec3 N, vec3 V) {
  float dotNV = max(dot(N, V), 0);
  vec3 R = 2 * dot(N, V) * N - V;
  R = R.xzy;
  vec3 color = textureLod(samplerEnvironment, R, roughness * 5).rgb;
  vec2 envBRDF = texture(samplerBRDFLUT, vec2(roughness, dotNV)).xy;
  return color * (fresnel * envBRDF.x + envBRDF.y);
}

void main() {
  vec3 albedo = texture(samplerAlbedo, inUV).xyz;
  vec4 frm = texture(samplerSpecular, inUV); // F0, roughness, metallic, material occlusion
  float specular = frm.x;
  float roughness = frm.y;
  float metallic = frm.z;

  vec3 normal = normalize(texture(samplerNormal, inUV).xyz);
  float depth = texture(samplerGbufferDepth, inUV).x;

  vec4 csPosition = cameraBuffer.projectionMatrixInverse * (vec4(inUV * 2 - 1, depth, 1));
  csPosition /= csPosition.w;

  vec3 camDir = -normalize(csPosition.xyz);

  OpenPBR m = sapienUnpackMaterial(texture(samplerMaterial, inUV), albedo, metallic, roughness);
  bool layered = openpbrHasLayers(m);
  vec3 diffuseAlbedo = albedo * (1 - metallic);
  vec3 fresnel = sapienBaseFresnel(m, albedo, specular, metallic);

  vec4 emission = texture(samplerEmission, inUV); // rgb colour, a strength
  vec3 color = emission.rgb * emission.a;
  if (layered) {
    color *= openpbrEmissionFactor(m, clamp(dot(normal, camDir), 1e-6, 1.0));
  }

  sapienAddDirectLighting(color, csPosition.xyz, normal, camDir, diffuseAlbedo, roughness,
                          fresnel, m, true);

  // environmental light; the material's occlusion texture darkens it
  vec3 wnormal = mat3(cameraBuffer.viewMatrixInverse) * normal;
  vec3 worldCamDir = mat3(cameraBuffer.viewMatrixInverse) * camDir;
  vec3 diffuseIndirect = frm.w * diffuseIBL(diffuseAlbedo, wnormal);
  vec3 specularIndirect =
      mix(1.0, frm.w, roughness) * specularIBL(fresnel, roughness, wnormal, worldCamDir);
  vec3 ambientIndirect = frm.w * sceneBuffer.ambientLight.rgb * albedo.rgb;
  if (layered) {
    float coatRoughness = sqrt(m.coatAlpha);
    vec3 irradiance = frm.w * (diffuseIBL(vec3(1.0), wnormal) + sceneBuffer.ambientLight.rgb);
    vec3 coatReflection = mix(1.0, frm.w, coatRoughness) *
                          specularIBL(vec3(sapienCoatF0(m)), coatRoughness, wnormal, worldCamDir);
    color += sapienLayeredIndirect(diffuseIndirect + specularIndirect + ambientIndirect, m,
                                   clamp(dot(wnormal, worldCamDir), 1e-6, 1.0), irradiance,
                                   coatReflection);
  } else {
    color += diffuseIndirect;
    color += specularIndirect;
    color += ambientIndirect;
  }

  if (depth == 1) {
    outLighting = vec4(getBackgroundColor((cameraBuffer.viewMatrixInverse * csPosition).xyz), 0.f);
  } else {
    outLighting = vec4(color, 1);
  }
}
