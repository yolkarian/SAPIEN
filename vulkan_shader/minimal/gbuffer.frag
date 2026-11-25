#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout (constant_id = 0) const int NUM_DIRECTIONAL_LIGHTS = 0;
layout (constant_id = 1) const int NUM_DIRECTIONAL_LIGHT_SHADOWS = 0;
layout (constant_id = 2) const int NUM_POINT_LIGHTS = 0;
layout (constant_id = 3) const int NUM_POINT_LIGHT_SHADOWS = 0;
layout (constant_id = 4) const int NUM_SPOT_LIGHTS = 0;
layout (constant_id = 5) const int NUM_SPOT_LIGHT_SHADOWS = 0;
layout (constant_id = 6) const int NUM_TEXTURED_LIGHT_SHADOWS = 0;

layout(set = 0, binding = 0) uniform CameraBuffer {
  mat4 viewMatrix;
  mat4 projectionMatrix;
  mat4 viewMatrixInverse;
  mat4 projectionMatrixInverse;
  float width;
  float height;
} cameraBuffer;

#define SET_NUM 1
#include "../common/scene.glsl"
#undef SET_NUM

#include "../common/light_data.glsl"

layout(set = 2, binding = 0) uniform SceneBuffer {
  vec4 ambientLight;
} sceneBuffer;

struct LightBuffer {
  mat4 viewMatrix;
  mat4 viewMatrixInverse;
  mat4 projectionMatrix;
  mat4 projectionMatrixInverse;
  int width;
  int height;
};

layout(set = 2, binding = 1) uniform ShadowBuffer {
  LightBuffer pointLightBuffers[60];
  LightBuffer directionalLightBuffers[3];
  LightBuffer spotLightBuffers[10];
  LightBuffer texturedLightBuffers[1];
} shadowBuffer;

layout(set = 2, binding = 2) uniform samplerCube samplerPointLightDepths[10];
layout(set = 2, binding = 3) uniform sampler2D samplerDirectionalLightDepths[3];
layout(set = 2, binding = 4) uniform sampler2D samplerSpotLightDepths[10];
layout(set = 2, binding = 5) uniform sampler2D samplerTexturedLightDepths[1];
layout(set = 2, binding = 6) uniform sampler2D samplerTexturedLightTextures[1];
layout(set = 2, binding = 7) uniform samplerCube samplerEnvironment;
layout(set = 2, binding = 8) uniform sampler2D samplerBRDFLUT;
// sized by the light counts, see default/scene_set.glsl
layout(set = 2, binding = 9) uniform PointLights {
  PointLight l[NUM_POINT_LIGHTS + 1];
} pointLights;
layout(set = 2, binding = 10) uniform DirectionalLights {
  DirectionalLight l[NUM_DIRECTIONAL_LIGHTS + 1];
} directionalLights;
layout(set = 2, binding = 11) uniform SpotLights {
  SpotLight l[NUM_SPOT_LIGHTS + NUM_TEXTURED_LIGHT_SHADOWS + 1];
} spotLights;

#include "../common/raster_lighting.glsl"

vec3 getBackgroundColor(vec3 texcoord) {
  texcoord = texcoord.xzy;
  return textureLod(samplerEnvironment, texcoord, 0).rgb;
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

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec2 inUV;
layout(location = 2) in flat uvec4 inSegmentation;
layout(location = 3) in vec3 objectCoord;
layout(location = 4) in mat3 inTbn;

layout(location = 0) out vec4 outColorRaw;
layout(location = 1) out ivec4 outPositionSegmentation;

void main() {
  vec3 emission = material.emissionColor * material.emissionStrength;
  vec4 albedo;
  vec4 frm;

  if (material.emissionTexture >= 0) {
    emission *= texture(textures[nonuniformEXT(material.emissionTexture)], materialUV(material.textureTransforms[4], inUV)).rgb;
  }

  albedo = material.baseColor;
  if (material.baseColorTexture >= 0) {
    albedo *= texture(textures[nonuniformEXT(material.baseColorTexture)], materialUV(material.textureTransforms[0], inUV));
  }

  albedo.a *=  (1.f - instance.transparency);

  if (albedo.a == 0) {
    discard;
  }
  albedo.rgb *= material.baseWeight;

  // the coat over the base lowers the IOR contrast of the base's interface
  frm.r = materialDielectricF0(material.ior / mix(1.0, material.coatIor, material.coatWeight),
                               material.specularWeight);

  if (material.roughnessTexture >= 0) {
    frm.g = texture(textures[nonuniformEXT(material.roughnessTexture)], materialUV(material.textureTransforms[1], inUV)).r;
  } else {
    frm.g = material.roughness;
  }

  if (material.metallicTexture >= 0) {
    frm.b = texture(textures[nonuniformEXT(material.metallicTexture)], materialUV(material.textureTransforms[3], inUV)).r;
  } else {
    frm.b = material.metallic;
  }

  OpenPBR m = openpbrSurface(material, albedo.rgb, frm.b, frm.g, 0.0, false);
  frm.g = sapienRasterRoughness(m, frm.g, material.specularAnisotropy);
  bool layered = openpbrHasLayers(m);

  vec3 normal;
  if (instance.shadeFlat == 0) {
    if (material.normalTexture >= 0) {
      normal = normalize(inTbn * (texture(textures[nonuniformEXT(material.normalTexture)], materialUV(material.textureTransforms[2], inUV)).xyz * 2 - 1));
    } else {
      normal = normalize(inTbn * vec3(0, 0, 1));
    }
  } else {
    vec4 fdx = dFdx(inPosition);
    vec4 fdy = dFdy(inPosition);
    normal = -normalize(cross(fdx.xyz, fdy.xyz));
  }

  float specular = frm.x;
  float roughness = frm.y;
  float metallic = frm.z;

  vec4 csPosition = inPosition;
  csPosition /= csPosition.w;

  vec3 camDir = -normalize(csPosition.xyz);

  vec3 diffuseAlbedo = albedo.rgb * (1 - metallic);
  vec3 fresnel = sapienBaseFresnel(m, albedo.rgb, specular, metallic);

  vec3 color = emission;
  if (layered) {
    color *= openpbrEmissionFactor(m, clamp(dot(normal, camDir), 1e-6, 1.0));
  }

  sapienAddDirectLighting(color, csPosition.xyz, normal, camDir, diffuseAlbedo, roughness,
                          fresnel, m, true);

  // the material's occlusion texture darkens the indirect light
  float occlusion = 1.0;
  if (material.occlusionTexture >= 0) {
    occlusion = texture(textures[nonuniformEXT(material.occlusionTexture)], materialUV(material.textureTransforms[6], inUV)).r;
  }
  vec3 wnormal = mat3(cameraBuffer.viewMatrixInverse) * normal;
  vec3 worldCamDir = mat3(cameraBuffer.viewMatrixInverse) * camDir;
  vec3 diffuseIndirect = occlusion * diffuseIBL(diffuseAlbedo, wnormal);
  vec3 specularIndirect =
      mix(1.0, occlusion, roughness) * specularIBL(fresnel, roughness, wnormal, worldCamDir);
  vec3 ambientIndirect = occlusion * sceneBuffer.ambientLight.rgb * albedo.rgb;
  if (layered) {
    float coatRoughness = sqrt(m.coatAlpha);
    vec3 irradiance = occlusion * (diffuseIBL(vec3(1.0), wnormal) + sceneBuffer.ambientLight.rgb);
    vec3 coatReflection = mix(1.0, occlusion, coatRoughness) *
                          specularIBL(vec3(sapienCoatF0(m)), coatRoughness, wnormal, worldCamDir);
    color += sapienLayeredIndirect(diffuseIndirect + specularIndirect + ambientIndirect, m,
                                   clamp(dot(wnormal, worldCamDir), 1e-6, 1.0), irradiance,
                                   coatReflection);
  } else {
    color += diffuseIndirect;
    color += specularIndirect;
    color += ambientIndirect;
  }

  outColorRaw = vec4(clamp(pow(color, vec3(1/2.2)), vec3(0), vec3(1)), albedo.a);
  outPositionSegmentation = ivec4(ivec3(inPosition * 1000), inSegmentation[1]);
}
