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
  int textureMask = material.textureMask;
  vec4 emission;
  vec4 albedo;
  vec4 frm;

  if ((material.textureMask & 16) != 0) {
    emission = texture(textures[nonuniformEXT(material.emissionTexture)], inUV * material.textureTransforms[4].zw + material.textureTransforms[4].xy);
  } else {
    emission = material.emission;
  }

  if ((material.textureMask & 1) != 0) {
    albedo = texture(textures[nonuniformEXT(material.baseColorTexture)], inUV * material.textureTransforms[0].zw + material.textureTransforms[0].xy);
  } else {
    albedo = material.baseColor;
  }

  albedo.a *=  (1.f - instance.transparency);

  if (albedo.a == 0) {
    discard;
  }

  frm.r = material.fresnel * 0.08;

  if ((material.textureMask & 2) != 0) {
    frm.g = texture(textures[nonuniformEXT(material.roughnessTexture)], inUV * material.textureTransforms[1].zw + material.textureTransforms[1].xy).r;
  } else {
    frm.g = material.roughness;
  }

  if ((material.textureMask & 8) != 0) {
    frm.b = texture(textures[nonuniformEXT(material.metallicTexture)], inUV * material.textureTransforms[3].zw + material.textureTransforms[3].xy).r;
  } else {
    frm.b = material.metallic;
  }

  vec3 normal;
  if (instance.shadeFlat == 0) {
    if ((material.textureMask & 4) != 0) {
      normal = normalize(inTbn * (texture(textures[nonuniformEXT(material.normalTexture)], inUV * material.textureTransforms[2].zw + material.textureTransforms[2].xy).xyz * 2 - 1));
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
  vec3 fresnel = specular * (1 - metallic) + albedo.rgb * metallic;

  vec3 color = emission.rgb;

  sapienAddDirectLighting(color, csPosition.xyz, normal, camDir, diffuseAlbedo, roughness,
                          fresnel, true);

  vec3 wnormal = mat3(cameraBuffer.viewMatrixInverse) * normal;
  color += diffuseIBL(diffuseAlbedo, wnormal);
  color += specularIBL(fresnel, roughness,
                       wnormal,
                       mat3(cameraBuffer.viewMatrixInverse) * camDir);
  color += sceneBuffer.ambientLight.rgb * albedo.rgb;

  outColorRaw = vec4(clamp(pow(color, vec3(1/2.2)), vec3(0), vec3(1)), albedo.a);
  outPositionSegmentation = ivec4(ivec3(inPosition * 1000), inSegmentation[1]);
}
