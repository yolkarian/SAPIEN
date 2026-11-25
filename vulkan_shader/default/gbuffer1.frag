#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout (constant_id = 0) const int NUM_DIRECTIONAL_LIGHTS = 0;
layout (constant_id = 1) const int NUM_POINT_LIGHTS = 0;
layout (constant_id = 2) const int NUM_DIRECTIONAL_LIGHT_SHADOWS = 1;
layout (constant_id = 3) const int NUM_POINT_LIGHT_SHADOWS = 3;
layout (constant_id = 4) const int NUM_TEXTURED_LIGHT_SHADOWS = 1;
layout (constant_id = 5) const int NUM_SPOT_LIGHT_SHADOWS = 10;
layout (constant_id = 6) const int NUM_SPOT_LIGHTS = 0;

#define SET_NUM 0
#include "./camera_set.glsl"
#undef SET_NUM

#define SET_NUM 1
#include "../common/scene.glsl"
#undef SET_NUM

#define SET_NUM 2
#include "./scene_set.glsl"
#undef SET_NUM

#include "../common/environment.glsl"
#include "../common/raster_lighting.glsl"

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec2 inUV;
layout(location = 2) in flat uvec4 inSegmentation;
layout(location = 3) in vec3 objectCoord;
layout(location = 4) in mat3 inTbn;

layout(location = 0) out vec4 outLighting;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out uvec4 outSegmentation;
layout(location = 3) out vec4 outPositionRaw;

void main() {
  int textureMask = material.textureMask;
  outSegmentation = inSegmentation;

  vec4 emission;
  vec4 albedo;
  vec4 frm;

  emission = material.emission;
  if ((material.textureMask & 16) != 0) {
    emission.rgb *= texture(textures[nonuniformEXT(material.emissionTexture)], inUV * material.textureTransforms[4].zw + material.textureTransforms[4].xy).rgb;
  }

  albedo = material.baseColor;
  if ((material.textureMask & 1) != 0) {
    albedo *= texture(textures[nonuniformEXT(material.baseColorTexture)], inUV * material.textureTransforms[0].zw + material.textureTransforms[0].xy);
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

  if (instance.shadeFlat == 0) {
    if ((material.textureMask & 4) != 0) {
      outNormal = vec4(normalize(inTbn * (texture(textures[nonuniformEXT(material.normalTexture)], inUV * material.textureTransforms[2].zw + material.textureTransforms[2].xy).xyz * 2 - 1)), 0);
    } else {
      outNormal = vec4(normalize(inTbn * vec3(0, 0, 1)), 0);
    }
  } else {
    vec4 fdx = dFdx(inPosition);
    vec4 fdy = dFdy(inPosition);
    vec3 normal = -normalize(cross(fdx.xyz, fdy.xyz));
    outNormal = vec4(normal, 0);
  }

  outNormal.xyz = faceforward(outNormal.xyz, inPosition.xyz, outNormal.xyz);

  outPositionRaw = inPosition;

  float specular = max(frm.x, 0.0);
  float roughness = clamp(frm.y, 0.045, 1.0);
  float metallic = clamp(frm.z, 0.0, 1.0);

  vec3 normal = outNormal.xyz;
  vec4 csPosition = inPosition;
  csPosition /= csPosition.w;

  vec3 camDir = -normalize(csPosition.xyz);

  vec3 diffuseAlbedo = albedo.rgb * (1 - metallic);
  vec3 fresnel = specular * (1 - metallic) + albedo.rgb * metallic;

  vec3 color = emission.rgb * emission.a;

  sapienAddDirectLighting(color, csPosition.xyz, normal, camDir, diffuseAlbedo, roughness,
                          fresnel, true);

  vec3 wnormal = mat3(cameraBuffer.viewMatrixInverse) * normal;
  vec3 worldCamDir = mat3(cameraBuffer.viewMatrixInverse) * camDir;
  float dotNV = clamp(dot(wnormal, worldCamDir), 0.0, 1.0);
  vec3 environmentFresnel =
      sapienFresnelSchlickRoughness(fresnel, roughness, dotNV);
  color += sapienDiffuseIBL((1.0 - environmentFresnel) * diffuseAlbedo, wnormal);
  color += sapienSpecularIBL(fresnel, roughness, wnormal, worldCamDir);

  color += sceneBuffer.ambientLight.rgb *
           (1.0 - environmentFresnel) * diffuseAlbedo;

  outLighting = vec4(color, albedo.a);
}
