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
  outSegmentation = inSegmentation;

  vec4 emission;
  vec4 albedo;
  vec4 frm;

  emission = vec4(material.emissionColor, material.emissionStrength);
  if (material.emissionTexture >= 0) {
    emission.rgb *= texture(textures[nonuniformEXT(material.emissionTexture)], materialUV(material.textureTransforms[4], inUV)).rgb;
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

  if (instance.shadeFlat == 0) {
    if (material.normalTexture >= 0) {
      outNormal = vec4(normalize(inTbn * (texture(textures[nonuniformEXT(material.normalTexture)], materialUV(material.textureTransforms[2], inUV)).xyz * 2 - 1)), 0);
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
  vec3 fresnel = sapienBaseFresnel(m, albedo.rgb, specular, metallic);

  vec3 color = emission.rgb * emission.a;
  if (layered) {
    color *= openpbrEmissionFactor(m, clamp(dot(normal, camDir), 1e-6, 1.0));
  }

  // transparent surfaces are not shadowed
  sapienAddDirectLighting(color, csPosition.xyz, normal, camDir, diffuseAlbedo, roughness,
                          fresnel, m, false);

  // the material's occlusion texture darkens the indirect light, like the deferred pass
  float occlusion = 1.0;
  if (material.occlusionTexture >= 0) {
    occlusion = texture(textures[nonuniformEXT(material.occlusionTexture)],
                        materialUV(material.textureTransforms[6], inUV)).r;
  }
  vec3 wnormal = mat3(cameraBuffer.viewMatrixInverse) * normal;
  vec3 worldCamDir = mat3(cameraBuffer.viewMatrixInverse) * camDir;
  float dotNV = clamp(dot(wnormal, worldCamDir), 0.0, 1.0);
  vec3 environmentFresnel =
      sapienFresnelSchlickRoughness(fresnel, roughness, dotNV);
  vec3 diffuseIndirect =
      occlusion * sapienDiffuseIBL((1.0 - environmentFresnel) * diffuseAlbedo, wnormal);
  vec3 specularIndirect = mix(1.0, occlusion, roughness) *
                          sapienSpecularIBL(fresnel, roughness, wnormal, worldCamDir);
  vec3 ambientIndirect = occlusion * sceneBuffer.ambientLight.rgb *
                         (1.0 - environmentFresnel) * diffuseAlbedo;
  if (layered) {
    float coatRoughness = sqrt(m.coatAlpha);
    vec3 irradiance =
        occlusion * (sapienDiffuseIBL(vec3(1.0), wnormal) + sceneBuffer.ambientLight.rgb);
    vec3 coatReflection = mix(1.0, occlusion, coatRoughness) *
                          sapienSpecularIBL(vec3(sapienCoatF0(m)), coatRoughness, wnormal,
                                            worldCamDir);
    color += sapienLayeredIndirect(diffuseIndirect + specularIndirect + ambientIndirect, m,
                                   max(dotNV, 1e-6), irradiance, coatReflection);
  } else {
    color += diffuseIndirect;
    color += specularIndirect;
    color += ambientIndirect;
  }

  outLighting = vec4(color, albedo.a);
}
