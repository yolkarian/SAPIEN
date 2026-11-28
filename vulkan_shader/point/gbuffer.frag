#version 450
#extension GL_EXT_nonuniform_qualifier : enable

#define SET_NUM 0
#include "./camera_set.glsl"
#undef SET_NUM

#define SET_NUM 1
#include "../common/scene.glsl"
#undef SET_NUM

#include "../common/raster_material.glsl"

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec2 inUV;
layout(location = 2) in flat uvec4 inSegmentation;
layout(location = 3) in vec3 objectCoord;
layout(location = 4) in mat3 inTbn;

layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outPosition;
layout(location = 2) out vec4 outSpecular;
layout(location = 3) out vec4 outNormal;
layout(location = 4) out uvec4 outSegmentation;
layout(location = 5) out vec4 outCustom;
layout(location = 6) out vec4 outEmission;
layout(location = 7) out uvec4 outMaterial; // what the lighting pass needs beyond the base

void main() {
  outCustom = vec4(objectCoord, 1);
  outSegmentation = inSegmentation;

  outPosition = inPosition;

  vec4 p1 = cameraBuffer.projectionMatrix * inPosition;
  p1 /= p1.w;
  vec2 p1s = p1.xy / p1.z;

  // emission colour times strength, like every other pack
  outEmission = vec4(material.emissionColor, material.emissionStrength);
  if (material.emissionTexture >= 0) {
    outEmission.rgb *= texture(textures[nonuniformEXT(material.emissionTexture)], materialUV(material.textureTransforms[4], inUV)).rgb;
  }

  outAlbedo = material.baseColor;
  float texelAlpha = 1.0;
  if (material.baseColorTexture >= 0) {
    vec4 texel = texture(textures[nonuniformEXT(material.baseColorTexture)], materialUV(material.textureTransforms[0], inUV));
    outAlbedo *= texel;
    texelAlpha = texel.a;
  }

  // geometry coverage follows the material's alpha mode: OPAQUE covers, MASK cuts out below the
  // cutoff, BLEND keeps the alpha (the deferred lighting pass blends by it)
  float coverage = materialCoverage(material, texelAlpha);
  if (coverage <= 0.0) {
    discard;
  }
  outAlbedo.a = coverage;
  outAlbedo.rgb *= material.baseWeight;

  if (material.roughnessTexture >= 0) {
    outSpecular.g = texture(textures[nonuniformEXT(material.roughnessTexture)], materialUV(material.textureTransforms[1], inUV)).r;
  } else {
    outSpecular.g = material.roughness;
  }

  if (material.metallicTexture >= 0) {
    outSpecular.b = texture(textures[nonuniformEXT(material.metallicTexture)], materialUV(material.textureTransforms[3], inUV)).r;
  } else {
    outSpecular.b = material.metallic;
  }

  OpenPBR m = openpbrSurface(material, outAlbedo.rgb, outSpecular.b, outSpecular.g, 0.0, false);
  outSpecular.r = materialDielectricF0(m.baseEta, 1.0);
  outSpecular.g = sapienRasterRoughness(m, outSpecular.g, material.specularAnisotropy);
  outMaterial = sapienPackMaterial(m);

  // occlusion of the indirect light, from glTF's occlusion texture (red channel)
  outSpecular.a = 1.0;
  if (material.occlusionTexture >= 0) {
    outSpecular.a = mix(1.0, texture(textures[nonuniformEXT(material.occlusionTexture)], materialUV(material.textureTransforms[6], inUV)).r, material.occlusionStrength);
  }

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
}
