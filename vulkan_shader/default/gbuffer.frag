#version 450
#extension GL_EXT_nonuniform_qualifier : enable

#define SET_NUM 0
#include "./camera_set.glsl"
#undef SET_NUM

#define SET_NUM 1
#include "../common/scene.glsl"
#undef SET_NUM

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec2 inUV;
layout(location = 2) in flat uvec4 inSegmentation;
layout(location = 3) in vec3 objectCoord;
layout(location = 4) in mat3 inTbn;

layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outPositionRaw;
layout(location = 2) out vec4 outSpecular;
layout(location = 3) out vec4 outNormal;
layout(location = 4) out uvec4 outSegmentation;
layout(location = 5) out vec4 outCustom;
layout(location = 6) out vec4 outEmission;

void main() {
  int textureMask = material.textureMask;

  outCustom = vec4(objectCoord, 1);
  outSegmentation = inSegmentation;

  outPositionRaw = inPosition;

  outEmission = material.emission;
  if ((textureMask & 16) != 0) {
    outEmission.rgb *= texture(textures[nonuniformEXT(material.emissionTexture)],
                               inUV * material.textureTransforms[4].zw +
                                   material.textureTransforms[4].xy).rgb;
  }

  outAlbedo = material.baseColor;
  if ((textureMask & 1) != 0) {
    outAlbedo *= texture(textures[nonuniformEXT(material.baseColorTexture)],
                          inUV * material.textureTransforms[0].zw +
                              material.textureTransforms[0].xy);
  }

  if (outAlbedo.a == 0) {
    discard;
  }

  outSpecular.r = material.fresnel * 0.08;

  if ((textureMask & 2) != 0) {
    outSpecular.g = texture(textures[nonuniformEXT(material.roughnessTexture)],
                            inUV * material.textureTransforms[1].zw +
                                material.textureTransforms[1].xy).r;
  } else {
    outSpecular.g = material.roughness;
  }

  if ((textureMask & 8) != 0) {
    outSpecular.b = texture(textures[nonuniformEXT(material.metallicTexture)],
                            inUV * material.textureTransforms[3].zw +
                                material.textureTransforms[3].xy).r;
  } else {
    outSpecular.b = material.metallic;
  }

  if (instance.shadeFlat == 0) {
    if ((textureMask & 4) != 0) {
      outNormal = vec4(normalize(inTbn * (texture(textures[nonuniformEXT(material.normalTexture)],
                                                  inUV * material.textureTransforms[2].zw +
                                                      material.textureTransforms[2].xy).xyz * 2 - 1)), 0);
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
}
