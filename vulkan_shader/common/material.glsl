#ifndef SAPIEN_COMMON_MATERIAL_GLSL
#define SAPIEN_COMMON_MATERIAL_GLSL

// The material record of every renderer (svulkan2/scene/material_data.h). 256 B, the same in
// std140 and std430 (no arrays of scalars), so a rasterizer reads one buffer both as the storage
// array materials[] and as a draw's uniform `material`. The first 160 B match
// SVMetallicMaterial::Buffer in C++; the texture fields are slots of the renderer's textures[],
// -1 for none; textureTransforms[i] (offset xy, scale zw) belong to the textures in field order:
// base colour, roughness, normal, metallic, emission, transmission. The tail is reserved for
// material fields to come.
struct Material {
  vec4 emission;
  vec4 baseColor;
  float fresnel;
  float roughness;
  float metallic;
  float transmission;
  float ior;
  float transmissionRoughness;
  int textureMask;
  int padding1;
  vec4 textureTransforms[6];
  int baseColorTexture;
  int roughnessTexture;
  int normalTexture;
  int metallicTexture;
  int emissionTexture;
  int transmissionTexture;
  int padding2;
  int padding3;
  vec4 reserved[4];
};

#endif
