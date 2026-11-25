#ifndef SAPIEN_COMMON_MATERIAL_GLSL
#define SAPIEN_COMMON_MATERIAL_GLSL

// The material record of every renderer (svulkan2/scene/material_data.h): the first batch of the
// OpenPBR Surface 1.1 parameters. 512 B, the same in std140 and std430 (a float after a vec3
// shares its 16 bytes in both; no arrays of scalars), so a rasterizer reads one buffer both as
// the storage array materials[] and as a draw's uniform `material`.
//
// Names follow OpenPBR except metallic (base_metalness), roughness (specular_roughness), ior
// (specular_ior) and emissionStrength (emission_luminance, a plain multiplier). The texture
// fields are slots of the renderer's textures[], -1 for none; textureTransforms[i] (offset xy,
// scale zw) belong to the textures in field order: base colour, roughness, normal, metallic,
// emission, transmission, occlusion. The tail is reserved for the OpenPBR parameters to come.
struct Material {
  vec4 baseColor; // rgb base_color, a geometry_opacity
  vec3 emissionColor;
  float emissionStrength;
  vec3 specularColor;
  float specularWeight;
  vec3 transmissionColor;
  float transmissionWeight;
  vec3 coatColor;
  float coatWeight;
  vec3 fuzzColor;
  float fuzzWeight;
  float baseWeight;
  float metallic;
  float roughness;
  float ior;
  float specularAnisotropy; // specular_roughness_anisotropy
  float transmissionDepth;
  float coatRoughness;
  float coatIor;
  float fuzzRoughness;
  int thinWalled;
  int padding0;
  int padding1;
  vec4 textureTransforms[7];
  int baseColorTexture;
  int roughnessTexture;
  int normalTexture;
  int metallicTexture;
  int emissionTexture;
  int transmissionTexture;
  int occlusionTexture;
  int padding2;
  vec4 reserved[14];
};

// a texture's UV from its textureTransforms entry
vec2 materialUV(vec4 transform, vec2 uv) { return uv * transform.zw + transform.xy; }

// normal-incidence reflectance of the dielectric base: OpenPBR's specular_weight times the
// Fresnel reflectance of specular_ior against air
float materialDielectricF0(float ior, float specularWeight) {
  float r = (ior - 1.0) / (ior + 1.0);
  return clamp(specularWeight * r * r, 0.0, 1.0);
}

#endif
