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
  float occlusionStrength; // glTF occlusionTexture.strength: mix(1, texture, strength)
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
  int twoSided;      // both faces visible (cull mode none)
  int alphaMode;     // base_color.a: 0 OPAQUE (ignored), 1 MASK, 2 BLEND (stochastic coverage)
  float alphaCutoff; // MASK threshold
  int padding3;
  vec4 reserved[13];
};

// a texture's UV from its textureTransforms entry
vec2 materialUV(vec4 transform, vec2 uv) { return uv * transform.zw + transform.xy; }

// geometry coverage of a surface from its base colour alpha: OPAQUE covers, MASK cuts out below
// alphaCutoff, BLEND keeps the alpha as a stochastic coverage. `texelAlpha` is the base colour
// texture's alpha (1 without one).
float materialCoverage(Material material, float texelAlpha) {
  if (material.alphaMode == 0) {
    return 1.0;
  }
  float alpha = clamp(material.baseColor.a * texelAlpha, 0.0, 1.0);
  if (material.alphaMode == 1) {
    return alpha < material.alphaCutoff ? 0.0 : 1.0;
  }
  return alpha;
}

// normal-incidence reflectance of the dielectric base: OpenPBR's specular_weight times the
// Fresnel reflectance of specular_ior against air
float materialDielectricF0(float ior, float specularWeight) {
  float r = (ior - 1.0) / (ior + 1.0);
  return clamp(specularWeight * r * r, 0.0, 1.0);
}

#endif
