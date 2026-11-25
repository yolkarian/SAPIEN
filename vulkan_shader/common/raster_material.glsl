#ifndef SAPIEN_RASTER_MATERIAL_GLSL
#define SAPIEN_RASTER_MATERIAL_GLSL

// The OpenPBR material of the rasterization packs. The base keeps the packs' own BRDF (Lambert
// plus GGX from F0 = specular_weight * F(ior) tinted by specular_color, or base_color for
// metals); the fuzz and coat layers of common/openpbr.glsl lie over it by albedo scaling. The
// deferred packs carry what the lighting pass needs beyond the base in the G-buffer target
// `Material` (four packUnorm4x8 words).

#include "./openpbr.glsl"

// the layers and base tint of material `m`: specular colour, metal Fresnel weight (up to 1),
// coat colour and weight, fuzz colour and weight, coat and fuzz roughness, coat IOR
uvec4 sapienPackMaterial(OpenPBR m) {
  return uvec4(packUnorm4x8(vec4(m.specularColor, min(m.specularWeight, 1.0))),
               packUnorm4x8(vec4(m.coatColor, m.coatWeight)),
               packUnorm4x8(vec4(m.fuzzColor, m.fuzzWeight)),
               packUnorm4x8(vec4(sqrt(m.coatAlpha), m.fuzzRoughness,
                                 clamp((m.coatEta - 1.0) * 0.5, 0.0, 1.0), 0.0)));
}

// the material of a G-buffer pixel: its base colour, metallic, roughness and `Material` texel
OpenPBR sapienUnpackMaterial(uvec4 packed, vec3 base, float metallic, float roughness) {
  vec4 specular = unpackUnorm4x8(packed.x);
  vec4 coat = unpackUnorm4x8(packed.y);
  vec4 fuzz = unpackUnorm4x8(packed.z);
  vec4 rest = unpackUnorm4x8(packed.w);
  OpenPBR m = openpbrDielectric(base, roughness, 1.5, 0.0);
  m.metallic = metallic;
  m.specularColor = specular.rgb;
  m.specularWeight = specular.a;
  m.coatColor = coat.rgb;
  m.coatWeight = coat.a;
  m.fuzzColor = fuzz.rgb;
  m.fuzzWeight = fuzz.a;
  m.coatAlpha = max(rest.x * rest.x, OPENPBR_MIN_ALPHA);
  m.fuzzRoughness = rest.y;
  m.coatEta = 1.0 + 2.0 * rest.z;
  return m;
}

// the base's F0: the dielectric reflectance `specular` tinted by specular_color, or the base
// colour scaled by specular_weight for metals
vec3 sapienBaseFresnel(OpenPBR m, vec3 albedo, float specular, float metallic) {
  return specular * m.specularColor * (1.0 - metallic) + albedo * (m.specularWeight * metallic);
}

// the base roughness the raster BRDF uses: the material's, or the OpenPBR one where the coat
// roughens the base or anisotropy stretches it (averaged, the raster BRDF is isotropic)
float sapienRasterRoughness(OpenPBR m, float roughness, float anisotropy) {
  if (m.coatWeight <= 0.0 && anisotropy <= 0.0) {
    return roughness;
  }
  return sqrt(length(m.alpha) * 0.70710678);
}

// The indirect light of a layered surface. `base` is the base's own indirect light (diffuse and
// specular image-based light, ambient), `irradiance` the cosine-weighted environment light and
// `coatReflection` the environment reflected by the coat (specular image-based light with the
// coat's F0 and roughness).
vec3 sapienLayeredIndirect(vec3 base, OpenPBR m, float NoV, vec3 irradiance,
                           vec3 coatReflection) {
  float fuzz = m.fuzzWeight * openpbrFuzzAlbedo(NoV, m.fuzzRoughness);
  return base * openpbrUnderLayers(m, NoV, NoV) + fuzz * m.fuzzColor * irradiance +
         (1.0 - fuzz) * m.coatWeight * coatReflection;
}

// normal-incidence reflectance of the coat against air
float sapienCoatF0(OpenPBR m) {
  float r = (m.coatEta - 1.0) / (m.coatEta + 1.0);
  return r * r;
}

#endif
