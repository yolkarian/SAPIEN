#ifndef SAPIEN_COMMON_OPENPBR_GLSL
#define SAPIEN_COMMON_OPENPBR_GLSL

// The first batch of OpenPBR Surface 1.1 (https://academysoftwarefoundation.github.io/OpenPBR/):
// fuzz over coat over a base substrate that mixes metal (F82-tint Fresnel) with a dielectric
// (specular reflection over Lambertian diffuse, or translucent), all GGX microfacet lobes with
// anisotropy on the base. Layers combine by albedo scaling: a layer scales what lies below by
// one minus its directional albedo at the view direction, and the coat also by its absorption
// along both directions. Approximations: the coat and dielectric albedos are their Fresnel
// factors at the view direction; the fuzz lobe is the Charlie sheen with the Estevez-Kulla
// shadowing, whose albedo is tabulated below; the diffuse lobe is Lambertian.
//
// Every direction is in the local shading frame (z = shading normal, x = tangent) and points
// away from the surface; wo is the view direction, above the surface (wo.z > 0). "f" functions
// return the BSDF times |cos(wi)|.

#include "./material.glsl"

const float OPENPBR_PI = 3.14159265358979;
// roughness 0.045, the smallest any lobe uses
const float OPENPBR_MIN_ALPHA = 0.002025;

struct OpenPBR {
  vec3 base;              // base_weight * base_color
  float metallic;
  vec3 specularColor;
  float specularWeight;   // metal Fresnel scale (the dielectric one is folded into eta)
  float eta;              // dielectric IOR ratio across the interface, inside over the side of wo
  float baseEta;          // the ratio the base's Fresnel uses: against the coat, never below 1
                          // outside a body (no total internal reflection at the base)
  vec2 alpha;             // GGX alpha along tangent and bitangent of the base
  float transmission;
  vec3 transmissionTint;  // tint of light entering through the interface
  bool thinWalled;
  bool interior;          // wo inside a translucent body: only the bare dielectric interface
  float coatWeight;
  vec3 coatColor;
  float coatAlpha;
  float coatEta;
  float fuzzWeight;
  vec3 fuzzColor;
  float fuzzRoughness;
};

float openpbrLuminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

float openpbrPow5(float x) {
  float x2 = x * x;
  return x2 * x2 * x;
}

// unpolarized Fresnel reflectance; eta = n(transmitted side) / n(incident side)
float openpbrFresnel(float cosI, float eta) {
  float c = abs(cosI);
  float g2 = eta * eta - 1.0 + c * c;
  if (g2 <= 0.0) {
    return 1.0; // total internal reflection
  }
  float g = sqrt(g2);
  float a = (g - c) / (g + c);
  float b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
  return 0.5 * a * a * (1.0 + b * b);
}

// OpenPBR's specular_weight: scale the normal-incidence reflectance through the IOR ratio
float openpbrModulatedEta(float eta, float weight) {
  float r = (1.0 - eta) / (1.0 + eta);
  float e = sign(eta - 1.0) * sqrt(clamp(weight * r * r, 0.0, 0.9999));
  return (1.0 + e) / (1.0 - e);
}

// keep the ratio of a refraction away from 1, where the half vector wo + eta * wi vanishes
float openpbrRefractionEta(float eta) {
  return abs(eta - 1.0) < 1e-4 ? 1.0001 : eta;
}

// F82-tint metal Fresnel: Schlick from F0 = base, pulled toward `tint` times Schlick near 82 deg.
// The weight is in [0, 1] for metals: above 1 the lobe would reflect more than it receives.
vec3 openpbrMetalFresnel(float mu, vec3 f0, vec3 tint, float weight) {
  const float muBar = 1.0 / 7.0;
  const float barWeight = muBar * 0.39656; // muBar * (1 - muBar)^6
  vec3 schlick = f0 + (1.0 - f0) * openpbrPow5(1.0 - mu);
  vec3 schlickBar = f0 + (1.0 - f0) * openpbrPow5(1.0 - muBar);
  // mu can reach 1 + eps: pow of a negative base is undefined
  float k = mu * pow(max(1.0 - mu, 0.0), 6.0) / barWeight;
  return clamp(weight, 0.0, 1.0) * clamp(schlick - k * (schlickBar - tint * schlickBar), 0.0, 1.0);
}

// ---- anisotropic GGX ----

float openpbrGgxD(vec3 h, vec2 a) {
  if (h.z <= 0.0) {
    return 0.0;
  }
  vec3 s = vec3(h.x / a.x, h.y / a.y, h.z);
  float d = dot(s, s);
  return 1.0 / (OPENPBR_PI * a.x * a.y * d * d);
}

float openpbrGgxLambda(vec3 w, vec2 a) {
  float z2 = max(w.z * w.z, 1e-12);
  float t = (a.x * a.x * w.x * w.x + a.y * a.y * w.y * w.y) / z2;
  return 0.5 * (sqrt(1.0 + t) - 1.0);
}

float openpbrGgxG1(vec3 w, vec2 a) { return 1.0 / (1.0 + openpbrGgxLambda(w, a)); }

float openpbrGgxG2(vec3 wo, vec3 wi, vec2 a) {
  return 1.0 / (1.0 + openpbrGgxLambda(wo, a) + openpbrGgxLambda(wi, a));
}

// a micronormal visible from wo (Heitz 2018), u in [0, 1)^2
vec3 openpbrSampleGgxVisible(vec3 wo, vec2 a, vec2 u) {
  vec3 vh = normalize(vec3(a.x * wo.x, a.y * wo.y, wo.z));
  float lensq = vh.x * vh.x + vh.y * vh.y;
  vec3 t1 = lensq > 0.0 ? vec3(-vh.y, vh.x, 0.0) * inversesqrt(lensq) : vec3(1.0, 0.0, 0.0);
  vec3 t2 = cross(vh, t1);
  float r = sqrt(u.x);
  float phi = 2.0 * OPENPBR_PI * u.y;
  float p1 = r * cos(phi);
  float p2 = r * sin(phi);
  float s = 0.5 * (1.0 + vh.z);
  p2 = (1.0 - s) * sqrt(max(1.0 - p1 * p1, 0.0)) + s * p2;
  vec3 nh = p1 * t1 + p2 * t2 + sqrt(max(1.0 - p1 * p1 - p2 * p2, 0.0)) * vh;
  return normalize(vec3(a.x * nh.x, a.y * nh.y, max(nh.z, 1e-6)));
}

// solid-angle pdf of the reflection of wo about a visible micronormal
float openpbrGgxReflectionPdf(vec3 wo, vec3 h, vec2 a) {
  return openpbrGgxG1(wo, a) * openpbrGgxD(h, a) / (4.0 * wo.z);
}

// D * G2 / (4 cos(wo)): a microfacet reflection lobe times cos(wi), without its Fresnel factor
float openpbrGgxReflection(vec3 wo, vec3 wi, vec3 h, vec2 a) {
  return openpbrGgxD(h, a) * openpbrGgxG2(wo, wi, a) / (4.0 * wo.z);
}

vec2 openpbrAlpha(float roughness, float anisotropy) {
  float a = roughness * roughness;
  float at = a * sqrt(2.0 / (1.0 + (1.0 - anisotropy) * (1.0 - anisotropy)));
  return max(vec2(at, (1.0 - anisotropy) * at), vec2(OPENPBR_MIN_ALPHA));
}

// ---- fuzz: Charlie sheen with the Estevez-Kulla shadowing ----

// directional albedo, [roughness][cos(wo)] at cell centres of a 16 x 16 grid over [0, 1]^2
const float OPENPBR_FUZZ_ALBEDO[256] = float[256](
  0.8318, 0.5651, 0.4053, 0.2961, 0.2170, 0.1583, 0.1143, 0.0812, 0.0565, 0.0384, 0.0253, 0.0159, 0.0094, 0.0050, 0.0023, 0.0007,
  0.7861, 0.5572, 0.4160, 0.3165, 0.2422, 0.1849, 0.1402, 0.1050, 0.0775, 0.0563, 0.0399, 0.0273, 0.0178, 0.0108, 0.0057, 0.0023,
  0.7380, 0.5562, 0.4396, 0.3544, 0.2879, 0.2341, 0.1897, 0.1527, 0.1218, 0.0964, 0.0754, 0.0577, 0.0429, 0.0304, 0.0201, 0.0116,
  0.7319, 0.5694, 0.4636, 0.3849, 0.3223, 0.2706, 0.2267, 0.1891, 0.1568, 0.1295, 0.1061, 0.0858, 0.0678, 0.0520, 0.0379, 0.0255,
  0.7422, 0.5889, 0.4881, 0.4127, 0.3521, 0.3013, 0.2576, 0.2196, 0.1863, 0.1579, 0.1331, 0.1110, 0.0912, 0.0731, 0.0566, 0.0415,
  0.7602, 0.6109, 0.5125, 0.4386, 0.3788, 0.3283, 0.2846, 0.2461, 0.2121, 0.1828, 0.1570, 0.1338, 0.1126, 0.0931, 0.0749, 0.0580,
  0.7815, 0.6335, 0.5360, 0.4626, 0.4030, 0.3525, 0.3085, 0.2695, 0.2348, 0.2048, 0.1783, 0.1543, 0.1322, 0.1116, 0.0923, 0.0741,
  0.8037, 0.6554, 0.5579, 0.4845, 0.4248, 0.3740, 0.3296, 0.2902, 0.2549, 0.2243, 0.1972, 0.1726, 0.1498, 0.1285, 0.1084, 0.0894,
  0.8251, 0.6758, 0.5778, 0.5041, 0.4441, 0.3930, 0.3483, 0.3083, 0.2726, 0.2416, 0.2140, 0.1889, 0.1656, 0.1438, 0.1232, 0.1037,
  0.8446, 0.6939, 0.5953, 0.5213, 0.4610, 0.4096, 0.3645, 0.3242, 0.2881, 0.2567, 0.2288, 0.2034, 0.1797, 0.1576, 0.1366, 0.1167,
  0.8613, 0.7093, 0.6102, 0.5358, 0.4753, 0.4236, 0.3783, 0.3378, 0.3014, 0.2698, 0.2417, 0.2160, 0.1922, 0.1698, 0.1486, 0.1286,
  0.8746, 0.7216, 0.6221, 0.5476, 0.4869, 0.4352, 0.3898, 0.3491, 0.3127, 0.2810, 0.2527, 0.2270, 0.2031, 0.1806, 0.1594, 0.1392,
  0.8838, 0.7305, 0.6310, 0.5565, 0.4960, 0.4444, 0.3990, 0.3584, 0.3219, 0.2902, 0.2621, 0.2363, 0.2124, 0.1900, 0.1688, 0.1487,
  0.8888, 0.7356, 0.6366, 0.5625, 0.5024, 0.4510, 0.4059, 0.3655, 0.3292, 0.2977, 0.2697, 0.2441, 0.2204, 0.1981, 0.1770, 0.1570,
  0.8890, 0.7370, 0.6389, 0.5656, 0.5060, 0.4552, 0.4106, 0.3706, 0.3347, 0.3035, 0.2757, 0.2504, 0.2269, 0.2048, 0.1840, 0.1642,
  0.8844, 0.7345, 0.6378, 0.5656, 0.5070, 0.4570, 0.4130, 0.3736, 0.3382, 0.3075, 0.2802, 0.2552, 0.2321, 0.2104, 0.1898, 0.1703
);

float openpbrFuzzAlbedo(float mu, float roughness) {
  vec2 p = clamp(vec2(mu, roughness) * 16.0 - 0.5, vec2(0.0), vec2(15.0));
  ivec2 i = min(ivec2(p), ivec2(14));
  vec2 t = p - vec2(i);
  float a = mix(OPENPBR_FUZZ_ALBEDO[i.y * 16 + i.x], OPENPBR_FUZZ_ALBEDO[i.y * 16 + i.x + 1], t.x);
  float b = mix(OPENPBR_FUZZ_ALBEDO[(i.y + 1) * 16 + i.x],
                OPENPBR_FUZZ_ALBEDO[(i.y + 1) * 16 + i.x + 1], t.x);
  return mix(a, b, t.y);
}

float openpbrFuzzLambda(float c, float r) {
  float t = (1.0 - r) * (1.0 - r);
  float a = mix(21.5473, 25.3245, t);
  float b = mix(3.82987, 3.32435, t);
  float cc = mix(0.19823, 0.16801, t);
  float d = mix(-1.97760, -1.27393, t);
  float e = mix(-4.32054, -4.85967, t);
  c = clamp(c, 1e-6, 1.0);
  float x = c < 0.5 ? c : 0.5;
  float l = a / (1.0 + b * pow(x, cc)) + d * x + e;
  if (c < 0.5) {
    return exp(l);
  }
  float x1 = 1.0 - c;
  float l1 = a / (1.0 + b * pow(x1, cc)) + d * x1 + e;
  return exp(2.0 * l - l1);
}

// the fuzz lobe times cos(wi) / cos(wi), from the cosines of wo, wi and their half vector
float openpbrFuzz(float cosO, float cosI, float cosH, float roughness) {
  float r = max(roughness, 0.07);
  float sinH = sqrt(max(1.0 - cosH * cosH, 0.0));
  float d = (2.0 + 1.0 / r) * pow(sinH, 1.0 / r) / (2.0 * OPENPBR_PI);
  float g = 1.0 / (1.0 + openpbrFuzzLambda(cosO, r) + openpbrFuzzLambda(cosI, r));
  return d * g / (4.0 * cosO);
}

// an isotropic GGX lobe D * G2 / (4 cos(wo)) from cosines; for the coat, and for rasterizers
// that shade without a tangent frame
float openpbrGgxReflectionIso(float cosO, float cosI, float cosH, float alpha) {
  float a2 = alpha * alpha;
  float d = cosH * cosH * (a2 - 1.0) + 1.0;
  float lo = 0.5 * (sqrt(1.0 + a2 * (1.0 - cosO * cosO) / max(cosO * cosO, 1e-12)) - 1.0);
  float li = 0.5 * (sqrt(1.0 + a2 * (1.0 - cosI * cosI) / max(cosI * cosI, 1e-12)) - 1.0);
  return a2 / (OPENPBR_PI * d * d) / (1.0 + lo + li) / (4.0 * cosO);
}

// ---- layering ----

// transmittance of one pass through a coat of `color` and IOR `eta` at external cosine mu;
// coat_color is the transmittance of the two passes at normal incidence
vec3 openpbrCoatTransmittance(vec3 color, float eta, float mu) {
  float mut = sqrt(max(1.0 - (1.0 - mu * mu) / (eta * eta), 1e-8));
  return pow(max(color, vec3(1e-6)), vec3(0.5 / mut));
}

bool openpbrHasLayers(OpenPBR m) { return m.coatWeight > 0.0 || m.fuzzWeight > 0.0; }

// what the fuzz and the coat let through to the base, for light leaving toward wo after
// crossing the coat at external cosine muI
vec3 openpbrUnderLayers(OpenPBR m, float muO, float muI) {
  float fuzz = 1.0 - m.fuzzWeight * openpbrFuzzAlbedo(muO, m.fuzzRoughness);
  vec3 coat = openpbrCoatTransmittance(m.coatColor, m.coatEta, muO) *
              openpbrCoatTransmittance(m.coatColor, m.coatEta, muI) *
              (1.0 - openpbrFresnel(muO, m.coatEta));
  return fuzz * mix(vec3(1.0), coat, m.coatWeight);
}

// the factor of emission from the base seen at cosine muO
vec3 openpbrEmissionFactor(OpenPBR m, float muO) {
  if (m.interior) {
    return vec3(1.0);
  }
  float fuzz = 1.0 - m.fuzzWeight * openpbrFuzzAlbedo(muO, m.fuzzRoughness);
  vec3 coat = openpbrCoatTransmittance(m.coatColor, m.coatEta, muO) *
              (1.0 - openpbrFresnel(muO, m.coatEta));
  return fuzz * mix(vec3(1.0), coat, m.coatWeight);
}

// the fuzz and coat lobes times cos(wi), from the cosines of wo and wi with the normal and
// with their half vector h; zero without layers
vec3 openpbrLayerLobes(OpenPBR m, float cosO, float cosI, float cosH, float cosOH) {
  vec3 f = vec3(0.0);
  if (m.fuzzWeight > 0.0) {
    f += m.fuzzWeight * m.fuzzColor * openpbrFuzz(cosO, cosI, cosH, m.fuzzRoughness);
  }
  if (m.coatWeight > 0.0) {
    float fuzzAbove = 1.0 - m.fuzzWeight * openpbrFuzzAlbedo(cosO, m.fuzzRoughness);
    f += fuzzAbove * m.coatWeight * openpbrFresnel(cosOH, m.coatEta) *
         openpbrGgxReflectionIso(cosO, cosI, cosH, m.coatAlpha);
  }
  return f;
}

// ---- evaluation ----

// the diffuse lobe alone, times cos(wi), for wi above the surface (the base's Lambertian lobe
// under the layers; zero inside a body)
vec3 openpbrEvalDiffuse(OpenPBR m, vec3 wo, vec3 wi) {
  if (wo.z <= 0.0 || wi.z <= 0.0 || m.interior) {
    return vec3(0.0);
  }
  return openpbrUnderLayers(m, wo.z, wi.z) *
         ((1.0 - m.metallic) * (1.0 - m.transmission) * (1.0 - openpbrFresnel(wo.z, m.baseEta))) *
         m.base * (wi.z / OPENPBR_PI);
}

// BSDF times cos(wi) for wi above the surface, split into the base diffuse lobe and the
// remaining (specular, coat, fuzz) lobes so callers do not evaluate the layers twice.
void openpbrEvalReflectionSplit(OpenPBR m, vec3 wo, vec3 wi, out vec3 diffuse,
                                out vec3 specular) {
  diffuse = vec3(0.0);
  specular = vec3(0.0);
  if (wo.z <= 0.0 || wi.z <= 0.0) {
    return;
  }
  vec3 h = normalize(wo + wi);
  float oh = max(dot(wo, h), 1e-6);
  float spec = openpbrGgxReflection(wo, wi, h, m.alpha);
  if (m.interior) {
    specular = vec3(openpbrFresnel(oh, m.eta) * spec);
    return;
  }
  vec3 under = openpbrUnderLayers(m, wo.z, wi.z);
  vec3 f = openpbrLayerLobes(m, wo.z, wi.z, h.z, oh);
  vec3 baseSpec =
      m.metallic * openpbrMetalFresnel(oh, m.base, m.specularColor, m.specularWeight) * spec;
  float dielectric = 1.0 - m.metallic;
  if (dielectric > 0.0) {
    baseSpec += dielectric * m.specularColor * (openpbrFresnel(oh, m.baseEta) * spec);
    diffuse = under *
              (dielectric * (1.0 - m.transmission) * (1.0 - openpbrFresnel(wo.z, m.baseEta))) *
              m.base * (wi.z / OPENPBR_PI);
  }
  specular = f + under * baseSpec;
}

// BSDF times cos(wi) for wi above the surface
vec3 openpbrEvalReflection(OpenPBR m, vec3 wo, vec3 wi) {
  vec3 diffuse, specular;
  openpbrEvalReflectionSplit(m, wo, wi, diffuse, specular);
  return diffuse + specular;
}

// the transmission lobe times |cos(wi)| for wi below the surface, without its lobe weight
float openpbrTransmissionLobe(OpenPBR m, vec3 wo, vec3 wi) {
  if (m.thinWalled) {
    // a thin sheet: the rough reflection mirrored through the surface, no refraction
    vec3 wr = vec3(wi.xy, -wi.z);
    vec3 h = normalize(wo + wr);
    return (1.0 - openpbrFresnel(dot(wo, h), m.baseEta)) *
           openpbrGgxReflection(wo, wr, h, m.alpha);
  }
  vec3 h = normalize(wo + wi * m.eta);
  h = h.z < 0.0 ? -h : h;
  float oh = dot(wo, h);
  float ih = dot(wi, h);
  if (oh <= 0.0 || ih >= 0.0) {
    return 0.0;
  }
  float denom = ih + oh / m.eta;
  float jacobian = abs(ih) / (denom * denom); // dh / dwi
  return openpbrGgxD(h, m.alpha) * openpbrGgxG2(wo, wi, m.alpha) *
         (1.0 - openpbrFresnel(oh, m.eta)) * oh * jacobian / (wo.z * m.eta * m.eta);
}

// the weight of the transmission lobe; light entering also takes the interface tint
vec3 openpbrTransmissionWeight(OpenPBR m, vec3 wo, vec3 wi) {
  if (m.interior) {
    return vec3(1.0);
  }
  return openpbrUnderLayers(m, wo.z, abs(wi.z)) * ((1.0 - m.metallic) * m.transmission) *
         m.transmissionTint;
}

// ---- sampling ----

struct OpenPBRLobes {
  float fuzz;         // cosine-weighted
  float coat;         // coat GGX reflection
  float specular;     // base GGX reflection (metal and dielectric)
  float diffuse;      // cosine-weighted
  float transmission; // base GGX transmission
};

// lobe selection probabilities: each lobe's estimated albedo toward wo, every lobe that can
// contribute kept at 10% or more of the total
OpenPBRLobes openpbrLobes(OpenPBR m, vec3 wo) {
  OpenPBRLobes p;
  if (m.interior) {
    float fr = openpbrFresnel(wo.z, m.eta);
    p.fuzz = 0.0;
    p.coat = 0.0;
    p.diffuse = 0.0;
    p.specular = clamp(fr, 0.1, 0.9);
    p.transmission = 1.0 - p.specular;
    return p;
  }
  float fuzzE = m.fuzzWeight * openpbrFuzzAlbedo(wo.z, m.fuzzRoughness);
  float fc = openpbrFresnel(wo.z, m.coatEta);
  vec3 coatThrough = openpbrCoatTransmittance(m.coatColor, m.coatEta, wo.z);
  float under = (1.0 - fuzzE) *
                mix(1.0, openpbrLuminance(coatThrough) * (1.0 - fc), m.coatWeight);
  float fd = openpbrFresnel(wo.z, m.baseEta);
  float dielectric = 1.0 - m.metallic;
  p.fuzz = fuzzE * openpbrLuminance(m.fuzzColor);
  p.coat = (1.0 - fuzzE) * m.coatWeight * fc;
  p.specular =
      under *
      (m.metallic * openpbrLuminance(openpbrMetalFresnel(wo.z, m.base, m.specularColor,
                                                           m.specularWeight)) +
       dielectric * fd * openpbrLuminance(m.specularColor));
  p.diffuse = under * dielectric * (1.0 - m.transmission) * (1.0 - fd) *
              openpbrLuminance(m.base);
  p.transmission = under * dielectric * m.transmission * (1.0 - fd) *
                   max(openpbrLuminance(m.transmissionTint), 0.05);
  float total = p.fuzz + p.coat + p.specular + p.diffuse + p.transmission;
  if (total <= 0.0) {
    p.fuzz = p.coat = p.diffuse = p.transmission = 0.0;
    p.specular = 1.0;
    return p;
  }
  float floorP = 0.1 * total;
  p.fuzz = p.fuzz > 0.0 ? max(p.fuzz, floorP) : 0.0;
  p.coat = p.coat > 0.0 ? max(p.coat, floorP) : 0.0;
  p.specular = p.specular > 0.0 ? max(p.specular, floorP) : 0.0;
  p.diffuse = p.diffuse > 0.0 ? max(p.diffuse, floorP) : 0.0;
  p.transmission = p.transmission > 0.0 ? max(p.transmission, floorP) : 0.0;
  total = p.fuzz + p.coat + p.specular + p.diffuse + p.transmission;
  p.fuzz /= total;
  p.coat /= total;
  p.specular /= total;
  p.diffuse /= total;
  p.transmission /= total;
  return p;
}

// solid-angle pdf of sampling wi above the surface
float openpbrPdfReflection(OpenPBR m, OpenPBRLobes p, vec3 wo, vec3 wi) {
  if (wo.z <= 0.0 || wi.z <= 0.0) {
    return 0.0;
  }
  vec3 h = normalize(wo + wi);
  float pdf = (p.fuzz + p.diffuse) * wi.z / OPENPBR_PI +
              p.specular * openpbrGgxReflectionPdf(wo, h, m.alpha);
  if (p.coat > 0.0) {
    pdf += p.coat * openpbrGgxReflectionPdf(wo, h, vec2(m.coatAlpha));
  }
  return pdf;
}

// solid-angle pdf of the transmission lobe sampling wi below the surface
float openpbrPdfTransmission(OpenPBR m, vec3 wo, vec3 wi) {
  if (m.thinWalled) {
    vec3 wr = vec3(wi.xy, -wi.z);
    return openpbrGgxReflectionPdf(wo, normalize(wo + wr), m.alpha);
  }
  vec3 h = normalize(wo + wi * m.eta);
  h = h.z < 0.0 ? -h : h;
  float oh = dot(wo, h);
  float ih = dot(wi, h);
  if (oh <= 0.0 || ih >= 0.0) {
    return 0.0;
  }
  float denom = ih + oh / m.eta;
  return openpbrGgxG1(wo, m.alpha) * oh * openpbrGgxD(h, m.alpha) / wo.z * abs(ih) /
         (denom * denom);
}

struct OpenPBRSample {
  vec3 wi;
  vec3 weight;        // BSDF * |cos(wi)| / pdf; zero when the sample is lost
  vec3 diffuseWeight; // the diffuse lobe's part of `weight`
  bool transmitted;   // wi crossed the surface
};

// one sample; u in [0, 1)^3
OpenPBRSample openpbrSample(OpenPBR m, vec3 wo, vec3 u) {
  OpenPBRSample s;
  s.weight = vec3(0.0);
  s.diffuseWeight = vec3(0.0);
  s.transmitted = false;
  OpenPBRLobes p = openpbrLobes(m, wo);
  float x = u.x;
  if (x < p.transmission) {
    vec3 h = openpbrSampleGgxVisible(wo, m.alpha, u.yz);
    if (m.thinWalled) {
      vec3 wr = reflect(-wo, h);
      s.wi = vec3(wr.xy, -wr.z);
    } else {
      s.wi = refract(-wo, h, 1.0 / m.eta);
      if (dot(s.wi, s.wi) == 0.0) {
        return s; // total internal reflection: the reflection lobes cover that direction
      }
    }
    s.transmitted = true;
    float pdf = p.transmission * openpbrPdfTransmission(m, wo, s.wi);
    if (s.wi.z < 0.0 && pdf > 0.0) {
      s.weight = openpbrTransmissionWeight(m, wo, s.wi) * openpbrTransmissionLobe(m, wo, s.wi) /
                 pdf;
    }
    return s;
  }
  x -= p.transmission;
  if (x < p.fuzz + p.diffuse) {
    float r = sqrt(u.y);
    float phi = 2.0 * OPENPBR_PI * u.z;
    s.wi = vec3(r * cos(phi), r * sin(phi), sqrt(max(1.0 - u.y, 0.0)));
  } else if (x < p.fuzz + p.diffuse + p.coat) {
    s.wi = reflect(-wo, openpbrSampleGgxVisible(wo, vec2(m.coatAlpha), u.yz));
  } else {
    s.wi = reflect(-wo, openpbrSampleGgxVisible(wo, m.alpha, u.yz));
  }
  float pdf = openpbrPdfReflection(m, p, wo, s.wi);
  if (s.wi.z > 0.0 && pdf > 0.0) {
    vec3 diffuse, specular;
    openpbrEvalReflectionSplit(m, wo, s.wi, diffuse, specular);
    s.weight = (diffuse + specular) / pdf;
    s.diffuseWeight = diffuse / pdf;
  }
  return s;
}

// ---- the shading parameters of a material ----

// `base`, `metallic`, `roughness` and `transmission` have the material's textures applied;
// `interior`: wo is inside the body of a translucent, not thin-walled material
OpenPBR openpbrSurface(Material mat, vec3 base, float metallic, float roughness,
                       float transmission, bool interior) {
  OpenPBR m;
  m.base = clamp(mat.baseWeight * base, 0.0, 1.0);
  m.metallic = clamp(metallic, 0.0, 1.0);
  m.specularColor = clamp(mat.specularColor, 0.0, 1.0);
  m.specularWeight = max(mat.specularWeight, 0.0);
  m.thinWalled = mat.thinWalled != 0;
  m.transmission = clamp(transmission, 0.0, 1.0);
  m.interior = interior && !m.thinWalled && m.transmission > 0.0;
  m.coatWeight = m.interior ? 0.0 : clamp(mat.coatWeight, 0.0, 1.0);
  m.coatColor = clamp(mat.coatColor, 0.0, 1.0);
  m.coatEta = max(mat.coatIor, 1.0);
  float coatRoughness = clamp(mat.coatRoughness, 0.0, 1.0);
  m.coatAlpha = max(coatRoughness * coatRoughness, OPENPBR_MIN_ALPHA);
  m.fuzzWeight = m.interior ? 0.0 : clamp(mat.fuzzWeight, 0.0, 1.0);
  m.fuzzColor = clamp(mat.fuzzColor, 0.0, 1.0);
  m.fuzzRoughness = clamp(mat.fuzzRoughness, 0.0, 1.0);

  // the coat roughens the base: convolve the two lobes' slope variances
  float r = clamp(roughness, 0.0, 1.0);
  float c4 = coatRoughness * coatRoughness * coatRoughness * coatRoughness;
  r = mix(r, pow(min(1.0, r * r * r * r + 2.0 * c4), 0.25), m.coatWeight);
  m.alpha = openpbrAlpha(r, clamp(mat.specularAnisotropy, 0.0, 1.0));

  // The base's IOR ratio against the medium above it: against the coat where the coat covers the
  // base (OpenPBR's specular_ior_ratio). A coat denser than the base would make the base's
  // Fresnel total-internal-reflect, which no ray entering through the coat can do: the spec
  // inverts that ratio instead of showing the bright ring it would otherwise produce.
  float n = max(mat.ior, 1.0);
  float ratio = n / m.coatEta;
  ratio = ratio < 1.0 ? 1.0 / ratio : ratio;
  m.baseEta = openpbrModulatedEta(mix(n, ratio, m.coatWeight), m.specularWeight);
  // Refraction and the 1 / eta^2 radiance scaling must use the same ratio entering and leaving a
  // body, or a round trip through it gains energy; the coat's ratio stays out of it.
  float refraction = openpbrRefractionEta(openpbrModulatedEta(n, m.specularWeight));
  m.eta = m.interior ? 1.0 / refraction : refraction;
  if (m.interior) {
    m.baseEta = m.eta; // the bare interface seen from inside the body
  }

  // transmission_depth 0 (or a thin wall) tints at the surface; otherwise the medium absorbs
  m.transmissionTint = (m.thinWalled || mat.transmissionDepth <= 0.0)
                           ? clamp(mat.transmissionColor, 0.0, 1.0)
                           : vec3(1.0);
  return m;
}

// a plain rough dielectric without coat or fuzz, for geometry that has no material record
OpenPBR openpbrDielectric(vec3 base, float roughness, float ior, float transmission) {
  OpenPBR m;
  m.base = clamp(base, 0.0, 1.0);
  m.metallic = 0.0;
  m.specularColor = vec3(1.0);
  m.specularWeight = 1.0;
  m.eta = openpbrRefractionEta(max(ior, 1.0));
  m.baseEta = m.eta;
  m.alpha = openpbrAlpha(clamp(roughness, 0.0, 1.0), 0.0);
  m.transmission = clamp(transmission, 0.0, 1.0);
  m.transmissionTint = vec3(1.0);
  m.thinWalled = false;
  m.interior = false;
  m.coatWeight = 0.0;
  m.coatColor = vec3(1.0);
  m.coatAlpha = OPENPBR_MIN_ALPHA;
  m.coatEta = 1.5;
  m.fuzzWeight = 0.0;
  m.fuzzColor = vec3(1.0);
  m.fuzzRoughness = 0.5;
  return m;
}

// absorption coefficient of the interior medium (zero without transmission_depth)
vec3 openpbrAbsorption(Material mat) {
  if (mat.thinWalled != 0 || mat.transmissionDepth <= 0.0) {
    return vec3(0.0);
  }
  return -log(clamp(mat.transmissionColor, vec3(1e-4), vec3(1.0))) / mat.transmissionDepth;
}

#endif
