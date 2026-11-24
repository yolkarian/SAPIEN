#include "./rt_random.glsl"

float schlickFresnel(float cosTheta) {
  float m = clamp(1.0 - cosTheta, 0.0, 1.0);
  float m2 = m * m;
  return m2 * m2 * m;
}

float dielectricFresnel(float cosTheta, float eta) {
// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf (eq. 22)
// eta: target ior / input ior
  float c = abs(cosTheta);
  float g = eta * eta - 1.0 + c * c;
  if (g > 0.0) {
    g = sqrt(g);
    float A = (g - c) / (g + c);
    float B = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return 0.5 * A * A * (1.0 + B * B);
  }
  return 1.0;
}

vec3 diffuseBRDF(in vec3 albedo, in float dotNL, in float dotNV) {
  return M_1_PI * albedo;
  // float f = (1.0 - 0.5 * schlickFresnel(dotNL)) * (1.0 - 0.5 * schlickFresnel(dotNV));
  // return  M_1_PI * f * albedo;
}

// D
float ggxNormalDistribution(float dotNH, float a2) {
  float d = max(dotNH * dotNH * (a2 - 1.0) + 1.0, 1e-5);
  return a2 / (d * d * M_PI);
}

// G
float smithGGXMaskingShadowing(float dotNL, float dotNV, float a2) {
  float A = dotNV * sqrt(a2 + (1.0 - a2) * dotNL * dotNL);
  float B = dotNL * sqrt(a2 + (1.0 - a2) * dotNV * dotNV);
  return 2.0 * dotNL * dotNV / max(A + B, 1e-6);
}

vec3 ggxBRDF(float dotNL, float dotNV, float dotNH, float dotVH, float roughness, vec3 F0) {
  float a = roughness * roughness;
  float a2 = a * a;

  vec3 F = F0 + (1.0 - F0) * schlickFresnel(dotVH);
  float D = ggxNormalDistribution(dotNH, a2);
  float G = smithGGXMaskingShadowing(dotNL, dotNV, a2);

  return F * D * G / max(4.0 * dotNL * dotNV, 1e-6);
}

vec3 evalDiffuse(in float dotNL, in float dotNV, in vec3 albedo) {
  return dotNL *  diffuseBRDF(albedo, dotNL, dotNV);
}

void sampleDiffuse(inout uint seed, in mat3 tbn, in vec3 V, in vec3 albedo, out vec3 L, out vec3 attenuation) {
  vec3 N = tbn[2];
  L = tbn * cosineSampleHemisphere(seed);  // PDF = dotNL / PI
  float dotNL = dot(N, L);
  float dotNV = dot(N, V);
  vec3 f = diffuseBRDF(albedo, dotNL, dotNV);
  attenuation = f * M_PI;  // attenuation = dotNL * BRDF / PDF
}


vec3 sampleNormalDistribution(inout uint seed, in float a2) {
  a2 = max(a2, 1e-6);
  float u0 = rnd(seed);
  float u1 = rnd(seed);

  // sample the GGX distribution
  float z1 = clamp((1.0 - u1) / (1.0 + (a2 - 1.0) * u1), 0.0, 1.0);
  float cosTheta = sqrt(z1);
  float sinTheta = sqrt(1.0 - z1);
  float phi = 2.0 * M_PI * u0;
  vec3 H = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

  return H;  // PDF = D * dotNH / (4 * dotVH)
}

vec3 evalGGXReflection(float dotNL, float dotNV, float dotNH, float dotVH, float roughness, vec3 F0) {
  return dotNL * ggxBRDF(dotNL, dotNV, dotNH, dotVH, roughness, F0);
}

void sampleGGXReflection(inout uint seed, in mat3 tbn, vec3 V, float roughness,vec3 F0,  out vec3 L, out vec3 attenuation) {
  vec3 N = tbn[2];
  float a = roughness * roughness;
  float a2 =  a*a;

  vec3 H = tbn * sampleNormalDistribution(seed, a2);

  L = 2 * dot(V, H) * H - V;

  float dotNL = dot(N, L);
  float dotNV = dot(N, V);
  if (dotNL <= 0.0 || dotNV <= 0.0) {
    attenuation = vec3(0.0);
    return;
  }

  float dotNH = clamp(dot(N, H), 1e-6, 1.0);
  float dotVH = clamp(dot(V, H), 1e-6, 1.0);
  vec3 F = F0 + (1.0 - F0) * schlickFresnel(dotVH);
  float G = smithGGXMaskingShadowing(dotNL, dotNV, a2);
  attenuation = F * G * dotVH / max(dotNH * dotNV, 1e-6);
}


vec3 evalGGXTransmission(vec3 N, vec3 L, vec3 V, float eta, float roughness) {
// eta is ior of source over ior of target, eta1 / eta2
// air to water: 1 / 1.3

  float a = roughness * roughness;
  float a2 = a * a;

  if (dot(N, L) < 0) {
    // refraction

    vec3 h = -(eta * V + L);
    float h2 = dot(h, h);
    if (h2 < 1e-6) {
      return vec3(0.0);
    }
    vec3 H = normalize(h);

    float dotVH = dot(V, H);
    if (dotVH < 1e-6) {
      return vec3(0.0);
    }

    float dotLH = clamp(abs(dot(L, H)), 1e-6, 1.0);
    float dotNL = clamp(abs(dot(N, L)), 1e-6, 1.0);
    float dotNV = clamp(abs(dot(N, V)), 1e-6, 1.0);
    float dotNH = clamp(abs(dot(N, H)), 1e-6, 1.0);

    float F = dielectricFresnel(dotVH, 1.0 / eta);
    float D = ggxNormalDistribution(dotNH, a2);
    float G = smithGGXMaskingShadowing(dotNL, dotNV, a2);

    float A = (dotLH * dotVH) / (dotNL * dotNV);
    float B = dotLH / eta + dotVH;
    B = clamp(B * B, 1e-6, 1.0);

    float result = A * (1.0 - F) * D * G / B;

    return dotNL * vec3(result);
  } else {
    // reflection
    vec3 H = normalize(V + L);

    float dotNH = clamp(dot(N, H), 1e-6, 1.0);
    float dotVH = clamp(dot(V, H), 1e-6, 1.0);
    float dotNL = clamp(dot(N, L), 1e-6, 1.0);
    float dotNV = clamp(dot(N, V), 1e-6, 1.0);

    float F = dielectricFresnel(dotVH, 1.0 / eta);
    float D = ggxNormalDistribution(dotNH, a2);
    float G = smithGGXMaskingShadowing(dotNL, dotNV, a2);

    float result = F * D * G / (4.0 * dotNL * dotNV);

    return dotNL * vec3(result);
  }
}

void sampleGGXTransmission(inout uint seed, in mat3 tbn, vec3 V, float eta, float roughness, out vec3 L, out vec3 attenuation) {
// eta is ior of camera side over ior of light side
// tracing air into water: 1 / 1.3
  vec3 N = tbn[2];

  float a = roughness * roughness;
  float a2 = a * a;

  vec3 H = tbn * sampleNormalDistribution(seed, a2);

  float dotVH = clamp(dot(V, H), 1e-6, 1.0);
  float dotNV = clamp(dot(N, V), 1e-6, 1.0);
  float dotNH = clamp(dot(N, H), 1e-6, 1.0);

  float F = dielectricFresnel(dotVH, 1.0 / eta);
  if (rnd(seed) <= F) {
    // reflection
    L = 2 * dot(V, H) * H - V;
    float dotNL = max(dot(N, L), 1e-6);
    float G = smithGGXMaskingShadowing(dotNL, dotNV, a2);
    float result = G * dotVH / (dotNV * dotNH);

    attenuation = vec3(result);
  } else {
    // refraction
    L = refract(-V, H, eta);
    float dotNL = dot(N, L);
    float dotLH = dot(L, H);

    float G = smithGGXMaskingShadowing(max(abs(dotNL), 1e-6), dotNV, a2);

    float result = G * dotVH / (dotNV * dotNH);

    attenuation = vec3(result);
  }
}


