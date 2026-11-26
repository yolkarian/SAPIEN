#ifndef SAPIEN_TONEMAPPING_GLSL
#define SAPIEN_TONEMAPPING_GLSL

vec3 sapienSanitizeRadiance(vec3 color) {
  if (any(isnan(color)) || any(isinf(color))) {
    return vec3(0.0);
  }
  return max(color, vec3(0.0));
}

vec3 sapienGammaOETF(vec3 color) {
  color = sapienSanitizeRadiance(color);
  return clamp(pow(color, vec3(1.0 / 2.2)), 0.0, 1.0);
}

vec3 sapienSRGBOETF(vec3 color) {
  color = sapienSanitizeRadiance(color);
  bvec3 useLinearSegment = lessThanEqual(color, vec3(0.0031308));
  vec3 linearSegment = 12.92 * color;
  vec3 powerSegment = 1.055 * pow(color, vec3(1.0 / 2.4)) - 0.055;
  return clamp(mix(powerSegment, linearSegment, useLinearSegment), 0.0, 1.0);
}

const mat3 SAPIEN_ACES_INPUT_MAT = mat3(
    0.59719, 0.35458, 0.04823,
    0.07600, 0.90834, 0.01566,
    0.02840, 0.13383, 0.83777
);

const mat3 SAPIEN_ACES_OUTPUT_MAT = mat3(
     1.60475, -0.53108, -0.07367,
    -0.10208,  1.10813, -0.00605,
    -0.00327, -0.07276,  1.07602
);

vec3 sapienACESFit(vec3 color) {
  vec3 a = color * (color + 0.0245786) - 0.000090537;
  vec3 b = color * (0.983729 * color + 0.4329510) + 0.238081;
  return a / b;
}

vec3 sapienACESSRGB(vec3 color) {
  color = sapienSanitizeRadiance(color) * SAPIEN_ACES_INPUT_MAT;
  color = sapienACESFit(color);
  color = color * SAPIEN_ACES_OUTPUT_MAT;
  return sapienSRGBOETF(clamp(color, 0.0, 1.0));
}

// Khronos PBR Neutral tone mapper (preserves albedo hue; linear in [0, 0.76], smooth highlight
// roll-off above) + sRGB OETF.
vec3 sapienPBRNeutralSRGB(vec3 color) {
  color = sapienSanitizeRadiance(color);
  const float startCompression = 0.8 - 0.04;
  const float desaturation = 0.15;
  float x = min(color.r, min(color.g, color.b));
  float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
  color -= offset;
  float peak = max(color.r, max(color.g, color.b));
  if (peak < startCompression) {
    return sapienSRGBOETF(color);
  }
  const float d = 1.0 - startCompression;
  float newPeak = 1.0 - d * d / (peak + d - startCompression);
  color *= newPeak / peak;
  float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
  return sapienSRGBOETF(mix(color, vec3(newPeak), g));
}

// AgX (Troy Sobotka / Blender 4.0) polynomial contrast curve.
vec3 sapienAgXDefaultContrastApprox(vec3 x) {
  vec3 x2 = x * x;
  vec3 x4 = x2 * x2;
  return + 15.5     * x4 * x2
         - 40.14    * x4 * x
         + 31.96    * x4
         - 6.868    * x2 * x
         + 0.4298   * x2
         + 0.1191   * x
         - 0.00232;
}

vec3 sapienAgXSRGB(vec3 color) {
  color = sapienSanitizeRadiance(color);
  const mat3 agxMat = mat3(
      0.842479062253094, 0.0423282422610123, 0.0423756549057051,
      0.0784335999999992, 0.878468636469772, 0.0784336,
      0.0792237451477643, 0.0791661274605434, 0.879142973793104);
  const mat3 agxMatInv = mat3(
       1.19687900512017,  -0.0528968517574562, -0.0529716355144438,
      -0.0980208811401368, 1.15190312990417,   -0.0980434501171241,
      -0.0990297440797205,-0.0989611768448433,  1.15107367264116);
  const float minEv = -12.47393;
  const float maxEv = 4.026069;
  color = agxMat * color;
  color = clamp(log2(max(color, vec3(1e-10))), minEv, maxEv);
  color = (color - minEv) / (maxEv - minEv);
  color = sapienAgXDefaultContrastApprox(color);
  color = agxMatInv * color;
  return clamp(color, 0.0, 1.0);
}

// Hable / Uncharted 2 filmic curve + sRGB OETF.
vec3 sapienHablePartial(vec3 x) {
  const float A = 0.15;
  const float B = 0.50;
  const float C = 0.10;
  const float D = 0.20;
  const float E = 0.02;
  const float F = 0.30;
  return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

vec3 sapienFilmicSRGB(vec3 color) {
  color = sapienSanitizeRadiance(color);
  const float exposureBias = 2.0;
  const float whitePoint = 11.2;
  vec3 curr = sapienHablePartial(exposureBias * color);
  vec3 whiteScale = vec3(1.0) / sapienHablePartial(vec3(whitePoint));
  return sapienSRGBOETF(clamp(curr * whiteScale, 0.0, 1.0));
}

// toneMapper: 0 = gamma 2.2, 1 = sRGB, 2 = ACES fitted + sRGB, 3 = Khronos PBR Neutral + sRGB,
// 4 = AgX, 5 = Filmic (Hable) + sRGB.
vec3 sapienToneMap(vec3 hdrColor, float exposure, int toneMapper) {
  vec3 exposed = sapienSanitizeRadiance(hdrColor) * max(exposure, 0.0);
  if (toneMapper == 0) {
    return sapienGammaOETF(exposed);
  }
  if (toneMapper == 1) {
    return sapienSRGBOETF(exposed);
  }
  if (toneMapper == 2) {
    return sapienACESSRGB(exposed);
  }
  if (toneMapper == 3) {
    return sapienPBRNeutralSRGB(exposed);
  }
  if (toneMapper == 4) {
    return sapienAgXSRGB(exposed);
  }
  return sapienFilmicSRGB(exposed);
}

#endif
