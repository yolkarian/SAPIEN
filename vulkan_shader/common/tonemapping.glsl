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

// toneMapper: 0 = gamma 2.2, 1 = sRGB, 2 = ACES fitted + sRGB.
vec3 sapienToneMap(vec3 hdrColor, float exposure, int toneMapper) {
  vec3 exposed = sapienSanitizeRadiance(hdrColor) * max(exposure, 0.0);
  if (toneMapper == 0) {
    return sapienGammaOETF(exposed);
  }
  if (toneMapper == 1) {
    return sapienSRGBOETF(exposed);
  }
  return sapienACESSRGB(exposed);
}

#endif
