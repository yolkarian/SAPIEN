#ifndef SAPIEN_GTAO_GLSL
#define SAPIEN_GTAO_GLSL

// Ground-Truth Ambient Occlusion (Jimenez et al. 2016) + depth-aware bilateral upsample.
// Runs in a half-resolution AO pass (ao.frag) and is bilaterally filtered/upsampled in
// deferred.frag so depth discontinuities stay crisp without foreground halos.

const float SAPIEN_GTAO_PI = 3.14159265359;
const float SAPIEN_GTAO_HALF_PI = 1.57079632679;

vec3 sapienReconstructViewPos(vec2 uv, float depth, mat4 invProj) {
  vec4 p = invProj * vec4(uv * 2.0 - 1.0, depth, 1.0);
  return p.xyz / p.w;
}

// Analytical cosine-weighted horizon arc integral from 0 to h (h >= 0 or h <= 0) for projected
// normal angle gamma relative to the view vector V.
float sapienGtaoIntegrateArc(float h, float gamma) {
  return 0.25 * (-cos(2.0 * h - gamma) + cos(gamma) + 2.0 * h * sin(gamma));
}

// The AO pass renders at half resolution (shader_pack.cpp sets its scale to 0.5): AO texel j
// covers full-resolution texels 2j and 2j + 1 and reads the first. The pass and the upsample
// both derive it from integer texel indices, so they read the same depth at every size.
vec2 sapienGtaoFullTexel(vec2 aoTexel, vec2 fullSize) {
  return min(aoTexel * 2.0, fullSize - vec2(1.0));
}

// Compute raw GTAO visibility in [0, 1] at the half-resolution pixel `fragCoord`.
float sapienComputeGTAO(sampler2D depthTex, sampler2D normalTex, vec2 fragCoord, mat4 proj,
                        mat4 invProj, float radius, float strength) {
  if (radius <= 1e-5 || strength <= 1e-5) {
    return 1.0;
  }

  // Read nearest-filtered depth/normal at exact full-resolution texel centers; unsnapped UVs
  // introduce sub-pixel parallax that tilts reconstructed neighbor vectors off planar surfaces.
  vec2 fullSize = vec2(textureSize(depthTex, 0));
  vec2 invFullSize = 1.0 / max(fullSize, vec2(1.0));
  vec2 centerTexel = sapienGtaoFullTexel(floor(fragCoord), fullSize);
  vec2 centerUV = (centerTexel + 0.5) * invFullSize;

  float centerDepth = texture(depthTex, centerUV).x;
  if (centerDepth >= 1.0) {
    return 1.0;
  }

  vec3 P = sapienReconstructViewPos(centerUV, centerDepth, invProj);
  vec3 rawN = texture(normalTex, centerUV).xyz;
  float nLen = length(rawN);
  if (nLen <= 1e-6) {
    return 1.0;
  }
  vec3 N = rawN / nLen;

  bool isOrtho = abs(proj[3][3]) > 0.5;
  vec3 V = isOrtho ? vec3(0.0, 0.0, 1.0) : normalize(-P);
  float denom = isOrtho ? 1.0 : max(-P.z, 1e-4);
  vec2 uvRadius = 0.5 * vec2(abs(proj[0][0]), abs(proj[1][1])) * radius / denom;
  // Scale uniformly so screenDir stays aligned with view-space sliceDirV on non-square viewports.
  float maxUvRadius = max(uvRadius.x, uvRadius.y);
  if (maxUvRadius > 0.25) {
    uvRadius *= 0.25 / maxUvRadius;
  }

  if (max(uvRadius.x * fullSize.x, uvRadius.y * fullSize.y) < 1.0) {
    return 1.0;
  }

  // Spatial interleaving across 4x4 half-res pixels; the 3x3 bilateral filter in deferred.frag
  // blends adjacent orientations into a smooth hemisphere estimate.
  float noiseAngle = fract(52.9829189 * fract(dot(fragCoord, vec2(0.06711056, 0.00583715))));
  float noiseStep = fract(dot(fragCoord, vec2(0.75487767, 0.56984029)));

  const int SLICES = 4;
  const int STEPS = 4;
  float visibilitySum = 0.0;
  float weightSum = 0.0;

  for (int s = 0; s < SLICES; ++s) {
    float phi = (float(s) + noiseAngle) * (SAPIEN_GTAO_PI / float(SLICES));
    vec2 screenDir = vec2(cos(phi), sin(phi));

    // Map screen-space UV direction to view space (+uv.x -> sign(proj[0][0]) X, +uv.y -> sign(proj[1][1]) Y).
    vec3 sliceDirV = vec3(screenDir.x * (proj[0][0] >= 0.0 ? 1.0 : -1.0),
                          screenDir.y * (proj[1][1] >= 0.0 ? 1.0 : -1.0), 0.0);
    vec3 planeNormal = cross(sliceDirV, V);
    float pnLen = length(planeNormal);
    if (pnLen <= 1e-6) {
      continue;
    }
    planeNormal /= pnLen;
    vec3 tangent = cross(V, planeNormal);

    vec3 projN = N - planeNormal * dot(N, planeNormal);
    float projNLen = length(projN);
    if (projNLen <= 1e-5) {
      continue;
    }
    vec3 projNHat = projN / projNLen;
    float cosGamma = clamp(dot(projNHat, V), -1.0, 1.0);
    float gamma = (dot(projNHat, tangent) >= 0.0 ? 1.0 : -1.0) * acos(cosGamma);

    float hPlusMax = clamp(gamma + SAPIEN_GTAO_HALF_PI, 0.0, SAPIEN_GTAO_PI);
    float hMinusMax = clamp(SAPIEN_GTAO_HALF_PI - gamma, 0.0, SAPIEN_GTAO_PI);
    float initCosPlus = cos(hPlusMax);
    float initCosMinus = cos(hMinusMax);
    float maxCosPlus = initCosPlus;
    float maxCosMinus = initCosMinus;

    for (int stepIdx = 0; stepIdx < STEPS; ++stepIdx) {
      // Quadratic step distribution concentrates samples near the receiver for crisp contact AO.
      float u = (float(stepIdx) + 0.15 + 0.85 * noiseStep) / float(STEPS);
      float t = u * u;
      vec2 duv = screenDir * uvRadius * t;

      // Positive side (+screenDir)
      vec2 uvPlus = centerUV + duv;
      if (all(greaterThan(uvPlus, vec2(0.0))) && all(lessThan(uvPlus, vec2(1.0)))) {
        vec2 texPlus = floor(uvPlus * fullSize);
        if (any(notEqual(texPlus, centerTexel))) {
          vec2 snapPlus = (texPlus + 0.5) * invFullSize;
          float dPlus = texture(depthTex, snapPlus).x;
          if (dPlus < 1.0) {
            vec3 D = sapienReconstructViewPos(snapPlus, dPlus, invProj) - P;
            float dist2 = dot(D, D);
            if (dist2 > 1e-8) {
              float dist = sqrt(dist2);
              if (dist < radius && dot(D, N) > 0.04 * dist) {
                float cosH = dot(D, V) / dist;
                float falloff = clamp((radius - dist) / max(0.35 * radius, 1e-4), 0.0, 1.0);
                maxCosPlus = max(maxCosPlus, mix(initCosPlus, cosH, falloff));
              }
            }
          }
        }
      }

      // Negative side (-screenDir)
      vec2 uvMinus = centerUV - duv;
      if (all(greaterThan(uvMinus, vec2(0.0))) && all(lessThan(uvMinus, vec2(1.0)))) {
        vec2 texMinus = floor(uvMinus * fullSize);
        if (any(notEqual(texMinus, centerTexel))) {
          vec2 snapMinus = (texMinus + 0.5) * invFullSize;
          float dMinus = texture(depthTex, snapMinus).x;
          if (dMinus < 1.0) {
            vec3 D = sapienReconstructViewPos(snapMinus, dMinus, invProj) - P;
            float dist2 = dot(D, D);
            if (dist2 > 1e-8) {
              float dist = sqrt(dist2);
              if (dist < radius && dot(D, N) > 0.04 * dist) {
                float cosH = dot(D, V) / dist;
                float falloff = clamp((radius - dist) / max(0.35 * radius, 1e-4), 0.0, 1.0);
                maxCosMinus = max(maxCosMinus, mix(initCosMinus, cosH, falloff));
              }
            }
          }
        }
      }
    }

    float h2 = clamp(acos(clamp(maxCosPlus, -1.0, 1.0)), 0.0, hPlusMax);
    float h1 = -clamp(acos(clamp(maxCosMinus, -1.0, 1.0)), 0.0, hMinusMax);
    visibilitySum += projNLen * (sapienGtaoIntegrateArc(h1, gamma) +
                                 sapienGtaoIntegrateArc(h2, gamma));
    weightSum += projNLen * (sapienGtaoIntegrateArc(-hMinusMax, gamma) +
                             sapienGtaoIntegrateArc(hPlusMax, gamma));
  }

  if (weightSum <= 1e-5) {
    return 1.0;
  }
  return clamp(visibilitySum / weightSum, 0.0, 1.0);
}

// Depth-aware 3x3 bilateral filter and upsample of the half-resolution AO target.
float sapienSampleBilateralAO(sampler2D aoTex, sampler2D depthTex, vec2 uv, float centerZ,
                              mat4 invProj, float strength, float radius) {
  float s = clamp(strength, 0.0, 1.0);
  if (s <= 1e-5 || radius <= 1e-5) {
    return 1.0;
  }

  vec2 aoSize = vec2(textureSize(aoTex, 0));
  vec2 invAoSize = 1.0 / max(aoSize, vec2(1.0));
  vec2 fullSize = vec2(textureSize(depthTex, 0));
  vec2 invFullSize = 1.0 / max(fullSize, vec2(1.0));
  // the AO pass evaluated full-resolution texel 2 * aoTexel (see sapienComputeGTAO), so derive this
  // pixel's AO texel from its own full-resolution texel rather than from uv * ceil(size / 2): at
  // odd sizes the two disagree for about a quarter of the rows and columns
  vec2 fullTexel = floor(uv * max(fullSize, vec2(1.0)));
  vec2 baseTexel = floor(fullTexel * 0.5);
  float absCenterZ = abs(centerZ);
  float depthSigma = max(0.05 * absCenterZ, 0.02);

  float sum = 0.0;
  float weightSum = 0.0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      vec2 aoTexel = clamp(baseTexel + vec2(float(dx), float(dy)), vec2(0.0), aoSize - 1.0);
      vec2 sampleUV = (aoTexel + 0.5) * invAoSize;
      vec2 depthUV = (sapienGtaoFullTexel(aoTexel, fullSize) + 0.5) * invFullSize;
      float sampleDepth = texture(depthTex, depthUV).x;
      if (sampleDepth >= 1.0) {
        continue;
      }
      float sampleZ = abs(sapienReconstructViewPos(depthUV, sampleDepth, invProj).z);
      float dz = abs(sampleZ - absCenterZ) / depthSigma;
      float depthWeight = exp(-dz * dz);
      float spatialWeight = (dx == 0 && dy == 0) ? 1.0 : ((dx == 0 || dy == 0) ? 0.5 : 0.25);
      float w = spatialWeight * depthWeight;
      sum += texture(aoTex, sampleUV).x * w;
      weightSum += w;
    }
  }

  float rawAO = weightSum > 1e-5 ? (sum / weightSum) : texture(aoTex, uv).x;
  return mix(1.0, clamp(rawAO, 0.0, 1.0), s);
}

#endif
