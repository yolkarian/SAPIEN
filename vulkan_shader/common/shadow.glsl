#ifndef SAPIEN_SHADOW_GLSL
#define SAPIEN_SHADOW_GLSL

const int SAPIEN_DIRECTIONAL_SHADOW_CASCADES = 3;

vec3 project(mat4 proj, vec3 point) {
  vec4 v = proj * vec4(point, 1);
  return v.xyz / v.w;
}

const int PCF_SAMPLE_COUNT = 16;
const vec2 PCF_SAMPLES[PCF_SAMPLE_COUNT] = {
  vec2(-0.94201624, -0.39906216), vec2(0.94558609, -0.76890725),
  vec2(-0.09418410, -0.92938870), vec2(0.34495938, 0.29387760),
  vec2(-0.91588581, 0.45771432), vec2(-0.81544232, -0.87912464),
  vec2(-0.38277543, 0.27676845), vec2(0.97484398, 0.75648379),
  vec2(0.44323325, -0.97511554), vec2(0.53742981, -0.47373420),
  vec2(-0.26496911, -0.41893023), vec2(0.79197514, 0.19090188),
  vec2(-0.24188840, 0.99706507), vec2(-0.81409955, 0.91437590),
  vec2(0.19984126, 0.78641367), vec2(0.14383161, -0.14100790)
};

// Interleaved gradient noise rotation in [0, 2*pi) for banding-free PCSS filtering.
mat2 sapienShadowRotation(vec2 fragCoord) {
  float n = fract(52.9829189 * fract(dot(fragCoord, vec2(0.06711056, 0.00583715))));
  float a = 6.28318530718 * n;
  float c = cos(a);
  float s = sin(a);
  return mat2(c, -s, s, c);
}

// Analytical receiver-plane depth gradient d(ndcZ)/d(uv) for an orthographic shadow projection,
// derived from the receiver normal in light view space (nL.z > 0 faces the light). At grazing
// incidence nL.z gets small; clamping it low keeps the gradient exact instead of shrinking the
// bias into self-shadowing (a clamp at 0.25 darkened unoccluded receivers below NoL ~ 0.1).
vec2 sapienOrthoReceiverPlaneBias(vec3 nL, mat4 shadowProj) {
  float nz = max(nL.z, 1e-3);
  float dzdu = (-nL.x / nz) * (2.0 / shadowProj[0][0]) * shadowProj[2][2];
  float dzdv = (-nL.y / nz) * (2.0 / shadowProj[1][1]) * shadowProj[2][2];
  return vec2(dzdu, dzdv);
}

// Evaluate one cascade layer of a directional shadow map array with receiver slope bias and
// optional PCSS soft shadows when softness > 0.
float sapienDirectionalCascadeShadow(sampler2DArray shadowArray, int cascade, vec3 coord,
                                     mat4 shadowView, mat4 shadowProj, vec3 worldNormal,
                                     float softness, vec2 fragCoord) {
  if (coord.z <= 0.0 || coord.z >= 1.0 ||
      any(lessThanEqual(coord.xy, vec2(0.0))) ||
      any(greaterThanEqual(coord.xy, vec2(1.0)))) {
    return 1.0;
  }

  float resolution = float(textureSize(shadowArray, 0).x);
  float invRes = 1.0 / max(resolution, 1.0);
  float absP00 = max(abs(shadowProj[0][0]), 1e-6);
  float absP22 = max(abs(shadowProj[2][2]), 1e-6);
  float depthRange = 1.0 / absP22;
  float texelWorld = 2.0 / (absP00 * max(resolution, 1.0));

  vec3 nL = normalize(mat3(shadowView) * worldNormal);
  float NoL = clamp(nL.z, 0.0, 1.0);
  float sinTheta = sqrt(max(1.0 - NoL * NoL, 0.0));
  float tanTheta = min(sinTheta / max(NoL, 0.05), 6.0);
  float baseBias = max((0.75 + 1.25 * tanTheta) * texelWorld / depthRange, 2e-5);
  vec2 planeBias = sapienOrthoReceiverPlaneBias(nL, shadowProj);

  float filterUV = 1.5 * invRes;
  mat2 rot = mat2(1.0);
  if (softness > 0.0) {
    rot = sapienShadowRotation(fragCoord);
    float searchDist = min(depthRange * 0.35, 6.0);
    float searchUV = clamp(softness * searchDist * 0.5 * absP00, 2.0 * invRes, 0.08);
    float blockerSum = 0.0;
    int blockerCount = 0;
    for (int i = 0; i < PCF_SAMPLE_COUNT; ++i) {
      vec2 offset = (rot * PCF_SAMPLES[i]) * searchUV;
      vec2 sampleUV = clamp(coord.xy + offset, vec2(invRes), vec2(1.0 - invRes));
      vec2 clampedOffset = sampleUV - coord.xy;
      float sampleDepth = texture(shadowArray, vec3(sampleUV, float(cascade))).x;
      float expectedReceiver = coord.z + dot(clampedOffset, planeBias) - baseBias;
      if (sampleDepth < expectedReceiver) {
        blockerSum += sampleDepth;
        blockerCount++;
      }
    }
    if (blockerCount == 0) {
      return 1.0;
    }
    float avgBlockerDepth = blockerSum / float(blockerCount);
    float blockerDist = max(coord.z - avgBlockerDepth, 0.0) * depthRange;
    float penumbraWorld = softness * blockerDist;
    filterUV = clamp(penumbraWorld * 0.5 * absP00, 1.5 * invRes, 0.08);
  }

  float visibility = 0.0;
  for (int i = 0; i < PCF_SAMPLE_COUNT; ++i) {
    vec2 offset = (rot * PCF_SAMPLES[i]) * filterUV;
    vec2 sampleUV = clamp(coord.xy + offset, vec2(invRes), vec2(1.0 - invRes));
    vec2 clampedOffset = sampleUV - coord.xy;
    float shadowDepth = texture(shadowArray, vec3(sampleUV, float(cascade))).x;
    float receiverDepth = coord.z + dot(clampedOffset, planeBias) - baseBias;
    visibility += shadowDepth >= receiverDepth ? 1.0 : 0.0;
  }
  return visibility / float(PCF_SAMPLE_COUNT);
}

// Sample a 3-cascade directional shadow map array, selecting the finest cascade covering the
// receiver and smoothly blending across cascade borders.
float sapienDirectionalShadow(sampler2DArray shadowArray,
                              mat4 view0, mat4 proj0,
                              mat4 view1, mat4 proj1,
                              mat4 view2, mat4 proj2,
                              vec4 worldPos, vec3 worldNormal,
                              float softness, vec2 fragCoord) {
  vec3 c0 = (proj0 * (view0 * worldPos)).xyz;
  c0.xy = c0.xy * 0.5 + 0.5;
  float edge0 = max(abs(c0.x - 0.5), abs(c0.y - 0.5));
  if (edge0 < 0.49 && c0.z > 0.0 && c0.z < 1.0) {
    float v0 = sapienDirectionalCascadeShadow(shadowArray, 0, c0, view0, proj0, worldNormal,
                                              softness, fragCoord);
    if (edge0 > 0.42) {
      vec3 c1 = (proj1 * (view1 * worldPos)).xyz;
      c1.xy = c1.xy * 0.5 + 0.5;
      float v1 = sapienDirectionalCascadeShadow(shadowArray, 1, c1, view1, proj1, worldNormal,
                                                softness, fragCoord);
      return mix(v0, v1, smoothstep(0.42, 0.49, edge0));
    }
    return v0;
  }

  vec3 c1 = (proj1 * (view1 * worldPos)).xyz;
  c1.xy = c1.xy * 0.5 + 0.5;
  float edge1 = max(abs(c1.x - 0.5), abs(c1.y - 0.5));
  if (edge1 < 0.49 && c1.z > 0.0 && c1.z < 1.0) {
    float v1 = sapienDirectionalCascadeShadow(shadowArray, 1, c1, view1, proj1, worldNormal,
                                              softness, fragCoord);
    if (edge1 > 0.42) {
      vec3 c2 = (proj2 * (view2 * worldPos)).xyz;
      c2.xy = c2.xy * 0.5 + 0.5;
      float v2 = sapienDirectionalCascadeShadow(shadowArray, 2, c2, view2, proj2, worldNormal,
                                                softness, fragCoord);
      return mix(v1, v2, smoothstep(0.42, 0.49, edge1));
    }
    return v1;
  }

  vec3 c2 = (proj2 * (view2 * worldPos)).xyz;
  c2.xy = c2.xy * 0.5 + 0.5;
  return sapienDirectionalCascadeShadow(shadowArray, 2, c2, view2, proj2, worldNormal,
                                        softness, fragCoord);
}

// Perspective 2D shadow map (spot and textured lights) with receiver slope bias and optional PCSS
// when lightRadius > 0.
float sapienPerspectiveShadow(sampler2D shadowTex, vec3 coord, mat4 shadowView, mat4 shadowProj,
                              vec3 worldNormal, float lightRadius, vec2 fragCoord) {
  if (coord.z <= 0.0 || coord.z >= 1.0 ||
      any(lessThanEqual(coord.xy, vec2(0.0))) ||
      any(greaterThanEqual(coord.xy, vec2(1.0)))) {
    return 1.0;
  }

  float resolution = float(textureSize(shadowTex, 0).x);
  float invRes = 1.0 / max(resolution, 1.0);
  float dReceiver = shadowProj[3][2] / (coord.z + shadowProj[2][2]);
  float f = max(abs(shadowProj[0][0]), 1e-6);
  float texelWorld = 2.0 * dReceiver / (f * max(resolution, 1.0));
  float dNdcDd = abs(shadowProj[3][2]) / max(dReceiver * dReceiver, 1e-6);

  vec3 nL = normalize(mat3(shadowView) * worldNormal);
  vec3 posL = vec3((2.0 * coord.x - 1.0) * dReceiver / shadowProj[0][0],
                   (2.0 * coord.y - 1.0) * dReceiver / shadowProj[1][1],
                   -dReceiver);
  float NoL = clamp(dot(nL, normalize(-posL)), 0.0, 1.0);
  float sinTheta = sqrt(max(1.0 - NoL * NoL, 0.0));
  float tanTheta = min(sinTheta / max(NoL, 0.05), 6.0);
  float biasWorld = texelWorld * (0.75 + 1.25 * tanTheta);
  float baseBias = max(biasWorld * dNdcDd, 1.5e-5);

  // Exact receiver-plane gradient in hyperbolic NDC depth: for a plane X.nL = posL.nL,
  // 1/d(u,v) = (ray(u,v).nL) / (posL.nL) is affine in (u,v), so d(ndcZ)/d(u,v) is exact
  // anywhere in the frustum (not only at the optical axis).
  vec2 planeBias = vec2(0.0);
  float posLDist = max(length(posL), 1e-6);
  float nDotNegPos = dot(nL, -posL);
  // the exact gradient only needs the receiver to face the light at all: gating it on a larger
  // cosine left grazing receivers with the flat centre-depth comparison, which shadows them
  if (nDotNegPos > 1e-4 * posLDist) {
    float scale = abs(shadowProj[3][2]) / nDotNegPos;
    planeBias = vec2(nL.x * (2.0 / shadowProj[0][0]),
                     nL.y * (2.0 / shadowProj[1][1])) * scale;
  }

  float filterUV = 1.5 * invRes;
  mat2 rot = mat2(1.0);
  if (lightRadius > 0.0) {
    rot = sapienShadowRotation(fragCoord);
    float searchUV = clamp(0.75 * f * lightRadius / max(dReceiver, 1e-3), 2.0 * invRes, 0.06);
    float blockerDepthSum = 0.0;
    int blockerCount = 0;
    for (int i = 0; i < PCF_SAMPLE_COUNT; ++i) {
      vec2 offset = (rot * PCF_SAMPLES[i]) * searchUV;
      vec2 sampleUV = clamp(coord.xy + offset, vec2(invRes), vec2(1.0 - invRes));
      vec2 clampedOffset = sampleUV - coord.xy;
      float sampleDepth = texture(shadowTex, sampleUV).x;
      float expectedReceiver = coord.z + dot(clampedOffset, planeBias) - baseBias;
      if (sampleDepth < expectedReceiver) {
        blockerDepthSum += shadowProj[3][2] / (sampleDepth + shadowProj[2][2]);
        blockerCount++;
      }
    }
    if (blockerCount == 0) {
      return 1.0;
    }
    float avgBlockerD = blockerDepthSum / float(blockerCount);
    float penumbraWorld = lightRadius * max(dReceiver - avgBlockerD, 0.0) / max(avgBlockerD, 1e-4);
    filterUV = clamp(penumbraWorld * 0.5 * f / max(dReceiver, 1e-3), 1.5 * invRes, 0.06);
  }

  float visibility = 0.0;
  for (int i = 0; i < PCF_SAMPLE_COUNT; ++i) {
    vec2 offset = (rot * PCF_SAMPLES[i]) * filterUV;
    vec2 sampleUV = clamp(coord.xy + offset, vec2(invRes), vec2(1.0 - invRes));
    vec2 clampedOffset = sampleUV - coord.xy;
    float shadowDepth = texture(shadowTex, sampleUV).x;
    float receiverDepth = coord.z + dot(clampedOffset, planeBias) - baseBias;
    visibility += shadowDepth >= receiverDepth ? 1.0 : 0.0;
  }
  return visibility / float(PCF_SAMPLE_COUNT);
}

// Omnidirectional cubemap shadow (point lights) with receiver slope bias, 16-tap cube PCF, and
// optional PCSS when lightRadius > 0. `wsl` is world-space (lightPos - receiverPos).
float sapienPointShadow(samplerCube shadowCube, vec3 wsl, mat4 shadowProj, vec3 worldNormal,
                        float lightRadius, vec2 fragCoord) {
  vec3 v = abs(wsl);
  float maxAxis = max(max(v.x, v.y), v.z);
  if (maxAxis <= 1e-6) {
    return 1.0;
  }
  vec4 p = shadowProj * vec4(0.0, 0.0, -maxAxis, 1.0);
  float pixelDepth = p.z / p.w;
  if (pixelDepth <= 0.0 || pixelDepth >= 1.0) {
    return 1.0;
  }

  float resolution = float(textureSize(shadowCube, 0).x);
  float dist = max(length(wsl), 1e-6);
  vec3 toLight = wsl / dist;
  float texelWorld = 2.0 * maxAxis / max(resolution, 1.0);
  float dNdcDd = abs(shadowProj[3][2]) / max(maxAxis * maxAxis, 1e-6);

  float NoL = clamp(dot(worldNormal, toLight), 0.0, 1.0);
  float sinTheta = sqrt(max(1.0 - NoL * NoL, 0.0));
  float tanTheta = min(sinTheta / max(NoL, 0.05), 6.0);
  float biasWorld = texelWorld * (0.75 + 1.25 * tanTheta);
  float baseBias = max(biasWorld * dNdcDd, 1.5e-5);
  float wslDotN = dot(wsl, worldNormal);

  vec3 up = abs(toLight.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 T = normalize(cross(up, toLight));
  vec3 B = cross(toLight, T);

  float filterWorld = 1.5 * texelWorld;
  mat2 rot = mat2(1.0);
  if (lightRadius > 0.0) {
    rot = sapienShadowRotation(fragCoord);
    float searchWorld = clamp(lightRadius * 0.75, 2.0 * texelWorld, 0.08 * maxAxis);
    float blockerDepthSum = 0.0;
    int blockerCount = 0;
    for (int i = 0; i < PCF_SAMPLE_COUNT; ++i) {
      vec2 disk = (rot * PCF_SAMPLES[i]) * searchWorld;
      vec3 deltaW = disk.x * T + disk.y * B;
      vec3 wSample = wsl - deltaW;
      float sampleDepth = texture(shadowCube, wSample).x;
      // Exact receiver tangent-plane cubemap depth along wSample: handles off-axis major-axis
      // change and hyperbolic depth without linearization error.
      float expectedReceiver = pixelDepth - baseBias;
      if (wslDotN > 1e-4 * dist) {
        float wDotN = max(dot(wSample, worldNormal), 0.25 * wslDotN);
        vec3 va = abs(wSample);
        float sampleD = max(max(va.x, va.y), va.z) * (wslDotN / wDotN);
        expectedReceiver = (-shadowProj[2][2] + shadowProj[3][2] / max(sampleD, 1e-4)) - baseBias;
      }
      if (sampleDepth < expectedReceiver) {
        blockerDepthSum += shadowProj[3][2] / (sampleDepth + shadowProj[2][2]);
        blockerCount++;
      }
    }
    if (blockerCount == 0) {
      return 1.0;
    }
    float avgBlockerD = blockerDepthSum / float(blockerCount);
    float penumbraWorld = lightRadius * max(maxAxis - avgBlockerD, 0.0) / max(avgBlockerD, 1e-4);
    filterWorld = clamp(penumbraWorld, 1.5 * texelWorld, 0.12 * maxAxis);
  }

  float visibility = 0.0;
  for (int i = 0; i < PCF_SAMPLE_COUNT; ++i) {
    vec2 disk = (rot * PCF_SAMPLES[i]) * filterWorld;
    vec3 deltaW = disk.x * T + disk.y * B;
    vec3 wSample = wsl - deltaW;
    float shadowDepth = texture(shadowCube, wSample).x;
    float receiverDepth = pixelDepth - baseBias;
    if (wslDotN > 1e-4 * dist) {
      float wDotN = max(dot(wSample, worldNormal), 0.25 * wslDotN);
      vec3 va = abs(wSample);
      float sampleD = max(max(va.x, va.y), va.z) * (wslDotN / wDotN);
      receiverDepth = (-shadowProj[2][2] + shadowProj[3][2] / max(sampleD, 1e-4)) - baseBias;
    }
    visibility += shadowDepth >= receiverDepth ? 1.0 : 0.0;
  }
  return visibility / float(PCF_SAMPLE_COUNT);
}

#endif
