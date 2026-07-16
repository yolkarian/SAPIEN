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

float ShadowMapPCF(
    sampler2D shadowTexture, vec3 projectedCoordinate, float resolution,
    float searchUV, float filterSize) {
  if (projectedCoordinate.z <= 0.0 || projectedCoordinate.z >= 1.0 ||
      any(lessThanEqual(projectedCoordinate.xy, vec2(0.0))) ||
      any(greaterThanEqual(projectedCoordinate.xy, vec2(1.0)))) {
    return 1.0;
  }

  // Offset receivers by roughly one shadow texel to suppress self-shadowing.
  float receiverDepth = projectedCoordinate.z - max(1.5 / resolution, 0.00015);
  float visibility = 0.0;
  for (int i = 0; i < PCF_SAMPLE_COUNT; ++i) {
    vec2 sampleUV = projectedCoordinate.xy +
                    2.0 * filterSize * PCF_SAMPLES[i] * searchUV;
    float shadowDepth = texture(shadowTexture, sampleUV).x;
    visibility += shadowDepth >= receiverDepth ? 1.0 : 0.0;
  }
  return visibility / float(PCF_SAMPLE_COUNT);
}
