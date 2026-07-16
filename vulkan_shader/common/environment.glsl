#ifndef SAPIEN_ENVIRONMENT_GLSL
#define SAPIEN_ENVIRONMENT_GLSL

float sapienMaxComponent(vec3 value) {
  return max(value.x, max(value.y, value.z));
}

vec3 sapienProceduralEnvironment(vec3 direction, float roughness) {
  direction = normalize(direction);
  roughness = clamp(roughness, 0.0, 1.0);

  vec3 ambient = max(sceneBuffer.ambientLight.rgb, vec3(0.0));
  vec3 ground = ambient * vec3(0.42, 0.40, 0.38);
  vec3 horizon = ambient * vec3(0.95, 0.98, 1.02);
  vec3 sky = ambient * vec3(0.62, 0.78, 1.05);

  float elevation = direction.z;
  vec3 color = mix(ground, horizon, smoothstep(-0.35, 0.05, elevation));
  color = mix(color, sky, smoothstep(0.0, 0.85, elevation));

  // A broad studio light gives untextured dielectrics a readable reflection
  // instead of the previous black environment response.
  vec3 keyDirection = normalize(vec3(-0.45, 0.35, 0.82));
  float highlightPower = mix(96.0, 3.0, roughness);
  float highlight = pow(max(dot(direction, keyDirection), 0.0), highlightPower);
  color += ambient * highlight * mix(2.5, 0.18, roughness);
  return color;
}

vec3 sapienSampleEnvironment(vec3 direction, float roughness) {
  direction = normalize(direction);
  float maxMipLevel = float(max(textureQueryLevels(samplerEnvironment) - 1, 0));
  vec3 sampled = textureLod(
      samplerEnvironment, direction.xzy, clamp(roughness, 0.0, 1.0) * maxMipLevel).rgb;

  // SAPIEN stores the procedural-fallback flag in ambientLight.a so explicit
  // environment maps retain genuinely black texels.
  if (sceneBuffer.ambientLight.a < 0.5 || sapienMaxComponent(sampled) > 1e-6) {
    return sampled;
  }
  return sapienProceduralEnvironment(direction, roughness);
}

vec3 sapienBackgroundColor(vec3 direction) {
  direction = normalize(direction);
  vec3 sampled = textureLod(samplerEnvironment, direction.xzy, 0.0).rgb;
  if (sceneBuffer.ambientLight.a < 0.5 || sapienMaxComponent(sampled) > 1e-6) {
    return sampled + sceneBuffer.ambientLight.rgb;
  }
  return sapienProceduralEnvironment(direction, 0.0);
}

vec3 sapienDiffuseIBL(vec3 albedo, vec3 normal) {
  return sapienSampleEnvironment(normal, 1.0) * albedo;
}

vec3 sapienFresnelSchlickRoughness(
    vec3 fresnel, float roughness, float normalViewDot) {
  return fresnel + (max(vec3(1.0 - roughness), fresnel) - fresnel) *
                       pow(1.0 - normalViewDot, 5.0);
}

vec3 sapienSpecularIBL(
    vec3 fresnel, float roughness, vec3 normal, vec3 viewDirection) {
  float normalViewDot = clamp(dot(normal, viewDirection), 0.0, 1.0);
  vec3 reflectionDirection = reflect(-viewDirection, normal);
  vec3 radiance = sapienSampleEnvironment(reflectionDirection, roughness);
  vec2 environmentBRDF = texture(
      samplerBRDFLUT, vec2(clamp(roughness, 0.045, 1.0), normalViewDot)).xy;
  return radiance * (fresnel * environmentBRDF.x + environmentBRDF.y);
}

#endif
