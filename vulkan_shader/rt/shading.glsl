#include "integrator_common.glsl"

// Shadow rays share the path payload. Transparent occluders scale ray.transmittance in the
// any-hit shader (camera.rahit) and let the ray go on; an opaque occluder ends it (opaque
// geometry such as point sets never runs any-hit), so only a ray that reaches the shadow miss
// shader, which clears isShadow, is unoccluded.
vec3 traceShadow(vec3 origin, vec3 L, float tMax) {
  ray.isShadow = 1;
  ray.transmittance = vec3(1.0);
  traceRayEXT(tlas, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT, 0xff,
              0, 0, 1, origin, 0.0, L, tMax, 0);
  vec3 transmittance = ray.isShadow == 0 ? ray.transmittance : vec3(0.0);
  ray.isShadow = 0;
  return transmittance;
}

struct DirectLighting {
  vec3 diffuse;
  vec3 specular;
};

// BSDF times cos(N, L) toward L, split into base diffuse and remaining (specular, coat, fuzz)
// lobes. Zero below the shading or the geometric surface: light sampling only covers directions
// above both, BSDF sampling everything else.
void evalBsdfSplit(Surface s, vec3 L, out vec3 diffuse, out vec3 specular) {
  if (dot(s.N, L) <= 0.0 || dot(s.Ng, L) <= 0.0) {
    diffuse = vec3(0.0);
    specular = vec3(0.0);
    return;
  }
  openpbrEvalReflectionSplit(s.m, s.wo, surfaceLocal(s, L), diffuse, specular);
}

// Solid-angle pdf of BSDF sampling producing L above the surface.
float evalBsdfPdf(Surface s, vec3 L) {
  if (dot(s.N, L) <= 0.0) {
    return 0.0;
  }
  return openpbrPdfReflection(s.m, s.lobes, s.wo, surfaceLocal(s, L));
}

bool isBlack(vec3 c) {
  return c.x <= 0.0 && c.y <= 0.0 && c.z <= 0.0;
}

// ---------------------------------------------------------------------------------------------
// Point, directional, spot and parallelogram lights have no geometry: BSDF rays cannot hit
// them, so they are only light-sampled, with MIS weight 1, in every lightSampling mode.
// ---------------------------------------------------------------------------------------------
void tracePointLights(Surface s, inout DirectLighting result) {
  for (int i = 0; i < pointLightCount; ++i) {
    vec3 d = pointLights.l[i].position - s.pos;
    float d2 = max(dot(d, d), 1e-12);
    float dist = sqrt(d2);
    vec3 L = d / dist;
    vec3 fDiff, fSpec;
    evalBsdfSplit(s, L, fDiff, fSpec);
    if (!isBlack(fDiff + fSpec)) {
      vec3 light = traceShadow(offsetRayOrigin(s.pos, s.Ng), L, dist) * pointLights.l[i].rgb / d2;
      result.diffuse += light * fDiff;
      result.specular += light * fSpec;
    }
  }
}

void traceDirectionalLights(Surface s, inout DirectLighting result) {
  for (int i = 0; i < directionalLightCount; ++i) {
    DirectionalLight light = directionalLights.l[i];
    vec3 L = normalize(-light.direction);
    if (light.softness > 0.0) {
      vec3 tangent = abs(L.z) < 0.999 ? normalize(cross(L, vec3(0, 0, 1))) : vec3(1, 0, 0);
      vec3 bitangent = cross(L, tangent);
      float radius = sqrt(rnd(ray.seed)) * light.softness;
      float angle = 2.0 * SAPIEN_PI * rnd(ray.seed);
      L = normalize(L + radius * (cos(angle) * tangent + sin(angle) * bitangent));
    }
    vec3 fDiff, fSpec;
    evalBsdfSplit(s, L, fDiff, fSpec);
    if (!isBlack(fDiff + fSpec)) {
      vec3 li = traceShadow(offsetRayOrigin(s.pos, s.Ng), L, 1.0e30) * light.rgb;
      result.diffuse += li * fDiff;
      result.specular += li * fSpec;
    }
  }
}

void traceSpotLights(Surface s, inout DirectLighting result) {
  for (int i = 0; i < spotLightCount; ++i) {
    SpotLight light = spotLights.l[i];
    vec3 centerDir = vec3(0.0, 0.0, -1.0) * mat3(light.viewMat);
    vec3 d = light.position - s.pos;
    float d2 = max(dot(d, d), 1e-12);
    float dist = sqrt(d2);
    vec3 L = d / dist;
    float cf1 = cos(light.fovInner / 2) + 1e-5;
    float cf2 = cos(light.fovOuter / 2);
    float visibility = clamp((dot(-L, centerDir) - cf2) / (cf1 - cf2), 0.0, 1.0);
    vec3 fDiff, fSpec;
    evalBsdfSplit(s, L, fDiff, fSpec);
    if (visibility <= 0.0 || isBlack(fDiff + fSpec)) {
      continue;
    }
    vec3 texColor = vec3(1.0);
    if (light.textureId >= 0) {
      vec4 texcoord = light.projMat * (light.viewMat * vec4(s.pos, 1));
      texcoord /= texcoord.w;
      vec2 uv = texcoord.xy * 0.5 + 0.5;
      texColor = texture(textures[nonuniformEXT(light.textureId)], uv).rgb;
    }
    vec3 li = traceShadow(offsetRayOrigin(s.pos, s.Ng), L, dist) * light.rgb / d2 * visibility *
              texColor;
    result.diffuse += li * fDiff;
    result.specular += li * fSpec;
  }
}

void traceParallelogramLights(Surface s, inout DirectLighting result) {
  for (int i = 0; i < parallelogramLightCount; ++i) {
    ParallelogramLight light = parallelogramLights.l[i];
    vec3 d = light.position + rnd(ray.seed) * light.edge0 + rnd(ray.seed) * light.edge1 - s.pos;
    float d2 = max(dot(d, d), 1e-12);
    float dist = sqrt(d2);
    vec3 L = d / dist;
    vec3 lnormal = cross(light.edge0, light.edge1);
    float area = length(lnormal);
    float cosL = dot(lnormal / area, L);
    vec3 fDiff, fSpec;
    evalBsdfSplit(s, L, fDiff, fSpec);
    if (cosL <= 0.0 || isBlack(fDiff + fSpec)) {
      continue;
    }
    // one uniform area sample: Li f cosL area / d^2
    vec3 li = traceShadow(offsetRayOrigin(s.pos, s.Ng), L, dist) * light.rgb * cosL * area / d2;
    result.diffuse += li * fDiff;
    result.specular += li * fSpec;
  }
}

// ---------------------------------------------------------------------------------------------
// The environment and emissive meshes can be found both ways. lightSampling selects:
//   LIGHT_SAMPLING_MIS : light sampling and BSDF sampling, weighted by the power heuristic
//   LIGHT_SAMPLING_NEE : light sampling only (BSDF-found light that could be sampled is dropped)
//   LIGHT_SAMPLING_BSDF: BSDF sampling only
// Each present technique runs with probability q = 1 / (number of present techniques) and
// divides by it, so q is part of its pdf and of the MIS weights.
// ---------------------------------------------------------------------------------------------
float lightTechniqueProbability() {
  int present = (envWidth > 0 ? 1 : 0) + (emissiveCount > 0 ? 1 : 0);
  return present > 0 ? 1.0 / float(present) : 0.0;
}

// q times the solid-angle pdf of light-sampling the environment toward d
float envLightPdf(vec3 d) {
  if (envWidth <= 0) {
    return 0.0;
  }
  return lightTechniqueProbability() * uintBitsToFloat(envAlias.e[envDirectionCell(d, envWidth)].z);
}

vec3 envRadiance(vec3 d) {
  return envmap != 0 ? textureLod(samplerEnvironment, d.xzy, 0.0).rgb : ambientLight;
}

void estimateEnvironment(Surface s, inout DirectLighting result) {
#if !defined(LIGHT_SAMPLING_BSDF)
  float q = lightTechniqueProbability();
  if (envWidth <= 0 || rnd(ray.seed) >= q) {
    return;
  }
  uint count = uint(envWidth * (envWidth / 2));
  uint cell = min(uint(rnd(ray.seed) * float(count)), count - 1);
  uint chosen = aliasSample(envAlias.e[cell].xy, cell, uint(rnd(ray.seed) * 4294967296.0));
  vec3 L = envCellDirection(chosen, envWidth, rnd(ray.seed), rnd(ray.seed));
  float pdf = q * uintBitsToFloat(envAlias.e[chosen].z);
  vec3 fDiff, fSpec;
  evalBsdfSplit(s, L, fDiff, fSpec);
  if (pdf <= 0.0 || isBlack(fDiff + fSpec)) {
    return;
  }
  vec3 li = traceShadow(offsetRayOrigin(s.pos, s.Ng), L, 1.0e30) * envRadiance(L) / pdf;
#if defined(LIGHT_SAMPLING_MIS)
  li *= misPowerHeuristic(pdf, evalBsdfPdf(s, L));
#endif
  result.diffuse += li * fDiff;
  result.specular += li * fSpec;
#endif
}

// q times the solid-angle pdf of light-sampling triangle `index` toward a point at squared
// distance d2 seen at cosine cosL, `area2` being twice the triangle's world area
float emissiveLightPdf(uint index, float d2, float cosL, float area2) {
  return lightTechniqueProbability() * emissiveTriangles.t[index].v0.w * d2 /
         (0.5 * area2 * cosL);
}

void estimateEmissiveTriangles(Surface s, inout DirectLighting result) {
#if !defined(LIGHT_SAMPLING_BSDF)
  float q = lightTechniqueProbability();
  if (emissiveCount <= 0 || rnd(ray.seed) >= q) {
    return;
  }
  uint count = uint(emissiveCount);
  uint cell = min(uint(rnd(ray.seed) * float(count)), count - 1);
  uint index = aliasSample(emissiveAlias.e[cell], cell, uint(rnd(ray.seed) * 4294967296.0));
  EmissiveTriangle tri = emissiveTriangles.t[index];
  TlasInstance m = tlasInstances.i[floatBitsToInt(tri.e1.w)];
  vec3 e1 = instanceVector(m, tri.e1.xyz);
  vec3 e2 = instanceVector(m, tri.e2.xyz);
  vec3 cr = cross(e1, e2);
  float area2 = length(cr);
  if (area2 <= 0.0) {
    return;
  }
  // uniform point on the triangle
  float su = sqrt(rnd(ray.seed));
  float v = rnd(ray.seed);
  vec3 y = instancePoint(m, tri.v0.xyz) + su * (1.0 - v) * e1 + su * v * e2;
  vec3 d = y - s.pos;
  float d2 = dot(d, d);
  float dist = sqrt(d2);
  vec3 L = d / dist;
  float cosL = abs(dot(cr, L)) / area2;
  vec3 fDiff, fSpec;
  evalBsdfSplit(s, L, fDiff, fSpec);
  if (cosL <= 1e-6 || isBlack(fDiff + fSpec)) {
    return;
  }
  float pdf = emissiveLightPdf(index, d2, cosL, area2);
  // stop short of the emitter, which would occlude its own sample
  vec3 li = traceShadow(offsetRayOrigin(s.pos, s.Ng), L, dist * (1.0 - 1e-3)) * tri.emission.rgb /
            pdf;
#if defined(LIGHT_SAMPLING_MIS)
  li *= misPowerHeuristic(pdf, evalBsdfPdf(s, L));
#endif
  result.diffuse += li * fDiff;
  result.specular += li * fSpec;
#endif
}

DirectLighting estimateDirectSplit(Surface s) {
  DirectLighting result;
  result.diffuse = vec3(0.0);
  result.specular = vec3(0.0);
  tracePointLights(s, result);
  traceDirectionalLights(s, result);
  traceSpotLights(s, result);
  traceParallelogramLights(s, result);
  estimateEnvironment(s, result);
  estimateEmissiveTriangles(s, result);
  return result;
}

// Store in the payload what the next vertex needs for MIS, for the BSDF direction L sampled
// at s, and the continuation ray. Its origin is the exact hit point (the Position output);
// the camera ray generator traces it with a small tMin.
void setContinuation(Surface s, vec3 L) {
  bool lightSampleable = dot(s.N, L) > 0.0 && dot(s.Ng, L) > 0.0;
  ray.bsdfPdf = lightSampleable ? evalBsdfPdf(s, L) : -1.0;
  ray.envLightPdf = lightSampleable ? envLightPdf(L) : 0.0;
  ray.scatterOrigin = s.pos;
  ray.origin = s.pos;
  ray.direction = L;
}

// Sample the BSDF for the path's next direction: the payload's attenuation and continuation.
void sampleBsdf(Surface s) {
  OpenPBRSample b = openpbrSample(s.m, s.wo, vec3(rnd(ray.seed), rnd(ray.seed), rnd(ray.seed)));
  vec3 L = surfaceWorld(s, b.wi);
  // a reflection must leave above the geometric surface, a transmission below it
  if ((dot(s.Ng, L) > 0.0) == b.transmitted) {
    b.weight = vec3(0.0);
    b.diffuseWeight = vec3(0.0);
  }
  setContinuation(s, L);
  ray.attenuation = b.weight;
  ray.diffuseAttenuation = b.diffuseWeight;
}
