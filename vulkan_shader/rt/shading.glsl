#include "../common/bsdf.glsl"
vec3 traceDirectionalLights(vec3 pos, vec3 normal, vec3 diffuseColor, vec3 specularColor, float roughness, float transmissionWeight, float eta) {
  vec3 result = vec3(0.0);
  for (uint i = 0; i < directionalLightCount; ++i) {
    DirectionalLight light = directionalLights.l[i];
    vec3 emission = light.rgb;
    vec3 L = normalize(-light.direction);
    vec3 V = normalize(-ray.direction);
    if (light.softness > 0.0) {
      vec3 tangent = abs(L.z) < 0.999 ? normalize(cross(L, vec3(0, 0, 1)))
                                     : vec3(1, 0, 0);
      vec3 bitangent = cross(L, tangent);
      float radius = sqrt(rnd(ray.seed)) * light.softness;
      float angle = 2.0 * M_PI * rnd(ray.seed);
      L = normalize(L + radius * (cos(angle) * tangent + sin(angle) * bitangent));
    }

    uint flags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;

    shadowRay.shadowed = true;
    traceRayEXT(tlas,
                flags,
                0xff,
                0,
                0,
                1,  // miss index
                pos,
                0.001,
                L,
                1.0e6,
                1  // payload location
    );

    if (!shadowRay.shadowed) {
      vec3 H = normalize(V + L);
      float dotNH = clamp(dot(normal, H), 1e-6, 1);
      float dotVH = clamp(dot(V, H), 1e-6, 1);
      float dotNL = clamp(dot(normal, L), 1e-6, 1);
      float dotNV = clamp(dot(normal, V), 1e-6, 1);

      result += evalDiffuse(dotNL, dotNV, diffuseColor) * emission;  // diffuse
      result += evalGGXReflection(dotNL, dotNV, dotNH, dotVH, roughness, specularColor) * emission;  // specular
      if (transmissionWeight > 1e-5) {
        result += evalGGXTransmission(normal, L, V, eta, roughness) * emission * transmissionWeight;
      }
    }
  }
  return result;
}


vec3 traceParallelogramLights(vec3 pos, vec3 normal, vec3 diffuseColor, vec3 specularColor, float roughness, float transmissionWeight, float eta) {
  vec3 result = vec3(0.0);
  for (uint i = 0; i < parallelogramLightCount; ++i) {
    ParallelogramLight light = parallelogramLights.l[i];
    vec3 emission = light.rgb;
    vec3 d = light.position + rnd(ray.seed) * light.edge0 + rnd(ray.seed) * light.edge1 - pos;
    float d2 = max(dot(d, d), 1e-6);
    vec3 L = normalize(d);
    vec3 V = normalize(-ray.direction);

    vec3 lnormal = cross(light.edge0, light.edge1);
    float area = length(lnormal);
    lnormal = normalize(lnormal);
    float dotlnld = dot(lnormal, L); // light normal dot light direction

    if (dotlnld > 0.0) {
      emission = emission * dotlnld * area / d2;

      uint flags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;
      shadowRay.shadowed = true;
      traceRayEXT(tlas,
                  flags,
                  0xff,
                  0,
                  0,
                  1,  // miss index
                  pos,
                  0.001,
                  L,
                  sqrt(d2) - 0.001,
                  1  // payload location
      );

      if (!shadowRay.shadowed) {
        vec3 H = normalize(V + L);
        float dotNH = clamp(dot(normal, H), 1e-6, 1);
        float dotVH = clamp(dot(V, H), 1e-6, 1);
        float dotNL = clamp(dot(normal, L), 1e-6, 1);
        float dotNV = clamp(dot(normal, V), 1e-6, 1);

        result += evalDiffuse(dotNL, dotNV, diffuseColor) * emission;  // diffuse
        result += evalGGXReflection(dotNL, dotNV, dotNH, dotVH, roughness, specularColor) * emission;  // specular
        if (transmissionWeight > 1e-5) {
          result += evalGGXTransmission(normal, L, V, eta, roughness) *
                    emission * transmissionWeight;
        }
      }

    }
  }
  return result;
}

vec3 tracePointLights(vec3 pos, vec3 normal, vec3 diffuseColor, vec3 specularColor, float roughness, float transmissionWeight, float eta) {
  vec3 result = vec3(0.0);
  for (uint i = 0; i < pointLightCount; ++i) {
    PointLight light = pointLights.l[i];
    vec3 emission = light.rgb;
    vec3 d = light.position - pos;
    float d2 = max(dot(d, d), 1e-6);
    vec3 L = normalize(d);
    vec3 V = normalize(-ray.direction);

    uint flags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;

    shadowRay.shadowed = true;
    traceRayEXT(tlas,
                flags,
                0xff,
                0,
                0,
                1,  // miss index
                pos,
                0.001,
                L,
                sqrt(d2) - 0.001,
                1  // payload location
    );
    //
    if (!shadowRay.shadowed) {
      vec3 H = normalize(V + L);
      float dotNH = clamp(dot(normal, H), 1e-6, 1);
      float dotVH = clamp(dot(V, H), 1e-6, 1);
      float dotNL = clamp(dot(normal, L), 1e-6, 1);
      float dotNV = clamp(dot(normal, V), 1e-6, 1);

      result += evalDiffuse(dotNL, dotNV, diffuseColor) * emission / d2;  // diffuse
      result += evalGGXReflection(dotNL, dotNV, dotNH, dotVH, roughness, specularColor) * emission / d2;  // specular
      if (transmissionWeight > 1e-5) {
        result += evalGGXTransmission(normal, L, V, eta, roughness) *
                  emission * transmissionWeight / d2;
      }
    }
  }
  return result;
}

vec3 traceSpotLights(vec3 pos, vec3 normal, vec3 diffuseColor, vec3 specularColor, float roughness, float transmissionWeight, float eta) {
  vec3 result = vec3(0.0);
  for (uint i = 0; i < spotLightCount; ++i) {
    SpotLight light = spotLights.l[i];

    vec3 emission = light.rgb;
    vec3 centerDir =  vec3(0.0, 0.0, -1.0) * mat3(light.viewMat);  // model rotation * forward

    vec3 d = light.position - pos;
    float d2 = max(dot(d, d), 1e-6);
    vec3 L = normalize(d);
    vec3 V = normalize(-ray.direction);

    float cf1 = cos(light.fovInner/2) + 1e-5;
    float cf2 = cos(light.fovOuter/2);

    float visibility = clamp((dot(-L, centerDir) - cf2) / (cf1 - cf2), 0.0, 1.0);

    vec3 H = normalize(V + L);
    float dotNH = clamp(dot(normal, H), 1e-6, 1);
    float dotVH = clamp(dot(V, H), 1e-6, 1);
    float dotNL = clamp(dot(normal, L), 1e-6, 1);
    float dotNV = clamp(dot(normal, V), 1e-6, 1);

    int texId = light.textureId;

    vec3 texColor = vec3(1.0);
    if (texId >= 0) {
      vec4 texcoord = light.projMat * (light.viewMat * vec4(pos, 1));
      texcoord /= texcoord.w;
      vec2 uv = texcoord.xy * 0.5 + 0.5;
      texColor = texture(textures[nonuniformEXT(texId)], uv).rgb;
    }

    float maxTexColor = maxVec3(texColor);

    if (visibility > 0.0) {
      uint flags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;
      shadowRay.shadowed = true;
      traceRayEXT(tlas,
                  flags,
                  0xff,
                  0,
                  0,
                  1,  // miss index
                  pos,
                  0.001,
                  L,
                  sqrt(d2) - 0.001,
                  1  // payload location
      );
      if (!shadowRay.shadowed && visibility > 1e-5 && maxTexColor > 1e-5) {
        result += evalDiffuse(dotNL, dotNV, diffuseColor) * emission / d2 * visibility * texColor;  // diffuse
        result += evalGGXReflection(dotNL, dotNV, dotNH, dotVH, roughness, specularColor) * emission / d2 * visibility * texColor;  // specular
        if (transmissionWeight > 1e-5) {
          result += evalGGXTransmission(normal, L, V, eta, roughness) * emission *
                    transmissionWeight * visibility * texColor / d2;
        }
      }
    }
  }
  return result;
}
