#ifndef SAPIEN_RASTER_LIGHTING_GLSL
#define SAPIEN_RASTER_LIGHTING_GLSL

// Direct lighting of the rasterization packs from the lights of their scene set.
//
// The including shader declares, before including this file: the scene set (ShadowBuffer, the
// PointLights / DirectionalLights / SpotLights uniform buffers and the shadow-map samplers, see
// default/scene_set.glsl), the CameraBuffer `cameraBuffer`, and the specialization constants
// NUM_POINT_LIGHTS, NUM_DIRECTIONAL_LIGHTS, NUM_SPOT_LIGHTS and their NUM_*_SHADOWS, and
// NUM_TEXTURED_LIGHT_SHADOWS. The light counts are compile-time constants on purpose: a loop of
// known length over a uniform buffer is measurably faster in a forward pass than one bounded by
// a value read at run time from a storage buffer.
//
// The first NUM_*_SHADOWS lights of each kind cast shadows into the shadow map of the same
// index; the NUM_TEXTURED_LIGHT_SHADOWS textured lights follow the spot lights in SpotLights.
// `position`, `normal` and `camDir` are in camera space. Without `shadows` no shadow map is
// sampled; textured lights still project their texture. `m` carries the OpenPBR layers (coat,
// fuzz) over the base BRDF of `diffuseAlbedo`, `roughness` and `fresnel`; a material without
// layers takes exactly the base path.

#include "./lights.glsl"
#include "./shadow.glsl"
#include "./raster_material.glsl"

// a light's base contribution `base` seen through the layers, plus the layers' own lobes;
// `radiance` is the light arriving from `lightDir`
vec3 sapienLayered(vec3 base, OpenPBR m, vec3 normal, vec3 camDir, vec3 lightDir,
                   vec3 radiance) {
  float NoL = dot(normal, lightDir);
  if (NoL <= 0.0) {
    return base;
  }
  float NoV = clamp(dot(normal, camDir), 1e-6, 1.0);
  vec3 hv = lightDir + camDir;
  float hLen2 = dot(hv, hv);
  vec3 H = hLen2 > 1e-12 ? hv * inversesqrt(hLen2) : normal;
  return base * openpbrUnderLayers(m, NoV, NoL) +
         radiance * openpbrLayerLobes(m, NoV, NoL, clamp(dot(normal, H), 0.0, 1.0),
                                      clamp(dot(camDir, H), 1e-6, 1.0));
}

vec3 sapienLayeredPoint(vec3 base, OpenPBR m, vec3 normal, vec3 camDir, vec3 l, vec3 rgb) {
  float l2 = dot(l, l);
  if (l2 <= 1e-12) {
    return base;
  }
  return sapienLayered(base, m, normal, camDir, l * inversesqrt(l2), rgb / max(l2, 1e-8));
}

// the light-space shadow map coordinates of a world-space position
vec3 sapienShadowMapCoordWorld(mat4 shadowView, mat4 shadowProj, vec4 worldPos) {
  vec4 coord = shadowProj * (shadowView * worldPos);
  if (abs(coord.w) <= 1e-6) {
    return vec3(-1.0);
  }
  coord /= coord.w;
  coord.xy = coord.xy * 0.5 + 0.5;
  return coord.xyz;
}

void sapienAddDirectLighting(inout vec3 color, vec3 position, vec3 normal, vec3 camDir,
                             vec3 diffuseAlbedo, float roughness, vec3 fresnel, OpenPBR m,
                             bool shadows) {
  bool layered = openpbrHasLayers(m);
  vec4 worldPos = cameraBuffer.viewMatrixInverse * vec4(position, 1.0);
  vec3 worldNormal = normalize(mat3(cameraBuffer.viewMatrixInverse) * normal);

  int pointShadows = shadows ? NUM_POINT_LIGHT_SHADOWS : 0;
  for (int i = 0; i < pointShadows; ++i) {
    vec3 l = (cameraBuffer.viewMatrix * vec4(pointLights.l[i].position, 1.f)).xyz - position;
    mat4 shadowProj = shadowBuffer.pointLightBuffers[6 * i].projectionMatrix;
    vec3 wsl = vec3(cameraBuffer.viewMatrixInverse * vec4(l, 0));
    float visibility = sapienPointShadow(samplerPointLightDepths[i], wsl, shadowProj, worldNormal,
                                         pointLights.l[i].radius, gl_FragCoord.xy);
    vec3 c = computePointLight(pointLights.l[i].rgb, l, normal, camDir, diffuseAlbedo,
                               roughness, fresnel);
    if (layered) {
      c = sapienLayeredPoint(c, m, normal, camDir, l, pointLights.l[i].rgb);
    }
    color += visibility * c;
  }
  for (int i = pointShadows; i < NUM_POINT_LIGHTS; ++i) {
    vec3 l = (cameraBuffer.viewMatrix * vec4(pointLights.l[i].position, 1.f)).xyz - position;
    vec3 c = computePointLight(pointLights.l[i].rgb, l, normal, camDir, diffuseAlbedo,
                               roughness, fresnel);
    if (layered) {
      c = sapienLayeredPoint(c, m, normal, camDir, l, pointLights.l[i].rgb);
    }
    color += c;
  }

  int directionalShadows = shadows ? NUM_DIRECTIONAL_LIGHT_SHADOWS : 0;
  for (int i = 0; i < directionalShadows; ++i) {
    int base = i * SAPIEN_DIRECTIONAL_SHADOW_CASCADES;
    float visibility = sapienDirectionalShadow(
        samplerDirectionalLightDepths[i],
        shadowBuffer.directionalLightBuffers[base + 0].viewMatrix,
        shadowBuffer.directionalLightBuffers[base + 0].projectionMatrix,
        shadowBuffer.directionalLightBuffers[base + 1].viewMatrix,
        shadowBuffer.directionalLightBuffers[base + 1].projectionMatrix,
        shadowBuffer.directionalLightBuffers[base + 2].viewMatrix,
        shadowBuffer.directionalLightBuffers[base + 2].projectionMatrix,
        worldPos, worldNormal, directionalLights.l[i].softness, gl_FragCoord.xy);
    vec3 direction = mat3(cameraBuffer.viewMatrix) * directionalLights.l[i].direction;
    vec3 c = computeDirectionalLight(direction, directionalLights.l[i].rgb, normal, camDir,
                                     diffuseAlbedo, roughness, fresnel);
    if (layered) {
      c = sapienLayered(c, m, normal, camDir, -direction, directionalLights.l[i].rgb);
    }
    color += visibility * c;
  }
  for (int i = directionalShadows; i < NUM_DIRECTIONAL_LIGHTS; ++i) {
    vec3 direction = mat3(cameraBuffer.viewMatrix) * directionalLights.l[i].direction;
    vec3 c = computeDirectionalLight(direction, directionalLights.l[i].rgb, normal, camDir,
                                     diffuseAlbedo, roughness, fresnel);
    if (layered) {
      c = sapienLayered(c, m, normal, camDir, -direction, directionalLights.l[i].rgb);
    }
    color += c;
  }

  for (int i = 0; i < NUM_SPOT_LIGHTS + NUM_TEXTURED_LIGHT_SHADOWS; ++i) {
    SpotLight light = spotLights.l[i];
    vec3 l = (cameraBuffer.viewMatrix * vec4(light.position, 1.f)).xyz - position;
    vec3 centerDir = mat3(cameraBuffer.viewMatrix) * spotLightDirection(light);
    float visibility = 1.0;
    if (shadows && i < NUM_SPOT_LIGHT_SHADOWS) {
      mat4 sv = shadowBuffer.spotLightBuffers[i].viewMatrix;
      mat4 sp = shadowBuffer.spotLightBuffers[i].projectionMatrix;
      vec3 coord = sapienShadowMapCoordWorld(sv, sp, worldPos);
      visibility = sapienPerspectiveShadow(samplerSpotLightDepths[i], coord, sv, sp, worldNormal,
                                           light.radius, gl_FragCoord.xy);
    } else if (i >= NUM_SPOT_LIGHTS) {
      int t = i - NUM_SPOT_LIGHTS;
      mat4 sv = shadowBuffer.texturedLightBuffers[t].viewMatrix;
      mat4 sp = shadowBuffer.texturedLightBuffers[t].projectionMatrix;
      vec3 coord = sapienShadowMapCoordWorld(sv, sp, worldPos);
      if (shadows && light.shadowEnabled != 0) {
        visibility = sapienPerspectiveShadow(samplerTexturedLightDepths[t], coord, sv, sp,
                                             worldNormal, light.radius, gl_FragCoord.xy);
      }
      if (light.textureId >= 0) {
        // a textured light without a texture is a plain spot light: the rasterizer binds the
        // default texture for it, but -1 must never index the array
        visibility *= texture(samplerTexturedLightTextures[light.textureId], coord.xy).x;
      }
    }
    vec3 c = computeSpotLight(light.fovInner, light.fovOuter, centerDir, light.rgb, l, normal,
                              camDir, diffuseAlbedo, roughness, fresnel);
    if (layered) {
      float l2 = dot(l, l);
      if (l2 > 1e-12) {
        float cf1 = cos(light.fovInner / 2) + 1e-6;
        float cf2 = cos(light.fovOuter / 2);
        float cone = clamp((dot(-l * inversesqrt(l2), centerDir) - cf2) / (cf1 - cf2), 0, 1);
        c = sapienLayeredPoint(c, m, normal, camDir, l, light.rgb * cone);
      }
    }
    color += visibility * c;
  }
}

#endif
