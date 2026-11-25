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
// sampled; textured lights still project their texture.

#include "./lights.glsl"
#include "./shadow.glsl"

// the light-space shadow map coordinates of a camera-space position
vec3 sapienShadowMapCoord(mat4 shadowView, mat4 shadowProj, vec3 position) {
  vec4 coord = shadowProj * (shadowView * cameraBuffer.viewMatrixInverse * vec4(position, 1));
  coord /= coord.w;
  coord.xy = coord.xy * 0.5 + 0.5;
  return coord.xyz;
}

void sapienAddDirectLighting(inout vec3 color, vec3 position, vec3 normal, vec3 camDir,
                             vec3 diffuseAlbedo, float roughness, vec3 fresnel, bool shadows) {
  int pointShadows = shadows ? NUM_POINT_LIGHT_SHADOWS : 0;
  for (int i = 0; i < pointShadows; ++i) {
    vec3 l = (cameraBuffer.viewMatrix * vec4(pointLights.l[i].position, 1.f)).xyz - position;
    mat4 shadowProj = shadowBuffer.pointLightBuffers[6 * i].projectionMatrix;
    vec3 wsl = vec3(cameraBuffer.viewMatrixInverse * vec4(l, 0));
    vec3 v = abs(wsl);
    vec4 p = shadowProj * vec4(0, 0, -max(max(v.x, v.y), v.z), 1);
    float pixelDepth = p.z / p.w;
    float shadowDepth = texture(samplerPointLightDepths[i], wsl).x;
    float visibility = step(pixelDepth - shadowDepth, 0);
    color += visibility * computePointLight(pointLights.l[i].rgb, l, normal, camDir,
                                            diffuseAlbedo, roughness, fresnel);
  }
  for (int i = pointShadows; i < NUM_POINT_LIGHTS; ++i) {
    vec3 l = (cameraBuffer.viewMatrix * vec4(pointLights.l[i].position, 1.f)).xyz - position;
    color += computePointLight(pointLights.l[i].rgb, l, normal, camDir, diffuseAlbedo,
                               roughness, fresnel);
  }

  int directionalShadows = shadows ? NUM_DIRECTIONAL_LIGHT_SHADOWS : 0;
  for (int i = 0; i < directionalShadows; ++i) {
    vec3 coord = sapienShadowMapCoord(shadowBuffer.directionalLightBuffers[i].viewMatrix,
                                      shadowBuffer.directionalLightBuffers[i].projectionMatrix,
                                      position);
    float resolution = textureSize(samplerDirectionalLightDepths[i], 0).x;
    float visibility =
        ShadowMapPCF(samplerDirectionalLightDepths[i], coord, resolution, 1 / resolution, 1);
    color += visibility * computeDirectionalLight(
                              mat3(cameraBuffer.viewMatrix) * directionalLights.l[i].direction,
                              directionalLights.l[i].rgb, normal, camDir, diffuseAlbedo,
                              roughness, fresnel);
  }
  for (int i = directionalShadows; i < NUM_DIRECTIONAL_LIGHTS; ++i) {
    color += computeDirectionalLight(
        mat3(cameraBuffer.viewMatrix) * directionalLights.l[i].direction,
        directionalLights.l[i].rgb, normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  for (int i = 0; i < NUM_SPOT_LIGHTS + NUM_TEXTURED_LIGHT_SHADOWS; ++i) {
    SpotLight light = spotLights.l[i];
    vec3 l = (cameraBuffer.viewMatrix * vec4(light.position, 1.f)).xyz - position;
    vec3 centerDir = mat3(cameraBuffer.viewMatrix) * spotLightDirection(light);
    float visibility = 1.0;
    if (shadows && i < NUM_SPOT_LIGHT_SHADOWS) {
      vec3 coord = sapienShadowMapCoord(shadowBuffer.spotLightBuffers[i].viewMatrix,
                                        shadowBuffer.spotLightBuffers[i].projectionMatrix,
                                        position);
      float resolution = textureSize(samplerSpotLightDepths[i], 0).x;
      visibility = ShadowMapPCF(samplerSpotLightDepths[i], coord, resolution, 1 / resolution, 1);
    } else if (i >= NUM_SPOT_LIGHTS) {
      int t = i - NUM_SPOT_LIGHTS;
      vec3 coord = sapienShadowMapCoord(shadowBuffer.texturedLightBuffers[t].viewMatrix,
                                        shadowBuffer.texturedLightBuffers[t].projectionMatrix,
                                        position);
      if (shadows) {
        float resolution = textureSize(samplerTexturedLightDepths[t], 0).x;
        visibility =
            ShadowMapPCF(samplerTexturedLightDepths[t], coord, resolution, 1 / resolution, 1);
      }
      visibility *= texture(samplerTexturedLightTextures[light.textureId], coord.xy).x;
    }
    color += visibility * computeSpotLight(light.fovInner, light.fovOuter, centerDir, light.rgb,
                                           l, normal, camDir, diffuseAlbedo, roughness, fresnel);
  }
}

#endif
