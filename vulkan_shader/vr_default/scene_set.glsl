// usage
// #define SET_NUM 1
// #include "scene_set.glsl"
// #undef SET_NUM

#include "../common/light_data.glsl"

// The lights are in the uniform buffers at bindings 9-11 (common/light_data.glsl), sized by the
// specialization constants NUM_POINT_LIGHTS, NUM_DIRECTIONAL_LIGHTS, NUM_SPOT_LIGHTS and
// NUM_TEXTURED_LIGHT_SHADOWS of the including shader (plus one entry, so no array is empty).
// The first NUM_*_SHADOWS lights of each kind cast shadows; the textured lights follow the
// NUM_SPOT_LIGHTS spot lights in SpotLights.
layout(set = SET_NUM, binding = 0) uniform SceneBuffer {
  vec4 ambientLight;
} sceneBuffer;

struct LightBuffer {
  mat4 viewMatrix;
  mat4 viewMatrixInverse;
  mat4 projectionMatrix;
  mat4 projectionMatrixInverse;
  int width;
  int height;
};

layout(set = SET_NUM, binding = 1) uniform ShadowBuffer {
  LightBuffer directionalLightBuffers[3];
  LightBuffer spotLightBuffers[10];
  LightBuffer pointLightBuffers[18];
  LightBuffer texturedLightBuffers[1];
} shadowBuffer;

layout(set = SET_NUM, binding = 2) uniform samplerCube samplerPointLightDepths[3];
layout(set = SET_NUM, binding = 3) uniform sampler2DArray samplerDirectionalLightDepths[1];
layout(set = SET_NUM, binding = 4) uniform sampler2D samplerTexturedLightDepths[1];
layout(set = SET_NUM, binding = 5) uniform sampler2D samplerSpotLightDepths[10];
layout(set = SET_NUM, binding = 6) uniform sampler2D samplerTexturedLightTextures[1];

layout(set = SET_NUM, binding = 7) uniform samplerCube samplerEnvironment;
layout(set = SET_NUM, binding = 8) uniform sampler2D samplerBRDFLUT;

layout(set = SET_NUM, binding = 9) uniform PointLights {
  PointLight l[NUM_POINT_LIGHTS + 1];
} pointLights;
layout(set = SET_NUM, binding = 10) uniform DirectionalLights {
  DirectionalLight l[NUM_DIRECTIONAL_LIGHTS + 1];
} directionalLights;
layout(set = SET_NUM, binding = 11) uniform SpotLights {
  SpotLight l[NUM_SPOT_LIGHTS + NUM_TEXTURED_LIGHT_SHADOWS + 1];
} spotLights;
