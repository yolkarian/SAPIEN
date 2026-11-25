#version 450

layout(set = 0, binding = 0) uniform sampler2D samplerColorRaw;
layout(set = 0, binding = 1) uniform sampler2D samplerGbufferDepth;

layout(set = 1, binding = 0) uniform CameraBuffer {
  mat4 viewMatrix;
  mat4 projectionMatrix;
  mat4 viewMatrixInverse;
  mat4 projectionMatrixInverse;
  float width;
  float height;
} cameraBuffer;

// the scene set of gbuffer.frag; only the environment is read here
layout(set = 2, binding = 0) uniform SceneBuffer {
  vec4 ambientLight;
} sceneBuffer;
layout(set = 2, binding = 7) uniform samplerCube samplerEnvironment;

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outColor;

vec3 getBackgroundColor(vec3 texcoord) {
  texcoord = texcoord.xzy;
  return textureLod(samplerEnvironment, texcoord, 0).rgb;
}

void main() {
  vec4 color = texture(samplerColorRaw, inUV);

  float depth = texture(samplerGbufferDepth, inUV).x;
  vec4 csPosition = cameraBuffer.projectionMatrixInverse * (vec4(inUV * 2 - 1, depth, 1));
  csPosition /= csPosition.w;
  vec3 bg = pow(getBackgroundColor((cameraBuffer.viewMatrixInverse * csPosition).xyz), vec3(1/2.2));

  outColor.rgb = mix(bg, color.rgb, color.a);
  outColor.a = color.a;
}
