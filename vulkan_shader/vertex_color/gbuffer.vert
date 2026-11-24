#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
#extension GL_EXT_nonuniform_qualifier : enable

layout(set = 0, binding = 0) uniform CameraBuffer {
  mat4 viewMatrix;
  mat4 projectionMatrix;
  mat4 viewMatrixInverse;
  mat4 projectionMatrixInverse;
  float width;
  float height;
} cameraBuffer;

#define SET_NUM 1
#include "../common/scene.glsl"
#undef SET_NUM

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec3 tangent;
layout(location = 4) in vec3 bitangent;
layout(location = 5) in vec4 color;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec2 outUV;
layout(location = 2) out flat uvec4 outSeg;
layout(location = 3) out vec4 outColor;

void main() {
  outSeg = instance.segmentation;
  outUV = uv;
  outColor = color;
  outPosition = cameraBuffer.viewMatrix * (transforms[instance.transformIndex] * vec4(position, 1));
  gl_Position = cameraBuffer.projectionMatrix * outPosition;
}
