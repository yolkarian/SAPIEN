#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

#define SET_NUM 0
#include "./camera_set.glsl"
#undef SET_NUM

#define SET_NUM 1
#include "./object_set.glsl"
#undef SET_NUM

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec3 tangent;
layout(location = 4) in vec3 bitangent;

layout(location = 0) out vec4 outPositionRaw;
layout(location = 1) out vec2 outUV;
layout(location = 2) out flat uvec4 outSegmentation;
layout(location = 3) out vec3 objectCoord;
layout(location = 4) out mat3 outTbn;

void ensureTbn(inout vec3 tangent, inout vec3 bitangent, inout vec3 normal) {
  if (length(tangent) < 0.01 || length(bitangent) < 0.01) {
    vec3 wx = vec3(1, 0, 0);
    if (abs(dot(normal, vec3(1, 0, 0))) > 0.95) {
      wx = vec3(0, 1, 0);
    }
    vec3 wy = normalize(cross(normal, wx));
    wx = cross(wy, normal);
    tangent = wx;
    bitangent = wy;
  }
}

void main() {
  outSegmentation = objectDataBuffer.segmentation;
  mat4 modelView = cameraBuffer.viewMatrix * objectTransformBuffer.modelMatrix;
  mat3 modelViewLinear = mat3(modelView);
  mat3 normalMatrix = transpose(inverse(modelViewLinear));

  vec3 T = tangent;
  vec3 B = bitangent;
  vec3 N = normal;
  ensureTbn(T, B, N);

  objectCoord = position;
  outPositionRaw = modelView * vec4(position, 1);
  outUV = uv;
  gl_Position = cameraBuffer.projectionMatrix * outPositionRaw;

  vec3 outNormal = normalize(normalMatrix * N);
  vec3 transformedTangent = modelViewLinear * T;
  transformedTangent -= outNormal * dot(outNormal, transformedTangent);
  if (dot(transformedTangent, transformedTangent) < 1e-6) {
    vec3 axis = abs(outNormal.x) < 0.95 ? vec3(1, 0, 0) : vec3(0, 1, 0);
    transformedTangent = cross(axis, outNormal);
  }
  vec3 outTangent = normalize(transformedTangent);
  vec3 transformedBitangent = modelViewLinear * B;
  float handedness = dot(cross(outNormal, outTangent), transformedBitangent) < 0.0 ? -1.0 : 1.0;
  vec3 outBitangent = handedness * normalize(cross(outNormal, outTangent));
  outTbn = mat3(outTangent, outBitangent, outNormal);
}
