#version 450

layout (constant_id = 0) const float ambientOcclusionStrength = 0.65;
layout (constant_id = 1) const float ambientOcclusionRadius = 0.3;

#define SET_NUM 0
#include "./camera_set.glsl"
#undef SET_NUM

layout(set = 1, binding = 0) uniform sampler2D samplerGbufferDepth;
layout(set = 1, binding = 1) uniform sampler2D samplerNormal;

layout(location = 0) in vec2 inUV;
layout(location = 0) out float outAmbientOcclusion;

#include "../common/gtao.glsl"

void main() {
  outAmbientOcclusion = sapienComputeGTAO(
      samplerGbufferDepth, samplerNormal, gl_FragCoord.xy, cameraBuffer.projectionMatrix,
      cameraBuffer.projectionMatrixInverse, ambientOcclusionRadius, ambientOcclusionStrength);
}
