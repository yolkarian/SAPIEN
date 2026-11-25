#version 450
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

// declared for the renderer, unused by this unlit pack
layout(set = 2, binding = 0) uniform SceneBuffer {
  vec4 ambientLight;
} sceneBuffer;

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec2 inUV;
layout(location = 2) in flat uvec4 inSeg;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outPosition;
layout(location = 2) out uvec4 outSegmentation;

void main() {
  outPosition = inPosition;
  outSegmentation = inSeg;
  outColor = material.baseColor;
  if (material.baseColorTexture >= 0) {
    outColor *= texture(textures[nonuniformEXT(material.baseColorTexture)],
                        materialUV(material.textureTransforms[0], inUV));
  }
}
