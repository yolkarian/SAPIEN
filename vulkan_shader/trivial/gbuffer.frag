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

struct PointLight {
  vec4 position;
  vec4 emission;
};

struct DirectionalLight {
  vec4 direction;
  vec4 emission;
};

struct SpotLight {
  vec4 position;
  vec4 direction;
  vec4 emission;
};

layout(set = 2, binding = 0) uniform SceneBuffer {
  vec4 ambientLight;
  DirectionalLight directionalLights[3];
  SpotLight spotLights[10];
  PointLight pointLights[10];
  SpotLight texturedLights[1];
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
  if ((material.textureMask & 1) != 0) {
    outColor = texture(textures[nonuniformEXT(material.baseColorTexture)], inUV);
  } else {
    outColor = material.baseColor;
  }
}
