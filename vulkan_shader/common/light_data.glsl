#ifndef SAPIEN_LIGHT_DATA_GLSL
#define SAPIEN_LIGHT_DATA_GLSL

// Light records of every renderer (svulkan2/scene/light_data.h). Within each kind, the lights
// that cast shadows come first; textured lights follow the plain spot lights.

// `direction` is the direction the light travels.
struct DirectionalLight {
  vec3 direction;
  float softness;
  vec3 rgb;
  float padding;
};

struct PointLight {
  vec3 position;
  float radius;
  vec3 rgb;
  float padding;
};

// The light shines along the -z axis of `viewMat` (world to light space). `projMat` is its
// shadow projection, which also maps a textured light's texture. Angles are full angles in
// radians; `textureId` is the renderer's texture slot of a textured light, -1 otherwise.
struct SpotLight {
  mat4 viewMat;
  mat4 projMat;
  vec3 rgb;
  int padding0;
  vec3 position;
  int padding1;
  float fovInner;
  float fovOuter;
  int textureId;
  int padding2;
};

struct ParallelogramLight {
  vec3 rgb;
  float padding0;
  vec3 position;
  float padding1;
  vec3 edge0;
  float padding2;
  vec3 edge1;
  float padding3;
};

// the world direction a spot light shines along
vec3 spotLightDirection(SpotLight light) {
  return vec3(0.0, 0.0, -1.0) * mat3(light.viewMat);
}

#endif
