layout(push_constant) uniform Constants {
  vec3 ambientLight;
  int frameCount;
  int spp;
  int maxDepth;

  float aperture;
  float focusPlane;

  int russianRoulette;
  int russianRouletteMinBounces;

  int pointLightCount;
  int directionalLightCount;
  int spotLightCount;
  int parallelogramLightCount;

  int envmap;

  // increases by one on every render of this camera; decorrelates noise between frames
  int frameSeed;

  // post processing
  float exposure;
  int toneMapper;
};
