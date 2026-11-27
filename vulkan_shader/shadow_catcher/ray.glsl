// One payload at location 0 for path and shadow rays so camera.rahit handles both.
struct Ray {
  vec3 origin;
  vec3 direction;
  vec3 albedo;
  vec3 normal;
  vec3 radiance;
  float alpha;
  vec3 attenuation;
  uvec4 segmentation;
  uint depth;
  uint seed;
  vec3 transmittance;
  int isShadow;
  int scattered;
};
