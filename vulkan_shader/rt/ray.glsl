// One payload at location 0 for every ray (camera, BSDF continuation and shadow), so the one
// any-hit shader can tell a shadow ray (isShadow) from a path ray.
struct Ray {
  vec3 origin;
  vec3 direction;
  vec3 albedo;
  vec3 normal;
  vec3 emission;           // self-emission or environment radiance found at this vertex
  vec3 directDiffuse;      // direct lighting at this vertex through the base diffuse lobe
  vec3 directSpecular;     // direct lighting at this vertex through specular, coat and fuzz lobes
  float alpha;
  vec3 attenuation;        // BSDF weight (f cos / pdf) carried into the next segment
  vec3 diffuseAttenuation; // diffuse lobe's share of attenuation
  vec3 prevWorldPos;       // hit point's world position in the previous frame
  uvec4 segmentation;
  uint depth;
  uint seed;

  // multiple importance sampling of the next vertex: the solid-angle pdf of the BSDF
  // direction that leads there (negative when light sampling cannot produce that direction),
  // the environment's light-sampling pdf of that direction, and the scattering vertex it left
  // (a pass-through of a transparent object keeps it)
  float bsdfPdf;
  float envLightPdf;
  vec3 scatterOrigin;

  // shadow rays: the light left after transparent occluders; the shadow miss clears isShadow
  vec3 transmittance;
  int isShadow;

  // 1 when the closest hit scattered light, 0 for a miss or for the pass-through of the
  // uncovered part of a surface (geometry_opacity / object transparency). AOV classification
  // counts scattering events, not raw bounces, so that such a pass-through neither advances nor
  // changes the class.
  int scattered;
};
