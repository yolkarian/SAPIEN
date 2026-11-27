#version 460
#extension GL_EXT_ray_tracing : require

#include "ray.glsl"
#include "integrator_common.glsl"
#include "push_constant.glsl"

layout(location = 0) rayPayloadInEXT Ray ray;

layout(set = 1, binding = 10) uniform samplerCube samplerEnvironment;

void main() {
  // the environment is a light: no clamp, weighted against light-sampling it (camera rays: 1)
  vec3 Li = envmap != 0 ? textureLod(samplerEnvironment, ray.direction.xzy, 0.0).rgb : ambientLight;
  ray.emission = Li * bsdfHitWeight(ray.bsdfPdf, ray.envLightPdf);
  ray.directDiffuse = vec3(0.0);
  ray.directSpecular = vec3(0.0);
  ray.albedo = Li;
  ray.normal = vec3(0.0);
  ray.segmentation = uvec4(0);
  ray.attenuation = vec3(0.0);
  ray.diffuseAttenuation = vec3(0.0);
  ray.prevWorldPos = vec3(0.0);
  ray.depth = maxDepth + 1;
  ray.origin = vec3(0.0);
  ray.alpha = 0.0;
  // nothing was hit: the path ended without a scattering event here
  ray.scattered = 0;
}
