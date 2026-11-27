#ifndef SAPIEN_RT_INTEGRATOR_COMMON_GLSL
#define SAPIEN_RT_INTEGRATOR_COMMON_GLSL

#include "../common/openpbr.glsl"

#define SAPIEN_PI 3.141592653589793

// One triangle of an emissive mesh, in the object space of its TLAS instance. std430: 64 B.
struct EmissiveTriangle {
  vec4 v0;        // xyz = vertex 0, w = probability of selecting this triangle
  vec4 e1;        // xyz = vertex 1 - vertex 0, w = TLAS instance (bit pattern of an int)
  vec4 e2;        // xyz = vertex 2 - vertex 0
  vec4 emission;  // rgb = emitted radiance, w > 0.5 when double-sided
};

// One row of the TLAS instance buffer (VkAccelerationStructureInstanceKHR): the 3x4 row-major
// object-to-world transform, then fields the shaders do not read. std430: 64 B.
struct TlasInstance {
  vec4 row0;
  vec4 row1;
  vec4 row2;
  uvec4 rest;
};

vec3 instancePoint(TlasInstance m, vec3 p) {
  vec4 h = vec4(p, 1.0);
  return vec3(dot(m.row0, h), dot(m.row1, h), dot(m.row2, h));
}

vec3 instanceVector(TlasInstance m, vec3 v) {
  return vec3(dot(m.row0.xyz, v), dot(m.row1.xyz, v), dot(m.row2.xyz, v));
}

// Everything the integrator needs about the current surface vertex. The local frame (T, B, N)
// is the OpenPBR shading frame: x along the anisotropy tangent, z the shading normal.
struct Surface {
  vec3 pos;
  vec3 N;   // shading normal
  vec3 Ng;  // geometric normal, on the same side as N
  vec3 T;
  vec3 B;
  vec3 wo;  // view direction in the local frame
  OpenPBR m;
  OpenPBRLobes lobes;
};

vec3 surfaceLocal(Surface s, vec3 d) { return vec3(dot(d, s.T), dot(d, s.B), dot(d, s.N)); }

vec3 surfaceWorld(Surface s, vec3 d) { return d.x * s.T + d.y * s.B + d.z * s.N; }

// The surface at `pos` with shading frame (T, B, N), seen from world direction V.
Surface makeSurface(vec3 pos, vec3 T, vec3 B, vec3 N, vec3 Ng, vec3 V, OpenPBR m) {
  Surface s;
  s.pos = pos;
  s.N = N;
  s.Ng = Ng;
  s.T = T;
  s.B = B;
  vec3 wo = surfaceLocal(s, V);
  s.wo = normalize(vec3(wo.xy, max(wo.z, 1e-4))); // a normal map can tilt N away from V
  s.m = m;
  s.lobes = openpbrLobes(m, s.wo);
  return s;
}

// Power heuristic (beta = 2) of two solid-angle pdfs, scaled to avoid float32 overflow/underflow.
float misPowerHeuristic(float a, float b) {
  if (a <= 0.0) {
    return 0.0;
  }
  if (b <= 0.0) {
    return 1.0;
  }
  if (a >= b) {
    float r = b / a;
    return 1.0 / (1.0 + r * r);
  }
  float r = a / b;
  float r2 = r * r;
  return r2 / (1.0 + r2);
}

// MIS weight of light that a path ray found with BSDF pdf `bsdfPdf`, `pdfLight` being its
// light-sampling pdf (0 when that light is not light-sampled). bsdfPdf < 0 marks a ray that
// light sampling cannot produce: camera rays and BSDF directions below the surface.
float bsdfHitWeight(float bsdfPdf, float pdfLight) {
  if (bsdfPdf < 0.0 || pdfLight <= 0.0) {
    return 1.0;
  }
#if defined(LIGHT_SAMPLING_NEE)
  return 0.0;
#elif defined(LIGHT_SAMPLING_MIS)
  return misPowerHeuristic(bsdfPdf, pdfLight);
#else
  return 1.0;
#endif
}

// Origin of a ray leaving `p` on the side of normal `n`, offset by a few ulps so that it does
// not hit the surface it starts on (Wachter and Binder, Ray Tracing Gems, chapter 6).
vec3 offsetRayOrigin(vec3 p, vec3 n) {
  const float origin = 1.0 / 32.0;
  const float floatScale = 1.0 / 65536.0;
  const float intScale = 256.0;
  ivec3 of = ivec3(intScale * n);
  vec3 pi = vec3(intBitsToFloat(floatBitsToInt(p.x) + (p.x < 0.0 ? -of.x : of.x)),
                 intBitsToFloat(floatBitsToInt(p.y) + (p.y < 0.0 ? -of.y : of.y)),
                 intBitsToFloat(floatBitsToInt(p.z) + (p.z < 0.0 ? -of.z : of.z)));
  return vec3(abs(p.x) < origin ? p.x + floatScale * n.x : pi.x,
              abs(p.y) < origin ? p.y + floatScale * n.y : pi.y,
              abs(p.z) < origin ? p.z + floatScale * n.z : pi.z);
}

// The environment importance table covers a lat-long grid of envWidth x envWidth/2 cells of
// EQUAL solid angle (phi and cos(theta) are the two uniform coordinates), so a uniform point in
// a cell is a uniform direction in it. Directions are world space; the environment cube is
// sampled at dir.xzy.
vec3 envCellDirection(uint cell, int envWidth, float u, float v) {
  int envHeight = envWidth / 2;
  int i = int(cell) % envWidth;
  int j = int(cell) / envWidth;
  float phi = 2.0 * SAPIEN_PI * (float(i) + u) / float(envWidth);
  float cosTheta = 1.0 - 2.0 * (float(j) + v) / float(envHeight);
  float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
  return vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

uint envDirectionCell(vec3 d, int envWidth) {
  int envHeight = envWidth / 2;
  float phi = atan(d.y, d.x);
  if (phi < 0.0) phi += 2.0 * SAPIEN_PI;
  float cosTheta = clamp(d.z, -1.0, 1.0);
  int i = clamp(int(phi / (2.0 * SAPIEN_PI) * float(envWidth)), 0, envWidth - 1);
  int j = clamp(int((1.0 - cosTheta) * 0.5 * float(envHeight)), 0, envHeight - 1);
  return uint(j * envWidth + i);
}

// O(1) draw from an alias table: `entry` = (alias, threshold), threshold the probability of
// keeping `cell` scaled by 2^32.
uint aliasSample(uvec2 entry, uint cell, uint randomWord) {
  return randomWord < entry.y ? cell : entry.x;
}

#endif
