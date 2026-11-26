#include "./point.glsl"
#include "./integrator_common.glsl"

#include "../common/scene_structs.glsl"

layout(set = SET_NUM, binding = 0) uniform accelerationStructureEXT tlas;

layout(set = SET_NUM, binding = 1) readonly buffer GeometryInstances
{
  GeometryInstance i[];
} geometryInstances;

// one record per material (common/material.glsl); binding 3 is unused
layout(set = SET_NUM, binding = 2) readonly buffer Materials
{
  Material m[];
} materials;

layout(set = SET_NUM, binding = 4) uniform sampler2D textures[];

layout(set = SET_NUM, binding = 5) readonly buffer PointLights
{
  PointLight l[];
} pointLights;

layout(set = SET_NUM, binding = 6) readonly buffer DirectionalLights
{
  DirectionalLight l[];
} directionalLights;

layout(set = SET_NUM, binding = 7) readonly buffer SpotLights
{
  SpotLight l[];
} spotLights;

layout(std430, set = SET_NUM, binding = 8) readonly buffer Vertices {
  Vertex v[];
} vertices[];

layout(set = SET_NUM, binding = 9) readonly buffer Indices {
  uint i[];
} indices[];

layout(set = SET_NUM, binding = 10) uniform samplerCube samplerEnvironment;

layout(set = SET_NUM, binding = 11) readonly buffer Objects {
  Object o[];
} objects;

layout(set = SET_NUM, binding = 12) readonly buffer ParallelogramLights
{
  ParallelogramLight l[];
} parallelogramLights;

layout(set = SET_NUM, binding = 13) readonly buffer PointInstances
{
  int pointInstances[];
};

layout(std430, set = SET_NUM, binding = 14) readonly buffer Points {
  Point p[];
} points[];

// Environment importance table, one entry per cell of the equal-solid-angle grid:
// (alias, threshold, solid-angle pdf of the cell's directions as float bits, unused).
layout(std430, set = SET_NUM, binding = 15) readonly buffer EnvAlias {
  uvec4 e[];
} envAlias;

// Triangles of emissive meshes, selected through EmissiveAlias with probability v0.w.
layout(std430, set = SET_NUM, binding = 16) readonly buffer EmissiveTriangles {
  EmissiveTriangle t[];
} emissiveTriangles;

layout(std430, set = SET_NUM, binding = 17) readonly buffer EmissiveAlias {
  uvec2 e[];
} emissiveAlias;

// Per geometry instance (gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT): the index of its
// first emissive triangle (its triangles follow in primitive order), or -1.
layout(std430, set = SET_NUM, binding = 18) readonly buffer EmissiveBase {
  int b[];
} emissiveBase;

// The TLAS instance rows, for the current transforms of emissive meshes.
layout(std430, set = SET_NUM, binding = 19) readonly buffer TlasInstances {
  TlasInstance i[];
} tlasInstances;

// The previous frame's TLAS instance rows, for screen-space motion vectors.
layout(std430, set = SET_NUM, binding = 20) readonly buffer PrevTlasInstances {
  TlasInstance i[];
} prevTlasInstances;
