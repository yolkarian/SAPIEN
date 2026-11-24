// Scene-wide descriptor set for rasterization shader packs.
//
// A pack declares this set once per pass with `#define SET_NUM n` before the include. It replaces
// the old per-object ObjectTransformBuffer/ObjectDataBuffer and per-material MaterialBuffer sets
// with one set per scene. Every draw binds it with the offsets of its own instance and material,
// so `instance` and `material` below are that draw's data, read from uniform buffers (constant
// cache; storage-buffer reads cost a fragment shader up to 10% of throughput). `materials[]`
// and `transforms[]` give access to any entry. A shadow pass draws the same instances, so it
// reads `transforms[instance.transformIndex]` too.
//
// To add per-object custom fields, define `SAPIEN_CUSTOM_OBJECT_DATA` (the struct declaration)
// before including this file in EVERY pass that reads the set, e.g.:
//   #define SAPIEN_CUSTOM_OBJECT_DATA struct CustomObjectData { float gain; int decal; };
// and read them as `customObjectData[instance.customDataIndex]`. The fields are written by name
// from `render_body.set_property(name, value)`. An `int` field receives the `textures[]` slot of
// `render_body.set_texture(name, tex)` (a 2D texture) and an `int` array field the slots of
// `render_body.set_texture_array(name, texs)`; unset fields are 0, the default (white) texture.
//
// `SET_NUM` must be defined by the including shader before the include (and #undef'd after).
// NOTE: no `#extension` here: it must appear before other tokens in the including shader, so
// packs that sample `textures[]` declare `GL_EXT_nonuniform_qualifier` themselves at the top.

#ifndef SAPIEN_COMMON_SCENE_GLSL
#define SAPIEN_COMMON_SCENE_GLSL

#ifndef SET_NUM
#error "define SET_NUM before including common/scene.glsl"
#endif

// One per drawable: a shape of a render body, a line set or a point set.
struct InstanceInfo {
  uvec4 segmentation;
  int transformIndex;   // index into transforms[]
  int materialIndex;    // index into materials[] (-1 for line and point sets)
  float transparency;
  int shadeFlat;
  int customDataIndex;  // index into customObjectData[] (-1 for line and point sets)
};

// 256 B, the same in std140 and std430 (no arrays of scalars): the storage array materials[] and
// the uniform `material` read the same buffer. The first 160 B match SVMetallicMaterial::Buffer
// in C++; texture indices are slots of textures[], -1 for none; the tail is reserved for
// material fields to come.
struct Material {
  vec4 emission;
  vec4 baseColor;
  float fresnel;
  float roughness;
  float metallic;
  float transmission;
  float ior;
  float transmissionRoughness;
  int textureMask;
  int padding1;
  vec4 textureTransforms[6];
  int baseColorTexture;
  int roughnessTexture;
  int normalTexture;
  int metallicTexture;
  int emissionTexture;
  int transmissionTexture;
  int padding2;
  int padding3;
  vec4 reserved[4];
};

#ifndef SAPIEN_CUSTOM_OBJECT_DATA
#define SAPIEN_CUSTOM_OBJECT_DATA struct CustomObjectData { int dummy; };
#endif
SAPIEN_CUSTOM_OBJECT_DATA

// Object transforms (CPU uploads or CUDA pose writes), std430 mat4[] stride = 64 B.
layout(std430, set = SET_NUM, binding = 0) readonly buffer ObjectTransformBuffer {
  mat4 transforms[];
};

// The draw's instance.
layout(set = SET_NUM, binding = 1) uniform DrawInstance {
  InstanceInfo instance;
};

layout(std430, set = SET_NUM, binding = 2) readonly buffer MaterialBuffer {
  Material materials[];
};

// Bindless sampler array of max_num_textures slots. A material texture index of -1 means
// "no texture"; slot 0 always holds the default (white) texture.
layout(set = SET_NUM, binding = 3) uniform sampler2D textures[];

// Optional per-pack custom per-object data, addressed by instance.customDataIndex.
layout(std430, set = SET_NUM, binding = 4) readonly buffer CustomBuffer {
  CustomObjectData customObjectData[];
};

// The draw's material (materials[instance.materialIndex]; unspecified for line and point sets).
layout(set = SET_NUM, binding = 5) uniform DrawMaterial {
  Material material;
};

#endif
