#pragma once
/* C interface of src/sapien_renderer/pack_manifest.zig, the parser of a shader pack's pack.json.
 *
 * pack.json declares the parameters a RenderConfig may set for the pack and the presets that set
 * several of them at once:
 *
 *   {
 *     "parameters": {
 *       "spp":     {"kind": "uniform", "type": "int", "default": 32, "minimum": 1},
 *       "shading": {"kind": "variant", "type": "enum", "values": ["lambert", "pbr"],
 *                   "default": "lambert",
 *                   "define": {"lambert": "SHADING_LAMBERT", "pbr": "SHADING_PBR"}}
 *     },
 *     "presets": {"pbr": {"shading": "pbr"}},
 *     "post": ["tonemap"]
 *   }
 *
 * A uniform parameter is a renderer property (bool, int or float), applied per frame. A variant
 * parameter (bool, int, float or enum) selects shader source through a preprocessor definition:
 * a bool defines its "define" when true, an int or float defines it to the value, and an enum
 * defines the name its "define" object maps the value to ("" defines nothing).
 * "post" lists the post-processing passes the pack runs, in order; a pass name consists of
 * letters, digits and '_' and names the compute shader <name>.comp.
 *
 * Ownership: every string and handle returned is owned by the manifest and valid until
 * sapien_pack_manifest_destroy. Functions returning int return 0 on success; on failure they write
 * a NUL-terminated message into `error` (truncated to `error_size`). Indices out of range are
 * programming errors and abort. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SapienPackManifest SapienPackManifest;

enum {
  SAPIEN_PACK_BOOL = 0,
  SAPIEN_PACK_INT = 1,
  SAPIEN_PACK_FLOAT = 2,
  /* an enum parameter; as a value type, a string */
  SAPIEN_PACK_ENUM = 3,
};

enum {
  SAPIEN_PACK_UNIFORM = 0,
  SAPIEN_PACK_VARIANT = 1,
};

/* One parameter value. `integer` holds bools (0 or 1) and ints, `real` floats, `string` enums. */
typedef struct SapienPackValue {
  int32_t type;
  int64_t integer;
  double real;
  const char *string;
  size_t string_len;
} SapienPackValue;

typedef struct SapienPackParameter {
  const char *name;
  size_t name_len;
  int32_t kind;
  int32_t type;
  SapienPackValue default_value;
  int32_t has_minimum;
  int32_t has_maximum;
  double minimum;
  double maximum;
  /* number of enum values, 0 for other types */
  size_t enum_count;
  const char *description;
  size_t description_len;
} SapienPackParameter;

int sapien_pack_manifest_parse(const char *json, size_t json_len, SapienPackManifest **out,
                               char *error, size_t error_size);
void sapien_pack_manifest_destroy(SapienPackManifest *manifest);

size_t sapien_pack_manifest_parameter_count(const SapienPackManifest *manifest);
void sapien_pack_manifest_parameter(const SapienPackManifest *manifest, size_t index,
                                    SapienPackParameter *out);
void sapien_pack_manifest_enum_value(const SapienPackManifest *manifest, size_t parameter,
                                     size_t index, const char **value, size_t *value_len);
/* index of the parameter called `name`, or -1 */
int64_t sapien_pack_manifest_find_parameter(const SapienPackManifest *manifest, const char *name,
                                            size_t name_len);

/* Check `value` for `parameter`: its type and range. An int given for a float parameter is
 * converted in place. */
int sapien_pack_manifest_check(const SapienPackManifest *manifest, size_t parameter,
                               SapienPackValue *value, char *error, size_t error_size);

/* The definition a checked `value` of variant `parameter` selects; `*define_len` is 0 when it
 * selects none. Fails for a uniform parameter. */
int sapien_pack_manifest_define(const SapienPackManifest *manifest, size_t parameter,
                                const SapienPackValue *value, const char **define,
                                size_t *define_len, char *error, size_t error_size);

size_t sapien_pack_manifest_preset_count(const SapienPackManifest *manifest);
void sapien_pack_manifest_preset_name(const SapienPackManifest *manifest, size_t preset,
                                      const char **name, size_t *name_len);
/* index of the preset called `name`, or -1 */
int64_t sapien_pack_manifest_find_preset(const SapienPackManifest *manifest, const char *name,
                                         size_t name_len);
size_t sapien_pack_manifest_preset_entry_count(const SapienPackManifest *manifest,
                                               size_t preset);
/* entry `index` of `preset`: the parameter it sets and its checked value */
void sapien_pack_manifest_preset_entry(const SapienPackManifest *manifest, size_t preset,
                                       size_t index, size_t *parameter, SapienPackValue *value);

size_t sapien_pack_manifest_post_count(const SapienPackManifest *manifest);
void sapien_pack_manifest_post_name(const SapienPackManifest *manifest, size_t index,
                                    const char **name, size_t *name_len);

#ifdef __cplusplus
}
#endif
