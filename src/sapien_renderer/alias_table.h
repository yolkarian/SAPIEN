#pragma once
/* C interface of src/sapien_renderer/alias_table.zig: CPU construction of an alias table for
 * discrete importance sampling. The renderer builds one for environment-map texels
 * (weight = luminance x solid angle) and one for emissive triangles (weight = area x emission
 * luminance), and copies `entries` plus `total` into a GPU storage buffer.
 *
 * Sampling with two uniform random values `u`, `v` (v a 32-bit unsigned integer):
 *   uint i = min(uint(u * count), count - 1);
 *   SapienAliasEntry e = entries[i];
 *   uint selected = (v < e.threshold) ? i : e.alias;
 *   float pdf = weight(selected) / total;
 *
 * `threshold` is the acceptance probability scaled by 2^32 (clamped to 2^32-1), so a bin that
 * is entirely its own has threshold 0xffffffff. `total` is the sum of the input weights; when
 * it is 0 the table has no positive weights and must not be sampled. Negative, NaN and
 * infinite weights are treated as zero.
 *
 * Ownership: `entries` is allocated by sapien_alias_table_build and freed by
 * sapien_alias_table_free. Functions returning int return 0 on success; on failure they write
 * a NUL-terminated message into `error` (truncated to `error_size`). */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SapienAliasEntry {
  uint32_t alias;
  uint32_t threshold;
} SapienAliasEntry;

typedef struct SapienAliasTable {
  SapienAliasEntry *entries;
  size_t count;
  float total;
} SapienAliasTable;

int sapien_alias_table_build(const float *weights, size_t count, SapienAliasTable *out,
                             char *error, size_t error_size);
void sapien_alias_table_free(SapienAliasTable *table);

#ifdef __cplusplus
}
#endif
