# TLAS isolation benchmark

Standalone Vulkan 1.3 micro-benchmark that measures two ways of isolating N
environments that share K bottom-level acceleration structures (BLASes), for a
batched hardware ray-tracing renderer. It is a measurement program, not part of
SAPIEN: no svulkan2 code is used, only the Vulkan C API.

Design context: `docs/design/render-upgrade-raster-rt.md` §6.1 ("第 0 步") and
§4.2 of the SAPIEN render-upgrade plan.

## What is compared

Both modes render the same scene: every environment has one large ground quad and
M poseable rigid bodies drawn from K = 16 shared procedural meshes (12 subdivided
spheres/boxes of ~1k triangles each, one ground quad, three spares, one of which
is the isolation marker). One camera per environment, square images, primary ray
query plus one shadow ray query toward a fixed directional light.

| | Mode A | Mode B |
|---|---|---|
| Top-level structures | **one** TLAS with all N·(M+2) instances | **N** TLASes of M+2 instances |
| Isolation mechanism | environment *i* translated by a centred-grid cell offset; camera origin gets the same offset; every ray's `tMax` is clipped to the cell boundary | descriptor array `accelerationStructureEXT tlas[]`, indexed by environment id (nonuniform) |
| AS update per frame | 1 build/refit call, N·(M+2) instances | 1 build/refit call carrying **N** build infos (not N calls) |
| Spatial assumption | spacing ≥ environment extent; geometry that leaves its cell disappears from its own image and appears in a neighbour's | none |

The shader is byte-identical between modes; only the TLAS index and whether the
cell offset is added differ. This is the point of the benchmark: any measured
difference comes from the isolation scheme, not from shading.

## What is measured

Per frame, after 10 warm-up frames, over ≥ 60 frames:

* **AS update** — GPU time of the TLAS build/refit, from `vkCmdWriteTimestamp`
  before the build to just after the following barrier.
* **trace** — GPU time of the single compute dispatch over all N cameras.
* **wall** — host clock around `vkQueueSubmit` + `vkWaitForFences`: GPU work plus
  submit/wait overhead. The CPU jitter happens before this window, so
  `wall ≈ as + trace + submit overhead` and does not include `cpu`.
* **cpu** — host time to rewrite the instance buffer (deterministic jitter, one
  transform per instance). Excluded from the GPU timings but reported, because it
  is what a real batched renderer pays per frame.

Median and p90 are reported for the GPU timings.

Two AS update strategies are measured for each mode:

* **rebuild** — `MODE_BUILD` with `PREFER_FAST_BUILD` every frame.
* **refit** — built once with `PREFER_FAST_TRACE` + `ALLOW_UPDATE`, then
  `MODE_UPDATE` every frame.

Also reported: `accelerationStructureSize` summed over the TLAS(es), the same over
all BLASes (shared by both modes), and the instance-buffer byte count.

## Matrix

`M ∈ {8, 32, 128}` × `N ∈ {256, 1024, 4096}` × `R ∈ {64, 128, 256}` ×
`mode ∈ {A, B}` × `update ∈ {rebuild, refit}` — 108 configurations. A
configuration that exceeds a device limit or the memory budget is recorded as
`skipped: <reason>` instead of aborting.

CSV columns:

```
mode,update,N,M,R,as_ms_med,as_ms_p90,trace_ms_med,trace_ms_p90,wall_ms_med,
cpu_ms_med,tlas_bytes,blas_bytes,instance_bytes,fps_images_per_s,status
```

`fps_images_per_s` is N / (AS + trace) — environment-images per second of GPU
time, so it is comparable across different N.

## Correctness guard

`--verify` runs N=16, M=8, R=64 in both modes and checks two things before any
timing is trusted:

1. **Mode agreement.** The two modes trace identical rays over identical geometry
   with the same shader, so their output buffers must be identical pixel for
   pixel. The differing-pixel fraction is reported; more than 0.5% is a failure.
   (There is no float-tolerance band in the comparison: the values compared are
   final RGBA8 bytes produced by the same instructions, so the only legitimate
   differences would be silhouette pixels where two triangles meet exactly — the
   0.5% budget covers them.)
2. **Environment isolation.** Environment 0 carries an unlit marker instance
   packed as `0x00FF00FF` — the alpha byte is zero, which the shading path can
   never emit (it always sets `0xFF`), so an exact match cannot be a shading
   coincidence. No other environment's image may contain that value and
   environment 0's must. Every other environment carries the same instance row
   with a zero ray mask, so instance counts stay uniform.

The guard was checked for sensitivity: disabling the cell clipping in the shader
makes it fail (17 leaked marker pixels spread over 7 environments), so it does
detect the failure it exists to catch.

## Building

Requirements:

* Zig **0.17.1** exactly.
* Vulkan-Headers (header-only C headers).
* The Vulkan loader `libvulkan.so.1` at runtime. There is no development symlink
  on typical end-user systems, so the build links the versioned `.so` directly.
* A GLSL→SPIR-V compiler with `GL_EXT_ray_query` support (Khronos `glslang`
  release binaries are enough). The resulting SPIR-V is embedded in the binary.

```sh
zig build -Doptimize=ReleaseFast \
    -Dglslang=/path/to/glslang \
    -Dvulkan-headers=/path/to/Vulkan-Headers/include \
    -Dvulkan-loader=/usr/lib/x86_64-linux-gnu/libvulkan.so.1
```

`zig build -Doptimize=ReleaseFast` alone works when `glslang` is on `PATH` and the
headers are installed system-wide (`/usr/include`).

## Running

```sh
# single configuration, summary on stderr
./zig-out/bin/tlas_isolation --mode A --update rebuild --n 1024 --m 32 --res 128

# correctness guard first, then a single configuration
./zig-out/bin/tlas_isolation --verify --mode B --n 1024 --m 32 --res 128

# full sweep to CSV
./zig-out/bin/tlas_isolation --matrix --csv results.csv
```

Useful flags: `--frames` (default 60), `--warmup` (default 10), `--spacing`
(mode A grid spacing in metres, default 16), `--validation` (enable validation
layers if the loader has them installed; off by default and off for the matrix).

The program picks the first discrete-NVIDIA physical device and prints its name,
driver version, and the acceleration-structure limits it found.

## Notes and limits

* The cell half-extent is fixed at 7 m and the camera orbit radius at 6 m; only
  the spacing is adjustable. Changing the scene extent means changing
  `cell_half` in `src/scene.zig` and the matching constant in the shader.
* Mode A's `tMax` clipping is a hard geometric cut: geometry that leaves its cell
  is silently missing from that environment. This is the documented weakness of
  the scheme, and the benchmark does not try to hide it — the correctness guard
  uses geometry that stays inside its cell.
* Rebuild uses `PREFER_FAST_BUILD` for both modes, so the AS-time comparison is
  like-for-like; refit uses `PREFER_FAST_TRACE` because that is how a refit-based
  renderer would build its TLAS once.
* Timings come from `VK_QUERY_TYPE_TIMESTAMP`; `timestampPeriod` is read from
  device limits. The wall-clock column includes queue-submit latency and fence
  wait, so it is always ≥ AS + trace.
