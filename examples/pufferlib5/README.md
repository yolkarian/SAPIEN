# SAPIEN in PufferLib 5.0's native trainer

PufferLib 5.0 has no Python API: an environment is C/CUDA code compiled into PufferLib's native
trainer binary. This example runs a SAPIEN PhysX-GPU environment there, a continuous CartPole with
4096 copies on one GPU. Python is needed only to install SAPIEN and to download the PhysX GPU
runtime once; the trainer process does not load Python.

## Layout

| Path | Role |
| --- | --- |
| `shim/sapien_cartpole.{h,cpp}` | `libsapien_cartpole.so`: builds the scene with SAPIEN's C++ API and exposes plain C functions (create, set stream, apply forces, step, fetch and apply state). |
| `env/sapien_cartpole/sapien_cartpole.cu` | The PufferLib GPU environment (`puf_vec_create`, `puf_bind_stream`, `puf_step`, ...). |
| `env/sapien_cartpole/sapien_cartpole_task.cuh` | Task kernels: actions to joint forces, rewards, terminals, in-step resets, observations. |
| `env/sapien_cartpole.ini` | Trainer configuration. |
| `pufferlib-env-capturable.patch` | Trainer change for environments that cannot be recorded into CUDA graphs. |
| `build.sh` | Builds the library and the trainer. |

## Build and run

Needs Linux, an NVIDIA GPU with a driver for CUDA 12.8 or newer, g++ 11 or newer, and what
PufferLib's `build.sh` needs for its native trainer (nvcc, ccache, NCCL, libomp, network access for
raylib).

```shell
python -m pip install sapien-*.whl
python -c "import sapien; sapien.physx.enable_gpu()"  # downloads the PhysX GPU runtime once

git clone -b 5.0 https://github.com/PufferAI/PufferLib.git
git -C PufferLib checkout ea7f989                     # the commit the patch targets

examples/pufferlib5/build.sh PufferLib                # writes PufferLib/puffer_sapien_cartpole
cd PufferLib && ./puffer_sapien_cartpole train
```

`build.sh PUFFERLIB_DIR [OUT] [--float]` uses the SAPIEN package that `$PYTHON` (default
`python3`) imports. It compiles the library with nothing but the package's include and library
directories, copies the environment and its configuration into the PufferLib tree, applies the patch
once, and runs PufferLib's `build.sh sapien_cartpole OUT --cu`. `--float` trains in float32 instead
of bfloat16. PufferLib passes the extra linker flags through `NVCC_EXTRA`, split on whitespace, so
the paths must not contain spaces.

## How it fits together

- **C interface.** PufferLib compiles the trainer and the environment as one nvcc C++17
  translation unit. SAPIEN's headers are C++20 and include PhysX, so the environment talks to SAPIEN
  through `libsapien_cartpole.so`, compiled separately with g++.
- **No Python at runtime.** `PhysxDefault::EnableGPU()` finds the PhysX GPU runtime in the cache
  `sapien.physx.enable_gpu()` downloads into, and PhysX loads it and the CUDA driver. The installed
  headers carry SAPIEN's compile options, and the installed libraries find their dependencies
  through their RUNPATHs, so the library needs no `-D` flags or loader paths.
- **Data stays on the GPU.** The trainer allocates the observation, action, reward and terminal
  buffers on the GPU and hands their pointers to `puf_vec_create`. Each `puf_step` runs CUDA kernels
  between these buffers and SAPIEN's articulation buffers, and SAPIEN's GPU calls, all on the
  stream the trainer binds with `puf_bind_stream`. Nothing is copied to the host.
- **CUDA graphs.** PufferLib records a GPU environment's whole rollout into one CUDA graph. PhysX
  waits for the GPU on the host in every step, and recording the step makes PhysX fail
  (`Fetching GPU Narrowphase failed! 900`, CUDA error 900 being
  `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`). The patch adds `PUF_ENV_CAPTURABLE`: the environment
  defines it as 0, and the trainer then graphs each policy step and the training epoch but runs
  `puf_step` outside the graphs. On an RTX 3090 with 4096 environments, float32 training ran at
  1.58M steps/s this way, 5.8% faster than turning graphs off entirely (`cudagraphs = -1`, 1.49M
  steps/s). With the shipped configuration, bfloat16 (1.93M steps/s) and float32 (1.60M steps/s)
  both reached the maximum return of 500 within 15M steps on seeds 1 and 2.

## Adapting it

- Build your scene in `sapien_cartpole_create` with SAPIEN's C++ components (entities, rigid
  bodies, `PhysxArticulationLinkComponent` and its joints). The URDF loader and the
  actor/articulation builders are Python-only.
- Keep one `PhysxSystemGpu` for all environments and use the whole-buffer `gpu_apply_*` calls in
  the step: the indexed overloads synchronize the stream to validate their indices.
- Change `OBS_SIZE`, `NUM_ATNS`, `ACT_SIZES`, the `Log` fields and the kernels in
  `sapien_cartpole_task.cuh` for your task. PufferLib has no truncation channel: time limits end
  episodes as terminals.
