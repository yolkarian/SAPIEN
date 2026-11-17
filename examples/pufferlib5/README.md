# SAPIEN in PufferLib 5.0's native trainer

PufferLib 5.0 has no Python API: an environment is C/CUDA code compiled into PufferLib's native
trainer binary. This example runs SAPIEN PhysX-GPU environments there, 4096 copies on one GPU:

- `sapien_cartpole`: a continuous CartPole.
- `sapien_robot_arm`: PufferLib's own GPU task `robot_arm` (a Franka Panda picks a cube and places
  it on a target) with SAPIEN as its simulator instead of the hand-written CUDA physics it ships with.

Python is needed only to install SAPIEN and to download the PhysX GPU runtime once; the trainer
process does not load Python.

## Layout

| Path | Role |
| --- | --- |
| `shim/sapien_cartpole.{h,cpp}` | `libsapien_cartpole.so`: builds the scene with SAPIEN's C++ API and exposes plain C functions (create, set stream, apply forces, step, fetch and apply state). |
| `env/sapien_cartpole/sapien_cartpole.cu` | The PufferLib GPU environment (`puf_vec_create`, `puf_bind_stream`, `puf_step`, ...). |
| `env/sapien_cartpole/sapien_cartpole_task.cuh` | Task kernels: actions to joint forces, rewards, terminals, in-step resets, observations. |
| `shim/sapien_robot_arm.{h,cpp}` | `libsapien_robot_arm.so`: robot_arm's arm, cube and table as SAPIEN components, one scene per environment; C functions to apply drive targets, step, fetch joint, cube and contact data, and apply state. |
| `env/sapien_robot_arm/sapien_robot_arm.cu` | The PufferLib GPU environment: robot_arm's task code (`ocean/robot_arm/robot_arm.h`, included unchanged) on SAPIEN's state. |
| `env/*.ini` | Trainer configurations. |
| `pufferlib-env-capturable.patch` | Trainer change for environments that cannot be recorded into CUDA graphs. |
| `build.sh` | Builds a library and its trainer. |

## Build and run

Needs Linux, an NVIDIA GPU with a driver for CUDA 12.8 or newer, g++ 11 or newer, and what
PufferLib's `build.sh` needs for its native trainer (nvcc, ccache, NCCL, libomp, libGL, network
access for raylib).

```shell
python -m pip install sapien-*.whl
python -c "import sapien; sapien.physx.enable_gpu()"  # downloads the PhysX GPU runtime once

git clone -b 5.0 https://github.com/PufferAI/PufferLib.git
git -C PufferLib checkout ea7f989                     # the commit the patch targets

examples/pufferlib5/build.sh PufferLib sapien_cartpole    # writes PufferLib/puffer_sapien_cartpole
examples/pufferlib5/build.sh PufferLib sapien_robot_arm   # writes PufferLib/puffer_sapien_robot_arm
cd PufferLib && ./puffer_sapien_robot_arm train
```

`build.sh PUFFERLIB_DIR ENV [OUT] [--float]` uses the SAPIEN package that `$PYTHON` (default
`python3`) imports. It compiles `libENV.so` with nothing but the package's include and library
directories, copies the environment and its configuration into the PufferLib tree, applies the patch
once, and runs PufferLib's `build.sh ENV OUT --cu`. `--float` trains in float32 instead of bfloat16.
PufferLib passes the extra linker flags through `NVCC_EXTRA`, split on whitespace, so the paths must
not contain spaces.

## How it fits together

- **C interface.** PufferLib compiles the trainer and the environment as one nvcc C++17
  translation unit. SAPIEN's headers are C++20 and include PhysX, so each environment talks to
  SAPIEN through a C library compiled separately with g++.
- **No Python at runtime.** `PhysxDefault::EnableGPU()` finds the PhysX GPU runtime in the cache
  `sapien.physx.enable_gpu()` downloads into, and PhysX loads it and the CUDA driver. The installed
  headers carry SAPIEN's compile options, and the installed libraries find their dependencies
  through their RUNPATHs, so the libraries need no `-D` flags or loader paths.
- **Data stays on the GPU.** The trainer allocates the observation, action, reward and terminal
  buffers on the GPU and hands their pointers to `puf_vec_create`. Each `puf_step` runs CUDA kernels
  between these buffers and SAPIEN's buffers, and SAPIEN's GPU calls, all on the stream the trainer
  binds with `puf_bind_stream`. Nothing is copied to the host.
- **CUDA graphs.** PufferLib records a GPU environment's whole rollout into one CUDA graph. PhysX
  waits for the GPU on the host in every step, and recording the step makes PhysX fail
  (`Fetching GPU Narrowphase failed! 900`, CUDA error 900 being
  `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`). The patch adds `PUF_ENV_CAPTURABLE`: the environment
  defines it as 0, and the trainer then graphs each policy step (per rollout slot when
  `async = 1`, PufferLib's default) and the training epoch but runs `puf_step` outside the graphs.
  Both `async = 1` and `async = 0` train.

## CartPole

On an otherwise idle RTX 3090 with 4096 environments, float32 training ran at 1.58M steps/s with
the patch, 5.8% faster than turning graphs off entirely (`cudagraphs = -1`, 1.49M steps/s; three
runs each, spread under 1%). With the shipped configuration, bfloat16 (1.93M steps/s) and float32
(1.60M steps/s) both reached the maximum return of 500 within 15M steps on seeds 1 and 2. The host
waits on every physics step, so other jobs on the machine cost throughput: a busy host gave 1.3M to
1.9M steps/s for the same run.

## Robot arm

`ocean/robot_arm` is the GPU-simulated task of PufferLib 5.0: per environment, a 7-dof Franka Panda
with a parallel gripper and a 7 cm cube on a table, 8 continuous actions (absolute joint targets
around a home pose and the gripper width), a 69-dim observation, shaped rewards for reaching,
grasping, lifting, carrying and placing, and episodes of up to 600 steps at 60 Hz. Its physics is
its own CUDA code (`robot_arm_cuda.cuh`: articulated-body dynamics, box contacts and compliant
finger pads, one environment per CUDA thread).

`sapien_robot_arm.cu` keeps everything of the task and replaces only that physics:

- It includes `ocean/robot_arm/robot_arm.h` unchanged and calls its `ra_reset`, `ra_observe`,
  `ra_stept`, `ra_logep`, `ra_fk`, `puf_log` and rendering, so observations, rewards, terminals,
  resets and logs are robot_arm's own.
- `libsapien_robot_arm.so` builds robot_arm's world from SAPIEN components: the arm's joint frames,
  limits, link masses and inertias and its box collision geometry (links, hand, fingers) come from
  `robot_arm.h`; robot_arm's joint PD motors and gripper motor become PhysX joint drives with the
  same gains and force limits. Every environment is a SAPIEN scene with its own PhysX environment
  ID; the table plane sits in one shared scene.
- Each step writes the joint targets of the actions into SAPIEN's drive-target buffer, runs the
  physics steps, fetches joints, cube and contact impulses (pad-cube pairs, and net impulses on the
  hand and fingers), converts them into robot_arm's `RaState`, and applies the reset state of
  finished episodes. SAPIEN's world is robot_arm's arm base frame, z up; robot_arm's is y up.

Differences from robot_arm's physics:

- PhysX rigid contacts instead of robot_arm's elastic finger pads (`RA_PAD_ELASTIC_MODULUS`).
  Under the gripper's 90-98 N the pads sink into the 70 mm cube until the gripper reads 63-69 mm
  wide, depending on the solver settings (table below).
- Grasps are detected once per control step from the last physics step's contacts; robot_arm
  checks every physics step and drops a grasp after 8 physics steps without a pinch, here after one
  control step.
- The wrist force and torque observations sum net contact impulses on the hand and fingers and take
  their moments at the bodies' origins; robot_arm sums contact points of all robot bodies.
- The energy term of the reward uses the joint motor torques at the end of the step.
- Only the pick-and-place mode: `stack = 1` and `basketball = 1` need a second cube, a ball, a hoop
  and a backboard the library does not build.

Physics settings are `[env]` keys of `sapien_robot_arm.ini`. With 4096 environments on an idle
RTX 3090, random actions, environment stepping only (no policy or training):

| `substeps` (physics rate) | solver iterations (position/velocity) | ms per step | environment steps/s | grasp width on the 70 mm cube |
| --- | --- | --- | --- | --- |
| 2 (120 Hz) | 16/4 | 23.1 | 177K | 63.2 mm |
| 4 (240 Hz) | 8/1 | 27.4 | 149K | 65.0 mm |
| 2 (120 Hz) | 32/4 | 37.9 | 108K | 67.4 mm |
| 4 (240 Hz) | 16/4 | 44.7 | 92K | 68.1 mm |
| 8 (480 Hz, robot_arm's rate) | 8/1 | 52.8 | 78K | 68.7 mm |

PhysX's steps take all of it: applying targets, fetching state and contacts and applying resets
cost 0.3 ms per control step together. The shipped configuration uses 4 substeps with 8/1
iterations.

Training with robot_arm.ini's policy and training settings (shared by `sapien_robot_arm.ini`),
4096 environments, bfloat16, seed 42, 1B steps on an RTX 3090, against PufferLib's own robot_arm
built from the same tree. Metrics are the last training log, at episode end:

| | robot_arm (its own CUDA physics) | sapien_robot_arm |
| --- | --- | --- |
| success rate | 0 | 0 |
| grasp rate | 6.8% | 2.4% |
| lift rate | 0 | 0 |
| gripper to cube | 0.13 m | 0.22 m |
| cube to target | 0.50 m | 0.62 m |
| episode return | -4.27 | -4.31 |
| training steps/s, alone on the GPU | 560K at the start, 230K-280K after 40M-50M steps | 165K-180K at the start, 150K from 300M steps on |

Neither simulator learns to pick within 1B steps with these settings: both policies learn to reach
the cube and close the gripper on it in a few percent of the episodes, and neither lifts it. The
integration runs the task end to end at about 150K training steps/s; comparing how well the task
trains on either physics first needs settings under which robot_arm learns on its own physics.
robot_arm's steps/s depends on the state of its environments: a fresh run fell from 560K to 280K
within its first 40M steps. At ea7f989, `ocean/robot_arm/robot_arm.cu` fails PufferLib's own
`build.sh` check (`must typedef obs_t`, the typedef is in `robot_arm.h`); the baseline was built
with that typedef added to `robot_arm.cu`.

## Adapting it

- Build your scene in the library's create function with SAPIEN's C++ components (entities, rigid
  bodies, `PhysxArticulationLinkComponent` and its joints). Configure joint types, anchors, limits
  and drives before adding the links to a scene: PhysX ignores joint motion changes once the
  articulation is in a scene. The URDF loader and the actor/articulation builders are Python-only.
- Keep one `PhysxSystemGpu` for all environments and use the whole-buffer `gpu_apply_*` calls in
  the step: the indexed overloads synchronize the stream to validate their indices. Environments
  whose bodies touch get a scene each under `PhysxSceneConfig::numScenes`; static geometry they
  share goes into one scene marked with `PhysxSystemGpu::markSceneShared`.
- Create contact queries once after `gpuInit()` and run them with `synchronize = false`: the
  impulses then land on the system's stream, where the environment's kernels read them.
- Change `OBS_SIZE`, `NUM_ATNS`, `ACT_SIZES`, the `Log` fields and the kernels for your task.
  PufferLib's `build.sh` requires the environment file itself to declare `obs_t`. PufferLib has no
  truncation channel: time limits end episodes as terminals.
