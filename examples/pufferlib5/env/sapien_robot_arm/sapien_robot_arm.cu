// PufferLib's robot_arm task with SAPIEN PhysX GPU as its simulator
// (./build.sh sapien_robot_arm --cu).
//
// The task is ocean/robot_arm/robot_arm.h of PufferLib ea7f989, included unchanged: its state,
// observations, rewards, terminals, resets, logs and rendering. robot_arm.h normally runs on its own
// CUDA physics (robot_arm_cuda.cuh); here libsapien_robot_arm.so (sapien_robot_arm.h) simulates
// the same arm, cube and table, and the kernels below translate between the two:
//   begin: actions -> joint targets in SAPIEN's drive buffer (robot_arm's ra_kbegin)
//   SAPIEN: apply targets, `substeps` physics steps, fetch joints, cube and contact impulses
//   post:  SAPIEN state -> RaState, grasp detection (robot_arm's ra_solve), ra_stept, and for
//          finished environments ra_logep and ra_reset, whose new state is written back to SAPIEN
// All of it is ordered on the trainer's stream; only the physics steps block the host.
//
// PhysxSystemGpu::step() waits for PhysX on the host and cannot be recorded into a CUDA graph, so
// the environment sets PUF_ENV_CAPTURABLE 0 (pufferlib-env-capturable.patch).
//
// Only the pick-and-place mode is supported: stack = 1 and basketball = 1 need robot_arm's second
// cube, ball, hoop and backboard, which libsapien_robot_arm.so does not build.
#ifndef PUFFER_SAPIEN_ROBOT_ARM_CU
#define PUFFER_SAPIEN_ROBOT_ARM_CU

#define PUF_BACKEND PUF_GPU
#define PUF_ENV_CAPTURABLE 0

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// PufferLib's build.sh requires the environment file to declare obs_t; robot_arm.h repeats this
// typedef (bfloat16 for bfloat16 training, float otherwise).
#if defined(from_float) && !defined(PRECISION_FLOAT)
typedef precision_t obs_t;
#else
typedef float obs_t;
#endif

#include "../robot_arm/robot_arm.h"
#include "sapien_robot_arm.h"

#define SRA_THREADS 128
// a pad touches the cube when its normal impulse exceeds this, as in ra_solve
#define SRA_PAD_CONTACT_IMPULSE 1.0e-7f

// The trainer's CPU-environment code also compiles against Env and needs agents[], tag,
// boundary_reached and rng; a GPU environment leaves them unused.
typedef struct Env {
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;
    RaState state;
    float actions[RA_ACTIONS];
} Env;

// SAPIEN (x, y, z) is robot_arm (x, z, -y): robot_arm's world is y-up with the arm base turned
// -90 degrees about x, SAPIEN's is that base frame.
__device__ __forceinline__ RaVec3 sra_to_ra(const float* v) {
    return ra_v3(v[0], v[2], -v[1]);
}
__device__ __forceinline__ void sra_from_ra(RaVec3 v, float* out) {
    out[0] = v.x;
    out[1] = -v.z;
    out[2] = v.y;
}
// A cube rotation in robot_arm's frame, from SAPIEN's (w, x, y, z): the same rotation axis,
// expressed in robot_arm's world.
__device__ __forceinline__ RaQuat sra_rotation_to_ra(const float* q) {
    RaVec3 axis = sra_to_ra(q + 1);
    return ra_quat(axis.x, axis.y, axis.z, q[0]);
}
__device__ __forceinline__ void sra_rotation_from_ra(RaQuat q, float* out) {
    out[0] = q.w;
    sra_from_ra(ra_v3(q.x, q.y, q.z), out + 1);
}

// ra_prep's joint motor, for the energy term of the reward
__device__ __forceinline__ float sra_motor(const RaState* s, int joint) {
    float kp = joint < 2 ? 4500.0f : (joint < 4 ? 3500.0f : 2000.0f);
    float kd = joint < 2 ? 450.0f : (joint < 4 ? 350.0f : 200.0f);
    return ra_clamp(kp * (s->target_q[joint] - s->q[joint]) - kd * s->qd[joint],
        -ra_mlim(joint), ra_mlim(joint));
}

// RaState -> SAPIEN joint, drive target and cube rows of environment i (after ra_reset).
__device__ void sra_write_state(const RaState* s, SapienRobotArmBuffers b, int i) {
    int row = b.articulation[i] * b.dofs;
    for (int joint = 0; joint < RA_DOF; ++joint) {
        b.qpos[row + joint] = s->q[joint];
        b.qvel[row + joint] = s->qd[joint];
        b.qtarget[row + joint] = s->target_q[joint];
    }
    for (int finger = RA_DOF; finger < RA_DOF + 2; ++finger) {
        b.qpos[row + finger] = 0.5f * s->gripper_width;
        b.qvel[row + finger] = 0.5f * s->gripper_velocity;
        b.qtarget[row + finger] = 0.5f * s->gripper_width;
    }
    float* cube = b.rigid_dynamic + b.cube[i] * 13;
    sra_from_ra(s->cube_position, cube);
    sra_rotation_from_ra(s->cube_rotation, cube + 3);
    sra_from_ra(s->cube_velocity, cube + 7);
    sra_from_ra(s->cube_angular_velocity, cube + 10);
}

__device__ void sra_observe(const RaState* s, obs_t* observations, int i) {
    float observation[OBS_SIZE];
    ra_observe(s, observation);
    for (int feature = 0; feature < OBS_SIZE; ++feature) {
        observations[i * OBS_SIZE + feature] = (obs_t)observation[feature];
    }
}

__global__ void sra_k_sync(Env* envs, int n, SapienRobotArmBuffers b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        sra_write_state(&envs[i].state, b, i);
    }
}

__global__ void sra_k_observe(Env* envs, int n, obs_t* observations, float* rewards,
        float* terminals) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    sra_observe(&envs[i].state, observations, i);
    rewards[i] = 0.0f;
    terminals[i] = 0.0f;
}

// ra_kbegin: the arm joints track home + action * span, the fingers half the commanded width.
__global__ void sra_k_begin(Env* envs, int n, const float* actions, SapienRobotArmBuffers b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    Env* env = &envs[i];
    RaState* s = &env->state;
    for (int action = 0; action < RA_ACTIONS; ++action) {
        env->actions[action] = actions[i * RA_ACTIONS + action];
    }
    float* target = b.qtarget + b.articulation[i] * b.dofs;
    for (int joint = 0; joint < RA_DOF; ++joint) {
        float action = ra_clamp(env->actions[joint], -1.0f, 1.0f);
        s->target_q[joint] = ra_clamp(ra_jhome(joint) + action * ra_aspan(joint),
            ra_jmin(joint), ra_jmax(joint));
        target[joint] = s->target_q[joint];
    }
    float grip_action = ra_clamp(env->actions[RA_DOF], -1.0f, 1.0f);
    float target_width = 0.004f + 0.076f * 0.5f * (grip_action + 1.0f);
    target[RA_DOF] = 0.5f * target_width;
    target[RA_DOF + 1] = 0.5f * target_width;
    if (s->grasp_cooldown > 0) {
        s->grasp_cooldown -= 1;
    }
    s->step += 1;
}

// After the physics steps: SAPIEN state -> RaState, grasp detection, reward, terminal, reset,
// observation.
__global__ void sra_k_post(Env* envs, int n, SapienRobotArmBuffers b, float physics_dt,
        obs_t* observations, float* rewards, float* terminals) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    Env* env = &envs[i];
    RaState* s = &env->state;
    int row = b.articulation[i] * b.dofs;
    for (int joint = 0; joint < RA_DOF; ++joint) {
        s->q[joint] = b.qpos[row + joint];
        s->qd[joint] = b.qvel[row + joint];
    }
    s->gripper_width = b.qpos[row + RA_DOF] + b.qpos[row + RA_DOF + 1];
    s->gripper_velocity = b.qvel[row + RA_DOF] + b.qvel[row + RA_DOF + 1];
    const float* cube = b.rigid_dynamic + b.cube[i] * 13;
    s->cube_position = sra_to_ra(cube);
    s->cube_rotation = sra_rotation_to_ra(cube + 3);
    s->cube_velocity = sra_to_ra(cube + 7);
    s->cube_angular_velocity = sra_to_ra(cube + 10);
    RaPose links[RA_LINKS];
    ra_fk(s->q, s->gripper_width, links, NULL, NULL, &s->end_effector);
    RaGripperCollisionFrame frame = ra_gripf(links, s->end_effector);

    // Contact impulses of SAPIEN's last physics step, rescaled to robot_arm's physics step so
    // that the observation scales of robot_arm.h apply.
    float to_ra_step = RA_PHYSICS_DT / physics_dt;
    for (int pad = 0; pad < 2; ++pad) {
        s->pad_normal_impulse[pad] = ra_length(sra_to_ra(b.pad_impulse + (2 * i + pad) * 3))
            * to_ra_step;
    }
    s->gripper_force = 0.5f * (s->pad_normal_impulse[0] + s->pad_normal_impulse[1])
        / RA_PHYSICS_DT;
    // robot_arm sums its contact impulses on the robot and their moments about the end effector;
    // SAPIEN reports net impulses per body, taken at the hand and finger origins.
    RaVec3 body_position[3] = {frame.hand.position, frame.left_finger.position,
        frame.right_finger.position};
    s->wrist_linear_impulse = ra_v3(0, 0, 0);
    s->wrist_angular_impulse = ra_v3(0, 0, 0);
    for (int body = 0; body < 3; ++body) {
        RaVec3 impulse = ra_scale(sra_to_ra(b.hand_impulse + (3 * i + body) * 3), to_ra_step);
        s->wrist_linear_impulse = ra_add(s->wrist_linear_impulse, impulse);
        s->wrist_angular_impulse = ra_add(s->wrist_angular_impulse,
            ra_cross(ra_sub(body_position[body], s->end_effector), impulse));
    }

    // ra_solve's grasp detection, once per control step: both pads press on a cube between them
    // while the gripper is commanded closed. One control step without that pinch (robot_arm's
    // RA_GRASP_LOSS_SUBSTEPS physics steps) loses the grasp.
    float grip_action = ra_clamp(env->actions[RA_DOF], -1.0f, 1.0f);
    int between_pads = 0;
    if (s->pad_normal_impulse[0] > SRA_PAD_CONTACT_IMPULSE
            && s->pad_normal_impulse[1] > SRA_PAD_CONTACT_IMPULSE) {
        RaVec3 left_inward = ra_scale(ra_rotate(
            frame.left_finger.rotation, ra_v3(0, 1, 0)), -1.0f);
        RaVec3 right_inward = ra_scale(ra_rotate(
            frame.right_finger.rotation, ra_v3(0, 1, 0)), -1.0f);
        between_pads =
            ra_dot(ra_sub(s->cube_position, frame.left_finger.position), left_inward) > 0.0f
            && ra_dot(ra_sub(s->cube_position, frame.right_finger.position), right_inward) > 0.0f;
    }
    int pad_pinch = between_pads && grip_action < 0.25f;
    int first_grasp = 0;
    int released = 0;
    if (pad_pinch) {
        s->episode_pinch_force += s->gripper_force;
        s->pinch_substeps += 1;
    }
    if (!s->grasped && s->grasp_cooldown == 0 && pad_pinch) {
        first_grasp = !s->ever_grasped;
        s->grasped = 1;
        s->ever_grasped = 1;
    } else if (s->grasped && !pad_pinch) {
        s->grasped = 0;
        if (grip_action > 0.25f) {
            s->grasp_cooldown = RA_GRASP_COOLDOWN_STEPS;
            released = 1;
        } else {
            s->grasp_cooldown = 0;
            s->slip_events += 1;
        }
    }

    float energy = 0.0f;
    for (int joint = 0; joint < RA_DOF; ++joint) {
        energy += fabsf(sra_motor(s, joint) * s->qd[joint]) * RA_CONTROL_DT;
    }
    float reward = ra_stept(s, env->actions, energy, first_grasp, released, links);
    float terminal = s->done ? 1.0f : 0.0f;
    if (s->done) {
        ra_logep(s, &env->log);
        ra_reset(s);
        sra_write_state(s, b, i);
    }
    sra_observe(s, observations, i);
    rewards[i] = reward;
    terminals[i] = terminal;
}

static int g_ra_no_timeout;
static const char* g_ra_model_glb = "resources/robot_arm/franka_panda.glb";
static RaRenderHost g_ra_render_host;

static struct {
    SapienRobotArm* sim;
    SapienRobotArmBuffers buffers;
    int substeps;
    float physics_dt;
    Env* envs;
    int n;
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    cudaStream_t stream;
} g_gpu;

static int sra_grid(int n) {
    return (n + SRA_THREADS - 1) / SRA_THREADS;
}

static void sra_check(int status, const char* what) {
    cudaError_t cuda = cudaGetLastError();
    if (status != 0 || cuda != cudaSuccess) {
        fprintf(stderr, "sapien_robot_arm %s failed: %s (cuda: %s)\n", what,
            status != 0 ? sapien_robot_arm_error(g_gpu.sim) : "-", cudaGetErrorString(cuda));
        abort();
    }
}

static void sra_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        fprintf(stderr, "sapien_robot_arm %s failed: %s\n", what, cudaGetErrorString(status));
        abort();
    }
}

static double sra_kwarg(Dict* kwargs, const char* key, double fallback) {
    DictItem* item = dict_find(kwargs, key);
    return item != NULL ? item->value : fallback;
}

Env* puf_vec_create(int n, Dict* env_kwargs,
        obs_t* observations, float* actions, float* rewards, float* terminals) {
    if (sra_kwarg(env_kwargs, "stack", 0) != 0 || sra_kwarg(env_kwargs, "basketball", 0) != 0) {
        fprintf(stderr, "sapien_robot_arm supports only the pick-and-place mode "
            "(stack = 0, basketball = 0)\n");
        abort();
    }
    g_ra_no_timeout = sra_kwarg(env_kwargs, "no_timeout", 0) != 0;
    DictItem* model = dict_find(env_kwargs, "model_glb");
    if (model != NULL && model->str != NULL && model->str[0] != '\0'
            && strcmp(model->str, "None") != 0) {
        g_ra_model_glb = model->str;
    }
    g_ra_render_host.model_glb = g_ra_model_glb;
    g_ra_render_host.camera_distance = 1.55f;
    g_ra_render_host.camera_yaw = 0.78f;
    g_ra_render_host.camera_pitch = 0.48f;

    g_gpu.substeps = (int)sra_kwarg(env_kwargs, "substeps", 4);
    g_gpu.physics_dt = RA_CONTROL_DT / (float)g_gpu.substeps;
    int position_iterations = (int)sra_kwarg(env_kwargs, "solver_position_iterations", 8);
    int velocity_iterations = (int)sra_kwarg(env_kwargs, "solver_velocity_iterations", 1);
    int device = 0;
    sra_cuda(cudaGetDevice(&device), "cudaGetDevice");
    char err[2048] = {0};
    g_gpu.sim = sapien_robot_arm_create(n, device, g_gpu.physics_dt, position_iterations,
        velocity_iterations, err, sizeof(err));
    if (!g_gpu.sim) {
        fprintf(stderr, "sapien_robot_arm_create failed: %s\n", err);
        abort();
    }
    g_gpu.buffers = sapien_robot_arm_buffers(g_gpu.sim);

    // robot_arm.cu's ra_fill: environment i starts from RNG state i + 1
    Env* host = (Env*)calloc(n, sizeof(Env));
    for (int i = 0; i < n; i++) {
        host[i].num_agents = 1;
        host[i].state.rng = (unsigned int)i + 1u;
        host[i].state.no_timeout = g_ra_no_timeout;
        ra_reset(&host[i].state);
    }
    sra_cuda(cudaMalloc((void**)&g_gpu.envs, (size_t)n * sizeof(Env)), "cudaMalloc");
    sra_cuda(cudaMemcpy(g_gpu.envs, host, (size_t)n * sizeof(Env), cudaMemcpyHostToDevice),
        "initial state upload");
    free(host);
    g_gpu.n = n;
    g_gpu.observations = observations;
    g_gpu.actions = actions;
    g_gpu.rewards = rewards;
    g_gpu.terminals = terminals;
    g_gpu.stream = 0;

    // SAPIEN starts from the initial RaStates.
    sra_k_sync<<<sra_grid(n), SRA_THREADS>>>(g_gpu.envs, n, g_gpu.buffers);
    sra_check(sapien_robot_arm_apply_state(g_gpu.sim), "initial state");
    sra_check(sapien_robot_arm_apply_targets(g_gpu.sim), "initial targets");
    sra_cuda(cudaDeviceSynchronize(), "initial state");
    return g_gpu.envs;
}

void puf_bind_stream(cudaStream_t stream) {
    if (stream == g_gpu.stream) {
        return;
    }
    g_gpu.stream = stream;
    sra_check(sapien_robot_arm_set_stream(g_gpu.sim, (void*)stream), "set_stream");
}

// GPU environments are created by puf_vec_create; the trainer's evaluation path still probes one.
void puf_init(Env* env, Dict*) {
    env->num_agents = 1;
}

// As robot_arm's ra_kinit: observe the current state; episodes reset inside puf_step.
void puf_reset(Env*) {
    sra_k_observe<<<sra_grid(g_gpu.n), SRA_THREADS, 0, g_gpu.stream>>>(
        g_gpu.envs, g_gpu.n, g_gpu.observations, g_gpu.rewards, g_gpu.terminals);
    sra_check(0, "reset");
}

void puf_step(Env*) {
    int grid = sra_grid(g_gpu.n);
    sra_k_begin<<<grid, SRA_THREADS, 0, g_gpu.stream>>>(
        g_gpu.envs, g_gpu.n, g_gpu.actions, g_gpu.buffers);
    sra_check(sapien_robot_arm_apply_targets(g_gpu.sim), "apply_targets");
    sra_check(sapien_robot_arm_simulate(g_gpu.sim, g_gpu.substeps), "simulate");
    sra_check(sapien_robot_arm_fetch(g_gpu.sim), "fetch");
    sra_k_post<<<grid, SRA_THREADS, 0, g_gpu.stream>>>(g_gpu.envs, g_gpu.n, g_gpu.buffers,
        g_gpu.physics_dt, g_gpu.observations, g_gpu.rewards, g_gpu.terminals);
    sra_check(sapien_robot_arm_apply_state(g_gpu.sim), "apply_state");
}

void puf_close(Env*) {
    ra_rclose(&g_ra_render_host);
    if (!g_gpu.sim) {
        return;
    }
    cudaDeviceSynchronize();
    sapien_robot_arm_destroy(g_gpu.sim);
    g_gpu.sim = NULL;
    cudaFree(g_gpu.envs);
    g_gpu.envs = NULL;
}

// robot_arm's puf_render for environment 0; a requested reset ends its episode at the next step.
void puf_render(Env*) {
    if (g_gpu.stream) {
        cudaStreamSynchronize(g_gpu.stream);
    }
    RaState state;
    sra_cuda(cudaMemcpy(&state, &g_gpu.envs->state, sizeof(RaState), cudaMemcpyDeviceToHost),
        "render download");
    RaPose links[RA_LINKS];
    ra_fk(state.q, state.gripper_width, links, NULL, NULL, &state.end_effector);
    ra_draw(&g_ra_render_host, &state, links);
    if (!g_ra_render_host.reset_requested) {
        return;
    }
    g_ra_render_host.reset_requested = 0;
    int done = 1;
    sra_cuda(cudaMemcpy(&g_gpu.envs->state.done, &done, sizeof(int), cudaMemcpyHostToDevice),
        "render reset");
}

#endif
