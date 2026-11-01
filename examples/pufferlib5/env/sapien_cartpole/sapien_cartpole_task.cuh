// CartPole task on the SAPIEN simulation of libsapien_cartpole.so: CUDA kernels moving data
// between the trainer's buffers and SAPIEN's articulation buffers. The including file defines
// `obs_t` (float or __nv_bfloat16) and `Agent` (pufferenv.h) first.
//
// action a -> cart force 10 N * clip(a, -1, 1); observation [x, theta, x_dot, theta_dot];
// reward 1 every step, including the last; an episode ends when |x| > 2.4, |theta| > 0.2095 or
// after 500 steps (PufferLib has no truncation channel, so all of them are terminals). Finished
// environments reset inside the step to qpos, qvel ~ U(-0.05, 0.05).
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sapien_cartpole.h"

#define SC_FORCE_MAG 10.0f
#define SC_X_LIMIT 2.4f
#define SC_THETA_LIMIT 0.2095f
#define SC_MAX_STEPS 500
#define SC_RESET_RANGE 0.05f
#define SC_THREADS 256

// Float fields only, `n` last: the trainer sums logs as floats and divides by n.
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

// The trainer's CPU-environment code also compiles against Env and needs agents[], tag and
// boundary_reached; a GPU environment leaves them unused.
struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    int tick;
    float ret;
    unsigned int rng;
};

__device__ __forceinline__ float sc_uniform(unsigned int* s) {
    *s = *s * 1664525u + 1013904223u;
    unsigned int x = *s;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return (x >> 8) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float sc_reset_value(unsigned int* s) {
    return (2.0f * sc_uniform(s) - 1.0f) * SC_RESET_RANGE;
}

__device__ __forceinline__ void sc_store(float* p, float v) {
    *p = v;
}
__device__ __forceinline__ void sc_store(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16(v);
}

// Trainer actions -> SAPIEN joint forces: the cart dof gets the force, the pole dof none.
__global__ void sc_k_actions(const float* __restrict__ act, float* __restrict__ qf, int n, int dofs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float a = fminf(fmaxf(act[i], -1.0f), 1.0f);
    qf[i * dofs] = SC_FORCE_MAG * a;
    for (int d = 1; d < dofs; d++) {
        qf[i * dofs + d] = 0.0f;
    }
}

// After the fetch: reward and terminal from the new state, and reset values for finished
// environments written into the fetched qpos/qvel buffers, which the host then applies.
__global__ void sc_k_post(Env* __restrict__ env, float* __restrict__ qpos, float* __restrict__ qvel,
        float* __restrict__ rew, float* __restrict__ term, int n, int dofs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    Env& e = env[i];  // field-wise updates only: the trainer clears e.log from another stream
    float x = qpos[i * dofs];
    float th = qpos[i * dofs + 1];
    e.tick += 1;
    e.ret += 1.0f;
    // !(<=) ends the episode on a NaN state instead of running forever
    bool done = !(fabsf(x) <= SC_X_LIMIT) || !(fabsf(th) <= SC_THETA_LIMIT) || e.tick >= SC_MAX_STEPS;
    rew[i] = 1.0f;
    term[i] = done ? 1.0f : 0.0f;
    if (done) {
        e.log.perf += e.ret / (float)SC_MAX_STEPS;
        e.log.score += e.ret;
        e.log.episode_return += e.ret;
        e.log.episode_length += (float)e.tick;
        e.log.n += 1.0f;
        qpos[i * dofs] = sc_reset_value(&e.rng);
        qpos[i * dofs + 1] = sc_reset_value(&e.rng);
        qvel[i * dofs] = sc_reset_value(&e.rng);
        qvel[i * dofs + 1] = sc_reset_value(&e.rng);
        e.tick = 0;
        e.ret = 0.0f;
    }
}

__global__ void sc_k_reset_all(Env* __restrict__ env, float* __restrict__ qpos, float* __restrict__ qvel,
        float* __restrict__ rew, float* __restrict__ term, int n, int dofs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    unsigned int s = env[i].rng;
    qpos[i * dofs] = sc_reset_value(&s);
    qpos[i * dofs + 1] = sc_reset_value(&s);
    qvel[i * dofs] = sc_reset_value(&s);
    qvel[i * dofs + 1] = sc_reset_value(&s);
    env[i].rng = s;
    env[i].tick = 0;
    env[i].ret = 0.0f;
    rew[i] = 0.0f;
    term[i] = 0.0f;
}

// SAPIEN state -> trainer observations [x, theta, x_dot, theta_dot]
__global__ void sc_k_obs(const float* __restrict__ qpos, const float* __restrict__ qvel,
        obs_t* __restrict__ obs, int n, int dofs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    sc_store(&obs[4 * i + 0], qpos[i * dofs]);
    sc_store(&obs[4 * i + 1], qpos[i * dofs + 1]);
    sc_store(&obs[4 * i + 2], qvel[i * dofs]);
    sc_store(&obs[4 * i + 3], qvel[i * dofs + 1]);
}

static inline int sc_grid(int n) {
    return (n + SC_THREADS - 1) / SC_THREADS;
}

// Device Env[n] with a fixed seed per environment.
static inline Env* sc_make_envs(int n, unsigned int seed) {
    Env* host = (Env*)calloc(n, sizeof(Env));
    for (int i = 0; i < n; i++) {
        unsigned int s = seed * 0x9E3779B9u + (unsigned int)i * 0x85EBCA6Bu + 0x165667B1u;
        s ^= s >> 15;
        s *= 0x2c1b3c6du;
        s ^= s >> 12;
        host[i].rng = s ? s : 1u;
        host[i].num_agents = 1;
    }
    Env* dev = NULL;
    if (cudaMalloc((void**)&dev, (size_t)n * sizeof(Env)) != cudaSuccess) {
        free(host);
        return NULL;
    }
    cudaMemcpy(dev, host, (size_t)n * sizeof(Env), cudaMemcpyHostToDevice);
    free(host);
    return dev;
}

// The trainer's buffers of one batch.
struct ScIo {
    Env* env;
    obs_t* obs;
    const float* act;
    float* rew;
    float* term;
    int n;
};

// Reset every environment: random state -> PhysX -> observations.
static inline int sc_reset(SapienCartpole* h, cudaStream_t s, const ScIo& io) {
    int dofs = sapien_cartpole_max_dofs(h);
    float* qpos = sapien_cartpole_qpos_ptr(h);
    float* qvel = sapien_cartpole_qvel_ptr(h);
    cudaMemsetAsync(sapien_cartpole_qf_ptr(h), 0, (size_t)io.n * dofs * sizeof(float), s);
    sc_k_reset_all<<<sc_grid(io.n), SC_THREADS, 0, s>>>(io.env, qpos, qvel, io.rew, io.term, io.n, dofs);
    if (sapien_cartpole_apply_state(h)) {
        return -1;
    }
    sc_k_obs<<<sc_grid(io.n), SC_THREADS, 0, s>>>(qpos, qvel, io.obs, io.n, dofs);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// One control step: actions -> joint forces -> physics step -> state -> reward, terminal and
// resets -> PhysX -> observations. Everything is ordered on stream `s`; only the physics step
// blocks the host.
static inline int sc_step(SapienCartpole* h, cudaStream_t s, const ScIo& io) {
    int dofs = sapien_cartpole_max_dofs(h);
    float* qpos = sapien_cartpole_qpos_ptr(h);
    float* qvel = sapien_cartpole_qvel_ptr(h);
    sc_k_actions<<<sc_grid(io.n), SC_THREADS, 0, s>>>(io.act, sapien_cartpole_qf_ptr(h), io.n, dofs);
    if (sapien_cartpole_apply_qf(h) || sapien_cartpole_step(h) || sapien_cartpole_fetch_state(h)) {
        return -1;
    }
    sc_k_post<<<sc_grid(io.n), SC_THREADS, 0, s>>>(io.env, qpos, qvel, io.rew, io.term, io.n, dofs);
    if (sapien_cartpole_apply_state(h)) {
        return -1;
    }
    sc_k_obs<<<sc_grid(io.n), SC_THREADS, 0, s>>>(qpos, qvel, io.obs, io.n, dofs);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
