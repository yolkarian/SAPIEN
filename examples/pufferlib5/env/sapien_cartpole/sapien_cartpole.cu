// SAPIEN CartPole as a PufferLib 5.0 GPU environment (./build.sh sapien_cartpole --cu).
//
// All environments are one SAPIEN PhysxSystemGpu behind libsapien_cartpole.so (C interface,
// sapien_cartpole.h). The trainer owns the observation, action, reward and terminal buffers on
// the GPU; puf_step moves data between them and SAPIEN's buffers with the kernels of
// sapien_cartpole_task.cuh, all on the stream the trainer binds, so nothing is copied to the host.
//
// PhysxSystemGpu::step() waits for PhysX on the host and cannot be recorded into a CUDA graph.
// PUF_ENV_CAPTURABLE 0 (added to src/pufferl.cu by pufferlib-env-capturable.patch) makes the
// trainer graph only each policy step and the training epoch and run puf_step outside the graphs.
#ifndef PUFFER_SAPIEN_CARTPOLE_CU
#define PUFFER_SAPIEN_CARTPOLE_CU

#define PUF_BACKEND PUF_GPU
#define PUF_ENV_CAPTURABLE 0

#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// observations follow the training precision: bfloat16 by default, float32 with --float
#ifdef PRECISION_FLOAT
typedef float obs_t;
#else
typedef __nv_bfloat16 obs_t;
#endif

#include "pufferenv.h"
#include "sapien_cartpole_task.cuh"

#define ACT_SIZES {1}
#define NUM_ATNS 1
#define OBS_SIZE 4

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}

static struct {
    SapienCartpole* sim;
    Env* envs;
    int n;
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    cudaStream_t stream;
} g_gpu;

static ScIo sc_io() {
    ScIo io = {};
    io.env = g_gpu.envs;
    io.obs = g_gpu.observations;
    io.act = g_gpu.actions;
    io.rew = g_gpu.rewards;
    io.term = g_gpu.terminals;
    io.n = g_gpu.n;
    return io;
}

Env* puf_vec_create(int n, Dict* env_kwargs,
        obs_t* observations, float* actions, float* rewards, float* terminals) {
    int device = 0;
    assert(cudaGetDevice(&device) == cudaSuccess);
    DictItem* seed_item = dict_find(env_kwargs, "seed");
    unsigned int seed = seed_item ? (unsigned int)seed_item->value : 1u;

    char err[2048] = {0};
    g_gpu.sim = sapien_cartpole_create(n, device, err, sizeof(err));
    if (!g_gpu.sim) {
        fprintf(stderr, "sapien_cartpole_create failed: %s\n", err);
        abort();
    }
    g_gpu.envs = sc_make_envs(n, seed);
    assert(g_gpu.envs != NULL);
    g_gpu.n = n;
    g_gpu.observations = observations;
    g_gpu.actions = actions;
    g_gpu.rewards = rewards;
    g_gpu.terminals = terminals;
    g_gpu.stream = 0;
    return g_gpu.envs;
}

void puf_bind_stream(cudaStream_t stream) {
    if (stream == g_gpu.stream) {
        return;
    }
    g_gpu.stream = stream;
    if (sapien_cartpole_set_stream(g_gpu.sim, (void*)stream)) {
        fprintf(stderr, "sapien_cartpole_set_stream failed: %s\n", sapien_cartpole_error(g_gpu.sim));
        abort();
    }
}

// GPU environments are created by puf_vec_create; the trainer's evaluation path still probes one.
void puf_init(Env* env, Dict*) {
    env->num_agents = 1;
}

void puf_reset(Env*) {
    ScIo io = sc_io();
    if (sc_reset(g_gpu.sim, g_gpu.stream, io)) {
        fprintf(stderr, "sapien_cartpole reset failed: %s\n", sapien_cartpole_error(g_gpu.sim));
        abort();
    }
}

void puf_step(Env*) {
    ScIo io = sc_io();
    if (sc_step(g_gpu.sim, g_gpu.stream, io)) {
        fprintf(stderr, "sapien_cartpole step failed: %s (cuda: %s)\n",
            sapien_cartpole_error(g_gpu.sim), cudaGetErrorString(cudaGetLastError()));
        abort();
    }
}

void puf_render(Env*) {
}

void puf_close(Env*) {
    if (g_gpu.stream) {
        cudaStreamSynchronize(g_gpu.stream);
    }
    cudaDeviceSynchronize();
    sapien_cartpole_destroy(g_gpu.sim);
    g_gpu.sim = NULL;
    cudaFree(g_gpu.envs);
    g_gpu.envs = NULL;
}

#endif
