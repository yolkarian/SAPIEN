/* C interface of libsapien_cartpole.so: N continuous CartPoles on one SAPIEN PhysxSystemGpu.
 *
 * PufferLib 5.0 compiles its trainer and the environment as one nvcc C++17 translation unit, which
 * cannot include SAPIEN's C++20 and PhysX headers. The library is built separately against the
 * installed SAPIEN package (shim/sapien_cartpole.cpp) and exposes plain C functions.
 *
 * The pointers returned by the *_ptr getters are CUDA device buffers owned by SAPIEN: float32
 * [num_envs, max_dofs] row-major, dof 0 = cart x, dof 1 = pole angle. They stay valid until
 * sapien_cartpole_destroy(). Every call that enqueues GPU work enqueues it on the stream given to
 * sapien_cartpole_set_stream (the legacy default stream until then). Functions return 0 on success
 * and -1 on error (message in sapien_cartpole_error()) and are not thread-safe. */
#pragma once

#define SAPIEN_CARTPOLE_API __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SapienCartpole SapienCartpole;

/* Enables GPU PhysX (PhysxDefault::EnableGPU with the runtime sapien.physx.enable_gpu()
 * downloaded), builds `num_envs` cartpoles in one scene on CUDA device `cuda_id` and initializes
 * the GPU simulation. Returns NULL on failure and writes the reason into `err`. */
SAPIEN_CARTPOLE_API SapienCartpole *sapien_cartpole_create(int num_envs, int cuda_id, char *err,
                                                           int err_len);
SAPIEN_CARTPOLE_API void sapien_cartpole_destroy(SapienCartpole *h);
SAPIEN_CARTPOLE_API const char *sapien_cartpole_error(const SapienCartpole *h);

SAPIEN_CARTPOLE_API int sapien_cartpole_max_dofs(const SapienCartpole *h);
SAPIEN_CARTPOLE_API float *sapien_cartpole_qpos_ptr(const SapienCartpole *h);
SAPIEN_CARTPOLE_API float *sapien_cartpole_qvel_ptr(const SapienCartpole *h);
SAPIEN_CARTPOLE_API float *sapien_cartpole_qf_ptr(const SapienCartpole *h);

/* PhysxSystemGpu::gpuSetCudaStream: a stream of the primary context of `cuda_id`. */
SAPIEN_CARTPOLE_API int sapien_cartpole_set_stream(SapienCartpole *h, void *stream);
/* Joint forces in the qf buffer -> PhysX (whole buffer, no host synchronization). */
SAPIEN_CARTPOLE_API int sapien_cartpole_apply_qf(SapienCartpole *h);
/* One physics step. Blocks the host until PhysX finished it, so it cannot be stream-captured. */
SAPIEN_CARTPOLE_API int sapien_cartpole_step(SapienCartpole *h);
/* PhysX -> qpos and qvel buffers. */
SAPIEN_CARTPOLE_API int sapien_cartpole_fetch_state(SapienCartpole *h);
/* qpos and qvel buffers -> PhysX (whole buffer), then recompute link poses. */
SAPIEN_CARTPOLE_API int sapien_cartpole_apply_state(SapienCartpole *h);

#ifdef __cplusplus
}
#endif
