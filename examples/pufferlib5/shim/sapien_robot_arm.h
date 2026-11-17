/* C interface of libsapien_robot_arm.so: the world of PufferLib's robot_arm task (a Franka Panda
 * picking a cube off a table) for N environments on one SAPIEN PhysxSystemGpu.
 *
 * Every environment is its own SAPIEN scene with one arm and one cube; a shared scene holds the
 * table plane. Frames, joint limits, link masses, inertias, collision boxes and joint gains follow
 * ocean/robot_arm/robot_arm.h of PufferLib ea7f989. Coordinates are SAPIEN's: robot_arm's base frame
 * with z up, so a SAPIEN point (x, y, z) is the robot_arm point (x, z, -y).
 *
 * The buffers of SapienRobotArmBuffers are CUDA device memory owned by the library and valid until
 * sapien_robot_arm_destroy(). Calls that enqueue GPU work enqueue it on the stream given to
 * sapien_robot_arm_set_stream (the legacy default stream until then). Functions return 0 on success
 * and -1 on error (message in sapien_robot_arm_error()) and are not thread-safe. */
#pragma once

#define SAPIEN_ROBOT_ARM_API __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SapienRobotArm SapienRobotArm;

/* Joint and body buffers of all environments. Joint rows hold 9 dofs: arm joints 1-7, then the
 * left and the right finger (opening along the hand's +y and -y axes). */
typedef struct SapienRobotArmBuffers {
  int dofs;               /* row length of qpos, qvel and qtarget */
  float* qpos;            /* [articulations, dofs] */
  float* qvel;            /* [articulations, dofs] */
  float* qtarget;         /* [articulations, dofs] joint drive targets */
  float* rigid_dynamic;   /* [cubes, 13]: position, rotation (w, x, y, z), linear and angular
                             velocity */
  const int* articulation; /* [n] joint row of environment i */
  const int* cube;         /* [n] rigid_dynamic row of environment i */
  const float* pad_impulse;  /* [n, 2, 3] normal contact impulse between the cube and the left,
                                right finger over the last physics step */
  const float* hand_impulse; /* [n, 3, 3] net contact impulse on the hand, the left and the right
                                finger over the last physics step */
} SapienRobotArmBuffers;

/* Enables GPU PhysX (PhysxDefault::EnableGPU with the runtime sapien.physx.enable_gpu()
 * downloaded) and builds `num_envs` environments on CUDA device `cuda_id`, stepping `physics_dt`
 * seconds per physics step with the given solver iteration counts, and initializes the GPU
 * simulation. The joint and body state stays as built until the caller writes it and calls
 * sapien_robot_arm_apply_state(). Returns NULL on failure and writes the reason into `err`. */
SAPIEN_ROBOT_ARM_API SapienRobotArm* sapien_robot_arm_create(int num_envs, int cuda_id,
    float physics_dt, int solver_position_iterations, int solver_velocity_iterations, char* err,
    int err_len);
SAPIEN_ROBOT_ARM_API void sapien_robot_arm_destroy(SapienRobotArm* h);
SAPIEN_ROBOT_ARM_API const char* sapien_robot_arm_error(const SapienRobotArm* h);
SAPIEN_ROBOT_ARM_API SapienRobotArmBuffers sapien_robot_arm_buffers(const SapienRobotArm* h);

/* PhysxSystemGpu::gpuSetCudaStream: a stream of the primary context of `cuda_id`. */
SAPIEN_ROBOT_ARM_API int sapien_robot_arm_set_stream(SapienRobotArm* h, void* stream);
/* qtarget -> PhysX joint drives (whole buffer). */
SAPIEN_ROBOT_ARM_API int sapien_robot_arm_apply_targets(SapienRobotArm* h);
/* `steps` physics steps. Each blocks the host until PhysX finished it, so they cannot be
 * stream-captured. */
SAPIEN_ROBOT_ARM_API int sapien_robot_arm_simulate(SapienRobotArm* h, int steps);
/* PhysX -> qpos, qvel, rigid_dynamic, pad_impulse and hand_impulse. */
SAPIEN_ROBOT_ARM_API int sapien_robot_arm_fetch(SapienRobotArm* h);
/* qpos, qvel and rigid_dynamic -> PhysX (whole buffers), then recompute link poses. */
SAPIEN_ROBOT_ARM_API int sapien_robot_arm_apply_state(SapienRobotArm* h);

#ifdef __cplusplus
}
#endif
