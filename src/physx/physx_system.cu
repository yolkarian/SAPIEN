#include "./physx_system.cuh"

#include <cstdio>

namespace sapien {
namespace physx {

__global__ void body_data_physx_to_sapien_kernel(SapienBodyData *__restrict__ sapien_data,
                                                 PhysxPose const *__restrict__ physx_pose,
                                                 Vec3 const *__restrict__ physx_linear_velocity,
                                                 Vec3 const *__restrict__ physx_angular_velocity,
                                                 Vec3 const *__restrict__ offset, int count) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count) {
    return;
  }

  sapien_data[g] = {
      physx_pose[g].p - offset[g],
      Quat(physx_pose[g].q.w, physx_pose[g].q.x, physx_pose[g].q.y, physx_pose[g].q.z),
      physx_linear_velocity[g],
      physx_angular_velocity[g],
  };
}

__global__ void body_data_sapien_to_physx_kernel(PhysxPose *__restrict__ physx_pose,
                                                 Vec3 *__restrict__ physx_linear_velocity,
                                                 Vec3 *__restrict__ physx_angular_velocity,
                                                 SapienBodyData const *__restrict__ sapien_data,
                                                 Vec3 const *__restrict__ offset, int count) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count) {
    return;
  }

  SapienBodyData sd = sapien_data[g];
  physx_pose[g] = {{sd.q.x, sd.q.y, sd.q.z, sd.q.w}, sd.p + offset[g]};
  physx_linear_velocity[g] = sd.v;
  physx_angular_velocity[g] = sd.w;
}

__global__ void body_data_sapien_to_physx_kernel(
    PhysxPose *__restrict__ physx_pose, Vec3 *__restrict__ physx_linear_velocity,
    Vec3 *__restrict__ physx_angular_velocity, uint32_t *__restrict__ physx_index,
    SapienBodyData const *__restrict__ sapien_data, uint32_t const *__restrict__ sapien_index,
    int const *__restrict__ apply_index, Vec3 const *__restrict__ offset, int count) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count) {
    return;
  }

  int i = apply_index[g];
  SapienBodyData sd = sapien_data[i];

  physx_pose[g] = {{sd.q.x, sd.q.y, sd.q.z, sd.q.w}, sd.p + offset[i]};
  physx_linear_velocity[g] = sd.v;
  physx_angular_velocity[g] = sd.w;
  physx_index[g] = sapien_index[i];
}

__global__ void link_pose_physx_to_sapien_kernel(SapienBodyData *__restrict__ sapien_data,
                                                 PhysxPose const *__restrict__ physx_pose,
                                                 Vec3 const *__restrict__ offset, int link_count,
                                                 int count) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count) {
    return;
  }

  int ai = g / link_count;

  sapien_data[g].p = physx_pose[g].p - offset[ai];
  sapien_data[g].q =
      Quat(physx_pose[g].q.w, physx_pose[g].q.x, physx_pose[g].q.y, physx_pose[g].q.z);
}

__global__ void root_pose_sapien_to_physx_kernel(PhysxPose *__restrict__ physx_pose,
                                                 SapienBodyData const *__restrict__ sapien_data,
                                                 int const *__restrict__ index,
                                                 Vec3 const *__restrict__ offset, int link_count,
                                                 int count) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count) {
    return;
  }

  int ai = index[g];
  SapienBodyData sd = sapien_data[ai * link_count];

  physx_pose[g] = {{sd.q.x, sd.q.y, sd.q.z, sd.q.w}, sd.p + offset[ai]};
}

__global__ void link_vel_physx_to_sapien_kernel(SapienBodyData *__restrict__ sapien_data,
                                                Vec3 const *__restrict__ physx_linear_velocity,
                                                Vec3 const *__restrict__ physx_angular_velocity,
                                                int count) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count) {
    return;
  }

  sapien_data[g].v = physx_linear_velocity[g];
  sapien_data[g].w = physx_angular_velocity[g];
}

__global__ void root_vel_sapien_to_physx_kernel(Vec3 *__restrict__ physx_linear_velocity,
                                                Vec3 *__restrict__ physx_angular_velocity,
                                                SapienBodyData const *__restrict__ sapien_data,
                                                int const *__restrict__ index, int link_count,
                                                int count) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count) {
    return;
  }

  int ai = index[g];
  SapienBodyData sd = sapien_data[ai * link_count];

  physx_linear_velocity[g] = sd.v;
  physx_angular_velocity[g] = sd.w;
}

__global__ void gather_blocks_kernel(uint32_t *__restrict__ dst, uint32_t const *__restrict__ src,
                                     int const *__restrict__ index, int block_size, int count) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  int total = count * block_size;
  if (g >= total) {
    return;
  }

  int dst_block = g / block_size;
  int block_offset = g % block_size;
  int src_block = index[dst_block];
  dst[g] = src[src_block * block_size + block_offset];
}

__global__ void pack_vec3_kernel(Vec3 *__restrict__ dst, float const *__restrict__ src,
                                 int stride, int count) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count) {
    return;
  }

  dst[g] = {src[g * stride], src[g * stride + 1], src[g * stride + 2]};
}

__global__ void scatter_articulation_jacobians_kernel(float *__restrict__ dst,
                                                      float const *__restrict__ src,
                                                      int const *__restrict__ index,
                                                      uint32_t const *__restrict__ shape,
                                                      int max_rows, int max_cols, int count) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  int block_size = max_rows * max_cols;
  int total = count * block_size;
  if (g >= total) {
    return;
  }

  int selection = g / block_size;
  int block_offset = g % block_size;
  int articulation = index[selection];
  int row = block_offset / max_cols;
  int col = block_offset % max_cols;

  uint32_t rows = shape[articulation * 2];
  uint32_t cols = shape[articulation * 2 + 1];

  float value = 0.f;
  if (static_cast<uint32_t>(row) < rows && static_cast<uint32_t>(col) < cols) {
    value = src[selection * block_size + row * cols + col];
  }
  dst[articulation * block_size + block_offset] = value;
}

__device__ int binary_search(ActorPairQuery const *__restrict__ arr, int count, ActorPair x) {
  int low = 0;
  int high = count - 1;
  while (low <= high) {
    int mid = low + (high - low) / 2;
    if (arr[mid].pair == x)
      return mid;
    if (arr[mid].pair < x)
      low = mid + 1;
    else
      high = mid - 1;
  }
  return -1;
}

__device__ int binary_search(ActorQuery const *__restrict__ arr, int count, ::physx::PxActor *x) {
  int low = 0;
  int high = count - 1;
  while (low <= high) {
    int mid = low + (high - low) / 2;
    if (arr[mid].actor == x)
      return mid;
    if (arr[mid].actor < x)
      low = mid + 1;
    else
      high = mid - 1;
  }
  return -1;
}

__global__ void handle_contacts_kernel(::physx::PxGpuContactPair *__restrict__ contacts,
                                       int contact_count, ActorPairQuery *__restrict__ query,
                                       int query_count, Vec3 *__restrict__ out_forces) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= contact_count) {
    return;
  }

  int order = 0;
  ActorPair pair = makeActorPair(contacts[g].actor0, contacts[g].actor1, order);

  int index = binary_search(query, query_count, pair);
  if (index < 0) {
    return;
  }
  uint32_t id = query[index].id;

  order *= query[index].order;

  ::physx::PxContactPatch *patches = (::physx::PxContactPatch *)contacts[g].contactPatches;
  ::physx::PxContact *points = (::physx::PxContact *)contacts[g].contactPoints;

  float *forces = contacts[g].contactForces;

  Vec3 force = Vec3(0.f);
  for (int pi = 0; pi < contacts[g].nbPatches; ++pi) {
    Vec3 normal(patches[pi].normal.x, patches[pi].normal.y, patches[pi].normal.z);
    for (int i = 0; i < patches[pi].nbContacts; ++i) {
      int ci = patches[pi].startContactIndex + i;
      float f = forces[ci];
      force += normal * (f * order);
      // printf("normal = %f %f %f, normal length2 = %f, separation = %f, force = %f\n", normal.x,
      //        normal.y, normal.z, normal.dot(normal), points[ci].separation, f);
    }
  }
  atomicAdd(&out_forces[id].x, force.x);
  atomicAdd(&out_forces[id].y, force.y);
  atomicAdd(&out_forces[id].z, force.z);
}

__global__ void handle_net_contact_force_kernel(::physx::PxGpuContactPair *__restrict__ contacts,
                                                int contact_count, ActorQuery *__restrict__ query,
                                                int query_count, Vec3 *__restrict__ out_forces) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= contact_count) {
    return;
  }

  ::physx::PxActor *actor0 = contacts[g].actor0;
  ::physx::PxActor *actor1 = contacts[g].actor1;

  int index0 = binary_search(query, query_count, actor0);
  int index1 = binary_search(query, query_count, actor1);

  if (index0 < 0 && index1 < 0) {
    return;
  }

  ::physx::PxContactPatch *patches = (::physx::PxContactPatch *)contacts[g].contactPatches;
  ::physx::PxContact *points = (::physx::PxContact *)contacts[g].contactPoints;

  float *forces = contacts[g].contactForces;

  Vec3 force = Vec3(0.f);
  for (int pi = 0; pi < contacts[g].nbPatches; ++pi) {
    Vec3 normal(patches[pi].normal.x, patches[pi].normal.y, patches[pi].normal.z);
    for (int i = 0; i < patches[pi].nbContacts; ++i) {
      int ci = patches[pi].startContactIndex + i;
      float f = forces[ci];
      force += normal * f;
    }
  }

  if (index0 >= 0) {
    int id = query[index0].id;
    atomicAdd(&out_forces[id].x, force.x);
    atomicAdd(&out_forces[id].y, force.y);
    atomicAdd(&out_forces[id].z, force.z);
  }
  if (index1 >= 0) {
    int id = query[index1].id;
    atomicAdd(&out_forces[id].x, -force.x);
    atomicAdd(&out_forces[id].y, -force.y);
    atomicAdd(&out_forces[id].z, -force.z);
  }
}

constexpr int BLOCK_SIZE = 128;

void body_data_physx_to_sapien(void *sapien_data, void *physx_pose, void *physx_linear_velocity,
                               void *physx_angular_velocity, void *offset, int count,
                               cudaStream_t stream) {
  body_data_physx_to_sapien_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                                     stream>>>(
      (SapienBodyData *)sapien_data, (PhysxPose *)physx_pose, (Vec3 *)physx_linear_velocity,
      (Vec3 *)physx_angular_velocity, (Vec3 *)offset, count);
}

void body_data_sapien_to_physx(void *physx_pose, void *physx_linear_velocity,
                               void *physx_angular_velocity, void *sapien_data, void *offset,
                               int count, cudaStream_t stream) {
  body_data_sapien_to_physx_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                                     stream>>>(
      (PhysxPose *)physx_pose, (Vec3 *)physx_linear_velocity, (Vec3 *)physx_angular_velocity,
      (SapienBodyData *)sapien_data, (Vec3 *)offset, count);
}

void body_data_sapien_to_physx(void *physx_pose, void *physx_linear_velocity,
                               void *physx_angular_velocity, void *physx_index,
                               void *sapien_data, void *sapien_index, void *apply_index,
                               void *offset, int count, cudaStream_t stream) {
  body_data_sapien_to_physx_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                                     stream>>>(
      (PhysxPose *)physx_pose, (Vec3 *)physx_linear_velocity, (Vec3 *)physx_angular_velocity,
      (uint32_t *)physx_index, (SapienBodyData *)sapien_data, (uint32_t *)sapien_index,
      (int *)apply_index, (Vec3 *)offset, count);
}

void link_pose_physx_to_sapien(void *sapien_data, void *physx_pose, void *offset, int link_count,
                               int count, cudaStream_t stream) {
  link_pose_physx_to_sapien_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                                     stream>>>(
      (SapienBodyData *)sapien_data, (PhysxPose *)physx_pose, (Vec3 *)offset, link_count, count);
}

void root_pose_sapien_to_physx(void *physx_pose, void *sapien_data, void *index, void *offset,
                               int link_count, int count, cudaStream_t stream) {
  root_pose_sapien_to_physx_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                                     stream>>>(
      (PhysxPose *)physx_pose, (SapienBodyData *)sapien_data, (int *)index, (Vec3 *)offset,
      link_count, count);
}

void link_vel_physx_to_sapien(void *sapien_data, void *physx_linear_velocity,
                              void *physx_angular_velocity, int count, cudaStream_t stream) {
  link_vel_physx_to_sapien_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                                    stream>>>(
      (SapienBodyData *)sapien_data, (Vec3 *)physx_linear_velocity,
      (Vec3 *)physx_angular_velocity, count);
}

void root_vel_sapien_to_physx(void *physx_linear_velocity, void *physx_angular_velocity,
                              void *sapien_data, void *index, int link_count, int count,
                              cudaStream_t stream) {
  root_vel_sapien_to_physx_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                                    stream>>>(
      (Vec3 *)physx_linear_velocity, (Vec3 *)physx_angular_velocity,
      (SapienBodyData *)sapien_data, (int *)index, link_count, count);
}

void gather_blocks(void *dst, void *src, void *index, int block_size, int count,
                   cudaStream_t stream) {
  int total = count * block_size;
  gather_blocks_kernel<<<(total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
      (uint32_t *)dst, (uint32_t *)src, (int *)index, block_size, count);
}

void pack_vec3(void *dst, void *src, int stride, int count, cudaStream_t stream) {
  pack_vec3_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
      (Vec3 *)dst, (float *)src, stride, count);
}

void scatter_articulation_jacobians(void *dst, void *src, void *index, void *shape,
                                    int max_rows, int max_cols, int count,
                                    cudaStream_t stream) {
  int total = count * max_rows * max_cols;
  scatter_articulation_jacobians_kernel<<<(total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                                          stream>>>((float *)dst, (float *)src, (int *)index,
                                                    (uint32_t *)shape, max_rows, max_cols,
                                                    count);
}

void handle_contacts(::physx::PxGpuContactPair *contacts, int contact_count, ActorPairQuery *query,
                     int query_count, Vec3 *out_forces, cudaStream_t stream) {
  handle_contacts_kernel<<<(contact_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
      contacts, contact_count, query, query_count, out_forces);
}

void handle_net_contact_force(::physx::PxGpuContactPair *contacts, int contact_count,
                              ActorQuery *query, int query_count, Vec3 *out_forces,
                              cudaStream_t stream) {
  handle_net_contact_force_kernel<<<(contact_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                                    stream>>>(contacts, contact_count, query, query_count,
                                              out_forces);
}

} // namespace physx
} // namespace sapien
