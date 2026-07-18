#include "staged_render_system.cuh"
#include <cuda_runtime.h>

namespace sapien::sapien_renderer {
namespace {

__global__ void pack_staged_poses_kernel(float *output, float const *poses, int poseStride,
                                         int const *sourceIndices, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  int source = sourceIndices[index] * poseStride;
  int destination = index * 7;
  for (int channel = 0; channel < 7; ++channel) {
    output[destination + channel] = poses[source + channel];
  }
}

} // namespace

void pack_staged_poses(float *output, float const *poses, int poseStride,
                       int const *sourceIndices, int count, CUstream_st *stream) {
  constexpr int blockSize = 128;
  pack_staged_poses_kernel<<<(count + blockSize - 1) / blockSize, blockSize, 0,
                             reinterpret_cast<cudaStream_t>(stream)>>>(
      output, poses, poseStride, sourceIndices, count);
}

} // namespace sapien::sapien_renderer
