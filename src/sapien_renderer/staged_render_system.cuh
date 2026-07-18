#pragma once

struct CUstream_st;

namespace sapien::sapien_renderer {

void pack_staged_poses(float *output, float const *poses, int poseStride,
                       int const *sourceIndices, int count, CUstream_st *stream);

} // namespace sapien::sapien_renderer
