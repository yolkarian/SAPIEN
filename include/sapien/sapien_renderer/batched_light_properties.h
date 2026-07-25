#pragma once
#include <Eigen/Eigen>
#include <memory>
#include <vector>

namespace sapien {
namespace sapien_renderer {
class SapienRenderLightComponent;

/** Batched CPU light setters, primarily for per-environment lighting randomization.
 *
 *  Conventions follow the sapien.physx batched property setters: one call takes a flat
 *  component list, every entry is validated before anything is mutated, and a failed
 *  call never leaves the batch partially applied.
 *
 *  Poses and directions write the light component's local pose (the world pose when the
 *  owning entity pose is identity). Lights sealed by a RenderSystemGroup must use pose
 *  mode 'cpu' for pose/direction writes; colors are CPU real-time in every mode. */

/** Set the local pose of each light. Each row of poses is [x, y, z, qw, qx, qy, qz];
 *  quaternions are normalized before being applied. */
void batchSetLightPoses(std::vector<std::shared_ptr<SapienRenderLightComponent>> const &lights,
                        Eigen::Matrix<float, Eigen::Dynamic, 7, Eigen::RowMajor> const &poses);

/** Point each directional/spot/textured light along a direction (SAPIEN lights shine
 *  along +x of their pose). Each row of directions is a non-zero [x, y, z]; the light's
 *  local position is kept. Point and parallelogram lights are rejected. */
void batchSetLightDirections(
    std::vector<std::shared_ptr<SapienRenderLightComponent>> const &lights,
    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> const &directions);

/** Set the color of each light. Each row of colors is a finite, non-negative
 *  [r, g, b]; values above 1 are valid HDR intensities. */
void batchSetLightColors(std::vector<std::shared_ptr<SapienRenderLightComponent>> const &lights,
                         Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> const &colors);

} // namespace sapien_renderer
} // namespace sapien
