#include "sapien/sapien_renderer/batched_light_properties.h"
#include "sapien/math/conversion.h"
#include "sapien/sapien_renderer/light_component.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace sapien {
namespace sapien_renderer {

// All functions validate every entry before mutating anything so that a failed call
// never leaves the batch partially applied.

static void checkSize(char const *what, size_t expected, Eigen::Index actual) {
  if (static_cast<Eigen::Index>(expected) != actual) {
    throw std::runtime_error(std::string("batched light setter failed: ") + what +
                             " must have one row per light (expected " +
                             std::to_string(expected) + ", got " + std::to_string(actual) + ")");
  }
}

static void checkLight(std::shared_ptr<SapienRenderLightComponent> const &light, size_t i) {
  if (!light) {
    throw std::runtime_error("batched light setter failed: light " + std::to_string(i) +
                             " is null");
  }
}

static void checkPoseMutable(std::shared_ptr<SapienRenderLightComponent> const &light,
                             size_t i) {
  if (light->isGroupStateSealed() && light->getPoseMode() == LightPoseMode::eStatic) {
    throw std::runtime_error(
        "batched light setter failed: light " + std::to_string(i) +
        " uses pose mode 'static' (default) and is sealed by a render system group; set pose "
        "mode 'cpu' before RenderSystemGroup.gpu_init() to move it");
  }
}

void batchSetLightPoses(std::vector<std::shared_ptr<SapienRenderLightComponent>> const &lights,
                        Eigen::Matrix<float, Eigen::Dynamic, 7, Eigen::RowMajor> const &poses) {
  checkSize("poses", lights.size(), poses.rows());

  std::vector<Pose> validated;
  validated.reserve(lights.size());
  for (size_t i = 0; i < lights.size(); ++i) {
    checkLight(lights[i], i);
    checkPoseMutable(lights[i], i);
    auto row = poses.row(static_cast<Eigen::Index>(i));
    for (int c = 0; c < 7; ++c) {
      if (!std::isfinite(row(c))) {
        throw std::runtime_error("batched light setter failed: pose " + std::to_string(i) +
                                 " contains a non-finite value");
      }
    }
    Quat q(row(3), row(4), row(5), row(6));
    float norm = q.length();
    if (!(norm > 1e-6f)) {
      throw std::runtime_error("batched light setter failed: pose " + std::to_string(i) +
                               " has a near-zero quaternion");
    }
    validated.emplace_back(Vec3(row(0), row(1), row(2)),
                           Quat(row(3) / norm, row(4) / norm, row(5) / norm, row(6) / norm));
  }

  for (size_t i = 0; i < lights.size(); ++i) {
    lights[i]->setLocalPose(validated[i]);
  }
}

void batchSetLightDirections(
    std::vector<std::shared_ptr<SapienRenderLightComponent>> const &lights,
    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> const &directions) {
  checkSize("directions", lights.size(), directions.rows());

  std::vector<Pose> validated;
  validated.reserve(lights.size());
  for (size_t i = 0; i < lights.size(); ++i) {
    checkLight(lights[i], i);
    checkPoseMutable(lights[i], i);
    if (!std::dynamic_pointer_cast<SapienRenderDirectionalLightComponent>(lights[i])) {
      throw std::runtime_error("batched light setter failed: light " + std::to_string(i) +
                               " is not a directional light; only directional lights are "
                               "supported");
    }
    auto row = directions.row(static_cast<Eigen::Index>(i));
    Vec3 direction(row(0), row(1), row(2));
    if (!std::isfinite(direction.x) || !std::isfinite(direction.y) ||
        !std::isfinite(direction.z) || !(direction.length() > 1e-6f)) {
      throw std::runtime_error("batched light setter failed: direction " + std::to_string(i) +
                               " must be a finite non-zero vector");
    }
    // SAPIEN lights shine along +x of their pose; keep the current local position.
    validated.emplace_back(lights[i]->getLocalPose().p, ShortestRotation(Vec3(1, 0, 0), direction));
  }

  for (size_t i = 0; i < lights.size(); ++i) {
    lights[i]->setLocalPose(validated[i]);
  }
}

void batchSetLightColors(std::vector<std::shared_ptr<SapienRenderLightComponent>> const &lights,
                         Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> const &colors) {
  checkSize("colors", lights.size(), colors.rows());

  for (size_t i = 0; i < lights.size(); ++i) {
    checkLight(lights[i], i);
    auto row = colors.row(static_cast<Eigen::Index>(i));
    for (int c = 0; c < 3; ++c) {
      if (!std::isfinite(row(c)) || row(c) < 0.f) {
        throw std::runtime_error("batched light setter failed: color " + std::to_string(i) +
                                 " must be finite and non-negative");
      }
    }
  }

  for (size_t i = 0; i < lights.size(); ++i) {
    auto row = colors.row(static_cast<Eigen::Index>(i));
    lights[i]->setColor(Vec3(row(0), row(1), row(2)));
  }
}

} // namespace sapien_renderer
} // namespace sapien
