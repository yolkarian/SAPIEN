#pragma once
#include "svulkan2/scene/scene.h"
#include <memory>

namespace svulkan2 {
namespace core {
class Context;
}
namespace scene {

/** load a scene file into a scene of `context` (its device objects are built on demand) */
std::shared_ptr<Scene> LoadScene(std::shared_ptr<core::Context> context, std::string const &path);

} // namespace scene
} // namespace svulkan2
