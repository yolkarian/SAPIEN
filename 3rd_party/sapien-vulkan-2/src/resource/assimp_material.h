#pragma once
#include "svulkan2/resource/material.h"
#include <assimp/scene.h>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace svulkan2 {
namespace resource {

/** Converts the materials of an assimp scene, one per `scene->mMaterials` entry and in that
 *  order. `filename` is the file the scene was read from: texture paths resolve against its
 *  directory. Textures read from files load asynchronously; their futures are appended to
 *  `futures` and must be waited for before the materials are used. `honorTwoSided` disables
 *  back-face culling for materials the file marks as two-sided. */
std::vector<std::shared_ptr<SVMaterial>>
loadAssimpMaterials(aiScene const *scene, std::string const &filename, bool honorTwoSided,
                    std::vector<std::future<void>> &futures);

} // namespace resource
} // namespace svulkan2
