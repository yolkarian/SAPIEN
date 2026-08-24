#include "sapien/system.h"
#include "sapien/scene.h"

#include <algorithm>

namespace sapien {

System::~System() {}

void System::onSceneClosed(Scene &scene) { internalRemoveScene(scene); }

void System::internalAddScene(Scene &scene) {
  if (std::find(mScenes.begin(), mScenes.end(), &scene) == mScenes.end()) {
    mScenes.push_back(&scene);
  }
}

void System::internalRemoveScene(Scene &scene) { std::erase(mScenes, &scene); }

} // namespace sapien
