#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace sapien {

class Scene;

/** System controls entity components in a scene.
 *  Components should register themselves to a system to enable lifecycle methods */
class System {
public:
  virtual ~System();
  virtual void step() = 0;
  virtual std::string getName() const = 0;

  /** Called when a scene that still references this system is closed. The default
   *  implementation detaches the system from the scene handle set. Subsystems with
   *  per-scene state (e.g. PhysX GPU offsets and environment IDs) should override and
   *  release it here. */
  virtual void onSceneClosed(Scene &scene);

protected:
  /** Back-pointer used by Scene::close() to detach systems. Not a public registration
   *  API; Scene::addSystem/removeSystem drive it. */
  virtual void internalAddScene(Scene &scene);
  virtual void internalRemoveScene(Scene &scene);
  friend class Scene;
  std::vector<Scene *> mScenes;
};

} // namespace sapien
