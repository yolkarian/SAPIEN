#include "./scene.h"

namespace svulkan2 {
namespace scene {

class SceneGroup : public Scene {
public:
  SceneGroup(std::vector<std::shared_ptr<Scene>> const &scenes,
             std::vector<Transform> const &transforms);
  std::vector<Object *> getObjects() override;
  std::vector<LineObject *> getLineObjects() override;
  std::vector<PointObject *> getPointObjects() override;
  std::vector<Camera *> getCameras() override;
  std::vector<PointLight *> getPointLights() override;
  std::vector<DirectionalLight *> getDirectionalLights() override;
  std::vector<SpotLight *> getSpotLights() override;
  std::vector<TexturedLight *> getTexturedLights() override;
  std::vector<ParallelogramLight *> getParallelogramLights() override;
  void uploadObjectTransforms() override;

private:
  std::vector<std::shared_ptr<Scene>> mScenes;
  std::vector<Transform> mTransforms;
};

} // namespace scene
} // namespace svulkan2
