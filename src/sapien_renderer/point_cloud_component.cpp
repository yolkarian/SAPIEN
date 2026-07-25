#include "sapien/sapien_renderer/point_cloud_component.h"
#include "sapien/entity.h"
#include "sapien/scene.h"

namespace sapien {
namespace sapien_renderer {

PointCloudComponent::PointCloudComponent(uint32_t capacity) {
  mEngine = SapienRenderEngine::Get();
  mPointSet = std::make_shared<svulkan2::resource::SVPointSet>(capacity);
}

std::shared_ptr<PointCloudComponent> PointCloudComponent::setVertices(
    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> const &vertices) {
  setAttribute("position", vertices);
  return std::static_pointer_cast<PointCloudComponent>(shared_from_this());
}

Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> PointCloudComponent::getVertices() {
  auto pos = mPointSet->getVertexAttribute("position");
  return Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>>(pos.data(),
                                                                              pos.size() / 3, 3);
}

std::shared_ptr<PointCloudComponent> PointCloudComponent::setAttribute(
    std::string const &name,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const &attribute) {
  checkSnapshotMutable("set point-cloud attribute");
  std::vector<float> data(attribute.data(), attribute.data() + attribute.size());
  mPointSet->setVertexAttribute(name, data);
  return std::static_pointer_cast<PointCloudComponent>(shared_from_this());
}

void PointCloudComponent::onAddToScene(Scene &scene) {
  auto system = scene.getSapienRendererSystem();
  auto s = system->getScene();
  mObject = &s->addPointObject(mPointSet, getTransform());
  system->registerComponent(std::static_pointer_cast<PointCloudComponent>(shared_from_this()));
}

void PointCloudComponent::onRemoveFromScene(Scene &scene) {
  auto system = scene.getSapienRendererSystem();
  auto s = system->getScene();
  s->removeNode(*mObject);
  mObject = nullptr;
  system->unregisterComponent(std::static_pointer_cast<PointCloudComponent>(shared_from_this()));
}

// called by system to sync pose
void PointCloudComponent::internalUpdate() {
  checkStaticPose();
  auto pose = getEntity()->getPose();
  mObject->setTransform({.position = {pose.p.x, pose.p.y, pose.p.z},
                         .rotation = {pose.q.w, pose.q.x, pose.q.y, pose.q.z},
                         .scale = mObject->getScale()});
}

void PointCloudComponent::checkSnapshotMutable(char const *operation) const {
  if (mStaticPoseSealCount > 0) {
    throw std::runtime_error(std::string("failed to ") + operation +
                             ": the point cloud is a static snapshot sealed by a render system "
                             "group");
  }
}

void PointCloudComponent::checkStaticPose() const {
  if (mStaticPoseSealCount == 0) {
    return;
  }
  Pose pose = getEntity()->getPose();
  if (pose.p.x != mSealedPose.p.x || pose.p.y != mSealedPose.p.y ||
      pose.p.z != mSealedPose.p.z || pose.q.w != mSealedPose.q.w ||
      pose.q.x != mSealedPose.q.x || pose.q.y != mSealedPose.q.y ||
      pose.q.z != mSealedPose.q.z) {
    throw std::runtime_error(
        "failed to update point cloud: its pose is a static snapshot sealed by a render system "
        "group; destroy the group before moving it");
  }
}

void PointCloudComponent::internalSealStaticPose() {
  if (mStaticPoseSealCount++ == 0) {
    mSealedPose = getEntity()->getPose();
  }
}

void PointCloudComponent::internalReleaseStaticPoseSeal() {
  if (mStaticPoseSealCount == 0) {
    throw std::runtime_error("point-cloud static pose seal count is already zero");
  }
  --mStaticPoseSealCount;
}

CudaArrayHandle PointCloudComponent::getCudaArray() const {
#ifdef SAPIEN_CUDA
  int count = mPointSet->getVertexCount();
  int channels = mPointSet->getVertexSize() / 4;
  int itemsize = 4;

  return CudaArrayHandle{.shape = {count, channels},
                         .strides = {channels * itemsize, itemsize},
                         .type = "f4",
                         .cudaId = mPointSet->getVertexBuffer().getCudaDeviceId(),
                         .ptr = mPointSet->getVertexBuffer().getCudaPtr()};
#else
  throw std::runtime_error("sapien is not copmiled with CUDA support");
#endif
}

std::optional<CudaArrayHandle> PointCloudComponent::getCudaAABBArray() const {
#ifdef SAPIEN_CUDA
  auto buffer = mPointSet->getAabbBuffer();
  if (!buffer) {
    return {};
  }

  int count = mPointSet->getVertexCount();

  return CudaArrayHandle{.shape = {count, 2, 3},
                         .strides = {6 * sizeof(float), 3 * sizeof(float), sizeof(float)},
                         .type = "f4",
                         .cudaId = buffer->getCudaDeviceId(),
                         .ptr = buffer->getCudaPtr()};
#else
  throw std::runtime_error("sapien is not copmiled with CUDA support");
#endif
}

svulkan2::scene::Transform PointCloudComponent::getTransform() const {
  auto pose = getPose();
  return {
      .position = {pose.p.x, pose.p.y, pose.p.z},
      .rotation = {pose.q.w, pose.q.x, pose.q.y, pose.q.z},
      .scale = {1.f, 1.f, 1.f},
  };
}

} // namespace sapien_renderer
} // namespace sapien
