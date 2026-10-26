#include "sapien/sapien_renderer/deformable_mesh_component.h"
#include "sapien/entity.h"
#include "sapien/sapien_renderer/material.h"
#include "sapien/scene.h"

#ifdef SAPIEN_CUDA
#include "sapien/utils/cuda.h"
#include <cuda_runtime.h>
#endif

namespace sapien {
namespace sapien_renderer {

CudaDeformableMeshComponent::CudaDeformableMeshComponent(uint32_t maxVertexCount,
                                                         uint32_t maxTriangleCount,
                                                         std::shared_ptr<Device> device) {
  mEngine = SapienRenderEngine::Get();
  mDevice = device ? device : mEngine->getDefaultDevice();
  mContext = mEngine->getContext(mDevice);
  mMesh = std::make_shared<svulkan2::resource::SVMeshDeformable>(maxVertexCount, maxTriangleCount);

#ifdef SAPIEN_CUDA
  vk::SemaphoreTypeCreateInfo timelineCreateInfo(vk::SemaphoreType::eTimeline, 0);
  vk::SemaphoreCreateInfo createInfo{};
  vk::ExportSemaphoreCreateInfo exportCreateInfo(
      vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd);
  createInfo.setPNext(&exportCreateInfo);
  exportCreateInfo.setPNext(&timelineCreateInfo);
  auto vulkanDevice = mContext->getDevice();
  mSem = vulkanDevice.createSemaphoreUnique(createInfo);

  int fd = vulkanDevice.getSemaphoreFdKHR(
      {mSem.get(), vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd});
  cudaExternalSemaphoreHandleDesc desc = {};
  desc.flags = 0;
  desc.handle.fd = fd;
  desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
  // import into the mesh device's context, not the caller's current one
  CudaDeviceScope scope(mDevice->cudaId);
  auto err = cudaImportExternalSemaphore(&mCudaSem, &desc);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to export semaphore to CUDA: " +
                             std::string(cudaGetErrorName(err)));
  }
#endif

  mMaterial = std::make_shared<SapienRenderMaterial>();
}

void CudaDeformableMeshComponent::setMaterial(std::shared_ptr<SapienRenderMaterial> material) {
  mMaterial = material;
}
std::shared_ptr<SapienRenderMaterial> CudaDeformableMeshComponent::getMaterial() const {
  return mMaterial;
}

uint32_t CudaDeformableMeshComponent::getVertexCount() const { return mMesh->getVertexCount(); }
void CudaDeformableMeshComponent::setVertexCount(uint32_t count) { mMesh->setVertexCount(count); }

uint32_t CudaDeformableMeshComponent::getTriangleCount() const {
  return mMesh->getTriangleCount();
}
void CudaDeformableMeshComponent::setTriangleCount(uint32_t count) {
  mMesh->setTriangleCount(count);
}

void CudaDeformableMeshComponent::setTriangles(
    Eigen::Matrix<uint32_t, Eigen::Dynamic, 3, Eigen::RowMajor> const &triangles) {
  setTriangleCount(triangles.rows());
  mMesh->getIndexBuffer(*mContext).upload(triangles.data(), triangles.size() * sizeof(uint32_t));
}

uint32_t CudaDeformableMeshComponent::getMaxVertexCount() const {
  return mMesh->getMaxVertexCount();
}

uint32_t CudaDeformableMeshComponent::getMaxTriangleCount() const {
  return mMesh->getMaxTriangleCount();
}

CudaArrayHandle CudaDeformableMeshComponent::getVertexBufferCudaArray() const {
#ifdef SAPIEN_CUDA
  int count = mMesh->getMaxVertexCount();
  int channels = mMesh->getVertexSize() / 4;
  int itemsize = 4;

  auto &buffer = mMesh->getVertexBuffer(*mContext);
  assert(static_cast<int>(buffer.getSize()) == count * channels * itemsize);

  return CudaArrayHandle{.shape = {count, channels},
                         .strides = {channels * itemsize, itemsize},
                         .type = "f4",
                         .cudaId = buffer.getCudaDeviceId(),
                         .ptr = buffer.getCudaPtr()};
#else
  throw std::runtime_error("sapien is not copmiled with CUDA support");
#endif
}

CudaArrayHandle CudaDeformableMeshComponent::getIndexBufferCudaArray() const {
#ifdef SAPIEN_CUDA
  int count = mMesh->getMaxTriangleCount();
  int channels = 3;
  int itemsize = 4;

  auto &buffer = mMesh->getIndexBuffer(*mContext);
  assert(static_cast<int>(buffer.getSize()) == count * channels * itemsize);

  return CudaArrayHandle{.shape = {count, channels},
                         .strides = {channels * itemsize, itemsize},
                         .type = "u4",
                         .cudaId = buffer.getCudaDeviceId(),
                         .ptr = buffer.getCudaPtr()};
#else
  throw std::runtime_error("sapien is not copmiled with CUDA support");
#endif
}

void CudaDeformableMeshComponent::onAddToScene(Scene &scene) {
  auto system = scene.getSapienRendererSystem();
  auto s = system->getScene();

  // The mesh buffers and the semaphore are bound to mDevice; the scene renders on another
  // device would sample buffers it cannot see.
  if (system->getContext() != mContext) {
    throw std::runtime_error(
        "failed to add CudaDeformableMeshComponent to scene: the component is bound to device \"" +
        mDevice->getAlias() + "\" but the scene renders on device \"" +
        system->getDevice()->getAlias() + "\"");
  }

  mObject = &s->addDeformableObject(svulkan2::resource::SVModel::FromData(
      {svulkan2::resource::SVShape::Create(mMesh, mMaterial->getMaterial())}));
  mObject->setCustomDataInt("shadeFlat", true);

  system->registerComponent(
      std::static_pointer_cast<CudaDeformableMeshComponent>(shared_from_this()));
}

void CudaDeformableMeshComponent::onRemoveFromScene(Scene &scene) {
  auto system = scene.getSapienRendererSystem();
  auto s = system->getScene();

  s->removeNode(*mObject);

  system->unregisterComponent(
      std::static_pointer_cast<CudaDeformableMeshComponent>(shared_from_this()));
}

void CudaDeformableMeshComponent::notifyVertexUpdated(uintptr_t stream) {
#ifdef SAPIEN_CUDA
  cudaExternalSemaphoreSignalParams sigParams{};
  sigParams.flags = 0;
  sigParams.params.fence.value = ++mSemValue;
  {
    CudaDeviceScope scope(mDevice->cudaId);
    cudaSignalExternalSemaphoresAsync(&mCudaSem, &sigParams, 1,
                                      reinterpret_cast<cudaStream_t>(stream));
  }
  vk::PipelineStageFlags flags = vk::PipelineStageFlagBits::eVertexInput;
  mContext->getQueue().submit({}, mSem.get(), flags, mSemValue, {}, {}, {});
#endif
}

void CudaDeformableMeshComponent::internalUpdate() {}

} // namespace sapien_renderer
} // namespace sapien
