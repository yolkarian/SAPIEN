#include "sapien/sapien_renderer/sapien_renderer_system.h"
#include "../logger.h"
#include "sapien/sapien_renderer/camera_component.h"
#include "sapien/sapien_renderer/cubemap.h"
#include "sapien/sapien_renderer/deformable_mesh_component.h"
#include "sapien/sapien_renderer/light_component.h"
#include "sapien/sapien_renderer/point_cloud_component.h"
#include "sapien/sapien_renderer/render_body_component.h"
#include "sapien/sapien_renderer/sapien_renderer_default.h"
#include <algorithm>
#include <svulkan2/core/context.h>
#include <svulkan2/core/physical_device.h>
#include <svulkan2/renderer/renderer.h>
#include <svulkan2/renderer/rt_renderer.h>
#include <svulkan2/scene/scene.h>

#ifdef SAPIEN_CUDA
#include "sapien/utils/cuda.h"
#include <cuda_runtime.h>
#endif

namespace sapien {
namespace sapien_renderer {

static std::weak_ptr<SapienRenderEngine> gRenderEngine;

std::shared_ptr<SapienRenderEngine> SapienRenderEngine::Get(std::shared_ptr<Device> device) {
  auto engine = gRenderEngine.lock();
  if (engine && engine->isShutdown()) {
    engine.reset();
    gRenderEngine.reset();
  }
  if (engine) {
    if (device && engine->mDevice != device) {
      throw std::runtime_error("failed to create renderer on device \"" + device->getAlias() +
                               "\": current SAPIEN version only supports single-GPU rendering.");
    }
    return engine;
  }
  gRenderEngine = engine = std::make_shared<SapienRenderEngine>(device);
  return engine;
}

std::shared_ptr<SapienRenderEngine> SapienRenderEngine::GetIfExists() {
  return gRenderEngine.lock();
}

std::shared_ptr<svulkan2::core::Context> SapienRenderEngine::getContext() const {
  if (mShutdown) {
    throw std::runtime_error("failed to use render engine: the engine was shut down");
  }
  return mContext;
}

std::shared_ptr<svulkan2::resource::SVResourceManager>
SapienRenderEngine::getResourceManager() const {
  if (mShutdown) {
    throw std::runtime_error("failed to use render engine: the engine was shut down");
  }
  return mResourceManager;
}

std::string SapienRenderEngine::getSummary() {
  try {
    std::shared_ptr<svulkan2::core::Context> context;
    try {
      context = svulkan2::core::Context::Get();
    } catch (std::runtime_error &) {
      context = svulkan2::core::Context::Create();
    }

    auto info = context->getInstance2()->summarizePhysicalDevices();
    std::stringstream ss;

    for (auto const &entry : info) {
      ss << "GPU: " << entry.name << "\n";
      ss << "  Supported: " << entry.supported << "\n";
      ss << "  Present:   " << entry.present << "\n";
      ss << "  cudaId:    " << entry.cudaId << "\n";
      ss << "  rayTrace:  " << entry.rayTracing << "\n";
      ss << "  cudaMode   " << entry.cudaComputeMode << "\n";
    }
    return ss.str();
  } catch (std::runtime_error &) {
    return "Failed to initialize Vulkan";
  }
}

SapienRenderEngine::SapienRenderEngine(std::shared_ptr<Device> device) {
  if (!device) {
    device = findBestRenderDevice();
  }
  if (!device) {
    throw std::runtime_error("failed to find a rendering device");
  }

  mDevice = device;
  auto &d = SapienRendererDefault::Get();
  mContext = svulkan2::core::Context::Create(d.getMaxNumMaterials(), d.getMaxNumTextures(),
                                             d.getDefaultMipMaps(), d.getDoNotLoadTexture(),
                                             device->getAlias(), d.getVREnabled());
  mResourceManager = mContext->createResourceManager();
}

std::shared_ptr<svulkan2::resource::SVMesh> SapienRenderEngine::getSphereMesh() {
  if (!mSphereMesh) {
    mSphereMesh = svulkan2::resource::SVMesh::CreateUVSphere(32, 16);
  }
  return mSphereMesh;
}

std::shared_ptr<svulkan2::resource::SVMesh> SapienRenderEngine::getPlaneMesh() {
  if (!mPlaneMesh) {
    mPlaneMesh = svulkan2::resource::SVMesh::CreateYZPlane();
  }
  return mPlaneMesh;
}

std::shared_ptr<svulkan2::resource::SVMesh> SapienRenderEngine::getBoxMesh() {
  if (!mBoxMesh) {
    mBoxMesh = svulkan2::resource::SVMesh::CreateCube();
  }
  return mBoxMesh;
}

void SapienRenderEngine::shutdown() {
  if (mShutdown) {
    return;
  }
  if (!getRenderSystems().empty()) {
    throw std::runtime_error(
        "failed to shut down render engine: renderer systems are still alive");
  }
  if (mContext) {
    mContext->getDevice().waitIdle();
  }
  mSphereMesh.reset();
  mPlaneMesh.reset();
  mBoxMesh.reset();
  mRenderSystems.clear();
  mResourceManager.reset();
  mContext.reset();
  mDevice.reset();
  mShutdown = true;
}

SapienRenderEngine::~SapienRenderEngine() {
  if (mShutdown) {
    return;
  }
  try {
    shutdown();
  } catch (...) {
    // The remaining objects still hold their own context/resource references. Avoid
    // throwing from a destructor; normal job boundaries call render.shutdown() first.
  }
}

SapienRendererSystem::SapienRendererSystem(std::shared_ptr<Device> device) {
  mEngine = SapienRenderEngine::Get(device);
  mScene = std::make_shared<svulkan2::scene::Scene>();
  mScene->setAmbientLight({0.12f, 0.12f, 0.12f, 1.f});
}

void SapienRendererSystem::setBatchedRenderShared(bool shared) {
  checkNotClosed();
  mScene->setBatchedRenderShared(shared);
}

bool SapienRendererSystem::isBatchedRenderShared() const {
  checkNotClosed();
  return mScene->isBatchedRenderShared();
}

void SapienRenderEngine::registerRenderSystem(
    std::shared_ptr<SapienRendererSystem> const &system) {
  std::erase_if(mRenderSystems, [](auto const &candidate) { return candidate.expired(); });
  for (auto const &candidate : mRenderSystems) {
    if (candidate.lock() == system) {
      return;
    }
  }
  mRenderSystems.push_back(system);
}

std::vector<std::shared_ptr<SapienRendererSystem>> SapienRenderEngine::getRenderSystems() {
  std::vector<std::shared_ptr<SapienRendererSystem>> systems;
  std::erase_if(mRenderSystems, [](auto const &candidate) { return candidate.expired(); });
  systems.reserve(mRenderSystems.size());
  for (auto const &candidate : mRenderSystems) {
    if (auto system = candidate.lock()) {
      systems.push_back(system);
    }
  }
  return systems;
}

std::shared_ptr<svulkan2::scene::Scene> SapienRendererSystem::getScene() {
  checkNotClosed();
  return mScene;
}

std::shared_ptr<Device> SapienRendererSystem::getDevice() const {
  checkNotClosed();
  return mEngine->getDevice();
}

Vec3 SapienRendererSystem::getAmbientLight() const {
  checkNotClosed();
  auto l = mScene->getAmbientLight();
  return {l.r, l.g, l.b};
}

void SapienRendererSystem::setAmbientLight(Vec3 l) {
  checkNotClosed();
  // Alpha is an internal flag indicating whether the raster shader should use its fallback IBL.
  auto ambient = mScene->getAmbientLight();
  mScene->setAmbientLight({l.x, l.y, l.z, ambient.a});
  internalNotifyLightStateChanged();
}

void SapienRendererSystem::setCubemap(std::shared_ptr<SapienRenderCubemap> cubemap) {
  checkNotClosed();
  mScene->setEnvironmentMap(cubemap ? cubemap->getCubemap() : nullptr);
  auto ambient = mScene->getAmbientLight();
  ambient.a = cubemap ? 0.f : 1.f;
  mScene->setAmbientLight(ambient);
  mCubemap = cubemap;
}
std::shared_ptr<SapienRenderCubemap> SapienRendererSystem::getCubemap() const {
  checkNotClosed();
  return mCubemap;
}

void SapienRendererSystem::registerComponent(std::shared_ptr<SapienRenderBodyComponent> c) {
  checkNotClosed();
  mRenderBodyComponents.insert(c);
}

void SapienRendererSystem::registerComponent(std::shared_ptr<SapienRenderCameraComponent> c) {
  checkNotClosed();
  mRenderCameraComponents.insert(c);
}

void SapienRendererSystem::registerComponent(std::shared_ptr<SapienRenderLightComponent> c) {
  checkNotClosed();
  mRenderLightComponents.insert(c);
}

void SapienRendererSystem::registerComponent(std::shared_ptr<PointCloudComponent> c) {
  checkNotClosed();
  mPointCloudComponents.insert(c);
}

void SapienRendererSystem::registerComponent(std::shared_ptr<CudaDeformableMeshComponent> c) {
  checkNotClosed();
  mCudaDeformableMeshComponents.insert(c);
}

void SapienRendererSystem::unregisterComponent(std::shared_ptr<SapienRenderBodyComponent> c) {
  mRenderBodyComponents.erase(c);
}

void SapienRendererSystem::unregisterComponent(std::shared_ptr<SapienRenderCameraComponent> c) {
  mRenderCameraComponents.erase(c);
}

void SapienRendererSystem::unregisterComponent(std::shared_ptr<SapienRenderLightComponent> c) {
  mRenderLightComponents.erase(c);
}

void SapienRendererSystem::unregisterComponent(std::shared_ptr<PointCloudComponent> c) {
  mPointCloudComponents.erase(c);
}

void SapienRendererSystem::unregisterComponent(std::shared_ptr<CudaDeformableMeshComponent> c) {
  mCudaDeformableMeshComponents.erase(c);
}

void SapienRendererSystem::step() {
  checkNotClosed();
  for (auto c : mRenderBodyComponents) {
    c->internalUpdate();
  }
  for (auto c : mRenderCameraComponents) {
    c->internalUpdate();
  }
  for (auto c : mRenderLightComponents) {
    c->internalUpdate();
  }
  for (auto c : mPointCloudComponents) {
    c->internalUpdate();
  }
  for (auto c : mCudaDeformableMeshComponents) {
    c->internalUpdate();
  }
  mScene->updateModelMatrices();
}

CudaArrayHandle SapienRendererSystem::getTransformCudaArray() {
  checkNotClosed();
  mScene->prepareObjectTransformBuffer();
  int offset = mScene->getGpuTransformBufferSize();

  auto buffer = mScene->getObjectTransformBuffer();
#ifdef SAPIEN_CUDA
  return CudaArrayHandle{.shape = {static_cast<int>(buffer->getSize() / offset), 4, 4},
                         .strides = {offset, 16, 4},
                         .type = "f4",
                         .cudaId = buffer->getCudaDeviceId(),
                         .ptr = buffer->getCudaPtr()};
#else
  return CudaArrayHandle{.shape = {static_cast<int>(buffer->getSize() / offset), 4, 4},
                         .strides = {offset, 16, 4},
                         .type = "f4"};
#endif
}

void SapienRendererSystem::internalAddScene(Scene &scene) {
  checkNotClosed();
  System::internalAddScene(scene);
}

void SapienRendererSystem::checkNotClosed() const {
  if (mClosed) {
    throw std::runtime_error("failed to use RenderSystem: the system is closed");
  }
}

void SapienRendererSystem::close() {
  if (mClosed) {
    return;
  }
  if (!mScenes.empty()) {
    throw std::runtime_error("failed to close RenderSystem: " +
                             std::to_string(mScenes.size()) +
                             " scenes are still attached; close every scene first");
  }
  size_t components = mRenderBodyComponents.size() + mRenderCameraComponents.size() +
                      mRenderLightComponents.size() + mPointCloudComponents.size() +
                      mCudaDeformableMeshComponents.size();
  if (components != 0) {
    throw std::runtime_error("failed to close RenderSystem: " +
                             std::to_string(components) +
                             " components remain registered");
  }
  mEngine->getContext()->getDevice().waitIdle();
  mCubemap.reset();
  mScene.reset();
  mEngine.reset();
  mClosed = true;
}

void SapienRendererSystem::closeNoThrow() {
  try {
    close();
  } catch (std::exception const &e) {
    logger::error("failed to close RenderSystem during destruction: {}", e.what());
    mCubemap.reset();
    mScene.reset();
    mEngine.reset();
    mClosed = true;
  }
}

SapienRendererSystem::~SapienRendererSystem() { closeNoThrow(); }

} // namespace sapien_renderer
} // namespace sapien
