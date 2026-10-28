#pragma once

#include "../component.h"
#include "../device.h"
#include "../system.h"
#include "cubemap.h"
#include "sapien/array.h"
#include "sapien/math/vec3.h"
#include <mutex>
#include <set>
#include <svulkan2/core/context.h>
#include <svulkan2/scene/scene.h>

typedef struct CUstream_st *cudaStream_t;
typedef struct CUexternalSemaphore_st *cudaExternalSemaphore_t;

namespace svulkan2::resource {
class SVResourceManager;
class SVMesh;
} // namespace svulkan2::resource

namespace sapien {
namespace sapien_renderer {
class SapienRendererSystem;
class SapienRenderBodyComponent;
class SapienRenderCameraComponent;
class SapienRenderLightComponent;
class PointCloudComponent;
class CudaDeformableMeshComponent;
class SapienRenderCubemap;

/** Process-wide render engine.
 *
 *  One engine serves every rendering device of the process. It owns the process-wide
 *  resource manager, meshes shared by all devices, the render-system registry and one
 *  svulkan2 context per device. Device-independent render objects (materials, textures,
 *  cubemaps, shapes) keep the engine alive and let each device upload them lazily. */
class SapienRenderEngine {
public:
  /** the process-wide engine, created on first use */
  static std::shared_ptr<SapienRenderEngine> Get();
  static std::shared_ptr<SapienRenderEngine> GetIfExists();

  /** the device used when a render object does not name one: the best render device */
  std::shared_ptr<Device> getDefaultDevice();

  /** the svulkan2 context of `device`, created on first request; null means the default
   *  device. Every live context is kept until shutdown(). */
  std::shared_ptr<svulkan2::core::Context> getContext(std::shared_ptr<Device> device = nullptr);

  std::shared_ptr<svulkan2::resource::SVResourceManager> getResourceManager() const;

  std::shared_ptr<svulkan2::resource::SVMesh> getSphereMesh();
  std::shared_ptr<svulkan2::resource::SVMesh> getPlaneMesh();
  std::shared_ptr<svulkan2::resource::SVMesh> getBoxMesh();

  std::string getSummary();

  void registerRenderSystem(std::shared_ptr<SapienRendererSystem> const &system);
  void unregisterRenderSystem(SapienRendererSystem const *system);
  /** every live render system, on every device */
  std::vector<std::shared_ptr<SapienRendererSystem>> getRenderSystems();
  /** live render systems of one device; a rendering group or window only sees these */
  std::vector<std::shared_ptr<SapienRendererSystem>>
  getRenderSystems(std::shared_ptr<svulkan2::core::Context> const &context);

  void shutdown();
  bool isShutdown() const { return mShutdown; }

  ~SapienRenderEngine();

private:
  SapienRenderEngine();

  std::shared_ptr<Device> mDefaultDevice;

  std::mutex mContextLock;
  std::vector<std::shared_ptr<svulkan2::core::Context>> mContexts;

  // declared after the contexts: device objects it caches die before their devices
  std::shared_ptr<svulkan2::resource::SVResourceManager> mResourceManager;

  std::shared_ptr<svulkan2::resource::SVMesh> mSphereMesh;
  std::shared_ptr<svulkan2::resource::SVMesh> mPlaneMesh;
  std::shared_ptr<svulkan2::resource::SVMesh> mBoxMesh;
  std::vector<std::weak_ptr<SapienRendererSystem>> mRenderSystems;
  bool mShutdown{false};
};

class SapienRendererSystem : public System {
public:
  /** `device` null selects the engine's default device */
  SapienRendererSystem(std::shared_ptr<Device> device);
  std::shared_ptr<Device> getDevice() const;
  std::shared_ptr<svulkan2::core::Context> getContext() const;

  std::shared_ptr<svulkan2::scene::Scene> getScene();
  void setBatchedRenderShared(bool shared);
  bool isBatchedRenderShared() const;

  Vec3 getAmbientLight() const;
  void setAmbientLight(Vec3 l);
  void setCubemap(std::shared_ptr<SapienRenderCubemap> cubemap);
  std::shared_ptr<SapienRenderCubemap> getCubemap() const;

  /** coarse dirty version of CPU scene/light state (light properties, cpu-mode light
   *  poses, ambient light); consumed by RenderSystemGroup.update_render() */
  uint64_t getLightStateVersion() const { return mLightStateVersion; }
  void internalNotifyLightStateChanged() { ++mLightStateVersion; }

  void registerComponent(std::shared_ptr<SapienRenderBodyComponent> c);
  void registerComponent(std::shared_ptr<SapienRenderCameraComponent> c);
  void registerComponent(std::shared_ptr<SapienRenderLightComponent> c);
  void registerComponent(std::shared_ptr<PointCloudComponent> c);
  void registerComponent(std::shared_ptr<CudaDeformableMeshComponent> c);

  void unregisterComponent(std::shared_ptr<SapienRenderBodyComponent> c);
  void unregisterComponent(std::shared_ptr<SapienRenderCameraComponent> c);
  void unregisterComponent(std::shared_ptr<SapienRenderLightComponent> c);
  void unregisterComponent(std::shared_ptr<PointCloudComponent> c);
  void unregisterComponent(std::shared_ptr<CudaDeformableMeshComponent> c);

  std::vector<std::shared_ptr<SapienRenderBodyComponent>> getRenderBodyComponents() const {
    return std::vector<std::shared_ptr<SapienRenderBodyComponent>>{mRenderBodyComponents.begin(),
                                                                   mRenderBodyComponents.end()};
  }
  std::vector<std::shared_ptr<PointCloudComponent>> getPointCloudComponents() const {
    return std::vector<std::shared_ptr<PointCloudComponent>>{mPointCloudComponents.begin(),
                                                             mPointCloudComponents.end()};
  }
  std::vector<std::shared_ptr<SapienRenderCameraComponent>> getCameraComponents() const {
    return std::vector<std::shared_ptr<SapienRenderCameraComponent>>{
        mRenderCameraComponents.begin(), mRenderCameraComponents.end()};
  }
  std::vector<std::shared_ptr<SapienRenderLightComponent>> getLightComponents() const {
    return std::vector<std::shared_ptr<SapienRenderLightComponent>>{mRenderLightComponents.begin(),
                                                                    mRenderLightComponents.end()};
  }

  void step() override;
  std::string getName() const override { return "render"; }

  CudaArrayHandle getTransformCudaArray();
  CudaArrayHandle trackView(CudaArrayHandle handle) const;
  int64_t outstandingCudaViewCount() const { return mViewLifecycle->viewCount(); }

  void close();
  bool isClosed() const { return mClosed; }

  ~SapienRendererSystem();

  uint64_t nextRenderId() { return mNextRenderId++; };

private:
  void internalAddScene(Scene &scene) override;
  void checkNotClosed() const;
  void closeNoThrow();

  uint64_t mNextRenderId{1};
  uint64_t mLightStateVersion{1};

  std::shared_ptr<SapienRenderEngine> mEngine;
  std::shared_ptr<Device> mDevice;
  std::shared_ptr<svulkan2::core::Context> mContext;
  std::shared_ptr<svulkan2::scene::Scene> mScene;

  std::set<std::shared_ptr<SapienRenderBodyComponent>, comp_cmp> mRenderBodyComponents;
  std::set<std::shared_ptr<SapienRenderCameraComponent>, comp_cmp> mRenderCameraComponents;
  std::set<std::shared_ptr<SapienRenderLightComponent>, comp_cmp> mRenderLightComponents;
  std::set<std::shared_ptr<PointCloudComponent>, comp_cmp> mPointCloudComponents;
  std::set<std::shared_ptr<CudaDeformableMeshComponent>, comp_cmp> mCudaDeformableMeshComponents;

  std::shared_ptr<SapienRenderCubemap> mCubemap;
  std::shared_ptr<CudaArrayLifecycle> mViewLifecycle{std::make_shared<CudaArrayLifecycle>()};
  bool mClosed{false};
};

} // namespace sapien_renderer
} // namespace sapien
