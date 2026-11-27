#pragma once
#include "camera.h"
#include "light.h"
#include "light_data.h"
#include "material_data.h"
#include "node.h"
#include "object.h"
#include "svulkan2/core/as.h"
#include "svulkan2/core/command_pool.h"
#include "svulkan2/core/context.h"
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace svulkan2 {

namespace resource {
class SVCubemap;
class SVRenderTarget;
} // namespace resource

namespace scene {

/** Scene-owned shadow map targets and light UBOs shared across all rasterization cameras
 *  viewing the same Scene. Directional lights allocate a kShadowCascadeCount-layer 2D array,
 *  point lights a 6-layer cubemap, and spot/textured lights a 2D depth target. */
struct RasterShadowMaps {
  vk::Format depthFormat{vk::Format::eUndefined};
  uint64_t sceneVersion{0};
  size_t lightBufferSize{0};

  std::vector<uint32_t> directionalShadowSizes;
  std::vector<uint32_t> pointShadowSizes;
  std::vector<uint32_t> spotShadowSizes;
  std::vector<uint32_t> texturedShadowSizes;

  std::vector<std::shared_ptr<resource::SVRenderTarget>> directionalReadTargets;
  std::vector<std::shared_ptr<resource::SVRenderTarget>> pointReadTargets;
  std::vector<std::shared_ptr<resource::SVRenderTarget>> spotReadTargets;
  std::vector<std::shared_ptr<resource::SVRenderTarget>> texturedReadTargets;

  std::vector<std::shared_ptr<resource::SVRenderTarget>> allWriteTargets;
  std::vector<uint32_t> allShadowSizes;

  std::vector<std::unique_ptr<core::Buffer>> lightBuffers;
  std::vector<uint8_t> lastLightBufferBytes;

  uint64_t renderedShadowEpoch{0};
  bool renderedWithPointShadow{false};

  bool matches(vk::Format format, size_t bufferSize,
               std::vector<uint32_t> const &dirSizes,
               std::vector<uint32_t> const &ptSizes,
               std::vector<uint32_t> const &spSizes,
               std::vector<uint32_t> const &texSizes) const {
    return depthFormat == format && lightBufferSize == bufferSize &&
           directionalShadowSizes == dirSizes && pointShadowSizes == ptSizes &&
           spotShadowSizes == spSizes && texturedShadowSizes == texSizes;
  }
};

class Scene {
public:
  inline Node &getRootNode() { return *mRootNode; };

  Node &addNode(Transform const &transform = {});
  Node &addNode(Node &parent, Transform const &transform = {});

  Object &addObject(std::shared_ptr<resource::SVModel> model, Transform const &transform = {});
  Object &addObject(Node &parent, std::shared_ptr<resource::SVModel> model,
                    Transform const &transform = {});

  Object &addDeformableObject(std::shared_ptr<resource::SVModel> model);

  LineObject &addLineObject(std::shared_ptr<resource::SVLineSet> lineSet,
                            Transform const &transform = {});
  LineObject &addLineObject(Node &parent, std::shared_ptr<resource::SVLineSet> lineSet,
                            Transform const &transform = {});
  PointObject &addPointObject(std::shared_ptr<resource::SVPointSet> pointSet,
                              Transform const &transform = {});
  PointObject &addPointObject(Node &parent, std::shared_ptr<resource::SVPointSet> pointSet,
                              Transform const &transform = {});
  Camera &addCamera(Transform const &transform = {});
  Camera &addCamera(Node &parent, Transform const &transform = {});

  PointLight &addPointLight();
  PointLight &addPointLight(Node &parent);

  DirectionalLight &addDirectionalLight();
  DirectionalLight &addDirectionalLight(Node &parent);

  SpotLight &addSpotLight();
  SpotLight &addSpotLight(Node &parent);

  TexturedLight &addTexturedLight();
  TexturedLight &addTexturedLight(Node &parent);

  ParallelogramLight &addParallelogramLight();
  ParallelogramLight &addParallelogramLight(Node &parent);

  void removeNode(Node &node);
  void clearNodes();
  void forceRemove();

  inline void setAmbientLight(glm::vec4 const &color) { mAmbientLight = color; }
  inline glm::vec4 getAmbientLight() const { return mAmbientLight; };

  virtual std::vector<Object *> getObjects();

  virtual std::vector<LineObject *> getLineObjects();
  virtual std::vector<PointObject *> getPointObjects();
  virtual std::vector<Camera *> getCameras();
  virtual std::vector<PointLight *> getPointLights();
  virtual std::vector<DirectionalLight *> getDirectionalLights();
  virtual std::vector<SpotLight *> getSpotLights();
  virtual std::vector<TexturedLight *> getTexturedLights();
  virtual std::vector<ParallelogramLight *> getParallelogramLights();

  void setEnvironmentMap(std::shared_ptr<resource::SVCubemap> map) { mEnvironmentMap = map; }

  std::shared_ptr<resource::SVCubemap> getEnvironmentMap() const { return mEnvironmentMap; }

  void setBatchedRenderShared(bool shared) { mBatchedRenderShared = shared; }
  bool isBatchedRenderShared() const { return mBatchedRenderShared; }

  explicit Scene(std::shared_ptr<core::Context> context);
  std::shared_ptr<core::Context> getContext() const { return mContext; }
  Scene(Scene const &other) = delete;
  Scene &operator=(Scene const &other) = delete;
  Scene(Scene &&other) = delete;
  Scene &operator=(Scene &&other) = delete;

  std::shared_ptr<RasterShadowMaps> getOrCreateRasterShadowMaps(vk::Format depthFormat,
                                                                size_t lightBufferSize);

  void uploadShadowToDevice(core::Buffer &shadowBuffer, RasterShadowMaps &shadowMaps,
                            StructDataLayout const &shadowLayout);

  /** Shadow state epoch incremented whenever geometry, transforms, or light shadow matrices
   *  change. Cameras sharing a Scene skip redundant shadow passes when renderedShadowEpoch
   *  already matches getShadowEpoch(). */
  uint64_t getShadowEpoch() const { return mShadowEpoch; }
  void markShadowsDirty();
  uint64_t getRasterShadowRenderCount() const { return mRasterShadowRenderCount; }
  void noteRasterShadowsRendered() { mRasterShadowRenderCount++; }

  /** call exactly once per time frame to update the object matrices
   *  this function also copies current matrices into the previous matrices
   */
  void updateModelMatrices();

  /** called to order shadow lights before non-shadow lights */
  void reorderLights();

  uint64_t getVersion() const { return mVersion; }
  void updateVersion();

  uint64_t getRenderVersion() const { return mRenderVersion; }

  /** call this function to notify something has changed in the scene but the scene does not know.
   * E.g., some deformable mesh is modified */
  void updateRenderVersion();

  /** Record the refits of this scene's deformable and point-set BLASes into `commandBuffer`
   *  without submitting; true when anything was recorded. The caller owns submission, the
   *  barriers, and the TLAS refit (core::TLAS::RecordUpdates). Requires built RT resources. */
  bool recordBLASRefits(vk::CommandBuffer commandBuffer);
  /** once per render version: PrevTlasInstances <- LastTlasInstances, then the TLAS instance
   *  rows of this version -> LastTlasInstances, so every camera of the scene and every render
   *  call of one version read the same previous-frame transforms; the first snapshot after the
   *  tables were built seeds Last with this version's rows, so that frame reports no motion.
   *  Record it into the command buffer of a frame that is submitted right away, after the
   *  instance buffer holds the frame's rows; true when anything was recorded. */
  bool recordPrevTLASSnapshot(vk::CommandBuffer commandBuffer);
  /** recordPrevTLASSnapshot() in a command buffer of its own, submitted to the queue at once:
   *  call it right before submitting a trace. Preparing resources records no snapshot, so rows
   *  written before the first rendered frame (the CPU state at gpu_init()) never become its
   *  previous frame. */
  void submitPrevTLASSnapshot();

  // ray tracing
  /** one scene of BuildAccelerationStructures */
  struct AccelerationStructureBuild {
    Scene *scene;
    /** the TLAS instance rows live in device memory exported to CUDA */
    bool externalInstanceBuffer;
  };
  /** Build the BLASes and TLASes of several scenes together: one queue submission per batch of
   *  structures instead of one per structure, which dominated the start-up of thousands of
   *  scenes. Scenes whose structures are current are skipped. The per-camera ray tracing storage
   *  buffers are not built (buildRTResources builds them). */
  static void BuildAccelerationStructures(std::vector<AccelerationStructureBuild> const &builds);

  /** Build the acceleration structures and the ray tracing tables: geometry instances, the
   *  materials[] table (common/material.glsl), textures[] and the lights. */
  void buildRTResources(StructDataLayout const &geometryInstanceBufferLayout,
                        bool externalInstanceBuffer = false);
  /** refit the acceleration structures and, where buildRTResources made them, upload the ray
   *  tracing light buffers */
  void updateRTResources(bool useDeviceTransforms = false);
  /** update BLAS/TLAS and instance transforms only; light storage buffers are
   *  untouched (grouped-GPU per-frame path) */
  void updateRTTransforms(bool useDeviceTransforms);
  /** upload only the RT light storage buffers from current CPU light state */
  void uploadRTLights();
  /** versions of the materials[] content; each changes only when the table did */
  struct RTMaterialVersions {
    uint64_t content{};  // any field of any material: accumulated samples are stale
    uint64_t emission{}; // the emission of some material: the emissive light tables are stale
    bool operator==(RTMaterialVersions const &) const = default;
  };
  /** Repack materials[] when any material changed since it was written. Returns nullopt when
   *  the tables are not built or a material now samples a texture outside textures[]: the
   *  caller must call buildRTResources again. */
  std::optional<RTMaterialVersions> uploadRTMaterials();
  /** bumped whenever buildRTResources replaces the ray tracing tables; a renderer whose
   *  descriptors still name older tables must rebind */
  uint64_t getRTTablesVersion();
  inline core::TLAS *getTLAS() const { return mTLAS.get(); }

  inline std::vector<vk::Buffer> const &getRTPointSetBuffers() { return mPointSetBuffers; }
  inline std::vector<vk::Buffer> const &getRTVertexBuffers() { return mVertexBuffers; }
  inline std::vector<vk::Buffer> const &getRTIndexBuffers() const { return mIndexBuffers; }
  inline vk::Buffer getRTMaterialBuffer() const { return mRTMaterialBuffer->getVulkanBuffer(); }
  inline std::vector<std::tuple<vk::ImageView, vk::Sampler>> const &getRTTextures() const {
    return mTextures;
  }
  inline vk::Buffer getRTGeometryInstanceBuffer() const {
    return mGeometryInstanceBuffer->getVulkanBuffer();
  }
  inline vk::Buffer getRTPointInstanceBuffer() const {
    return mPointInstanceBuffer->getVulkanBuffer();
  }
  inline vk::Buffer getRTPointLightBuffer() const {
    return mRTPointLightBuffer->getVulkanBuffer();
  }
  inline vk::Buffer getRTDirectionalLightBuffer() const {
    return mRTDirectionalLightBuffer->getVulkanBuffer();
  }
  inline vk::Buffer getRTSpotLightBuffer() const { return mRTSpotLightBuffer->getVulkanBuffer(); }
  inline vk::Buffer getRTParallelogramLightBuffer() const {
    return mRTParallelogramLightBuffer->getVulkanBuffer();
  }
  inline core::Buffer &getRTPrevTLASInstanceBuffer() const { return *mPrevTLASInstanceBuffer; }

  void registerAccessFence(vk::Fence fence);
  void unregisterAccessFence(vk::Fence fence);

  size_t getGpuTransformBufferSize();
  void prepareObjectTransformBuffer();
  std::shared_ptr<core::Buffer> getObjectTransformBuffer();
  virtual void uploadObjectTransforms();

  /** Keep CPU-managed renderers from uploading object transforms while one or
   *  more grouped-GPU camera owners share this scene's transform resources. */
  void acquireExternalTransformOwnership(void const *owner);
  void releaseExternalTransformOwnership(void const *owner);
  bool hasExternalTransformOwnership() const { return !mExternalTransformOwners.empty(); }

protected:
  std::shared_ptr<core::Context> mContext;

  std::vector<std::unique_ptr<Node>> mNodes{};
  std::vector<std::unique_ptr<Object>> mObjects{};
  std::vector<std::unique_ptr<Object>> mDeformableObjects{};
  std::vector<std::unique_ptr<LineObject>> mLineObjects{};
  std::vector<std::unique_ptr<PointObject>> mPointObjects{};
  std::vector<std::unique_ptr<Camera>> mCameras{};
  std::vector<std::unique_ptr<PointLight>> mPointLights{};
  std::vector<std::unique_ptr<DirectionalLight>> mDirectionalLights{};
  std::vector<std::unique_ptr<SpotLight>> mSpotLights{};
  std::vector<std::unique_ptr<TexturedLight>> mTexturedLights{};
  std::vector<std::unique_ptr<ParallelogramLight>> mParallelogramLights{};

  Node *mRootNode{nullptr};

  bool mRequireForceRemove{};
  std::unordered_set<void const *> mExternalTransformOwners;

  // lighting
  glm::vec4 mAmbientLight{};
  std::shared_ptr<resource::SVCubemap> mEnvironmentMap{};

  bool mBatchedRenderShared{false};

  /** when anything is added or removed, the version changes */
  uint64_t mVersion{1l};

  /** when models in the scene are updated, the version changes */
  uint64_t mRenderVersion{1l};

  /** incremented whenever shadow maps need re-rendering */
  uint64_t mShadowEpoch{1l};
  uint64_t mRasterShadowRenderCount{0l};
  std::map<vk::Format, std::shared_ptr<RasterShadowMaps>> mRasterShadowMaps;

  // state of the world matrices, see updateModelMatrices()
  uint64_t mModelMatricesVersion{0l};
  bool mModelMatricesSettling{false};

  // for transform update and AS update
  std::unique_ptr<core::CommandPool> mCommandPool;
  core::CommandPool &getCommandPool();

  // transform buffers
  size_t mGpuTransformBufferSize{0l};
  uint64_t mTransformBufferVersion{0l};
  uint64_t mTransformBufferRenderVersion{0l};
  std::shared_ptr<core::Buffer> mTransformBufferCpu;
  std::shared_ptr<core::Buffer> mTransformBuffer;
  vk::UniqueCommandBuffer mTransformUpdateCommandBuffer;
  vk::UniqueFence mTransformUpdateFence{};

  // ray tracing helpers
  /** wait until no renderer uses the scene, then drop removed objects; requires
   *  mRTResourcesLock */
  void prepareRTBuildLocked();
  /** append unbuilt BLASes of models without one, skipping models in `seen`; point sets are
   *  built right away */
  void createMissingBLAS(std::vector<core::BLAS *> &blases,
                         std::unordered_set<resource::SVModel *> &seen);
  /** a new unbuilt TLAS over the current objects */
  core::TLAS *createTLAS(bool externalInstanceBuffer);
  /** BuildAccelerationStructures with every scene's mRTResourcesLock held */
  static void buildAccelerationStructuresLocked(
      std::vector<AccelerationStructureBuild> const &builds);

  void updateDynamicBLAS();
  void updateTLAS(bool useDeviceTransforms);
  /** record the refits of the deformable and point-set BLASes; true when any was recorded */
  bool recordDynamicBLASUpdates(vk::CommandBuffer commandBuffer);
  void createRTStorageBuffers(StructDataLayout const &geometryInstanceBufferLayout);
  void updateRTStorageBuffers();

  // ray tracing resources
  std::mutex mRTResourcesLock;
  std::unique_ptr<core::TLAS> mTLAS;
  std::unique_ptr<core::Buffer> mPrevTLASInstanceBuffer;
  // the TLAS instance rows of the last render version, the source of the next prev snapshot
  std::unique_ptr<core::Buffer> mLastTLASInstanceBuffer;
  // render version mPrevTLASInstanceBuffer already holds the predecessor of
  uint64_t mTLASSnapshotRenderVersion{0};
  // false until the buffers hold rows of a rendered frame; the first snapshot seeds both
  bool mTLASSnapshotInitialized{false};
  std::vector<vk::Buffer> mVertexBuffers;
  std::vector<vk::Buffer> mIndexBuffers;
  std::vector<std::tuple<vk::ImageView, vk::Sampler>> mTextures;
  std::unique_ptr<core::Buffer> mGeometryInstanceBuffer;

  std::vector<vk::Buffer> mPointSetBuffers;
  std::unique_ptr<core::Buffer> mPointInstanceBuffer;

  // the materials and lights of common/material.glsl and common/light_data.glsl; their
  // textures index mTextures through mRTTextureSlots
  std::vector<std::shared_ptr<resource::SVMetallicMaterial>> mRTMaterials;
  std::unique_ptr<core::Buffer> mRTMaterialBuffer;
  std::vector<GpuMaterial> mRTMaterialRecords; // the content of mRTMaterialBuffer
  uint64_t mRTMaterialVersion{};               // SVMetallicMaterial global version packed
  RTMaterialVersions mRTMaterialVersions;      // bumped when mRTMaterialRecords changes
  uint64_t mRTTablesVersion{};                 // bumped by createRTStorageBuffers
  LightTable mRTLights;
  // textures index mTextures through mRTTextureSlots. The keys own the textures: mTextures holds
  // only image views, and a raw-pointer key would let a freed texture's address be reused by a
  // new one, silently reusing a stale slot and its destroyed image view.
  std::unordered_map<std::shared_ptr<resource::SVTexture>, int> mRTTextureSlots;
  void writeRTLightBuffers();

  std::unique_ptr<core::Buffer> mRTPointLightBuffer;
  std::unique_ptr<core::Buffer> mRTDirectionalLightBuffer;
  std::unique_ptr<core::Buffer> mRTSpotLightBuffer;
  std::unique_ptr<core::Buffer> mRTParallelogramLightBuffer;

  // scene version the acceleration structures and the storage buffers were built for
  uint64_t mRTAccelerationStructureVersion{0l};
  uint64_t mRTResourcesVersion{0l};
  uint64_t mRTResourcesRenderVersion{0l};

  // indicates whether this scene is being accessed
  std::vector<vk::Fence> mAccessFences;

  // used to update AS
  vk::UniqueCommandBuffer mASUpdateCommandBuffer;
  // records and submits submitPrevTLASSnapshot(); signalled when the last one finished
  vk::UniqueCommandBuffer mPrevTLASSnapshotCommandBuffer;
  vk::UniqueFence mPrevTLASSnapshotFence;
};

} // namespace scene
} // namespace svulkan2
