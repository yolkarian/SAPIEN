#pragma once
#include <cstddef>
#include "renderer_base.h"
#include "svulkan2/core/buffer.h"
#include "svulkan2/core/context.h"
#include "svulkan2/core/descriptor_pool.h"
#include "svulkan2/resource/cubemap.h"
#include "svulkan2/scene/scene.h"
#include "svulkan2/shader/batch_rt.h"
#include "svulkan2/shader/glsl_compiler.h"
#include <map>
#include <variant>
#include <vector>

namespace svulkan2 {
namespace renderer {

/** one row of the batched camera table; the pose kernel writes viewMatrix (byte 0) and
 *  viewMatrixInverse (byte 128) in place */
struct BatchCameraRow {
  glm::mat4 viewMatrix{1};
  glm::mat4 projectionMatrix{1};
  glm::mat4 viewMatrixInverse{1};
  glm::mat4 projectionMatrixInverse{1};
  int envIndex{0};
  int padding0{0};
  int padding1{0};
  int padding2{0};
};
static_assert(sizeof(BatchCameraRow) == 16 * sizeof(float) * 4 + 16);
// update_camera_transforms (SAPIEN's CUDA pose kernel) writes these two offsets
static_assert(offsetof(BatchCameraRow, viewMatrix) == 0);
static_assert(offsetof(BatchCameraRow, viewMatrixInverse) == 128);

/** one row of the batched environment table, mirroring `BatchEnvironment` of batch.comp */
struct BatchEnvironmentRow {
  int pointLightBegin{0};
  int pointLightEnd{0};
  int directionalLightBegin{0};
  int directionalLightEnd{0};
  int spotLightBegin{0};
  int spotLightEnd{0};
  int parallelogramLightBegin{0};
  int parallelogramLightEnd{0};
  int objectOffset{0};
  int geometryInstanceOffset{0};
  int pointInstanceOffset{0};
  int environmentMap{0};
  glm::vec4 ambientLight{0.f, 0.f, 0.f, 0.f};
};
static_assert(sizeof(BatchEnvironmentRow) == 64);

/** Group-level batched ray tracing: one command buffer and one queue submit render every camera
 *  of a camera group.
 *
 *  Every camera owns a row of the camera table and names one environment. Per frame the renderer
 *  records the refit of every environment's top-level acceleration structure and one compute
 *  dispatch over all cameras, in one command buffer, and writes each picture straight into a
 *  device buffer whose memory exports to CUDA. Nothing is copied afterwards.
 *
 *  All cameras share one resolution and the rasterizer's picture layout:
 *  `[camera][height][width][channel]`. Users create one renderer per resolution.
 *
 *  Scene tables (materials, textures, geometry instances, objects, point instances, lights) are
 *  built once by `initialize()`, concatenated across environments with per environment offsets;
 *  only instance transforms and light tables change per frame. */
class BatchRTRenderer {
public:
  /** Compile `batch.comp` of `shaderDir` with `defines` plus the picture selection: each name of
   *  `pictures` compiles in (define OUT_<NAME>), and its entry in `formats` selects the format
   *  (define <NAME>_FORMAT_<FORMAT>). Throws when a picture is not one of the pictures of
   *  batch.comp, when the pack does not declare it, or when its declared element type cannot
   *  produce the requested format. */
  BatchRTRenderer(std::shared_ptr<core::Context> context, std::string const &shaderDir,
                  ShaderDefines const &defines, int width, int height,
                  std::vector<std::string> const &pictures,
                  std::map<std::string, vk::Format> const &formats);

  /** Build the RT resources (BLAS, TLAS) and the scene tables of `environments`, allocate
   *  the camera table and the exported picture buffers, and write the descriptor sets.
   *  `cameraEnvironments` gives the environment index of each camera. Call once, before the first
   *  frame. */
  void initialize(std::vector<std::shared_ptr<scene::Scene>> const &environments, int cameraCount,
                  std::vector<int> const &cameraEnvironments);

  int getEnvironmentCount() const { return static_cast<int>(mEnvironments.size()); }
  /** group camera index of camera row `row` (rows are sorted by environment) */
  int getCameraOfRow(int row) const {
    if (row < 0 || row >= static_cast<int>(mCameraOrder.size())) {
      throw std::runtime_error("batched ray tracing camera row out of range");
    }
    return mCameraOrder.at(row);
  }
  int getCameraCount() const { return mCameraCount; }
  int getWidth() const { return mWidth; }
  int getHeight() const { return mHeight; }

  std::vector<std::string> const &getPictureNames() const { return mPictureNames; }
  vk::Format getPictureFormat(std::string const &name) const;
  /** device buffer of picture `name`: [camera][height][width][channel], exportable to CUDA */
  core::Buffer &getPictureBuffer(std::string const &name);
  /** bytes of one camera's `name` picture */
  vk::DeviceSize getPictureBytes(std::string const &name) const;

#ifdef SVULKAN2_CUDA_INTEROP
  /** CUDA pointer of camera row `row` (a BatchCameraRow): a pose kernel writes the view matrix
   *  at byte 0 and its inverse at byte 128 in place */
  void *getCameraRowCudaPtr(int row);
#endif

  /** write the view and projection matrices and their inverses of camera row `row`; waits for
   *  the transfer, so call it only while no frame of this renderer is in flight */
  void uploadCamera(int row, glm::mat4 const &viewMatrix, glm::mat4 const &projectionMatrix);

  /** declared type of `name` in the push constant of the pack; eNone when the pack does not
   *  declare it, or when the renderer owns it (cameraCount, width, height) */
  CustomPropertyType getCustomPropertyType(std::string const &name) const;
  void setCustomProperty(std::string const &name, int value);
  void setCustomProperty(std::string const &name, float value);

  /** re-upload the concatenated light tables, the per environment ambient light and the
   *  environment map selection from the current CPU state of the environment scenes */
  void uploadSceneLightState();

  /** record every environment's acceleration structure refit and the group dispatch, then
   *  submit once: `wait` at `waitValue` must be reached first, and `signal` is signalled with
   *  `signalValue` when the frame completed */
  void render(vk::Semaphore wait, uint64_t waitValue, vk::Semaphore signal, uint64_t signalValue);

  /** host wait until every submitted frame finished */
  void waitIdle();

  std::string summary() const;

  ~BatchRTRenderer();

  BatchRTRenderer(BatchRTRenderer const &) = delete;
  BatchRTRenderer &operator=(BatchRTRenderer const &) = delete;

private:
  void checkInitialized() const;
  void checkCameraRow(int row) const;
  void allocatePictures();
  void buildPipeline();
  /** descriptor count of scene binding `name`: the group table size for arrays, else 1 */
  uint32_t getSceneBindingCount(std::string const &name) const;
  /** default black cube followed by the distinct environment cubemaps */
  std::vector<vk::DescriptorImageInfo> environmentMapInfos();
  void writeDescriptors();
  void refreshEnvironmentRows();
  void refreshLightRows();
  void updatePushConstant();

  std::shared_ptr<core::Context> mContext;
  std::shared_ptr<shader::BatchShaderPack> mShaderPack;
  std::string mShaderDir;

  int mWidth{};
  int mHeight{};
  int mCameraCount{};
  bool mInitialized{};

  std::vector<std::shared_ptr<scene::Scene>> mEnvironments;
  std::vector<BatchEnvironmentRow> mEnvironmentRows;
  std::vector<std::shared_ptr<resource::SVCubemap>> mEnvironmentMaps;
  std::vector<int> mCameraOrder;    // camera group index of every camera row

  std::vector<std::string> mPictureNames;
  std::map<std::string, vk::Format> mPictureFormats;
  std::map<std::string, std::unique_ptr<core::Buffer>> mPictureBuffers;
  vk::DeviceSize mPictureBytes{};

  std::vector<vk::Buffer> mMaterialBuffers;
  std::vector<std::pair<vk::ImageView, vk::Sampler>> mTextures;
  std::unique_ptr<core::Buffer> mCameraBuffer;
  vk::DeviceSize mCameraSlotSize{};
  std::unique_ptr<core::Buffer> mEnvironmentBuffer;
  std::unique_ptr<core::Buffer> mCameraOrderBuffer;
  std::unique_ptr<core::Buffer> mGeometryInstanceBuffer;
  std::unique_ptr<core::Buffer> mTextureIndexBuffer;
  std::unique_ptr<core::Buffer> mObjectBuffer;
  std::unique_ptr<core::Buffer> mPointInstanceBuffer;
  std::vector<vk::Buffer> mVertexBuffers;
  std::vector<vk::Buffer> mIndexBuffers;
  std::vector<vk::Buffer> mPointSetBuffers;
  std::unique_ptr<core::Buffer> mPointLightBuffer;
  std::unique_ptr<core::Buffer> mDirectionalLightBuffer;
  std::unique_ptr<core::Buffer> mSpotLightBuffer;
  std::unique_ptr<core::Buffer> mParallelogramLightBuffer;

  // every environment's lights, concatenated in environment order
  scene::LightTable mLights;

  std::map<std::string, std::variant<int, float>> mProperties;

  std::unique_ptr<core::DynamicDescriptorPool> mOutputPool;
  std::unique_ptr<core::DynamicDescriptorPool> mScenePool;
  core::DescriptorSet mOutputSet;
  core::DescriptorSet mSceneSet;
  vk::UniqueDescriptorSetLayout mOutputLayout;
  vk::UniqueDescriptorSetLayout mSceneLayout;
  vk::UniquePipelineLayout mPipelineLayout;
  vk::UniquePipeline mPipeline;
  std::unique_ptr<core::CommandPool> mCommandPool;
  vk::UniqueCommandBuffer mCommandBuffer;
  std::vector<uint8_t> mPushConstantBuffer;
};

} // namespace renderer
} // namespace svulkan2
