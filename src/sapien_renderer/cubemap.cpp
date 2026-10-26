#include "sapien/sapien_renderer/cubemap.h"
#include "sapien/sapien_renderer/sapien_renderer_system.h"
#include <svulkan2/resource/manager.h>

namespace sapien {
namespace sapien_renderer {

SapienRenderCubemap::SapienRenderCubemap(std::string const &filename) {
  mEngine = SapienRenderEngine::Get();
  mCubemap = mEngine->getResourceManager()->CreateCubemapFromFile(filename, 5);
}

SapienRenderCubemap::SapienRenderCubemap(std::string const &px, std::string const &nx,
                                         std::string const &py, std::string const &ny,
                                         std::string const &pz, std::string const &nz) {
  mEngine = SapienRenderEngine::Get();
  mCubemap = mEngine->getResourceManager()->CreateCubemapFromFiles({px, nx, py, ny, pz, nz}, 5);
}

void SapienRenderCubemap::exportKtx(std::string const &filename) {
  // Export needs GPU state; a cubemap is device-independent, so use the default device.
  auto context = mEngine->getContext();
  mCubemap->load();
  mCubemap->uploadToDevice(*context);
  mCubemap->exportKTX(*context, filename);
}

} // namespace sapien_renderer
} // namespace sapien
