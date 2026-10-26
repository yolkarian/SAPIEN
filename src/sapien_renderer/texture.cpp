#include "sapien/sapien_renderer/texture.h"
#include "sapien/sapien_renderer/sapien_renderer_system.h"
#include <cstring>
#include <svulkan2/resource/manager.h>

namespace sapien {
namespace sapien_renderer {

static SapienRenderTexture::FilterMode convertFilterMode(vk::Filter filter) {
  switch (filter) {
  case vk::Filter::eLinear:
    return SapienRenderTexture::FilterMode::eLINEAR;
  case vk::Filter::eNearest:
    return SapienRenderTexture::FilterMode::eNEAREST;
  default:
    throw std::runtime_error("unsupported filter mode");
  }
}

static vk::Filter convertFilterMode(SapienRenderTexture::FilterMode filterMode) {
  switch (filterMode) {
  case SapienRenderTexture::FilterMode::eNEAREST:
    return vk::Filter::eNearest;
  case SapienRenderTexture::FilterMode::eLINEAR:
    return vk::Filter::eLinear;
  default:
    // unreachable
    return vk::Filter::eLinear;
  }
}

static SapienRenderTexture::AddressMode convertAddressMode(vk::SamplerAddressMode address) {
  switch (address) {
  case vk::SamplerAddressMode::eRepeat:
    return SapienRenderTexture::AddressMode::eREPEAT;
  case vk::SamplerAddressMode::eClampToBorder:
    return SapienRenderTexture::AddressMode::eBORDER;
  case vk::SamplerAddressMode::eClampToEdge:
    return SapienRenderTexture::AddressMode::eEDGE;
  case vk::SamplerAddressMode::eMirroredRepeat:
    return SapienRenderTexture::AddressMode::eMIRROR;
  default:
    throw std::runtime_error("unsupported filter mode");
  }
}

static vk::SamplerAddressMode convertAddressMode(SapienRenderTexture::AddressMode addressMode) {
  switch (addressMode) {
  case SapienRenderTexture::AddressMode::eREPEAT:
    return vk::SamplerAddressMode::eRepeat;
  case SapienRenderTexture::AddressMode::eBORDER:
    return vk::SamplerAddressMode::eClampToBorder;
  case SapienRenderTexture::AddressMode::eEDGE:
    return vk::SamplerAddressMode::eClampToEdge;
  case SapienRenderTexture::AddressMode::eMIRROR:
    return vk::SamplerAddressMode::eMirroredRepeat;
  default:
    // unreachable
    return vk::SamplerAddressMode::eRepeat;
  }
}

SapienRenderTexture::SapienRenderTexture(uint32_t width, uint32_t height, uint32_t depth,
                                         vk::Format format, std::vector<char> const &rawData,
                                         int dim, uint32_t mipLevels, FilterMode filterMode,
                                         AddressMode addressMode, bool srgb) {
  mEngine = SapienRenderEngine::Get();
  vk::Filter filter = convertFilterMode(filterMode);
  vk::SamplerAddressMode address = convertAddressMode(addressMode);
  mTexture = svulkan2::resource::SVTexture::FromRawData(width, height, depth, format, rawData, dim,
                                                        mipLevels, filter, filter, address,
                                                        address, address, srgb);
}

vk::Format SapienRenderTexture::getFormat() const {
  if (!mTexture->isLoaded()) {
    mTexture->loadAsync().get();
  }
  return mTexture->getImage()->getFormat();
}

int SapienRenderTexture::getDim() const { return mTexture->getDescription().dim; }
int SapienRenderTexture::getWidth() const {
  if (!mTexture->isLoaded()) {
    mTexture->loadAsync().get();
  }
  return mTexture->getImage()->getWidth();
}
int SapienRenderTexture::getHeight() const {
  if (!mTexture->isLoaded()) {
    mTexture->loadAsync().get();
  }
  return mTexture->getImage()->getHeight();
}
int SapienRenderTexture::getDepth() const {
  if (!mTexture->isLoaded()) {
    mTexture->loadAsync().get();
  }
  return mTexture->getImage()->getDepth();
}
int SapienRenderTexture::getChannels() const {
  if (!mTexture->isLoaded()) {
    mTexture->loadAsync().get();
  }
  return mTexture->getImage()->getChannels();
}

SapienRenderTexture::FilterMode SapienRenderTexture::getMinFilterMode() const {
  return convertFilterMode(mTexture->getDescription().minFilter);
}
SapienRenderTexture::FilterMode SapienRenderTexture::getMagFilterMode() const {
  return convertFilterMode(mTexture->getDescription().magFilter);
}

SapienRenderTexture::AddressMode SapienRenderTexture::getAddressModeU() const {
  return convertAddressMode(mTexture->getDescription().addressModeU);
}
SapienRenderTexture::AddressMode SapienRenderTexture::getAddressModeV() const {
  return convertAddressMode(mTexture->getDescription().addressModeV);
}
SapienRenderTexture::AddressMode SapienRenderTexture::getAddressModeW() const {
  return convertAddressMode(mTexture->getDescription().addressModeW);
}
int SapienRenderTexture::getMipmapLevels() const { return mTexture->getDescription().mipLevels; }

bool SapienRenderTexture::getIsSrgb() const { return mTexture->getDescription().srgb; }

SapienRenderTexture2D::SapienRenderTexture2D(uint32_t width, uint32_t height, vk::Format format,
                                             std::vector<char> const &rawData, uint32_t mipLevels,
                                             FilterMode filterMode, AddressMode addressMode,
                                             bool srgb) {
  mEngine = SapienRenderEngine::Get();
  vk::Filter vkf;
  vk::SamplerAddressMode vka;
  switch (filterMode) {
  case FilterMode::eNEAREST:
    vkf = vk::Filter::eNearest;
    break;
  case FilterMode::eLINEAR:
    vkf = vk::Filter::eLinear;
    break;
  }

  switch (addressMode) {
  case AddressMode::eREPEAT:
    vka = vk::SamplerAddressMode::eRepeat;
    break;
  case AddressMode::eBORDER:
    vka = vk::SamplerAddressMode::eClampToBorder;
    break;
  case AddressMode::eEDGE:
    vka = vk::SamplerAddressMode::eClampToEdge;
    break;
  case AddressMode::eMIRROR:
    vka = vk::SamplerAddressMode::eMirroredRepeat;
    break;
  }

  mTexture = mEngine->getResourceManager()->CreateTextureFromRawData(
      width, height, 1, format, rawData, 2, mipLevels, vkf, vkf, vka, vka, vka, srgb);
}

SapienRenderTexture2D::SapienRenderTexture2D(std::string_view filename, uint32_t mipLevels,
                                             FilterMode filterMode, AddressMode addressMode,
                                             bool srgb) {
  mEngine = SapienRenderEngine::Get();
  vk::Filter vkf;
  vk::SamplerAddressMode vka;
  switch (filterMode) {
  case FilterMode::eNEAREST:
    vkf = vk::Filter::eNearest;
    break;
  case FilterMode::eLINEAR:
    vkf = vk::Filter::eLinear;
    break;
  }

  switch (addressMode) {
  case AddressMode::eREPEAT:
    vka = vk::SamplerAddressMode::eRepeat;
    break;
  case AddressMode::eBORDER:
    vka = vk::SamplerAddressMode::eClampToBorder;
    break;
  case AddressMode::eEDGE:
    vka = vk::SamplerAddressMode::eClampToEdge;
    break;
  case AddressMode::eMIRROR:
    vka = vk::SamplerAddressMode::eMirroredRepeat;
    break;
  }

  mTexture = mEngine->getResourceManager()->CreateTextureFromFile(
      {filename.begin(), filename.end()}, mipLevels, vkf, vkf, vka, vka, srgb);
  mTexture->loadAsync().get();
}

SapienRenderTexture2D::SapienRenderTexture2D(std::shared_ptr<svulkan2::resource::SVTexture> tex) {
  mEngine = SapienRenderEngine::Get();
  mTexture = tex;
}

std::string SapienRenderTexture2D::getFilename() const {
  if (mTexture->getDescription().source ==
      svulkan2::resource::SVTextureDescription::SourceType::eFILE) {
    return mTexture->getDescription().filename;
  }
  return "";
}

void SapienRenderTexture::download(void *data) {
  if (!mTexture->isLoaded()) {
    mTexture->loadAsync().get();
  }
  // Host pixels are the authoritative copy; every device holding the image mirrors them.
  auto image = mTexture->getImage();
  auto const &rawData = image->getRawData();
  if (rawData.empty()) {
    throw std::runtime_error("failed to download texture: it has no host pixel data");
  }
  size_t size = static_cast<size_t>(image->getWidth()) * image->getHeight() * image->getDepth() *
                svulkan2::getFormatSize(image->getFormat());
  if (rawData.front().size() < size) {
    throw std::runtime_error("failed to download texture: host pixel data is incomplete");
  }
  std::memcpy(data, rawData.front().data(), size);
}

void SapienRenderTexture::upload(void *data) {
  if (!mTexture->isLoaded()) {
    mTexture->loadAsync().get();
  }
  auto image = mTexture->getImage();
  size_t size = static_cast<size_t>(image->getWidth()) * image->getHeight() * image->getDepth() *
                svulkan2::getFormatSize(image->getFormat());
  image->updateBaseLevel(data, size);
}

} // namespace sapien_renderer
} // namespace sapien
