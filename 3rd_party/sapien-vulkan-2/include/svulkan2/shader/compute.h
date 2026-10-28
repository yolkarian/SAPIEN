#pragma once
#include "svulkan2/core/image.h"
#include <string>
#include <vector>

namespace svulkan2 {
namespace shader {

/** GPU image helpers; each runs on `context` and waits for completion */
std::unique_ptr<core::Image> generateBRDFLUT(core::Context &context, uint32_t size);

void prefilterCubemap(core::Context &context, core::Image &image);

std::unique_ptr<core::Image> latlongToCube(core::Context &context, core::Image &latlong,
                                           int mipLevels);

} // namespace shader
} // namespace svulkan2
