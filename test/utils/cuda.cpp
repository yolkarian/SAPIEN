#include "sapien/utils/cuda.h"
#include <gtest/gtest.h>

#ifdef SAPIEN_CUDA
TEST(CudaEvent, EmptySynchronizationIsNoOp) {
  sapien::CudaEvent event;
  EXPECT_NO_THROW(event.synchronize());
  EXPECT_EQ(event.event, nullptr);
}
#endif
