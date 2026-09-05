#include "sapien/array.h"
#include <gtest/gtest.h>

#ifdef SAPIEN_CUDA
#include <dlpack/dlpack.h>

TEST(CudaArrayDLPack, ViewReleaseRetainsLifecycleUntilLastExport) {
  auto lifecycle = std::make_shared<sapien::CudaArrayLifecycle>();
  sapien::CudaArrayHandle handle{.shape = {2, 3},
                                .strides = {12, 4},
                                .type = "f4",
                                .cudaId = 0};
  handle.viewGuard = std::make_shared<sapien::CudaArrayViewGuard>(lifecycle);
  auto first = handle.toDLPack();
  auto second = handle.toDLPack();
  handle.viewGuard.reset();

  EXPECT_EQ(first->dl_tensor.ndim, 2);
  EXPECT_EQ(first->dl_tensor.shape[1], 3);
  EXPECT_EQ(first->dl_tensor.strides[0], 3);
  EXPECT_EQ(lifecycle->viewCount(), 1);
  first->deleter(first);
  EXPECT_EQ(lifecycle->viewCount(), 1);
  second->deleter(second);
  EXPECT_EQ(lifecycle->viewCount(), 0);
}

TEST(CudaArrayDLPack, OwningExportMovesStorageAndReleasesMetadata) {
  sapien::CudaArray array({2, 3}, "f4");
  auto pointer = array.ptr;
  auto tensor = array.moveToDLPack();
  EXPECT_EQ(array.ptr, nullptr);
  EXPECT_TRUE(array.shape.empty());
  EXPECT_EQ(tensor->dl_tensor.data, pointer);
  EXPECT_EQ(tensor->dl_tensor.ndim, 2);
  EXPECT_EQ(tensor->dl_tensor.shape[0], 2);
  EXPECT_EQ(tensor->dl_tensor.shape[1], 3);
  EXPECT_EQ(tensor->dl_tensor.strides, nullptr);
  tensor->deleter(tensor);
}
#endif
