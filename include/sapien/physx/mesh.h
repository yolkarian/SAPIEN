#pragma once
#include "sapien/math/bounding_box.h"
#include "sapien/physx/physx_default.h"
#include <Eigen/Eigen>
#include <PxPhysicsAPI.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sapien {
namespace physx {
class PhysxEngine;

using Vertices = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Triangles = Eigen::Matrix<uint32_t, Eigen::Dynamic, 3, Eigen::RowMajor>;
using HeightFieldSamples = Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class PhysxConvexMesh {
public:
  PhysxConvexMesh(Vertices const &vertices);
  PhysxConvexMesh(Vertices const &vertices, std::string const &filename);
  PhysxConvexMesh(Vertices const &vertices, std::string const &filename, int part);
  PhysxConvexMesh(std::string const &filename);

  static std::vector<std::shared_ptr<PhysxConvexMesh>>
  LoadByConnectedParts(std::string const &filename);

  static std::shared_ptr<PhysxConvexMesh> CreateCylinder();

  ::physx::PxConvexMesh *getPxMesh() const { return mMesh; }
  bool hasFilename() { return mFilename.has_value(); }
  std::string getFilename() {
    if (hasFilename()) {
      return mFilename.value();
    }
    throw std::runtime_error("no filename is associated with the mesh");
  }
  bool hasPart() { return mPart.has_value(); }
  int getPart() {
    if (hasPart()) {
      return mPart.value();
    }
    throw std::runtime_error("no part is associated with the mesh");
  }

  Vertices getVertices() const;
  Triangles getTriangles() const;
  AABB const &getAABB() const { return mAABB; }

  ~PhysxConvexMesh() {
    if (mMesh) {
      mMesh->release();
    }
  }

private:
  void loadMesh(Vertices const &vertices);
  std::shared_ptr<PhysxEngine> mEngine;
  ::physx::PxConvexMesh *mMesh{};
  std::optional<std::string> mFilename;
  std::optional<int> mPart;

  AABB mAABB;

  PhysxConvexMesh() {}
};

class PhysxHeightField {
public:
  PhysxHeightField(HeightFieldSamples const &samples);

  ::physx::PxHeightField *getPxHeightField() const { return mHeightField; }
  HeightFieldSamples getSamples() const { return mSamples; }
  uint32_t getRows() const { return mSamples.rows(); }
  uint32_t getColumns() const { return mSamples.cols(); }
  int16_t getMinHeight() const { return mSamples.minCoeff(); }
  int16_t getMaxHeight() const { return mSamples.maxCoeff(); }

  ~PhysxHeightField() {
    if (mHeightField) {
      mHeightField->release();
    }
  }

private:
  std::shared_ptr<PhysxEngine> mEngine;
  ::physx::PxHeightField *mHeightField{};
  HeightFieldSamples mSamples;
};

class PhysxTriangleMesh {
public:
  PhysxTriangleMesh(Vertices const &vertices, Triangles const &triangles,
                    bool generateSDF = false,
                    std::optional<PhysxSDFShapeConfig> sdfConfig = std::nullopt);
  PhysxTriangleMesh(Vertices const &vertices, Triangles const &triangles,
                    std::string const &filename, bool generateSDF = false,
                    std::optional<PhysxSDFShapeConfig> sdfConfig = std::nullopt);
  PhysxTriangleMesh(std::string const &filename, bool generateSDF = false,
                    std::optional<PhysxSDFShapeConfig> sdfConfig = std::nullopt);

  ::physx::PxTriangleMesh *getPxMesh() const { return mMesh; }
  bool hasFilename() { return mFilename.has_value(); }
  std::string getFilename() {
    if (hasFilename()) {
      return mFilename.value();
    }
    throw std::runtime_error("no filename is associated with the mesh");
  }

  Vertices getVertices() const;
  Triangles getTriangles() const;
  AABB const &getAABB() const { return mAABB; }

  bool getSDFEnabled() const { return mSDF; }
  float getSDFSpacing() const { return mSDFSpacing; }
  uint32_t getSDFSubgridSize() const { return mSDFSubgridSize; }
  PhysxSDFShapeConfig const &getSDFConfig() const { return mSDFConfig; }

  ~PhysxTriangleMesh() {
    if (mMesh) {
      mMesh->release();
    }
  }

private:
  void loadMesh(Vertices const &vertices, Triangles const &triangles, bool generateSDF,
                std::optional<PhysxSDFShapeConfig> sdfConfig = std::nullopt);
  std::shared_ptr<PhysxEngine> mEngine;
  ::physx::PxTriangleMesh *mMesh{};
  std::optional<std::string> mFilename;

  bool mSDF{};
  float mSDFSpacing{};
  uint32_t mSDFSubgridSize{};
  PhysxSDFShapeConfig mSDFConfig{};

  AABB mAABB;
  PhysxTriangleMesh() {}
};

} // namespace physx
} // namespace sapien
