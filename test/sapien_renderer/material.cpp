#include "sapien/sapien_renderer/sapien_renderer.h"
#include <filesystem>
#include <gtest/gtest.h>

using namespace sapien;
using namespace sapien::sapien_renderer;

TEST(SapienRenderTexture, CreateMaterialDefaults) {
  auto material = std::make_shared<SapienRenderMaterial>();

  auto expectEqual = [](std::array<float, 4> const &actual, std::array<float, 4> const &expected) {
    for (int i = 0; i < 4; ++i) {
      EXPECT_FLOAT_EQ(actual[i], expected[i]);
    }
  };
  auto expectEqual3 = [](std::array<float, 3> const &actual,
                         std::array<float, 3> const &expected) {
    for (int i = 0; i < 3; ++i) {
      EXPECT_FLOAT_EQ(actual[i], expected[i]);
    }
  };

  expectEqual(material->getEmission(), {0.f, 0.f, 0.f, 0.f});
  expectEqual(material->getBaseColor(), {1.f, 1.f, 1.f, 1.f});
  EXPECT_FLOAT_EQ(material->getBaseWeight(), 1.f);
  EXPECT_FLOAT_EQ(material->getSpecularWeight(), 1.f);
  expectEqual3(material->getSpecularColor(), {1.f, 1.f, 1.f});
  EXPECT_FLOAT_EQ(material->getSpecularAnisotropy(), 0.f);
  EXPECT_FLOAT_EQ(material->getMetallic(), 0.f);
  EXPECT_FLOAT_EQ(material->getRoughness(), 0.45f);
  EXPECT_FLOAT_EQ(material->getIOR(), 1.5f);
  EXPECT_FLOAT_EQ(material->getTransmission(), 0.f);
  expectEqual3(material->getTransmissionColor(), {1.f, 1.f, 1.f});
  EXPECT_FLOAT_EQ(material->getTransmissionDepth(), 0.f);
  EXPECT_FLOAT_EQ(material->getCoatWeight(), 0.f);
  expectEqual3(material->getCoatColor(), {1.f, 1.f, 1.f});
  EXPECT_FLOAT_EQ(material->getCoatRoughness(), 0.f);
  EXPECT_FLOAT_EQ(material->getCoatIor(), 1.6f);
  EXPECT_FLOAT_EQ(material->getFuzzWeight(), 0.f);
  expectEqual3(material->getFuzzColor(), {1.f, 1.f, 1.f});
  EXPECT_FLOAT_EQ(material->getFuzzRoughness(), 0.5f);
  EXPECT_FALSE(material->getThinWalled());

  EXPECT_EQ(material->getEmissionTexture(), nullptr);
  EXPECT_EQ(material->getBaseColorTexture(), nullptr);
  EXPECT_EQ(material->getRoughnessTexture(), nullptr);
  EXPECT_EQ(material->getMetallicTexture(), nullptr);
  EXPECT_EQ(material->getNormalTexture(), nullptr);
  EXPECT_EQ(material->getTransmissionTexture(), nullptr);
  EXPECT_EQ(material->getOcclusionTexture(), nullptr);
}

TEST(SapienRenderTexture, CreateMaterial) {
  auto meshfile =
      std::filesystem::path(__FILE__).parent_path().parent_path() / "assets" / "test_texture.png";
  auto texture = std::make_shared<SapienRenderTexture2D>(
      meshfile.string(), 3, SapienRenderTexture2D::FilterMode::eNEAREST,
      SapienRenderTexture2D::AddressMode::eMIRROR, false);

  auto material = std::make_shared<SapienRenderMaterial>(std::array<float, 4>{0.1, 0.2, 0.3, 0.9},
                                                         std::array<float, 4>{0.4, 0.5, 0.6, 0.8},
                                                         0.1, 0.2, 0.3, 0.4, 0.5);
  EXPECT_FLOAT_EQ(std::get<0>(material->getEmission()), 0.1);
  EXPECT_FLOAT_EQ(std::get<1>(material->getEmission()), 0.2);
  EXPECT_FLOAT_EQ(std::get<2>(material->getEmission()), 0.3);
  EXPECT_FLOAT_EQ(std::get<3>(material->getEmission()), 0.9);

  EXPECT_FLOAT_EQ(std::get<0>(material->getBaseColor()), 0.4);
  EXPECT_FLOAT_EQ(std::get<1>(material->getBaseColor()), 0.5);
  EXPECT_FLOAT_EQ(std::get<2>(material->getBaseColor()), 0.6);
  EXPECT_FLOAT_EQ(std::get<3>(material->getBaseColor()), 0.8);

  EXPECT_FLOAT_EQ(material->getSpecularWeight(), 0.1);
  EXPECT_FLOAT_EQ(material->getRoughness(), 0.2);
  EXPECT_FLOAT_EQ(material->getMetallic(), 0.3);
  EXPECT_FLOAT_EQ(material->getTransmission(), 0.4);
  EXPECT_FLOAT_EQ(material->getIOR(), 0.5);

  // setters
  material->setBaseWeight(0.11);
  EXPECT_FLOAT_EQ(material->getBaseWeight(), 0.11);
  material->setSpecularColor({0.12, 0.13, 0.14});
  EXPECT_FLOAT_EQ(std::get<0>(material->getSpecularColor()), 0.12);
  EXPECT_FLOAT_EQ(std::get<1>(material->getSpecularColor()), 0.13);
  EXPECT_FLOAT_EQ(std::get<2>(material->getSpecularColor()), 0.14);
  material->setSpecularAnisotropy(0.15);
  EXPECT_FLOAT_EQ(material->getSpecularAnisotropy(), 0.15);
  material->setTransmissionColor({0.16, 0.17, 0.18});
  EXPECT_FLOAT_EQ(std::get<0>(material->getTransmissionColor()), 0.16);
  EXPECT_FLOAT_EQ(std::get<1>(material->getTransmissionColor()), 0.17);
  EXPECT_FLOAT_EQ(std::get<2>(material->getTransmissionColor()), 0.18);
  material->setTransmissionDepth(0.19);
  EXPECT_FLOAT_EQ(material->getTransmissionDepth(), 0.19);
  material->setCoatWeight(0.2);
  EXPECT_FLOAT_EQ(material->getCoatWeight(), 0.2);
  material->setCoatColor({0.21, 0.22, 0.23});
  EXPECT_FLOAT_EQ(std::get<0>(material->getCoatColor()), 0.21);
  EXPECT_FLOAT_EQ(std::get<1>(material->getCoatColor()), 0.22);
  EXPECT_FLOAT_EQ(std::get<2>(material->getCoatColor()), 0.23);
  material->setCoatRoughness(0.24);
  EXPECT_FLOAT_EQ(material->getCoatRoughness(), 0.24);
  material->setCoatIor(0.25);
  EXPECT_FLOAT_EQ(material->getCoatIor(), 0.25);
  material->setFuzzWeight(0.26);
  EXPECT_FLOAT_EQ(material->getFuzzWeight(), 0.26);
  material->setFuzzColor({0.27, 0.28, 0.29});
  EXPECT_FLOAT_EQ(std::get<0>(material->getFuzzColor()), 0.27);
  EXPECT_FLOAT_EQ(std::get<1>(material->getFuzzColor()), 0.28);
  EXPECT_FLOAT_EQ(std::get<2>(material->getFuzzColor()), 0.29);
  material->setFuzzRoughness(0.3);
  EXPECT_FLOAT_EQ(material->getFuzzRoughness(), 0.3);
  material->setThinWalled(true);
  EXPECT_TRUE(material->getThinWalled());
  material->setThinWalled(false);
  EXPECT_FALSE(material->getThinWalled());

  material->setEmission({0.2, 0.3, 0.4, 0.8});
  EXPECT_FLOAT_EQ(std::get<0>(material->getEmission()), 0.2);
  EXPECT_FLOAT_EQ(std::get<1>(material->getEmission()), 0.3);
  EXPECT_FLOAT_EQ(std::get<2>(material->getEmission()), 0.4);
  EXPECT_FLOAT_EQ(std::get<3>(material->getEmission()), 0.8);

  material->setBaseColor({0.5, 0.6, 0.7, 0.9});
  EXPECT_FLOAT_EQ(std::get<0>(material->getBaseColor()), 0.5);
  EXPECT_FLOAT_EQ(std::get<1>(material->getBaseColor()), 0.6);
  EXPECT_FLOAT_EQ(std::get<2>(material->getBaseColor()), 0.7);
  EXPECT_FLOAT_EQ(std::get<3>(material->getBaseColor()), 0.9);

  material->setSpecularWeight(0.5);
  EXPECT_FLOAT_EQ(material->getSpecularWeight(), 0.5);
  material->setRoughness(0.6);
  EXPECT_FLOAT_EQ(material->getRoughness(), 0.6);
  material->setMetallic(0.7);
  EXPECT_FLOAT_EQ(material->getMetallic(), 0.7);
  material->setTransmission(0.8);
  EXPECT_FLOAT_EQ(material->getTransmission(), 0.8);
  material->setIOR(0.9);
  EXPECT_FLOAT_EQ(material->getIOR(), 0.9);

  material->setEmissionTexture(texture);
  ASSERT_TRUE(material->getEmissionTexture());
  EXPECT_EQ(material->getEmissionTexture()->getTexture(), texture->getTexture());
  material->setEmissionTexture(nullptr);
  EXPECT_EQ(material->getEmissionTexture(), nullptr);

  material->setBaseColorTexture(texture);
  ASSERT_TRUE(material->getBaseColorTexture());
  EXPECT_EQ(material->getBaseColorTexture()->getTexture(), texture->getTexture());
  material->setBaseColorTexture(nullptr);
  EXPECT_EQ(material->getBaseColorTexture(), nullptr);

  material->setRoughnessTexture(texture);
  ASSERT_TRUE(material->getRoughnessTexture());
  EXPECT_EQ(material->getRoughnessTexture()->getTexture(), texture->getTexture());
  material->setRoughnessTexture(nullptr);
  EXPECT_EQ(material->getRoughnessTexture(), nullptr);

  material->setMetallicTexture(texture);
  ASSERT_TRUE(material->getMetallicTexture());
  EXPECT_EQ(material->getMetallicTexture()->getTexture(), texture->getTexture());
  material->setMetallicTexture(nullptr);
  EXPECT_EQ(material->getMetallicTexture(), nullptr);

  material->setNormalTexture(texture);
  ASSERT_TRUE(material->getNormalTexture());
  EXPECT_EQ(material->getNormalTexture()->getTexture(), texture->getTexture());
  material->setNormalTexture(nullptr);
  EXPECT_EQ(material->getNormalTexture(), nullptr);

  material->setTransmissionTexture(texture);
  ASSERT_TRUE(material->getTransmissionTexture());
  EXPECT_EQ(material->getTransmissionTexture()->getTexture(), texture->getTexture());
  material->setTransmissionTexture(nullptr);
  EXPECT_EQ(material->getTransmissionTexture(), nullptr);

  material->setOcclusionTexture(texture);
  ASSERT_TRUE(material->getOcclusionTexture());
  EXPECT_EQ(material->getOcclusionTexture()->getTexture(), texture->getTexture());
  material->setOcclusionTexture(nullptr);
  EXPECT_EQ(material->getOcclusionTexture(), nullptr);
}
