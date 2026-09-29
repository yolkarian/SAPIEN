#include "sapien/sapien_renderer/sapien_renderer.h"
#include "sapien/scene.h"
#include <gtest/gtest.h>

using namespace sapien;
using namespace sapien::sapien_renderer;

TEST(SapienRenderLight, AttachedPropertiesReachNodeAndInvalidateResources) {
  auto system = std::make_shared<SapienRendererSystem>(nullptr);
  auto scene = std::make_shared<Scene>(std::vector<std::shared_ptr<System>>{system});
  auto entity = std::make_shared<Entity>();
  auto light = std::make_shared<SapienRenderDirectionalLightComponent>();
  entity->addComponent(light);
  scene->addEntity(entity);
  auto node = system->getScene()->getDirectionalLights().at(0);
  light->setShadowEnabled(false);
  EXPECT_FALSE(node->isShadowEnabled());
  light->setShadowEnabled(true);
  EXPECT_TRUE(node->isShadowEnabled());
  auto version = system->getScene()->getVersion();
  light->setShadowMapSize(512);
  EXPECT_EQ(node->getShadowMapSize(), 512);
  EXPECT_GT(system->getScene()->getVersion(), version);
  light->internalSealGroupState();
  EXPECT_THROW(light->setShadowEnabled(false), std::runtime_error);
  EXPECT_THROW(light->setShadowMapSize(256), std::runtime_error);
  EXPECT_TRUE(node->isShadowEnabled());
  EXPECT_EQ(node->getShadowMapSize(), 512);
  light->internalReleaseGroupStateSeal();
  scene->close();
  system->close();
}

TEST(SapienRenderLight, AttachedPointAndSpotPropertiesReachNodes) {
  auto system = std::make_shared<SapienRendererSystem>(nullptr);
  auto scene = std::make_shared<Scene>(std::vector<std::shared_ptr<System>>{system});
  auto entity = std::make_shared<Entity>();
  auto point = std::make_shared<SapienRenderPointLightComponent>();
  auto spot = std::make_shared<SapienRenderSpotLightComponent>();
  entity->addComponent(point);
  entity->addComponent(spot);
  scene->addEntity(entity);
  auto pointNode = system->getScene()->getPointLights().at(0);
  auto spotNode = system->getScene()->getSpotLights().at(0);
  point->setShadowEnabled(false);
  spot->setShadowEnabled(false);
  EXPECT_FALSE(pointNode->isShadowEnabled());
  EXPECT_FALSE(spotNode->isShadowEnabled());
  point->setShadowMapSize(256);
  spot->setShadowMapSize(512);
  point->setShadowEnabled(true);
  spot->setShadowEnabled(true);
  EXPECT_TRUE(pointNode->isShadowEnabled());
  EXPECT_TRUE(spotNode->isShadowEnabled());
  EXPECT_EQ(pointNode->getShadowMapSize(), 256);
  EXPECT_EQ(spotNode->getShadowMapSize(), 512);
  scene->close();
  EXPECT_NO_THROW(point->setShadowEnabled(false));
  EXPECT_NO_THROW(spot->setShadowMapSize(128));
  system->close();
}
