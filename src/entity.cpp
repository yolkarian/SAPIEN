#include "sapien/entity.h"
#include "./logger.h"
#include "sapien/component.h"
#include "sapien/scene.h"

namespace sapien {

static uint64_t gNextEntityId = 1ul;
static uint64_t gEntityCount = 0;

Entity::Entity() : mId(gNextEntityId++) {
  gEntityCount++;
  logger::info("Created Entity {}, total {}", mId, gEntityCount);
}

std::shared_ptr<Scene> Entity::getScene() { return mScene ? mScene->shared_from_this() : nullptr; }

void Entity::setPose(Pose const &pose) {
  mPose = pose;

  // TODO: defer the sync?
  for (auto &c : mComponents) {
    c->onSetPose(mPose);
  }
}

std::shared_ptr<Entity> Entity::addComponent(std::shared_ptr<Component> component) {
  if (component->getEntity()) {
    throw std::runtime_error("failed to add component: component already added.");
  }

  component->internalSetEntity(shared_from_this());
  mComponents.push_back(component);
  component->onSetPose(mPose);

  if (mScene && component->getEnabled()) {
    try {
      component->onAddToScene(*mScene);
    } catch (...) {
      // A rejected add must leave the component detached, as if it was never added.
      mComponents.pop_back();
      component->internalSetEntity(nullptr);
      throw;
    }
  }
  return shared_from_this();
}

void Entity::internalSwapInComponent(uint32_t index, std::shared_ptr<Component> component) {
  if (component->getEntity()) {
    throw std::runtime_error("failed to add component: component already added.");
  }
  component->internalSetEntity(shared_from_this());
  mComponents[index] = component;
  if (mScene) {
    component->onAddToScene(*mScene);
  }
}

void Entity::removeComponent(std::shared_ptr<Component> component) {
  if (component->getEntity().get() != this) {
    throw std::runtime_error("failed to add component: component already added.");
  }

  if (mScene && component->getEnabled()) {
    component->onRemoveFromScene(*mScene);
  }

  std::erase(mComponents, component);
  component->internalSetEntity(nullptr);
}

void Entity::onAddToScene(Scene &scene) {
  size_t added = 0;
  try {
    for (; added < mComponents.size(); ++added) {
      if (mComponents[added]->getEnabled()) {
        mComponents[added]->onAddToScene(scene);
      }
    }
  } catch (...) {
    // Undo the components that were already added, newest first, so a component that
    // rejects the scene leaves no partially added entity behind.
    for (size_t i = added; i-- > 0;) {
      if (!mComponents[i]->getEnabled()) {
        continue;
      }
      try {
        mComponents[i]->onRemoveFromScene(scene);
      } catch (std::exception const &e) {
        logger::error("failed to roll back component after a rejected scene add: {}", e.what());
      }
    }
    throw;
  }
}
void Entity::onRemoveFromScene(Scene &scene) {
  for (auto it = mComponents.rbegin(); it != mComponents.rend(); ++it) {
    auto &c = *it;
    if (c->getEnabled()) {
      c->onRemoveFromScene(scene);
    }
  }
}

Pose Entity::getPose() const { return mPose; }
void Entity::internalSyncPose(Pose const &pose) { mPose = pose; }

/** same as Scene::addEntity */
std::shared_ptr<Entity> Entity::addToScene(Scene &scene) {
  scene.addEntity(shared_from_this());
  return shared_from_this();
}
void Entity::removeFromScene() {
  if (!mScene) {
    throw std::runtime_error(
        "failed to remove entity from scene: the entity is not added to any scene.");
  }
  mScene->removeEntity(shared_from_this());
}

uint64_t Entity::liveCount() { return gEntityCount; }

Entity::~Entity() {
  gEntityCount--;
  logger::info("Deleting Entity {}, total {}", mId, gEntityCount);
}

} // namespace sapien
