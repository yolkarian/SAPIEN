#include "svulkan2/scene/node.h"
#include "svulkan2/scene/object.h"
#include <algorithm>

namespace svulkan2 {
namespace scene {

Node::Node(std::string const &name) : mName(name) {}

void Node::removeChild(Node &child) {
  auto it = std::find(mChildren.begin(), mChildren.end(), &child);
  if (it == mChildren.end()) {
    throw std::invalid_argument("child node does not exist");
  }
  mChildren.erase(it);
}

void Node::clearChild() { mChildren = {}; }

void Node::setTransform(Transform const &transform) {
  if (transform.position == mTransform.position && transform.rotation == mTransform.rotation &&
      transform.scale == mTransform.scale) {
    return;
  }
  mTransform = transform;
  markTransformDirty();
}

void Node::markTransformDirty() {
  for (Node *node = this; node && !node->mSubtreeDirty; node = node->mParent) {
    node->mSubtreeDirty = true;
  }
}

void Node::setPosition(glm::vec3 const &pos) {
  if (mTransform.position == pos) {
    return;
  }
  mTransform.position = pos;
  markTransformDirty();
}
void Node::setRotation(glm::quat const &rot) {
  if (mTransform.rotation == rot) {
    return;
  }
  mTransform.rotation = rot;
  markTransformDirty();
}
void Node::setScale(glm::vec3 const &scale) {
  if (mTransform.scale == scale) {
    return;
  }
  mTransform.scale = scale;
  markTransformDirty();
}

void Node::updateGlobalModelMatrixRecursive() {
  glm::mat4 localMatrix = glm::translate(glm::mat4(1), mTransform.position) *
                          glm::toMat4(mTransform.rotation) *
                          glm::scale(glm::mat4(1), mTransform.scale);
  glm::mat4 worldMatrix = mParent ? (mParent->mTransform.worldModelMatrix * localMatrix)
                                  : localMatrix;
  if (!mHasWorldModelMatrix || mInitializePrevOnNextUpdate) {
    mTransform.prevWorldModelMatrix = worldMatrix;
    mHasWorldModelMatrix = true;
    mInitializePrevOnNextUpdate = false;
  } else {
    mTransform.prevWorldModelMatrix = mTransform.worldModelMatrix;
  }
  mTransform.worldModelMatrix = worldMatrix;
  mSubtreeDirty = false;
  for (auto c : mChildren) {
    c->updateGlobalModelMatrixRecursive();
  }
}

void Node::initializeWorldModelMatrixOnNextUpdate() {
  mInitializePrevOnNextUpdate = true;
  markTransformDirty();
}

std::vector<Object *> Node::getObjectsRecursive() const {
  std::vector<Object *> result;
  for (auto c : mChildren) {
    if (auto object = dynamic_cast<Object *>(c))
      if (object) {
        result.push_back(object);
      }
    auto childObjects = c->getObjectsRecursive();
    result.insert(result.end(), childObjects.begin(), childObjects.end());
  }
  return result;
}

glm::mat4 Node::computeWorldModelMatrix() const {
  glm::mat4 localMatrix = glm::translate(glm::mat4(1), mTransform.position) *
                          glm::toMat4(mTransform.rotation) *
                          glm::scale(glm::mat4(1), mTransform.scale);
  if (!mParent) {
    return localMatrix;
  }
  return mParent->computeWorldModelMatrix() * localMatrix;
}

void Node::markRemovedRecursive() {
  mRemoved = true;
  for (auto c : mChildren) {
    c->markRemovedRecursive();
  }
}

} // namespace scene
} // namespace svulkan2
