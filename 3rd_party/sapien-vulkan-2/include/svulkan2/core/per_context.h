#pragma once
#include "context.h"
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace svulkan2 {
namespace core {

/** Device state of a device-independent resource, one entry per Context.
 *
 *  Entries are keyed by Context::getId(), which is never reused, so an entry can never
 *  be mistaken for the state of a later context. Thread-safe; a returned reference
 *  stays valid until its entry is erased or the container is cleared. */
template <typename T> class PerContext {
public:
  /** the state of `context`, default-constructed on first access */
  T &get(Context const &context) {
    std::scoped_lock lock(mMutex);
    for (auto &[id, state] : mEntries) {
      if (id == context.getId()) {
        return *state;
      }
    }
    return *mEntries.emplace_back(context.getId(), std::make_unique<T>()).second;
  }

  /** the state of `context`, or nullptr when it has none */
  T *find(Context const &context) const {
    std::scoped_lock lock(mMutex);
    for (auto &[id, state] : mEntries) {
      if (id == context.getId()) {
        return state.get();
      }
    }
    return nullptr;
  }

  void erase(Context const &context) {
    std::scoped_lock lock(mMutex);
    std::erase_if(mEntries, [&](auto const &entry) { return entry.first == context.getId(); });
  }

  void clear() {
    std::scoped_lock lock(mMutex);
    mEntries.clear();
  }

  /** call f(T &) for every entry */
  template <typename F> void forEach(F &&f) {
    std::scoped_lock lock(mMutex);
    for (auto &entry : mEntries) {
      f(*entry.second);
    }
  }

private:
  mutable std::mutex mMutex;
  std::vector<std::pair<uint64_t, std::unique_ptr<T>>> mEntries;
};

} // namespace core
} // namespace svulkan2
