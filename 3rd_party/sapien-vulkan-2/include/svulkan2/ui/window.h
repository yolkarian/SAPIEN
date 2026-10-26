#pragma once
#include "svulkan2/common/glm.h"
#include "widget.h"

namespace svulkan2 {
namespace ui {

UI_CLASS(Window) {
  UI_DECLARE_APPEND(Window);
  UI_DECLARE_LABEL(Window);
  UI_ATTRIBUTE(Window, glm::vec2, Pos);
  UI_ATTRIBUTE(Window, glm::vec2, Size);

public:
  bool isExpanded() const { return mExpanded; }
  void build() override;

private:
  bool mExpanded{true};
};

} // namespace ui
} // namespace svulkan2
