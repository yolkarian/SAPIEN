#pragma once
#include "widget.h"
#include <functional>

namespace svulkan2 {
namespace ui {

UI_CLASS(Button) {
  UI_DECLARE_LABEL(Button);
  UI_ATTRIBUTE(Button, std::function<void(std::shared_ptr<Button>)>, Callback);
  UI_ATTRIBUTE(Button, float, Width);

public:
  void build() override;
};

} // namespace ui
} // namespace svulkan2
