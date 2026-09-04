#include "taxiout/wake.hpp"

namespace taxiout {

std::optional<WakeCategory> parse_wake_category(std::string_view code) {
  if (code.size() != 1) return std::nullopt;
  switch (code[0]) {
    case 'L': case 'l': return WakeCategory::Light;
    case 'M': case 'm': return WakeCategory::Medium;
    case 'H': case 'h': return WakeCategory::Heavy;
    case 'J': case 'j': return WakeCategory::Super;
    default: return std::nullopt;
  }
}

std::string_view to_string(WakeCategory category) {
  switch (category) {
    case WakeCategory::Light: return "L";
    case WakeCategory::Medium: return "M";
    case WakeCategory::Heavy: return "H";
    case WakeCategory::Super: return "J";
  }
  return "?";
}

}  // namespace taxiout
