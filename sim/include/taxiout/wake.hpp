#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace taxiout {

// ICAO wake turbulence categories, ordered light -> heavy so that the enum
// value can index the separation matrix directly.
enum class WakeCategory : std::uint8_t { Light = 0, Medium = 1, Heavy = 2, Super = 3 };

inline constexpr std::size_t kWakeCategoryCount = 4;

// Parses the WK_TBL_CAT_flt field ("L", "M", "H", "J"). Returns nullopt for
// anything unrecognised so the caller can decide on a fallback.
std::optional<WakeCategory> parse_wake_category(std::string_view code);

std::string_view to_string(WakeCategory category);

}  // namespace taxiout
