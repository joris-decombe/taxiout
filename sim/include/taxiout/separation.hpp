#pragma once

#include <array>

#include "taxiout/wake.hpp"

namespace taxiout {

// Minimum time between the starts of two consecutive departure rolls on the
// same runway, indexed by the wake category of the leader and the follower.
//
// The defaults are the usual ICAO wake minima converted to time at typical
// European hub speeds. They are a calibration starting point, not ground
// truth -- fit them per airport against 2025 data before trusting them.
class SeparationMatrix {
 public:
  static SeparationMatrix icao_defaults();

  double seconds(WakeCategory leader, WakeCategory follower) const;
  void set(WakeCategory leader, WakeCategory follower, double seconds);

 private:
  static std::size_t index(WakeCategory leader, WakeCategory follower);

  std::array<double, kWakeCategoryCount * kWakeCategoryCount> seconds_{};
};

}  // namespace taxiout
