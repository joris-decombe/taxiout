#include "taxiout/separation.hpp"

namespace taxiout {

std::size_t SeparationMatrix::index(WakeCategory leader, WakeCategory follower) {
  return static_cast<std::size_t>(leader) * kWakeCategoryCount + static_cast<std::size_t>(follower);
}

double SeparationMatrix::seconds(WakeCategory leader, WakeCategory follower) const {
  return seconds_[index(leader, follower)];
}

void SeparationMatrix::set(WakeCategory leader, WakeCategory follower, double seconds) {
  seconds_[index(leader, follower)] = seconds;
}

SeparationMatrix SeparationMatrix::icao_defaults() {
  using W = WakeCategory;
  SeparationMatrix matrix;

  // A heavier leader imposes a longer wait on a lighter follower; a light
  // leader imposes only the runway-occupancy floor on anyone.
  const double table[kWakeCategoryCount][kWakeCategoryCount] = {
      // follower:  L      M      H      J
      /* L */     { 60.0,  60.0,  60.0,  60.0},
      /* M */     {120.0,  60.0,  60.0,  60.0},
      /* H */     {180.0, 120.0,  90.0,  60.0},
      /* J */     {240.0, 180.0, 120.0,  90.0},
  };

  for (std::size_t leader = 0; leader < kWakeCategoryCount; ++leader) {
    for (std::size_t follower = 0; follower < kWakeCategoryCount; ++follower) {
      matrix.set(static_cast<W>(leader), static_cast<W>(follower), table[leader][follower]);
    }
  }
  return matrix;
}

}  // namespace taxiout
