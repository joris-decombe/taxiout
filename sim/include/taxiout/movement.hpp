#pragma once

#include <cstdint>
#include <string>

#include "taxiout/wake.hpp"

namespace taxiout {

enum class Phase : std::uint8_t { Departure, Arrival };

// One airport movement, already resolved to the fields the surface model
// needs. All times are seconds since an arbitrary but consistent epoch.
struct Movement {
  std::uint64_t id = 0;
  Phase phase = Phase::Departure;
  WakeCategory wake = WakeCategory::Medium;
  std::string runway;

  // Departures: actual off-block time (AOBT) and the queue-free stand ->
  // runway transit estimated from the stand/runway pair.
  double off_block_time = 0.0;
  double unimpeded_taxi_sec = 0.0;

  // Arrivals: touchdown time. Unused for departures.
  double landing_time = 0.0;

  // When a departure reaches the runway and takes its place in the queue.
  double queue_entry_time() const { return off_block_time + unimpeded_taxi_sec; }
};

// What the simulator predicts for one departure.
struct Prediction {
  std::uint64_t id = 0;
  double takeoff_time = 0.0;
  double taxi_out_sec = 0.0;     // takeoff_time - off_block_time
  double queue_delay_sec = 0.0;  // taxi_out_sec - unimpeded_taxi_sec
};

}  // namespace taxiout
