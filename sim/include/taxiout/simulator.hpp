#pragma once

#include <vector>

#include "taxiout/movement.hpp"
#include "taxiout/separation.hpp"

namespace taxiout {

struct SimulatorConfig {
  SeparationMatrix separation = SeparationMatrix::icao_defaults();

  // How long a landing keeps the runway unusable for a departure. At airports
  // where arrivals cross the departure runway rather than share it, this is
  // the crossing block rather than the landing roll.
  double arrival_occupancy_sec = 50.0;

  // Runway occupancy of the departure roll itself: a floor on the headway
  // between successive departures regardless of wake category.
  double departure_occupancy_sec = 40.0;
};

// Event-driven model of the departure surface at one airport.
//
// Aircraft push back, spend unimpeded_taxi_sec crossing the apron, then join a
// FIFO queue at their runway. The runway serves departures subject to wake
// separation, and arrivals preempt departures when they land.
//
// Why simulate rather than regress directly: on the ranking set the takeoff
// times are blanked, so surface-congestion features cannot be measured -- they
// depend on the very quantity being predicted. Running every movement forward
// through this model reconstructs them self-consistently.
class Simulator {
 public:
  explicit Simulator(SimulatorConfig config = {});

  // Movements may be supplied in any order; they are sorted internally.
  // Returns one prediction per departure, in takeoff order.
  std::vector<Prediction> run(std::vector<Movement> movements) const;

 private:
  SimulatorConfig config_;
};

}  // namespace taxiout
