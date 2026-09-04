#include "taxiout/simulator.hpp"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace taxiout {
namespace {

constexpr double kNegativeInfinity = -std::numeric_limits<double>::infinity();

// The state of one runway as the event timeline advances.
struct RunwayState {
  double free_at = kNegativeInfinity;
  std::optional<WakeCategory> last_departure_wake;
  double last_departure_start = kNegativeInfinity;
};

// The moment a movement enters the timeline: queue arrival for departures,
// touchdown for arrivals.
double event_time(const Movement& movement) {
  return movement.phase == Phase::Departure ? movement.queue_entry_time() : movement.landing_time;
}

}  // namespace

Simulator::Simulator(SimulatorConfig config) : config_(std::move(config)) {}

std::vector<Prediction> Simulator::run(std::vector<Movement> movements) const {
  // Single chronological pass over the merged arrival/departure stream. Sorting
  // is the event queue: nothing scheduled during the pass can land earlier than
  // the event being processed.
  std::sort(movements.begin(), movements.end(), [](const Movement& a, const Movement& b) {
    const double ta = event_time(a);
    const double tb = event_time(b);
    if (ta != tb) return ta < tb;
    return a.id < b.id;  // deterministic tie-break
  });

  std::unordered_map<std::string, RunwayState> runways;
  std::vector<Prediction> predictions;
  predictions.reserve(movements.size());

  for (const Movement& movement : movements) {
    RunwayState& runway = runways[movement.runway];

    if (movement.phase == Phase::Arrival) {
      // Arrivals have priority: a landing simply pushes back whatever
      // departure would otherwise have gone next.
      runway.free_at = std::max(runway.free_at, movement.landing_time) + config_.arrival_occupancy_sec;
      continue;
    }

    double start = std::max(movement.queue_entry_time(), runway.free_at);
    if (runway.last_departure_wake) {
      const double separated = runway.last_departure_start +
                               config_.separation.seconds(*runway.last_departure_wake, movement.wake);
      start = std::max(start, separated);
    }

    runway.free_at = start + config_.departure_occupancy_sec;
    runway.last_departure_wake = movement.wake;
    runway.last_departure_start = start;

    Prediction prediction;
    prediction.id = movement.id;
    prediction.takeoff_time = start;
    prediction.taxi_out_sec = start - movement.off_block_time;
    prediction.queue_delay_sec = prediction.taxi_out_sec - movement.unimpeded_taxi_sec;
    predictions.push_back(prediction);
  }

  return predictions;
}

}  // namespace taxiout
