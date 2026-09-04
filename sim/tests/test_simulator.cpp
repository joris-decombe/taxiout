#include <doctest/doctest.h>

#include <vector>

#include "taxiout/simulator.hpp"

using taxiout::Movement;
using taxiout::Phase;
using taxiout::Prediction;
using taxiout::Simulator;
using taxiout::SimulatorConfig;
using W = taxiout::WakeCategory;

namespace {

Movement departure(std::uint64_t id, double off_block, double unimpeded = 300.0,
                   W wake = W::Medium, std::string runway = "09L") {
  Movement movement;
  movement.id = id;
  movement.phase = Phase::Departure;
  movement.wake = wake;
  movement.runway = std::move(runway);
  movement.off_block_time = off_block;
  movement.unimpeded_taxi_sec = unimpeded;
  return movement;
}

Movement arrival(std::uint64_t id, double landing, std::string runway = "09L") {
  Movement movement;
  movement.id = id;
  movement.phase = Phase::Arrival;
  movement.runway = std::move(runway);
  movement.landing_time = landing;
  return movement;
}

}  // namespace

TEST_CASE("an uncongested departure taxis out unimpeded") {
  const std::vector<Prediction> predictions = Simulator().run({departure(1, 0.0, 300.0)});

  REQUIRE(predictions.size() == 1);
  CHECK(predictions[0].taxi_out_sec == doctest::Approx(300.0));
  CHECK(predictions[0].queue_delay_sec == doctest::Approx(0.0));
}

TEST_CASE("a queue behind a heavy costs the follower its separation") {
  // Both reach the runway at t=300; the medium must wait out the heavy's wake.
  const std::vector<Prediction> predictions =
      Simulator().run({departure(1, 0.0, 300.0, W::Heavy), departure(2, 0.0, 300.0, W::Medium)});

  REQUIRE(predictions.size() == 2);
  CHECK(predictions[0].id == 1);
  CHECK(predictions[0].queue_delay_sec == doctest::Approx(0.0));
  // Heavy -> Medium is 120s, which dominates the 40s occupancy floor.
  CHECK(predictions[1].queue_delay_sec == doctest::Approx(120.0));
}

TEST_CASE("only the runway occupancy floor applies between equals") {
  const std::vector<Prediction> predictions =
      Simulator().run({departure(1, 0.0, 300.0, W::Light), departure(2, 0.0, 300.0, W::Light)});

  REQUIRE(predictions.size() == 2);
  // Light -> Light separation is 60s, above the 40s occupancy floor.
  CHECK(predictions[1].queue_delay_sec == doctest::Approx(60.0));
}

TEST_CASE("an arrival preempts the departure queue") {
  SimulatorConfig config;
  config.arrival_occupancy_sec = 50.0;

  const std::vector<Prediction> predictions =
      Simulator(config).run({arrival(99, 290.0), departure(1, 0.0, 300.0)});

  REQUIRE(predictions.size() == 1);
  // Runway free at 290+50=340; the departure was ready at 300.
  CHECK(predictions[0].takeoff_time == doctest::Approx(340.0));
  CHECK(predictions[0].queue_delay_sec == doctest::Approx(40.0));
}

TEST_CASE("independent runways do not interact") {
  const std::vector<Prediction> predictions =
      Simulator().run({departure(1, 0.0, 300.0, W::Super, "09L"),
                       departure(2, 0.0, 300.0, W::Light, "27R")});

  REQUIRE(predictions.size() == 2);
  for (const Prediction& prediction : predictions) {
    CHECK(prediction.queue_delay_sec == doctest::Approx(0.0));
  }
}

TEST_CASE("input order does not change the outcome") {
  const std::vector<Movement> forward = {departure(1, 0.0), departure(2, 60.0), departure(3, 120.0)};
  std::vector<Movement> reversed(forward.rbegin(), forward.rend());

  const std::vector<Prediction> a = Simulator().run(forward);
  const std::vector<Prediction> b = Simulator().run(reversed);

  REQUIRE(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    CHECK(a[i].id == b[i].id);
    CHECK(a[i].takeoff_time == doctest::Approx(b[i].takeoff_time));
  }
}

TEST_CASE("congestion accumulates across a departure bank") {
  // Twenty mediums all ready at once: the queue drains at the 60s headway, so
  // the last one waits 19 slots. This is the saturation behaviour the model
  // exists to reproduce.
  std::vector<Movement> bank;
  for (std::uint64_t i = 0; i < 20; ++i) bank.push_back(departure(i, 0.0, 300.0));

  const std::vector<Prediction> predictions = Simulator().run(bank);

  REQUIRE(predictions.size() == 20);
  CHECK(predictions.back().queue_delay_sec == doctest::Approx(19 * 60.0));
  CHECK(predictions.back().taxi_out_sec == doctest::Approx(300.0 + 19 * 60.0));
}
