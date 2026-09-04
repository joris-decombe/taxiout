#pragma once

#include <istream>
#include <ostream>
#include <vector>

#include "taxiout/movement.hpp"

namespace taxiout {

// Reads movements from the flat CSV the Python pipeline exports. Header:
//   id,phase,wake,runway,off_block_time,unimpeded_taxi_sec,landing_time
// phase is DEP or ARR; wake is L/M/H/J; times are seconds since any epoch, as
// long as the same epoch is used throughout one run.
//
// Throws std::runtime_error on a malformed row -- silently skipping bad rows
// would quietly change the congestion picture the simulator depends on.
std::vector<Movement> read_movements_csv(std::istream& input);

// Writes: id,takeoff_time,taxi_out_sec,queue_delay_sec
void write_predictions_csv(std::ostream& output, const std::vector<Prediction>& predictions);

}  // namespace taxiout
