#include "taxiout/csv_io.hpp"

#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace taxiout {
namespace {

std::vector<std::string> split(const std::string& line, char delimiter) {
  std::vector<std::string> fields;
  std::string field;
  std::istringstream stream(line);
  while (std::getline(stream, field, delimiter)) fields.push_back(field);
  return fields;
}

[[noreturn]] void fail(std::size_t line_number, const std::string& reason) {
  throw std::runtime_error("movements CSV line " + std::to_string(line_number) + ": " + reason);
}

}  // namespace

std::vector<Movement> read_movements_csv(std::istream& input) {
  std::vector<Movement> movements;
  std::string line;
  std::size_t line_number = 0;

  if (!std::getline(input, line)) return movements;  // empty file, no header
  ++line_number;

  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty()) continue;

    const std::vector<std::string> fields = split(line, ',');
    if (fields.size() != 7) fail(line_number, "expected 7 fields, got " + std::to_string(fields.size()));

    Movement movement;
    try {
      movement.id = std::stoull(fields[0]);
      movement.off_block_time = std::stod(fields[4]);
      movement.unimpeded_taxi_sec = std::stod(fields[5]);
      movement.landing_time = std::stod(fields[6]);
    } catch (const std::exception&) {
      fail(line_number, "unparseable numeric field");
    }

    if (fields[1] == "DEP") {
      movement.phase = Phase::Departure;
    } else if (fields[1] == "ARR") {
      movement.phase = Phase::Arrival;
    } else {
      fail(line_number, "phase must be DEP or ARR, got '" + fields[1] + "'");
    }

    const std::optional<WakeCategory> wake = parse_wake_category(fields[2]);
    if (!wake) fail(line_number, "unknown wake category '" + fields[2] + "'");
    movement.wake = *wake;

    movement.runway = fields[3];
    movements.push_back(std::move(movement));
  }

  return movements;
}

void write_predictions_csv(std::ostream& output, const std::vector<Prediction>& predictions) {
  // Default stream precision is 6 significant digits, which silently rounds a
  // Unix-epoch takeoff_time (~1.8e9) to the nearest few hundred seconds.
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  output << "id,takeoff_time,taxi_out_sec,queue_delay_sec\n";
  for (const Prediction& prediction : predictions) {
    output << prediction.id << ',' << prediction.takeoff_time << ',' << prediction.taxi_out_sec << ','
           << prediction.queue_delay_sec << '\n';
  }
}

}  // namespace taxiout
