#include <exception>
#include <fstream>
#include <iostream>

#include "taxiout/csv_io.hpp"
#include "taxiout/simulator.hpp"

// Usage: taxiout_sim <movements.csv> <predictions.csv>
//
// One airport per invocation: the surface model is only meaningful within a
// single aerodrome, and running them separately parallelises for free.
int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " <movements.csv> <predictions.csv>\n";
    return 2;
  }

  try {
    std::ifstream input(argv[1]);
    if (!input) {
      std::cerr << "cannot open " << argv[1] << '\n';
      return 1;
    }

    const std::vector<taxiout::Movement> movements = taxiout::read_movements_csv(input);

    const taxiout::Simulator simulator;
    const std::vector<taxiout::Prediction> predictions = simulator.run(movements);

    std::ofstream output(argv[2]);
    if (!output) {
      std::cerr << "cannot write " << argv[2] << '\n';
      return 1;
    }
    taxiout::write_predictions_csv(output, predictions);

    std::cerr << "simulated " << movements.size() << " movements -> " << predictions.size()
              << " departures\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
