#include <doctest/doctest.h>

#include "taxiout/separation.hpp"

using taxiout::SeparationMatrix;
using W = taxiout::WakeCategory;

TEST_CASE("separation grows as the leader gets heavier") {
  const SeparationMatrix matrix = SeparationMatrix::icao_defaults();

  CHECK(matrix.seconds(W::Heavy, W::Light) > matrix.seconds(W::Medium, W::Light));
  CHECK(matrix.seconds(W::Super, W::Light) > matrix.seconds(W::Heavy, W::Light));
}

TEST_CASE("separation shrinks as the follower gets heavier") {
  const SeparationMatrix matrix = SeparationMatrix::icao_defaults();

  CHECK(matrix.seconds(W::Heavy, W::Light) > matrix.seconds(W::Heavy, W::Medium));
  CHECK(matrix.seconds(W::Heavy, W::Medium) > matrix.seconds(W::Heavy, W::Heavy));
}

TEST_CASE("a light leader never imposes more than the floor") {
  const SeparationMatrix matrix = SeparationMatrix::icao_defaults();

  for (const W follower : {W::Light, W::Medium, W::Heavy, W::Super}) {
    CHECK(matrix.seconds(W::Light, follower) == doctest::Approx(60.0));
  }
}

TEST_CASE("calibrated values round-trip") {
  SeparationMatrix matrix = SeparationMatrix::icao_defaults();
  matrix.set(W::Medium, W::Medium, 73.5);

  CHECK(matrix.seconds(W::Medium, W::Medium) == doctest::Approx(73.5));
  // Setting one cell must not disturb its transpose.
  CHECK(matrix.seconds(W::Medium, W::Heavy) == doctest::Approx(60.0));
}
