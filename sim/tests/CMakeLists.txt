include(FetchContent)

# Pinned so a CI run cannot pick up a different doctest than a local one.
# 2.4.12 is the floor: 2.4.11 declares cmake_minimum_required(VERSION 3.0),
# which CMake 4.x refuses outright.
FetchContent_Declare(doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.12
)
FetchContent_MakeAvailable(doctest)

add_executable(taxiout_tests
  main.cpp
  test_separation.cpp
  test_simulator.cpp
)
target_link_libraries(taxiout_tests PRIVATE taxiout_core doctest::doctest)

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(taxiout_tests)
