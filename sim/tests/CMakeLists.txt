include(FetchContent)

# Pinned so a CI run cannot pick up a different doctest than a local one.
FetchContent_Declare(doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.11
)
FetchContent_MakeAvailable(doctest)

add_executable(taxiout_tests
  test_separation.cpp
  test_simulator.cpp
)
target_link_libraries(taxiout_tests PRIVATE taxiout_core doctest::doctest)
target_compile_definitions(taxiout_tests PRIVATE DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN)

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(taxiout_tests)
