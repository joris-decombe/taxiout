// The single translation unit that compiles doctest itself.
//
// DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN must be defined in exactly one file. As
// a target-wide compile definition it lands in every test .cpp, and the
// linker then sees main() and the whole doctest runtime defined twice.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
