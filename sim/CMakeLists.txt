cmake_minimum_required(VERSION 3.20)
project(taxiout_sim CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(taxiout_core
  src/wake.cpp
  src/separation.cpp
  src/simulator.cpp
  src/csv_io.cpp
)
target_include_directories(taxiout_core PUBLIC include)

add_executable(taxiout_sim src/main.cpp)
target_link_libraries(taxiout_sim PRIVATE taxiout_core)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
