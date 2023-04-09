cmake_minimum_required(VERSION 3.16)

project(Example CXX)

set (CMAKE_CXX_STANDARD 17)


# message(STATUS "BOOOOO: ${CMAKE_SOURCE_DIR}")


# # add_subdirectory(kokkos)
# # include_directories(${Kokkos_INCLUDE_DIRS_RET})

# add_subdirectory(dep/ArborX)


# add_executable(example program.cpp)
# #target_link_libraries(example Kokkos::kokkos)
# target_link_libraries(example ArborX::ArborX)

# spack install trilinos +cuda cuda_arch=70 cxxstd=17 +debug +exodus +stk +mpi +openmp +nox +zoltan2 +gtest

include(FetchContent)
FetchContent_Declare(
  Kokkos
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dep/Kokkos
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(Kokkos)

FetchContent_Declare(
  ArborX
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dep/ArborX
)
FetchContent_MakeAvailable(ArborX)

add_executable(example program.cpp)
target_link_libraries(example ArborX::ArborX)