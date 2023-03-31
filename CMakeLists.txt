cmake_minimum_required(VERSION 3.23)

project(Example CXX)


message(STATUS "BOOOOO: ${CMAKE_SOURCE_DIR}")


add_subdirectory(kokkos)
include_directories(${Kokkos_INCLUDE_DIRS_RET})


add_executable(example program.cpp)
target_link_libraries(example Kokkos::kokkos)

# spack install trilinos +cuda cuda_arch=70 cxxstd=17 +debug +exodus +stk +mpi +openmp +nox +zoltan2 +gtest
