cmake_minimum_required(VERSION 3.16)

project(Example CXX)

set (CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  ArborX
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/packages/ArborX
)
FetchContent_MakeAvailable(ArborX)

add_executable(example program.cpp)
target_link_libraries(example PUBLIC ArborX::ArborX)


set(CMAKE_PREFIX_PATH ${Trilinos_DIR} ${CMAKE_PREFIX_PATH})
find_package(Trilinos QUIET REQUIRED)
message(STATUS "smundy: Found Trilinos = ${Trilinos_LIBRARY_DIRS}")
message(STATUS "Smundy-Trilinos_LIBRARIES: ${Trilinos_LIBRARIES}")
message(STATUS "Smundy-Trilinos_INCLUDE_DIRS: ${Trilinos_INCLUDE_DIRS}")
message(STATUS "Smundy-Trilinos_TPL_INCLUDE_DIRS: ${Trilinos_TPL_INCLUDE_DIRS}")

target_link_libraries(example PUBLIC ${Trilinos_LIBRARIES})
target_include_directories(example SYSTEM PUBLIC ${Trilinos_INCLUDE_DIRS})
target_include_directories(example SYSTEM PUBLIC ${Trilinos_TPL_INCLUDE_DIRS})