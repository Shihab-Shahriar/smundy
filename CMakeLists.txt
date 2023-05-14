cmake_minimum_required(VERSION 3.16)

project(Example CXX)

set (CMAKE_CXX_STANDARD 17)

message(STATUS "Eign: ${Eigen3_DIR}")


include(FetchContent)
FetchContent_Declare(
  ArborX
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/packages/ArborX
)
FetchContent_MakeAvailable(ArborX)

## Tests

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

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




message(STATUS "Fetched & declared gtest")

enable_testing()

add_executable(
  hello_test
  tests.cpp
)
target_link_libraries(
  hello_test
  GTest::gtest_main
)

target_link_libraries(
  hello_test
  ArborX::ArborX
)

target_link_libraries(hello_test  ${Trilinos_LIBRARIES})
target_include_directories(hello_test SYSTEM PUBLIC ${Trilinos_INCLUDE_DIRS})
target_include_directories(hello_test SYSTEM PUBLIC ${Trilinos_TPL_INCLUDE_DIRS})

include(GoogleTest)
gtest_discover_tests(hello_test)




find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries (example PUBLIC Eigen3::Eigen)
target_link_libraries (hello_test Eigen3::Eigen)
