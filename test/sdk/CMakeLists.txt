# A program outside this repository that uses SAPIEN's C++ SDK through CMake;
# scripts/test_sdk.sh configures it with CMAKE_PREFIX_PATH set to an unpacked SDK archive.
cmake_minimum_required(VERSION 3.18)
project(sapien_sdk_test LANGUAGES CXX)

find_package(sapien REQUIRED)

add_executable(sdk_test main.cpp)
# sapien::sapien brings the include directories, compile definitions and C++20
target_link_libraries(sdk_test PRIVATE sapien::sapien)
