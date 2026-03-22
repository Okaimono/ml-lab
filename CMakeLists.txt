cmake_minimum_required(VERSION 3.10)
project(ml-lab)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O2")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(main ${SOURCES})
target_include_directories(main PRIVATE src)