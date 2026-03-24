cmake_minimum_required(VERSION 3.18)
project(ml-lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O2 -Wno-unused-result")
set(CMAKE_CUDA_ARCHITECTURES 86)

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

add_executable(main ${SOURCES})
target_include_directories(main PRIVATE src)