# min version of CMake needed and main project
cmake_minimum_required(VERSION 3.18)
project(gpu_inference_microservice LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(TRITON_SERVER_DIR $ENV{TRITON_SERVER_DIR})
set(TRITON_CLIENT_DIR $ENV{TRITON_CLIENT_DIR})

add_subdirectory(backend)
add_subdirectory(client)
