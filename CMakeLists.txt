# min version of CMake needed
cmake_minimum_required(VERSION 3.18)

# main project
project(gpu_inference_microservice)

# subdirectories that contain other CMakeLists.txt files
add_subdirectory(backend)
add_subdirectory(client)
