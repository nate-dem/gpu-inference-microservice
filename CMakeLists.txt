# min version of CMake needed
cmake_minimum_required(VERSION 3.18)

# main project
project(gpu_inference_microservice)

set(Triton_DIR "/opt/tritonserver/lib/cmake/Triton")
find_package(Triton REQUIRED)

# subdirectories that contain other CMakeLists.txt files
add_subdirectory(backend)
add_subdirectory(client)

target_link_libraries(custom_backend PRIVATE Triton::tritonbackend)
target_link_libraries(triton_client PRIVATE Triton::grpcclient)
