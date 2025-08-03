cmake_minimum_required(VERSION 3.18)
project(custom_backend LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_library(custom_backend SHARED src/backend.cpp src/kernel.cu)

set_target_properties(custom_backend PROPERTIES
   OUTPUT_NAME triton_custom_backend
   CUDA_ARCHITECTURES "86"
)

target_include_directories(custom_backend PRIVATE
   ${TRITON_SERVER_DIR}/include
   ${TRITON_SERVER_DIR}/backends)

target_link_libraries(custom_backend PRIVATE
   CUDAToolkit::cudart)