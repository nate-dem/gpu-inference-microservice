cmake_minimum_required(VERSION 3.18)

# project uses C++ and and CUDA
project(custom_backend LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# lets Triton find the shared library
add_library(custom_backend SHARED
  src/backend.cpp
  src/kernel.cu
)

# connects backend to CUDA runtime
target_link_libraries(custom_backend PRIVATE CUDAToolkit::cudart)

# lets Triton find the necessary functions
set_target_properties(custom_backend PROPERTIES
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  CUDA_VISIBILITY_PRESET hidden
)
