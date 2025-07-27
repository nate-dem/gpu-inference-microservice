# project uses C++ and and CUDA
project(custom_backend LANGUAGES CXX CUDA)

# lets Triton find the shared library
add_library(custom_backend SHARED
  src/backend.cpp
  src/kernel.cu
)

# lets Triton find the necessary functions
set_target_properties(custom_backend PROPERTIES
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  CUDA_VISIBILITY_PRESET hidden
)
