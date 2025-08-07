# ---------------------------------------------------------------------------
# backend/CMakeLists.txt
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.18)

# If the parent CMakeLists.txt already defined TRITON_ROOT, reuse it;
# otherwise fall back to the default path used in NVIDIA’s containers.
if (NOT DEFINED TRITON_ROOT)
  set(TRITON_ROOT /opt/tritonserver)
endif()

# ---------------------------------------------------------------------------
# Enable CUDA and C++
# ---------------------------------------------------------------------------
project(custom_backend LANGUAGES CXX CUDA)
enable_language(CUDA)              # ensures nvcc is available

# ---------------------------------------------------------------------------
# Include / link paths for Triton headers and shared libs
# ---------------------------------------------------------------------------
include_directories(${TRITON_ROOT}/include)
link_directories(${TRITON_ROOT}/lib)

# ---------------------------------------------------------------------------
# Source files for the backend
# ---------------------------------------------------------------------------
set(BACKEND_SRCS
   src/backend.cpp
   src/kernel.cu)

add_library(custom_backend SHARED ${BACKEND_SRCS})

# ---------------------------------------------------------------------------
# Build properties
# ---------------------------------------------------------------------------
set_target_properties(custom_backend PROPERTIES
   CXX_STANDARD 17
   POSITION_INDEPENDENT_CODE ON
   CUDA_SEPARABLE_COMPILATION ON)

# Expose GPU helpers in tritonbackend.h
target_compile_definitions(custom_backend
   PRIVATE TRITON_ENABLE_GPU=1)

# Optional: a little extra perf from fast-math
target_compile_options(custom_backend PRIVATE
   $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcompiler=-fPIC>)

# ---------------------------------------------------------------------------
# Link against Triton server and the CUDA runtime
# ---------------------------------------------------------------------------
find_library(TRITONSERVER_LIB tritonserver
   HINTS ${TRITON_ROOT}/lib
   REQUIRED)

target_link_libraries(custom_backend
   PUBLIC
      ${TRITONSERVER_LIB}
      cuda_runtime)               # libcudart.so


# cmake_minimum_required(VERSION 3.18)
# project(custom_backend LANGUAGES CXX CUDA)
# set(CMAKE_CXX_STANDARD 17)

# find_package(CUDAToolkit REQUIRED)

# add_library(custom_backend SHARED src/backend.cpp src/kernel.cu)

# set_target_properties(custom_backend PROPERTIES
#    OUTPUT_NAME triton_custom_backend
#    CUDA_ARCHITECTURES "86"
#    CUDA_SEPARABLE_COMPILATION ON
# )

# target_include_directories(custom_backend PRIVATE
#    ${TRITON_SERVER_DIR}/include
#    ${TRITON_SERVER_DIR}/backends)

# target_link_libraries(custom_backend PRIVATE
#    CUDA::cudart
#    ${TRITON_SERVER_DIR}/lib/stubs/libtritonserver.so
# )

# set_target_properties(custom_backend PROPERTIES
#     BUILD_RPATH "$ORIGIN"
# )
