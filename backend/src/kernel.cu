#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx];
    }
}

# called in backend.cpp
extern "C" void launch_kernel(const float* input, float* output, int size) {
    std::cout << "Launching CUDA kernel..." << std::endl;
}
