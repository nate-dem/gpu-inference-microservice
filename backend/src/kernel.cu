#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx];
    }
}

extern "C" void launch_kernel(const float* input, float* output, size_t size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    kernel<<<blocks, threads, 0, stream>>>(input, output, size);
}
