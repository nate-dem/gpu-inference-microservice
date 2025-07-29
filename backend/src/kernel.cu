#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx];
    }
}

extern "C" void launch_kernel(const float* input, float* output, size_t size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;    

    float* d_in = nullptr;
    float* d_out = nullptr;
    size_t bytes = size * sizeof(float);

    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, bytes);

    cudaMemcpy(d_in, input, bytes, cudaMemcpyHostToDevice);

    kernel<<<gridSize,blockSize>>>(d_in, d_out, size);

    cudaMemcpy(output, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);      
}
