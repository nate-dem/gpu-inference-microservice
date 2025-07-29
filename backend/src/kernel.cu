#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx];
    }
}

# called in backend.cpp
extern "C" void launch_kernel(const float* input, float* output, size_t size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;    

    float* deviceInput = nullptr;
    float * deviceOutput = nullptr;
    size_t bytes = size * sizeof(float);

    cudaMalloc((void**)&deviceInput, bytes);
    cudaMalloc((void**)&deviceOutput, bytes);

    cudaMemcpy(deviceInput, input, bytes, cudaMemcpyHostToDevice);

    kernel<<<gridSize,blockSize>>>(deviceInput, deviceOutput, size);

    cudaMemcpy(output, deviceOutput, bytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceInput);
    cudaFree(deviceOutput); 
     
}
