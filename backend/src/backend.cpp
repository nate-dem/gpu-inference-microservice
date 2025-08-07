#define TRITON_ENABLE_GPU 1
#include <iostream>
#include <cuda_runtime.h>
#include "triton/core/tritonbackend.h"

extern "C" void launch_kernel(const float* input, float* output, int size, cudaStream_t stream);

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests, const uint32_t request_count) {
    cudaStream_t stream = 0;
     
    int32_t device_id = 0;
    TRITONSERVER_Error* error = TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id);
    
    if (error != nullptr) {
	return error;
    }	

    for (uint32_t r = 0; r < request_count; ++r) {
        TRITONBACKEND_Request* request = requests[r];
        
	TRITONBACKEND_Input* input_tensor;
	error = TRITONBACKEND_RequestInputByIndex(request, 0, &input_tensor);
		
	if (error != nullptr) {
	    return error;
	}

	// retrieve properties of input tensor to create output tensor with same properties
	const int64_t* input_shape;
	uint32_t input_dims_count;
	TRITONSERVER_DataType input_data_type;
	uint64_t input_bytes;

	error = TRITONBACKEND_InputProperties(
	    input_tensor,
	    nullptr, 
	    &input_data_type,
	    &input_shape, 
	    &input_dims_count, 
	    &input_bytes, 
	    nullptr
	);

	if (error != nullptr) {
	    return error;
	}

	// need pointer to access input data buffer on the CPU
	const void* input_buffer;
	uint32_t idx = 0;
	TRITONSERVER_MemoryType input_memory_type;
	int64_t input_type_id;

	error = TRITONBACKEND_InputBuffer(
	    input_tensor, 
	    idx, 
	    &input_buffer, 
	    &input_bytes, 
	    &input_memory_type, 
	    &input_type_id
	);

	if (error != nullptr) {
	    return error;
	}

	void* device_input = const_cast<void*>(input_buffer);
	bool is_alloc = false;

	if (input_memory_type != TRITONSERVER_MEMORY_GPU) {
	    cudaMallocAsync(&device_input, input_bytes, stream);
	    cudaMemcpyAsync(device_input, input_buffer, input_bytes, cudaMemcpyDefault, stream);
	    is_alloc = true;
	}

	// creates response and output tensor then adds the tensor to the response	
	TRITONBACKEND_Response* response;
	TRITONBACKEND_ResponseNew(&response, request);
	TRITONBACKEND_Output* output_tensor;
	
	error = TRITONBACKEND_ResponseOutput(
	    response,
	    &output_tensor,
	    "OUTPUT0",
	    input_data_type,
	    input_shape,
	    input_dims_count
	); 
				
	if (error != nullptr) {
	    return error;
	}

	// allocate memory buffer for output tensor which will be passed into CUDA kernel
	void* output_buffer;
	TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_GPU;
	int64_t output_memory_id = static_cast<int64_t>(device_id);

	error = TRITONBACKEND_OutputBuffer(
	    output_tensor,
	    &output_buffer,
	    input_bytes,
	    &output_memory_type,
	    &output_memory_id
	);

	if (error != nullptr) {
	    return error;
	}

	size_t num_elems = input_bytes / sizeof(float);
	launch_kernel(
	    static_cast<const float*>(device_input),
	    static_cast<float*>(output_buffer),
	    static_cast<int>(num_elems),
	    stream
	);

	if (is_alloc) {
	    cudaFreeAsync(device_input, stream);
        }
	
	TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr);
    	TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL);
    }

    return nullptr;
}

// lifestyle functions for Triton
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend) {
    std::cout << "Custom backend initialized." << std::endl;
    return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend) {
    std::cout << "Custom backend finalized." << std::endl;
    return nullptr; 
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model) {
    std::cout << "Custom model initialized." << std::endl;
    return nullptr; 
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) {
    std::cout << "Custom model finalized." << std::endl;
    return nullptr; 
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance) {
    std::cout << "Custom model instance initialized." << std::endl;
    return nullptr; 
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance) {
    std::cout << "Custom model instance finalized." << std::endl;
    return nullptr;
} 
