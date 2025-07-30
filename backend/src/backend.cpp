#include <iostream>
#include "triton/core/tritonbackend.h"

extern "C" void launch_kernel(const float* input, float* output, int size);
extern "C" TRITONSERVER_Error*

TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests, const uint32_t request_count) {
    for (uint32_t r = 0; r < request_count; ++r) {
        TRITONBACKEND_Request* request = requests[r];
        
	TRITONBACKEND_Input* input_tensor;
	TRITONSERVER_Error* error = TRITONBACKEND_RequestInputByIndex(request, 0, &input_tensor);
		
	if (error != nullptr) {
	    return error;
	}

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

	const void* input_buffer;
	uint32_t idx = 0;
	TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
	int64_t memory_type_id = 0;

	error = TRITONBACKEND_InputBuffer(
	    input_tensor, 
	    idx, 
	    &input_buffer, 
	    &input_bytes, 
	    &memory_type, 
	    &memory_type_id
	);

	if (error != nullptr) {
	    return error;
	}	

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

	void* output_buffer;

	error = TRITONBACKEND_OutputBuffer(
	    output_tensor,
	    &output_buffer,
	    input_bytes,
	    &memory_type,
	    &memory_type_id
	);

	if (error != nullptr) {
	    return error;
	}

	size_t num_elems = input_bytes / sizeof(float);

	launch_kernel(
	    static_cast<const float*>(input_buffer),
	    static_cast<float*>(output_buffer),
	    num_elems
	);	
	TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr);
    	TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL);
    }

    return nullptr;
}

// lifestyle functions for Triton
extern "C" TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend) {
    std::cout << "Custom backend initialized." << std::endl;
    return nullptr;
}

extern "C" TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend) {
    std::cout << "Custom backend finalized." << std::endl;
    return nullptr; 
}

extern "C" TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model) {
    std::cout << "Custom model initialized." << std::endl;
    return nullptr; 
}

extern "C" TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) {
    std::cout << "Custom model finalized." << std::endl;
    return nullptr; 
}

extern "C" TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance) {
    std::cout << "Custom model instance initialized." << std::endl;
    return nullptr; 
}

extern "C" TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance) {
    std::cout << "Custom model instance finalized." << std::endl;
    return nullptr;
} 
