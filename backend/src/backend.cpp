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

        TRITONBACKEND_Response* response;
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

extern "C" TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance) {
    std::cout << "Custom model instance finalized." << std::endl;
    return nullptr;
} 
