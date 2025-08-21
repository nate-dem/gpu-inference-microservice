#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <string>
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

extern "C" void launch_kernel(const float* input, float* output, size_t size, cudaStream_t stream);

constexpr TRITONSERVER_DataType EXPECTED_DATA_TYPE = TRITONSERVER_TYPE_FP32;
constexpr const char* EXPECTED_DATA_TYPE_STR = "FP32";

struct InstanceState {
    int device_id = 0;
    cudaStream_t stream = nullptr;
};

static TRITONSERVER_Error* SafeElemCount(const int64_t* dims, uint32_t nd, size_t* out) {
  size_t n = 1;
  for (uint32_t i = 0; i < nd; ++i) {
    const int64_t d = dims[i];
    if (d < 0) {
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "negative runtime dimension");
    }
    const size_t dd = static_cast<size_t>(d);
    if (dd != 0 && n > (std::numeric_limits<size_t>::max() / dd)) {
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "shape size overflow");
    }
    n *= dd;
  }
  *out = n;
  return nullptr;
}

static size_t ByteSizeOf(TRITONSERVER_DataType t) {
  switch (t) {
    case TRITONSERVER_TYPE_FP32: return 4;
    case TRITONSERVER_TYPE_BF16: return 2;
    default: return 0;
  }
}

extern "C" TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance) {
    auto* state = new InstanceState();

    TRITONSERVER_InstanceGroupKind kind;

    // note: triton returns nullptr for success 
    if (auto* err = TRITONBACKEND_ModelInstanceKind(instance, &kind); err != nullptr) {
        delete state;
        return err;	
    }

    if (kind == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
		if (auto* err = TRITONBACKEND_ModelInstanceDeviceId(instance, &state->device_id); err != nullptr) {
			delete state;
			return err;
		}

		// cuda returns cudaError_t type
		if (auto cuda_err = cudaSetDevice(state->device_id); cuda_err != cudaSuccess) {
			delete state;
			return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(cuda_err));
		}

		if (auto cuda_err = cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking); cuda_err != cudaSuccess) {
			delete state;
			return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(cuda_err));
		}
    }

    if (auto* err = TRITONBACKEND_ModelInstanceSetState(instance, state); err != nullptr) {
        if (state->stream) {
	    	cudaStreamDestroy(state->stream);
		}
		delete state;
		return err;
    }

    return nullptr;
}

extern "C" TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance) {
    void* state = nullptr;
    if (auto* err = TRITONBACKEND_ModelInstanceState(instance, &state); err != nullptr) {
		return err;
    }
    auto* new_state = reinterpret_cast<InstanceState*>(state);
    if (new_state != nullptr) {
        if (new_state->stream) {
	    	cudaStreamDestroy(new_state->stream);
		}
		delete new_state;
    }
    return nullptr;
}

extern "C" TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests, const uint32_t request_count) {
    void* state = nullptr;
    if (auto* err = TRITONBACKEND_ModelInstanceState(instance, &state); err != nullptr) {
       	return err;
    }
    auto* new_state = reinterpret_cast<InstanceState*>(state);

    for (uint32_t i = 0; i < request_count; ++i) {
		TRITONBACKEND_Request* request = requests[i];
		TRITONBACKEND_Response* response = nullptr;

		if (auto* err = TRITONBACKEND_ResponseNew(&response, request); err != nullptr) {
			return err;
		}

		TRITONBACKEND_Input* input_tensor = nullptr;
		if (auto* err = TRITONBACKEND_RequestInputByIndex(request, 0, &input_tensor); err != nullptr) {
			return err;
		}

		const char* input_name = nullptr;
		TRITONSERVER_DataType data_type;
		const int64_t* input_shape = nullptr;
		uint32_t dim_count = 0;
		uint64_t input_size = 0; // size is returned in bytes
		uint32_t buffer_count = 0;

		if (auto* err = TRITONBACKEND_InputProperties(input_tensor, &input_name, &data_type, &input_shape, &dim_count, &input_size, &buffer_count); err != nullptr) {
			return err;
		}

		if (data_type != EXPECTED_DATA_TYPE) {
			std::string err_message = std::string("Expected ") + EXPECTED_DATA_TYPE_STR + " input";
			return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, err_message.c_str());
		}

		size_t elem_count = 0;
		if (auto* err = SafeElemCount(input_shape, dim_count, &elem_count); err != nullptr) {
			return err;
		}

		// custom sizeof function for switching between FP32, BF16, etc.
		const size_t bytes = elem_count * ByteSizeOf(data_type);

		TRITONBACKEND_Output* output_tensor = nullptr;
		if (auto* err = TRITONBACKEND_ResponseOutput(response, &output_tensor, "OUTPUT0", EXPECTED_DATA_TYPE, input_shape, dim_count); err != nullptr) {
			return err;
		}

		if (auto cuda_err = cudaSetDevice(new_state->device_id); cuda_err != cudaSuccess) {
			return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(cuda_err));
		}
		/* 
		* Triton may return input tensors divided up across multiple buffers and/or located on the host or a different 
		* device from the one we're using. Although my project is built for a single GPU, I'm including these checks now 
		* in case I want to expand to multiple GPUs later.
		*/
		const float* device_input = nullptr;
		float* device_input_owner = nullptr;
		size_t copied = 0;
		if (bytes != 0) {
		    for (uint32_t i = 0; i < buffer_count; ++i) {
				const void* buffer = nullptr;
				size_t buffer_size = 0;
				TRITONSERVER_MemoryType mem_type;
				int64_t mem_type_id;
				if (auto* err = TRITONBACKEND_InputBuffer(input_tensor, i, &buffer, &buffer_size, &mem_type, &mem_type_id); err != nullptr) {
					return err;
				}
				
				// buffer wasn't updated 
				if (!buffer || buffer_size == 0) {
					continue;
				}

				// check if memory is contiguous and on current device
				// if true, we can use the buffer ptr directly
				const bool can_alias = ((buffer_count == 1) && (mem_type == TRITONSERVER_MEMORY_GPU) && (mem_type_id == new_state->device_id));

				if (can_alias) {
					device_input = static_cast<const float*>(buffer);
					copied = buffer_size;
				} else {
					if (!device_input_owner) {
						if (auto cuda_err = cudaMallocAsync(&device_input_owner, bytes, new_state->stream); cuda_err != cudaSuccess) {
							return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(cuda_err));
						}
					}
					const cudaMemcpyKind kind = (mem_type == TRITONSERVER_MEMORY_CPU) ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;

					if (auto cuda_err = cudaMemcpyAsync(reinterpret_cast<char*>(device_input_owner) + copied, buffer, buffer_size, kind, new_state->stream); cuda_err != cudaSuccess) {
						return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(cuda_err));
					}
		    		copied += buffer_size;
				}
	    	}
			if (!device_input) {
				device_input = device_input_owner;
			}

			if ((device_input_owner && copied != bytes) || (!device_input && bytes != 0)) {
				return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "Input byte count mismatch");
			}
		}

	    void* output_buffer = nullptr;

	    if (auto* err = TRITONBACKEND_OutputBuffer(output_tensor, &output_buffer, bytes, TRITONSERVER_MEMORY_GPU, new_state->device_id); err != nullptr) {
			return err;
	    }

	    TRITONSERVER_BufferAttributes* attributes = nullptr;
	    if (auto* err = TRITONBACKEND_OutputBufferAttributes(output_tensor, &attributes); err != nullptr) {
			return err;
	    }

	    TRITONSERVER_MemoryType output_mem_type;
	    int64_t out_mem_type_id = 0;

	    if (auto* err = TRITONSERVER_BufferAttributesQuery(attributes, TRITONSERVER_BUFFER_ATTRIBUTES_MEMORY_TYPE, &output_mem_type); err != nullptr) {
			return err;
	    }

	    if (auto* err = TRITONSERVER_BufferAttributesQuery(attributes, TRITONSERVER_BUFFER_ATTRIBUTES_MEMORY_TYPE_ID, &out_mem_type_id); err != nullptr) {
			return err;
	    }

	    float* device_output = nullptr;
	    float* device_output_owner = nullptr;
	    void* host_output = nullptr;

	    if (bytes != 0) {
			if (output_mem_type == TRITONSERVER_MEMORY_GPU && out_mem_type_id == new_state->device_id) {
				device_output = static_cast<float*>(output_buffer);
			} else {
				host_output = output_buffer;
				if (auto cuda_err = cudaMallocAsync(&device_output_owner, bytes, new_state->stream); cuda_err != cudaSuccess) {
					return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(cuda_err));
				}
				device_output = device_output_owner;
			}
	    }

	    // launch CUDA kernel if data is available
	    if (bytes != 0) {
			launch_kernel(device_input, device_output, elem_count, new_state->stream);

			if (auto cuda_err = cudaGetLastError(); cuda_err != cudaSuccess) {
		    	return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(cuda_err));
			}
	    }

	    // check if output is on host
	    if (host_output != nullptr && bytes != 0) {
			if (auto cuda_err = cudaMemcpyAsync(host_output, device_output, bytes, cudaMemcpyDeviceToHost, new_state->stream); cuda_err != cudaSuccess) {
		    	return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(cuda_err));
			}
	    }

	    if (auto cuda_err = cudaStreamSynchronize(new_state->stream); cuda_err != cudaSuccess) {
			return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(cuda_err));
	    }

	    if (device_input_owner) {
			cudaFreeAsync(device_input_owner, new_state->stream);
	    }

	    if (device_output_owner) {
			cudaFreeAsync(device_output_owner, new_state->stream);
	    }

	    if (auto* err = TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE, nullptr); err != nullptr) {
			return err;
	    }
	    TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL);	
    }   
    return nullptr; 
}