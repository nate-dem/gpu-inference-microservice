#include <iostream>
#include <vector>
#include "grpc_client.h"

namespace tc = triton::client;

int main(int argc, char** argv) {
    const std::string server_url = "localhost:8001";
    const std::string model_name = "custom_model";

    triton::client::InferOptions options("custom_model");
    options.model_version_ = "1";
    options.client_timeout_ = 0;

    // client used to communicate with Triton server
    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    tc::Error error = tc::InferenceServerGrpcClient::Create(&client, server_url);

    if (!error.IsOk()) {
        std::cerr << "Error with creating Triton client: " << error << std::endl;
	return 1;
    }

    // properties of input tensor (used to create output tensor)
    std::vector<int64_t> shape = {1, 16};
    const std::string input_name = "INPUT0";
    const std::string output_name = "OUTPUT0";
    const std::string data_type = "FP32";
    const size_t size = 16;

    tc::InferInput* input_tensor;
    tc::InferInput::Create(&input_tensor, input_name, shape, data_type);

    std::vector<float> input_data(size);
    for (size_t i = 0; i < size; ++i) {
	input_data[i] = static_cast<float>(i);
    }

    input_tensor->AppendRaw(reinterpret_cast<const uint8_t*>(input_data.data()),
                      input_data.size() * sizeof(float));

    tc::InferRequestedOutput* output_tensor;
    tc::InferRequestedOutput::Create(&output_tensor, output_name);

    // server will send request to backend which will then call the CUDA kernel	
    tc::InferResult* result;
    error = client->Infer(&result, options, {input_tensor}, {output_tensor});

    if (!error.IsOk()) {
	return 1;
    }

    size_t output_bytes = 0;
    const uint8_t* output_buffer = nullptr;
    error = result->RawData(output_name, &output_buffer, &output_bytes);

    if (!error.IsOk()) {
	return 1;
    }
    
    std::cout << "Input data: ";
    for (const auto& val : input_data) { std::cout << val << " "; }
    std::cout << std::endl;
 
    std::cout << "Output data: ";
    const float* output_data_ptr = reinterpret_cast<const float*>(output_buffer);
    for (size_t i = 0; i < output_bytes / sizeof(float) ; ++i) {
	std::cout << output_data_ptr[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
