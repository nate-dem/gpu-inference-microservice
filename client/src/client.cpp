#include <iostream>
#include <vector>
#include "grpc_client.h"

namespace tc = triton::client;

int main(int argc, char** argv) {
    const std::string server_url = "localhost:8001";
    const std::string model_name = "custom_model";

    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    tc::Error error = tc::InferenceServerGrpcClient::Create(&client, server_url);

    if (!error.IsOk()) {
        std::cerr << "Error with creating Triton client: " << error << std::endl;
	return 1;
    }

    std::vector<int64_t> shape = {1, 16};
    const std::string input_name = "INPUT0";
    const std::string output_name = "OUTPUT0";
    const std::string data_type = "FP32";
    const float size = 16;

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

	
    tc::InferResult* result;
    err = client->Infer(&result, {input_tensor}, {output_tensor});
    if (!err.IsOk()) {
	return 1;
    }

    std::vector<float> output_data;
    err = result->RawData(output_name, (const uint8_t**)&output_data, &output_byte_size);
    if (!err.IsOk()) {
	return 1;
    }

    for (const auto& val : input_data) { std::cout << val << " "; }

    for (size_t i = 0; i < output_byte_size / sizeof(float) ; ++i) {
	std::cout << output_data[i] << " ";
    }

    return 0;
}
