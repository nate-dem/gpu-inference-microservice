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

    return 0;
}
