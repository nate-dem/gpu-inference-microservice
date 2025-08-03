cmake_minimum_required(VERSION 3.18)
project(triton_client LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(triton_client src/client.cpp)

target_include_directories(triton_client PRIVATE ${TRITON_CLIENT_DIR}/include)
target_link_directories(triton_client PRIVATE  ${TRITON_CLIENT_DIR}/lib)
