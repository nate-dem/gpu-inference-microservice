cmake_minimum_required(VERSION 3.18)
project(triton_client LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(triton_client src/client.cpp)

target_include_directories(triton_client PRIVATE ${TRITON_CLIENT_DIR}/include)

find_library(GRPC_CLIENT_LIB grpcclient PATHS ${TRITON_CLIENT_DIR}/lib REQUIRED)
find_library(HTTP_CLIENT_LIB httpclient PATHS ${TRITON_CLIENT_DIR}/lib REQUIRED)

target_link_libraries(triton_client PRIVATE
    ${GRPC_CLIENT_LIB}
    ${HTTP_CLIENT_LIB}
    pthread
)

set_target_properties(triton_client PROPERTIES
    BUILD_RPATH ${TRITON_CLIENT_DIR}/lib
)
