cmake_minimum_required(VERSION 3.10)

project(cudaMatmul CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.3/bin/nvcc")
enable_language(CXX)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(cudaMatmul kernel.cu queue.cpp)

set_target_properties(cudaMatmul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 11
)