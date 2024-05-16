#pragma once
#include "utils/tensor.cuh"
#include "ops/op_elemwise.cuh"
#include <cmath> 

template <typename T>
__global__ void layernorm_kernel(const Tensor<T> A, Tensor<T> B, int h, int w)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    T mean = 0.0;
    T var = 0.0; 

    if (row < h) {
        for(int col = 0; col < w; col++) {
            mean += Index(A, row, col);
        }

        mean = mean/w; 

        for(int col = 0; col < w; col++) {
            var += pow((Index(A, row, col) - mean), 2); 
        }

        var = var/ w;
        

        for(int col = 0; col < w; col++) {
            Index(B, row, col) = (Index(A, row, col) - mean)/var;
        }

    }
}

template <typename T>
void op_layernorm(const Tensor<T> &A, Tensor<T> &B)
{
    assert(A.h == B.h && A.w == B.w);
    assert(A.on_device && B.on_device); 

    int threadsPerBlock = 256;
    int numBlocks = static_cast<int>((ceil(static_cast<float>(A.w)/threadsPerBlock)));

    layernorm_kernel<<<numBlocks, threadsPerBlock>>>(A, B, A.h, A.w);
}
