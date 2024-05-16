#pragma once
#include "utils/tensor.cuh"
#include "ops/op_elemwise.cuh"
#include <cmath> 

template <typename T>
__global__ void softmax_kernel(const Tensor<T> A, Tensor<T> B, int h, int w)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < h) {
        T max = Index(A, row, 0);
        for(int col = 1; col < w; col++) {
            if(Index(A, row, col) > max) {
                max = Index(A, row, col);
            }
        }

        T sum = 0.0; 
        for(int col=0; col < w; col++) {
            Index(B, row, col) = exp(Index(A, row, col) - max);
            sum += Index(B, row, col);
        }

        for (int col=0; col < w; col++) {
            Index(B, row, col) = Index(B, row, col) / sum;
        }
    }
}

template <typename T>
void op_softmax(const Tensor<T> &A, Tensor<T> &B)
{
    assert(A.h == B.h && A.w == B.w);
    assert(A.on_device && B.on_device); 

    int threadsPerBlock = 256;
    int numBlocks = static_cast<int>((ceil(static_cast<float>(A.w)/threadsPerBlock)));

    softmax_kernel_new<<<numBlocks, threadsPerBlock>>>(A, B, A.h, A.w);
}
