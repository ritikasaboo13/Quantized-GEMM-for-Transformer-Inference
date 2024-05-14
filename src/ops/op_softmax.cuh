#pragma once
#include "utils/tensor.cuh"
#include "ops/op_elemwise.cuh"
#include <cmath> 

template <typename T>
__global__ void softmax_kernel(const Tensor<T> A, Tensor<T> B, int batch_size, int num_classes)
{
    int row = blockIdx.x; // 0, 1, .... seq_length-1
    if (row >= batch_size) return; // batch_size = seq_length

    T max_logit = Index(A, row, 0); 
    for (int i = 1; i < num_classes; i++) {
        if (Index(A, row, i) > max_logit) {
            max_logit = Index(A, row, i);
        }
    }

    T sum_exp = 0.0;
    for (int i = 0; i < num_classes; i++) {
        Index(B, row, i) = exp(Index(A, row, i) - max_logit);
        sum_exp += exp(Index(A, row, i) - max_logit);
    }

    for (int i = 0; i < num_classes; i++) {
        Index(B, row, i) = Index(B, row, i) / sum_exp;
    }
}

template <typename T>
void op_softmax(const Tensor<T> &A, Tensor<T> &B)
{
    assert(A.h == B.h && A.w == B.w);
    assert(A.on_device && B.on_device); 

    int batch_size = A.h;
    int num_classes = A.w;

    int blocksPerGrid = batch_size; // seq_length
    int threadsPerBlock = num_classes; // seq_length

    // why launch a kernel with blocksPerGrid, threadsPerBlock = seq_length x seq_length ??? 
    // this is wasteful because only the first thread of each block is used and rest remain unused
    // also, assuming seq_lengths are of 2048 sizes, is it right to launch 2048 blocks with 2048 threads?  
    // there might be limits to doing so? 
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, batch_size, num_classes);
}

 // ===================================== // 
template <typename T>
__global__ void softmax_kernel_new(const Tensor<T> A, Tensor<T> B, int h, int w)
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
void op_softmax_new(const Tensor<T> &A, Tensor<T> &B)
{
    assert(A.h == B.h && A.w == B.w);
    assert(A.on_device && B.on_device); 

    int threadsPerBlock = 256;
    int numBlocks = static_cast<int>((ceil(static_cast<float>(A.w)/threadsPerBlock)));

    softmax_kernel_new<<<numBlocks, threadsPerBlock>>>(A, B, A.h, A.w);
}
