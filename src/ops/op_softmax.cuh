#pragma once
#include "utils/tensor.cuh"
#include "ops/op_elemwise.cuh"

template <typename T>
__global__ void softmax_kernel(const Tensor<T> A, Tensor<T> B, int batch_size, int num_classes)
{
    int row = blockIdx.x;
    if (row >= batch_size) return;

    T max_logit = Index(A, row, 0);
    for (int i = 1; i < num_classes; i++) {
        if (Index(A, row, i) > max_logit) {
            max_logit = Index(A, row, i);
        }
    }

    T sum_exp = 0.0;
    for (int i = 0; i < num_classes; i++) {
        sum_exp += exp(Index(A, row, i) - max_logit);
    }

    for (int i = 0; i < num_classes; i++) {
        Index(B, row, i) = exp(Index(A, row, i) - max_logit) / sum_exp;
    }
}

template <typename T>
void op_softmax(const Tensor<T> &A, Tensor<T> &B)
{
    assert(A.h == B.h && A.w == B.w);
    assert(A.on_device && B.on_device); 

    int batch_size = A.h;
    int num_classes = A.w;

    int blocksPerGrid = batch_size;
    int threadsPerBlock = num_classes;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, batch_size, num_classes);
}
