#pragma once

#include "utils/assert.cuh"
#include "utils/tensor.cuh"

#define ELEMWISE_BLOCK_DIM 32 // thread block has 32x32 threads
#define TILE_WIDTH 32

#define CEIL_DIV(M, N) ((M + (N + 1))/(N));

#define threshold 6.0;

//This operator compute C = A@B using quantization
template <typename T>
void op_mm_quantize(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C)
{
    assert(A.h == C.h && B.w == C.w && A.w == B.h);
    assert(A.on_device && B.on_device && C.on_device);



    dim3 gridDim;
    gridDim.x = CEIL_DIV(C.w, blockDim.x);
    gridDim.y = CEIL_DIV(C.h, blockDim.y);
    dim3 blockDim(TILE_WIDTH * TILE_WIDTH, 1, 1);

    // identify outlier indices
    // initialize outlier_indices array of size X.w; each thread operates on an element
    // and checks if it is > threshold, returns its index and we store it in outlierIndices
    op_kernel_identify_outlier_indices<<<gridDim, blockDim>>>(X, W);

    // define matrix X_regular, W_regular based on outlierIndices
    op_regular_kernel<<<gridDim, blockDim>>>(X, W, X_outlier, W_outlier);

    // compute Cx

    // compute Cw

    // compute X_int8 and W_int8

    // Int8 matmul




}
