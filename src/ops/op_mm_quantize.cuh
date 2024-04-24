#pragma once

#include "utils/assert.cuh"
#include "utils/tensor.cuh"

#define ELEMWISE_BLOCK_DIM 32 // thread block has 32x32 threads
#define TILE_WIDTH 32


template <typename T>
__global__ void op_matmul_kernel_naive(Tensor<T> Mat_A, Tensor<T> Mat_B, Tensor<T> Mat_C){

    const uint M = Mat_A.h;
    const uint K = Mat_A.w;
    const uint N = Mat_B.w;

    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;
    
    __shared__ float A_tile_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile_shared[TILE_WIDTH][TILE_WIDTH];

    const uint threadRow = threadIdx.x / TILE_WIDTH; // row to which thread in C belongs
    const uint threadCol = threadIdx.x % TILE_WIDTH; // col to which thread in C belongs

    // advance pointers to the starting positions
    Mat_A += cRow * TILE_WIDTH * K;                    // row=cRow, col=0
    Mat_B += cCol * TILE_WIDTH;                        // row=0, col=cCol
    Mat_C += cRow * TILE_WIDTH * N + cCol * TILE_WIDTH; // row=cRow, col=cCol

    T res = 0.0;

    for (int k = 0; k < K; k += TILE_WIDTH){
        // Have each thread load one of the elements in A & B
        // Make the threadCol (=threadIdx.x) the consecutive index
        // to allow global memory access coalescing
        Index(As, threadRow * TILE_WIDTH, threadCol) = Index(A, threadRow * K, threadCol);
        Index(Bs, threadRow * TILE_WIDTH, threadCol) = Index(B, threadRow * N, threadCol);

        // block threads in this block until cache is fully populated
        __syncthreads();
        Mat_A += TILE_WIDTH;
        Mat_B += TILE_WIDTH * N;

        // execute the dotproduct on the currently cached block
        for (int dotIdx = 0; dotIdx < TILE_WIDTH; ++dotIdx) {
            T tmp = Index(As, threadRow * TILE_WIDTH,  dotIdx) * Index(Bs, dotIdx * TILE_WIDTH,  threadCol);
            res += tmp;
        }
        
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        __syncthreads();
    }

    Index(Mat_C, threadRow * N, threadCol) = res;
}

//This operator compute C = A@B
template <typename T>
void op_mm(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C)
{
    assert(A.h == C.h && B.w == C.w && A.w == B.h);
    assert(A.on_device && B.on_device && C.on_device);

    //Lab-1: please complete this
    //You need to define separate kernel function(s) and launch them here
    //delete assert(0) when you are finished    
    // Approach - tiling
    dim3 gridDim;
    gridDim.x = (C.w + blockDim.x - 1)/blockDim.x;
    gridDim.y = (C.h + blockDim.y - 1)/blockDim.y;
    dim3 blockDim(TILE_WIDTH * TILE_WIDTH, 1, 1);
    op_matmul_kernel_naive<<<gridDim, blockDim>>>(A, B, C);
    

}
