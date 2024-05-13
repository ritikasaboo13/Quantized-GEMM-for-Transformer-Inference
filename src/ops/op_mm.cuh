#pragma once

#include "utils/assert.cuh"
#include "utils/tensor.cuh"

#define TILE_WIDTH 32 // thread block has 32x32 threads


template <typename T, typename OutT>
__global__ void op_matmul_kernel(Tensor<T> Mat_A, Tensor<T> Mat_B, Tensor<OutT> Mat_C){

    OutT res = 0;
    int row = blockIdx.y*TILE_WIDTH + threadIdx.y; // row to which thread in C belongs
    int col = blockIdx.x*TILE_WIDTH + threadIdx.x; // col to which thread in C belongs

    __shared__ float A_tile_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile_shared[TILE_WIDTH][TILE_WIDTH];

    for (int k = 0; k < (Mat_A.w - 1)/TILE_WIDTH+1; k++){
        // Initialize A_tile_shared
        if ((k*TILE_WIDTH + threadIdx.x) < Mat_A.w && row < Mat_A.h){
            A_tile_shared[threadIdx.y][threadIdx.x] = Index(Mat_A, row, ((k*TILE_WIDTH) + threadIdx.x)); //load tile from A from row into A_tile_shared
        }
        else{
            A_tile_shared[threadIdx.y][threadIdx.x] = 0;
        }
        // Initialize B_tile_shared
        if ((k*TILE_WIDTH + threadIdx.y) < Mat_B.h && col < Mat_B.w){
            B_tile_shared[threadIdx.y][threadIdx.x] = Index(Mat_B, (k*TILE_WIDTH + threadIdx.y), col); //load tile from B from col into B_tile_shared
        }
        else{
            B_tile_shared[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for(int t=0; t < TILE_WIDTH; t++){
            res += A_tile_shared[threadIdx.y][t] * B_tile_shared[t][threadIdx.x];
        }
        __syncthreads();
    }

    if(row < Mat_A.h && col < Mat_B.w){
        Index(Mat_C, row, col) = res;
    }   
}

//This operator compute C = A@B
template <typename T, typename OutT>
void op_mm(const Tensor<T>& A, const Tensor<T>& B, Tensor<OutT>& C)
{
    assert(A.h == C.h && B.w == C.w && A.w == B.h);
    assert(A.on_device && B.on_device && C.on_device);

    //Lab-1: please complete this
    //You need to define separate kernel function(s) and launch them here
    //delete assert(0) when you are finished    
    // Approach - tiling
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim;
    gridDim.x = (C.w + blockDim.x - 1)/blockDim.x;
    gridDim.y = (C.h + blockDim.y - 1)/blockDim.y;
    op_matmul_kernel<<<gridDim, blockDim>>>(A, B, C);

}
