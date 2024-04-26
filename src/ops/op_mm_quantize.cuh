#pragma once

#include "utils/assert.cuh"
#include "utils/tensor.cuh"

#define ELEMWISE_BLOCK_DIM 32 // thread block has 32x32 threads
#define TILE_WIDTH 32

#define CEIL_DIV(M, N) ((M + (N + 1))/(N));

// //This is the GPU kernel function 
// template <typename T>
// __global__ void op_kernel_1(Tensor<AT> t, Tensor<T> ind, Tensor<AT> out)
// {
//     // each thread accesses 1 element of t
//     int i = threadIdx.y + ELEMWISE_BLOCK_DIM * blockIdx.y; // row
//     int j = threadIdx.x + ELEMWISE_BLOCK_DIM * blockIdx.x; // col
//     if (i < t.h && j < t.w){
//         if (Index(ind, 0, j) == 1){
//             Index(out, i, j) = Index(t, i, j);
//         }
//     }
// }




// // Extracts n number of rows/columns from A, rows/columns to extract are indices in C where value is 1/0
// template <typename AT, typename T>
// void op_Matrix_extractor(const Tensor<AT>& A, const Tensor<T>& C, const T& n, Tensor<AT>& O)
// {
//     assert(A.h == O.h)
//     assert(A.w == C.w && O.w == A.w-numOutliers)
//     assert(C.h == 1)

//     // create as many blocks as necessary to map all of A
//     // 32 * 32 = 1024 threadsPerBlock - organization of dimBlock 
//     dim3 dimBlock(ELEMWISE_BLOCK_DIM, ELEMWISE_BLOCK_DIM, 1);
//     dim3 gridDim;
//     gridDim.x = CEIL_DIV(A.w, blockDim.x);
//     gridDim.y = CEIL_DIV(A.h, blockDim.y);

//     // extract all columns from A where index of column is 1 in C, store result in O 
//     if (C.h == 1){
//         op_kernel_1<<<>>>(A, C, O);
//     }
//     if (C.w == 1){
//         op_kernel_2<<<>>>(A, C, O);
//     }
    


// }


//This operator compute C = A@B using quantization
template <typename T>
void op_mm_quantize(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C)
{
    assert(A.h == C.h && B.w == C.w && A.w == B.h);
    assert(A.on_device && B.on_device && C.on_device);


    dim3 blockDim(TILE_WIDTH * TILE_WIDTH, 1, 1);
    dim3 gridDim;
    gridDim.x = CEIL_DIV(C.w, blockDim.x);
    gridDim.y = CEIL_DIV(C.h, blockDim.y);
    

    // define matrix X_regular, W_regular based on outlierIndices
    // op_regular_kernel<<<gridDim, blockDim>>>(X, W, X_outlier, W_outlier);

    // compute Cx

    // compute Cw

    // compute X_int8 and W_int8

    // Int8 matmul




}
