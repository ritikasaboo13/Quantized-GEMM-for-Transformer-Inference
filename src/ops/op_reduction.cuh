#pragma once

#include "utils/tensor.cuh"

#define ELEMWISE_BLOCK_DIM 32 // thread block has 32x32 threads

template <typename T>
class MaxFunc
{
public:
    //This function adds input x to the current accumulated sum value stored in accum
    //The accumu's value is updated (to add x).  The ind_x and ind_accum arguments are not used. 
    __host__ __device__ void operator()(const T &x, const int &ind_x, T &accum, int &ind_accum)
    {
      //Lab-1: add your code here
      if (x > accum){
        accum = x;
        }
    }
};

template <typename T>
class MaxAccumFunc
{
public:
    //This function compares input x with the current accumulated maximum value stored in accum
    //If x is bigger than accum, stores x in accum and stores x's index (ind_x) to ind_accum
    __host__ __device__ void operator()(const T &x, const int &ind_x, T &accum, int &ind_accum)
    {
      //Lab-1: add your code here
      if (x > accum){
        accum = x;
        ind_accum = ind_x;
      }
    }
};

template <typename T>
class SumAccumFunc
{
public:
    //This function adds input x to the current accumulated sum value stored in accum
    //The accumu's value is updated (to add x).  The ind_x and ind_accum arguments are not used. 
    __host__ __device__ void operator()(const T &x, const int &ind_x, T &accum, int &ind_accum)
    {
      //Lab-1: add your code here
      accum += x;
    }
};

//This kernel function performs column-wise reduction of the "in" tensor and stores the result in "out" tensor.
//If "get_index" is true, then the index accumulator values are stored in "out_index" and "out" is not touched.
template <typename OpFunc, typename T>
__global__ void op_reduction_kernel_colwise(OpFunc f, Tensor<T> in, Tensor<T> out, Tensor<int> out_index, bool get_index)
{
    // //Lab-1: add your code here
    // consider this as operating on one row of input
    // each thread independently computes 32*32 elements of the output
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    if (idx < in.h){
        if (!get_index){
            Index(out, idx, 0) = Index(in, idx, 0);
            for (int i=1; i < in.w; i++){
                f(Index(in, idx, i), i, Index(out, idx, 0), idx);
            }
        }
        else{
            Index(out_index, idx, 0) = 0;
            for (int i=1; i < in.w; i++){
                f(Index(in, idx, i), i, Index(in, idx, Index(out_index, idx, 0)), Index(out_index, idx, 0));
            }
        }
    }
}

//This kernel function performs row-wise reduction of the "in" tensor and stores the result in "out" tensor.
//If "get_index" is true, then the index accumulator values are stored in "out_index" and "out" is not touched.
template <typename OpFunc, typename T>
__global__ void op_reduction_kernel_rowwise(OpFunc f, Tensor<T> in, Tensor<T> out, Tensor<int> out_index, bool get_index)
{
    //Lab-1: add your code here
    // consider this as operating on one column of input
    // each thread independently computes 32*32 elements of the output
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    if (idx < in.w){
        if (!get_index){
            Index(out, 0, idx) = Index(in, 0, idx);
            for (int i=1; i < in.h; i++){
                f(Index(in, i, idx), i, Index(out, 0, idx), idx);
            }
        }
        else{
            Index(out_index, 0, idx) = 0;
            for (int i=1; i < in.h; i++){
                f(Index(in, i, idx), i, Index(in, Index(out_index, 0, idx), idx), Index(out_index, 0, idx));
            }
        }
    }
}    

template <typename OpFunc, typename T>
void op_reduction_gpu(OpFunc f, const Tensor<T> &in, Tensor<T> &out, Tensor<int> &out_index, bool get_index = false)
{
    int out_h = out.h;
    if (!get_index) {
        assert((out.h == 1 && in.w == out.w) || (out.w == 1 && in.h == out.h));
    } else {
        out_h = out_index.h;
        assert((out_index.h == 1 && in.w == out_index.w) || (out_index.w == 1 && in.h == out_index.h));
    }
    // std::cout << "In shape: [" << in.h << "," << in.w << "]" << std::endl;
    // std::cout << in.str() << std::endl;
    // std::cout << "Out shape: [" << out.h << "," << out.w << "]" << std::endl;
    // std::cout << out.str() << std::endl;
    // std::cout << "Out_index shape: [" << out_index.h << "," << out_index.w << "]" << std::endl;
    // std::cout << out_index.str() << std::endl;

    // create as many blocks as necessary to map all of out
    int M_ = (in.w + ELEMWISE_BLOCK_DIM - 1)/ ELEMWISE_BLOCK_DIM;
    int N_ = (in.h + ELEMWISE_BLOCK_DIM - 1)/ ELEMWISE_BLOCK_DIM;
    dim3 dimGrid(M_*N_, 1);
    // 32 * 32 = 1024 thread per block
    dim3 dimBlock(ELEMWISE_BLOCK_DIM*ELEMWISE_BLOCK_DIM, 1, 1); //threadsPerBlock - organization of dimBlock changes according to the #inputs being used in the output

    if (in.h > out_h) // out shape [1, n]
    {
      //Lab-1: add your code here to launch op_reduction_kernel_rowwise
      //delete assert(0) when you are finished
        // std::cout << "launching rowwise kernel" << std::endl; //colwise compression
        op_reduction_kernel_rowwise<<<dimGrid, dimBlock>>>(f, in, out, out_index, get_index);
    }
    else // out shape [m, 1]
    {
      //Lab-1: add your code here to launch op_reduction_kernel_colwise
      //delete assert(0) when you are finished
        // std::cout << "launching colwise kernel" << std::endl; //colwise compression
        op_reduction_kernel_colwise<<<dimGrid, dimBlock>>>(f, in, out, out_index, get_index);
    }
    // std::cout << "---------------------------------------------------------" << std::endl;
}



template <typename T>
void op_sum(const Tensor<T> &in, Tensor<T> &out)
{
    Tensor<int> out_index;
    SumAccumFunc<T> f;
    if (in.on_device && out.on_device) {
        op_reduction_gpu(f, in, out, out_index, false);
    } else
        assert(0);
}

template <typename T>
void op_argmax(const Tensor<T> &in, Tensor<int> &out_index)
{
    Tensor<T> out;
    MaxAccumFunc<T> f;
    if (in.on_device && out_index.on_device) {
        op_reduction_gpu(f, in, out, out_index, true);
    } else
        assert(0);
}

template <typename T>
void op_max(const Tensor<T> &in, Tensor<T> &out)
{
    Tensor<int> out_index;
    MaxFunc<T> f;
    if (in.on_device && out.on_device) {
        op_reduction_gpu(f, in, out, out_index, false);
    } else
        assert(0);
}
