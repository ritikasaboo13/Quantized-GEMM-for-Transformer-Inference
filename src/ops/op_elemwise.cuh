#pragma once

#include "utils/tensor.cuh"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cmath>

#define ELEMWISE_BLOCK_DIM 32 // thread block has 32x32 threads
#define N 256

extern unsigned long long randgen_seed;

class RandGenGPU
{
public:
    RandGenGPU(unsigned long long s)
    {
        curandAssert(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandAssert(curandSetPseudoRandomGeneratorSeed(gen, s));
    }
    curandGenerator_t gen;
};

//This functor calculates the SGD operation 
//given input t (one element of the parameter tensor) 
//and its gradient dt
template <typename T>
class SGDFunc
{
public:
    __host__ __device__ T operator()(T t, T dt)
    {
        //Lab-1: add your code here (delete return 0)
        T elem = lr * dt;
        return t - elem;
    }
    const float lr;
};

//This functor adds normalize elements "a" by subtracting b
template <typename T>
class NormalizeFunc
{
public:
    __host__ __device__ T operator()(T a, T b)
    {
        //Lab-1: add your code here (delete return 0)
        return a - b;
    }
};



//This functor adds two input elements "a" and "b" together
template <typename T>
class AddFunc
{
public:
    __host__ __device__ T operator()(T a, T b)
    {
        //Lab-1: add your code here (delete return 0)
        return a + b;
    }
};

//This functor adds constant "b" to the input element
template <typename T>
class AddConstFunc
{
public:
    __host__ __device__ T operator()(T a)
    {
        //Lab-1: add your code here (delete return 0)
        return a + b;
    }
    const T b;
};

//This functor multiplies two input elements x and b together
template <typename T>
class MultiplyFunc
{
public:
    __host__ __device__ T operator()(T x, T a)
    {
        //Lab-1: add your code here (delete return 0)
        return x * a;

    }
};

template <typename T1, typename T2>
class DequantizeFunc
{
public:
    __host__ __device__ T2 operator()(T1 x, T2 a)
    {
        //Lab-1: add your code here (delete return 0)
        return (static_cast<T2>(x)) * a; 

    }
};


template <typename T, typename OutT>
class MultiplyWithTypecastFunc
{
public:
    __host__ __device__ OutT operator()(T x, T a)
    {
        return static_cast<OutT> (x * a);
    }
};


//This functor multiplies constant "b" to the input element
template <typename T>
class MultiplyConstFunc
{
public:
    __host__ __device__ T operator()(T x)
    {
        //Lab-1: add your code here (delete return 0)
        return x * b;

    }
    const T b;
};

//This functor divides constant "b" by the input element, hence the name inv(erse) divide
template <typename T>
class InvDivideConstFunc
{
public:
    __host__ __device__ T operator()(T x)
    {
        //Lab-1: add your code here (delete return 0)
        return b / x;

    }
    const T b;
};

//This functor returns 1 if inputs "a" and "b" are equal
//and returns 0 otherwise. 
template <typename AT, typename BT, typename OutT>
class EqualityFunc
{
public:
    __host__ __device__ OutT operator()(AT a, BT b)
    {
        //Lab-2: add your code here (delete return 0)
        if (a == b){
            return 1;
        }
        else{
            return 0;
        }
    }

};


//This functor rounds to nearest integer value 
template <typename AT>
class RoundInt8Func
{
public:
    __host__ __device__ int8_t operator()(AT a)
    {
        //Lab-2: add your code here (delete return 0)
        return round(a); 
    }

};

//This functor implements the ReLu operation 
//for a single element
template <typename T>
class ReluFunc
{
public:
    __host__ __device__ T operator()(T x)
    {
        //Lab-2: add your code here (delete return 0)
        if (x < 0){
            return 0;
        }
        else{
            return x;
        }

    }
};

//This functor implements the backwards of the 
//ReLu operation for a single element.
template <typename T>
class ReluBackFunc
{
public:
    __host__ __device__ T operator()(T x, T dy)
    {
        //Lab-2: add your code here (delete return 0)
        if (x > 0){
            return dy;
        }
        else{
            return 0;
        }
    }
};

//This functor subtracts "b" from "a"
template <typename T>
class SubtractFunc
{
public:
    __host__ __device__ T operator()(T a, T b)
    {
        //Lab-1: add your code here (delete return 0)
        return a - b;
    }
};

//This functor subtracts constant "b" to the input element
template <typename T>
class SubtractConstFunc
{
public:
    __host__ __device__ T operator()(T a)
    {
        //Lab-1: add your code here (delete return 0)
        return a - b;
    }
    const T b;
};

//This functor implements the exponent operation 
//for a single element
template <typename T>
class ExpFunc
{
public:
    __host__ __device__ T operator()(T x)
    {
        //Lab-2: add your code here (delete return 0)
        return exp(x);
    }
};

//This functor implements the logarithm (base e) operation 
//for a single element
template <typename T>
class LogFunc
{
public:
    __host__ __device__ T operator()(T x)
    {
        //Lab-2: add your code here (delete return 0)
        return log(x);
    }
};

template <typename T>
class ConstInitFunc
{
public:
    __host__ __device__ T operator()(T x)
    {
        return val;
    }
    const float val;
};

template <typename T>
class UniformInitFuncCPU
{
public:
    UniformInitFuncCPU(T min, T max) : dist(min, max) {}
    T operator()(T x)
    {
        static std::default_random_engine gen(randgen_seed);
        return dist(gen);
    }
    std::uniform_real_distribution<T> dist;
};

// This functor compares constant "b" to the absolute value of input element,
// returns 0 if input < b else 1
template <typename AT, typename BT, typename OutT>
class AbsCompareLTEConstFunc
{
public:
    __host__ __device__ OutT operator()(AT a)
    {
        if (((a >= 0) &(a <= b)) | ((a <= 0) & (-a <= b))){
            return 0;
        }
        else{
            return 1;
        }
    }
    const BT b;
};

//This is the GPU kernel function for performing element wise operation 
//that takes a single argument "t" and stores the result in "out"
template <typename OpFunc, typename AT, typename T>
__global__ void op_elemwise_unary_kernel(OpFunc f, Tensor<AT> t, Tensor<T> out)
{
  //Lab-1: add your code here
    // each thread accesses 1 element of t
    int i = threadIdx.y + ELEMWISE_BLOCK_DIM * blockIdx.y; // row
    int j = threadIdx.x + ELEMWISE_BLOCK_DIM * blockIdx.x; // col
    if (i < t.h && j < t.w){
        Index(out, i, j) = f(Index(t, i, j));
    }
}

//This function launches the GPU kernel to perform element wise operation 
//that takes a single argument "t" and stores the result in "out"
template <typename OpFunc, typename AT, typename T>
void op_elemwise_unary_gpu(OpFunc f, const Tensor<AT> &t, Tensor<T> &out)
{
  //Lab-1:add your code here. Somewhere in this function, 
  //you need to call op_elemwise_unary_kernel<<<???, ???>>>(f, t, out);
  //delete assert(0) when you are done
    // op_elemwise_unary_kernel<<<1, ELEMWISE_BLOCK_DIM*ELEMWISE_BLOCK_DIM>>>(f, t, out);
    int M_ = (t.w + ELEMWISE_BLOCK_DIM - 1)/ ELEMWISE_BLOCK_DIM;
    int N_ = (t.h + ELEMWISE_BLOCK_DIM - 1)/ ELEMWISE_BLOCK_DIM;
    dim3 gridDim(M_, N_, 1);
    dim3 blockDim(ELEMWISE_BLOCK_DIM, ELEMWISE_BLOCK_DIM, 1);
    op_elemwise_unary_kernel<<<gridDim, blockDim>>>(f, t, out);
}


//This is the GPU kernel function for performing element wise operation 
//that takes a single argument "t" and stores the result in "out"
template <typename OpFunc, typename T>
__global__ void op_elemwise_unary_kernel(OpFunc f, Tensor<T> t, Tensor<T> out)
{
  //Lab-1: add your code here
    // each thread accesses 1 element of t
    int i = threadIdx.y + ELEMWISE_BLOCK_DIM * blockIdx.y; // row
    int j = threadIdx.x + ELEMWISE_BLOCK_DIM * blockIdx.x; // col
    if (i < t.h && j < t.w){
        Index(out, i, j) = f(Index(t, i, j));
    }
}

//This function launches the GPU kernel to perform element wise operation 
//that takes a single argument "t" and stores the result in "out"
template <typename OpFunc, typename T>
void op_elemwise_unary_gpu(OpFunc f, const Tensor<T> &t, Tensor<T> &out)
{
  //Lab-1:add your code here. Somewhere in this function, 
  //you need to call op_elemwise_unary_kernel<<<???, ???>>>(f, t, out);
  //delete assert(0) when you are done
    // op_elemwise_unary_kernel<<<1, ELEMWISE_BLOCK_DIM*ELEMWISE_BLOCK_DIM>>>(f, t, out);
    int M_ = (t.w + ELEMWISE_BLOCK_DIM - 1)/ ELEMWISE_BLOCK_DIM;
    int N_ = (t.h + ELEMWISE_BLOCK_DIM - 1)/ ELEMWISE_BLOCK_DIM;
    dim3 gridDim(M_, N_, 1);
    dim3 blockDim(ELEMWISE_BLOCK_DIM, ELEMWISE_BLOCK_DIM, 1);
    op_elemwise_unary_kernel<<<gridDim, blockDim>>>(f, t, out);
}


template <typename OpFunc, typename AT, typename BT, typename OutT>
__global__ void op_elemwise_binary_w_bcast_w_typeconversion_kernel(OpFunc f, Tensor<AT> in1, Tensor<BT> in2, Tensor<OutT> out)
{
  //Lab-1: add your code here
    int i = threadIdx.y + ELEMWISE_BLOCK_DIM * blockIdx.y; // row
    int j = threadIdx.x + ELEMWISE_BLOCK_DIM * blockIdx.x; // col

    if (i < in1.h && j < in1.w){
        if (in2.h == 1 && in1.w == in2.w && in1.h != in2.h) {
            // repeat in2 at i=0 across all indices of w
            Index(out, i, j) = __float2int_ru(f(Index(in1, i, j), Index(in2, 0, j)));
        }
        else if (in2.w == 1 && in1.h == in2.h && in1.w != in2.w) {
            // repeat in2 at j=0 across all indices of h
            Index(out, i, j) = __float2int_ru(f(Index(in1, i, j), Index(in2, i, 0)));
        }  
        else {
            Index(out, i, j) = __float2int_ru(f(Index(in1, i, j), Index(in2, i, j)));
        }
    }
}

//This is the GPU kernel function for performing element wise operation with 
//two input arguments "in1" and "in2" with potential broadcasting. 
// Input tensor "in2" is always the one to be 
// broadcasted when broadcasting is necessary.  Broadcasting is needed if 
// "in2" only have one dimension (instead of both dimensions) in common with "in1" 
// and its other dimension has size 1. In this case, to perform elemwise operation,
// we essentially broadcast the values of "in2" along the dimension with size 1 
// to match the dimension size of "in1". 
// Example1: a = [[1, 2, 3], [4, 5, 6]] and b = [[10],[20]], 
// then a+b = [[11, 12, 13], [24, 25, 26]]
// Example2: a = [[1, 2, 3], [4, 5, 6]] and b = [[10,20,30]]
// then a+b = [[11,22,33], [14, 25, 36]] 
template <typename OpFunc, typename AT, typename BT, typename OutT>
__global__ void op_elemwise_binary_w_bcast_kernel(OpFunc f, Tensor<AT> in1, Tensor<BT> in2, Tensor<OutT> out)
{
  //Lab-1: add your code here
    int i = threadIdx.y + ELEMWISE_BLOCK_DIM * blockIdx.y; // row
    int j = threadIdx.x + ELEMWISE_BLOCK_DIM * blockIdx.x; // col

    if (i < in1.h && j < in1.w){
        if (in2.h == 1 && in1.w == in2.w && in1.h != in2.h) {
            // repeat in2 at i=0 across all indices of w
            Index(out, i, j) = f(Index(in1, i, j), Index(in2, 0, j));
        }
        else if (in2.w == 1 && in1.h == in2.h && in1.w != in2.w) {
            // repeat in2 at j=0 across all indices of h
            Index(out, i, j) = f(Index(in1, i, j), Index(in2, i, 0));
        }  
        else {
            Index(out, i, j) = f(Index(in1, i, j), Index(in2, i, j));
        }
    }
}

//This function launches the GPU kernel that performs elementwise operation 
//(with potential broadcast) with two input tensor arguments "in1" and "in2",
// and stores the result in "out".  
template <typename OpFunc, typename AT, typename BT, typename OutT>
void op_elemwise_binary_w_bcast_gpu(OpFunc f, const Tensor<AT> &in1, const Tensor<BT> &in2, Tensor<OutT> &out)
{
    //Lab-1: add your code here. Somewhere in this function
   //you need to call op_elemwise_binary_w_bcast_kernel<<<???, ???>>>(f, in1, in2, out);
   //delete assert(0) when you are done
    int M_ = (in1.w + ELEMWISE_BLOCK_DIM - 1)/ ELEMWISE_BLOCK_DIM;
    int N_ = (in1.h + ELEMWISE_BLOCK_DIM - 1)/ ELEMWISE_BLOCK_DIM;
    dim3 gridDim(M_, N_, 1);
    dim3 blockDim(ELEMWISE_BLOCK_DIM, ELEMWISE_BLOCK_DIM, 1);
    op_elemwise_binary_w_bcast_kernel<<<gridDim, blockDim>>>(f, in1, in2, out);
    // if (typeConversion) {
    //     std::cout << "doing float to int type conversion" << std::endl;
    //     op_elemwise_binary_w_bcast_w_typeconversion_kernel<<<gridDim, blockDim>>>(f, in1, in2, out);
    // }
    // else{
    //     op_elemwise_binary_w_bcast_kernel<<<gridDim, blockDim>>>(f, in1, in2, out);
    // }

}

/*----------------------- tensor operators-----------------------*/

//This operator implements ReLu and stores the result in "out".
//Suppose y = Relu(x) Then y = x if x >=0.  y= 0 if x < 0.
template <typename T>
void op_relu(const Tensor<T> &t, Tensor<T> &out)
{
    assert(out.h == t.h && out.w == t.w);
    ReluFunc<T> f;
    if (t.on_device && out.on_device) {
        op_elemwise_unary_gpu(f, t, out);
    } else {
        assert(0);
    }
}

//This operator is the "backward" function of ReLu. Let out = ReLu(in).
//Let "d_out" represents the gradient of "out". Calculate the gradient 
//of "in" using the chain rule and store the result in "d_in".
template <typename T>
void op_relu_back(const Tensor<T> &in, const Tensor<T> &d_out, Tensor<T> &d_in)
{
    assert(d_in.h == in.h && d_out.w == in.w);
    assert(in.h == d_out.h && in.w == d_out.w);
    ReluBackFunc<T> f;
    if (d_in.on_device && in.on_device && d_out.on_device) {
        op_elemwise_binary_w_bcast_gpu(f, in, d_out, d_in);
    } else {
        assert(0);
    }
}

//This operator performs the "SGD" operation, aka calculating out = t - lr*dt;
//and stores the result in "out" tensor. lr is the learning rate. dt tensor should
//contain the gradient of parameter tensor "t".
template <typename T>
void op_sgd(const Tensor<T> &t, const Tensor<T> &dt, Tensor<T> &out, float lr)
{
    assert(out.h == t.h && out.w == t.w);
    assert(t.h == dt.h && t.w == dt.w);

    SGDFunc<T> f{lr};
    if (t.on_device && dt.on_device && out.on_device) {
        op_elemwise_binary_w_bcast_gpu(f, t, dt, out);
    } else {
        assert(0);
    }
}

//This operator performs element-wise addition of "a" and "b" and 
//stores the result in tensor "out"
template <typename T>
void op_add(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out)
{
    assert(out.h == a.h && out.w == a.w);
    assert((a.h == b.h && a.w == b.w) || (a.h == b.h && b.w == 1) || (a.w == b.w && b.h == 1));
    AddFunc<T> f;
    if (a.on_device && b.on_device && out.on_device) {
        op_elemwise_binary_w_bcast_gpu(f, a, b, out);
    } else {
        assert(0);
    }
}


//This operator performs element-wise addition of "a" and constant b
//stores the result in tensor "out"
template <typename T>
void op_add(const Tensor<T> &a, T b, Tensor<T> &out)
{
    assert(out.h == a.h && out.w == a.w);
    AddConstFunc<T> f{b};
    if (a.on_device && out.on_device) {
        op_elemwise_unary_gpu(f, a, out);
    } else {
        assert(0);
    }
}

//This operator performs element-wise subtraction of "b" from "a" and 
//stores the result in tensor "out"
template <typename T>
void op_subtract(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out)
{
    assert(out.h == a.h && out.w == a.w);
    assert((a.h == b.h && a.w == b.w) || (a.h == b.h && b.w == 1) || (a.w == b.w && b.h == 1));
    SubtractFunc<T> f;
    if (a.on_device && b.on_device && out.on_device) {
        op_elemwise_binary_w_bcast_gpu(f, a, b, out);
    } else {
        assert(0);
    }
}

//This operator performs element-wise subtraction of constant b from "a"
//stores the result in tensor "out"
template <typename T>
void op_subtract(const Tensor<T> &a, T b, Tensor<T> &out)
{
    assert(out.h == a.h && out.w == a.w);
    SubtractConstFunc<T> f{b};
    if (a.on_device && out.on_device) {
        op_elemwise_unary_gpu(f, a, out);
    } else {
        assert(0);
    }
}

//This operator performs exponentiation of the tensor
template <typename T>
void op_exp(const Tensor<T> &t, Tensor<T> &out)
{
    assert(out.h == t.h && out.w == t.w);
    ExpFunc<T> f;
    if (t.on_device && out.on_device) {
        op_elemwise_unary_gpu(f, t, out);
    } else {
        assert(0);
    }
}

//This operator performs exponentiation of the tensor with scaling
template <typename T>
void op_exp_with_scaling(const Tensor<T> &t, Tensor<T> &out)
{
    assert(out.h == t.h && out.w == t.w);
    ExpFunc<T> f;
    if (t.on_device && out.on_device) {
        op_elemwise_unary_gpu(f, t, out);
    } else {
        assert(0);
    }
}

//This operator performs logarithm (base e) of the tensor
template <typename T>
void op_log(const Tensor<T> &t, Tensor<T> &out)
{
    assert(out.h == t.h && out.w == t.w);
    LogFunc<T> f;
    if (t.on_device && out.on_device) {
        op_elemwise_unary_gpu(f, t, out);
    } else {
        assert(0);
    }
}

//This operator performs element-wise multiplication of "a" and "b" and 
//stores the result in tensor "out"
template <typename T>
void op_multiply(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out)
{
    assert(out.h == a.h && out.w == a.w);
    assert((a.h == b.h && a.w == b.w) || (a.h == b.h && b.w == 1) || (a.w == b.w && b.h == 1));
    MultiplyFunc<T> f;
    if (a.on_device && b.on_device && out.on_device) {
        op_elemwise_binary_w_bcast_gpu(f, a, b, out);
    } else {
        assert(0);
    }
}

//This operator performs element-wise multiplication of "a" and "b" and 
//stores the result in tensor "out"
template <typename T1, typename T2>
void op_dequantize(const Tensor<T1> &a, const Tensor<T2> &b, Tensor<T2> &out)
{
    assert(out.h == a.h && out.w == a.w);
    assert((a.h == b.h && a.w == b.w) || (a.h == b.h && b.w == 1) || (a.w == b.w && b.h == 1));
    DequantizeFunc<T1, T2> f;
    if (a.on_device && b.on_device && out.on_device) {
        op_elemwise_binary_w_bcast_gpu(f, a, b, out);
    } else {
        assert(0);
    }
}

//This operator performs element-wise multiplication of "a" and "b" and 
//stores the result in tensor "out"
template <typename T, typename OutT>
void op_multiply(const Tensor<T> &a, const Tensor<T> &b, Tensor<OutT> &out)
{
    assert(out.h == a.h && out.w == a.w);
    assert((a.h == b.h && a.w == b.w) || (a.h == b.h && b.w == 1) || (a.w == b.w && b.h == 1));
    MultiplyWithTypecastFunc<T, OutT> f;
    if (a.on_device && b.on_device && out.on_device) {
        op_elemwise_binary_w_bcast_gpu(f, a, b, out);
    } else {
        assert(0);
    }
}

//This operator performs element-wise multiplication of "a" and constant b
//stores the result in tensor "out"
template <typename T>
void op_multiply(const Tensor<T> &a, T b, Tensor<T> &out)
{
    assert(out.h == a.h && out.w == a.w);
    MultiplyConstFunc<T> f{b};
    if (a.on_device && out.on_device) {
        op_elemwise_unary_gpu(f, a, out);
    } else {
        assert(0);
    }
}
//This operator performs element-wise inv(erse)division of "a" and constant b i.e. b/a
//stores the result in tensor "out"
template <typename T>
void op_inv_divide(const Tensor<T> &a, T b, Tensor<T> &out)
{
    assert(out.h == a.h && out.w == a.w);
    InvDivideConstFunc<T> f{b};
    if (a.on_device && out.on_device) {
        op_elemwise_unary_gpu(f, a, out);
    } else {
        assert(0);
    }
}

//This operator checks if tensor "a" and "b" are the same
//and stores in the "out" tensor value 0 at places where "a" and "b" are not equal 
//and 1 at places where "a" and "b" are equal.
template <typename AT, typename BT, typename OutT>
void op_equal(const Tensor<AT> &a, const Tensor<BT> &b, Tensor<OutT> &out)
{
    assert(out.h == a.h && out.w == a.w);
    assert((a.h == b.h && a.w == b.w));
    EqualityFunc<AT, BT, OutT> f;
    if (a.on_device && b.on_device && out.on_device) {
        op_elemwise_binary_w_bcast_gpu(f, a, b, out);
    } else {
        assert(0);
    }
}

template <typename AT>
void op_round_int8(const Tensor<AT> &a, Tensor<int8_t> &out)
{
    assert(out.h == a.h && out.w == a.w);
    RoundInt8Func<AT> f;
    if (a.on_device && out.on_device) {
        op_elemwise_unary_gpu(f, a, out);
    } else {
        assert(0);
    }
}


template <typename AT, typename T, typename OutT>
void op_outlier_extractor(const Tensor<AT> &a, T b, Tensor<OutT> &out)
{
    assert(out.h == a.h && out.w == a.w);
    AbsCompareLTEConstFunc<AT, T, OutT> f{b};
    if (a.on_device && out.on_device) {
        op_elemwise_unary_gpu(f, a, out);
    } else {
        assert(0);
    }
}

//This operator initializes tensor with constant values
template <typename T>
void op_const_init(Tensor<T> &t, float init_val)
{
    ConstInitFunc<float> f{init_val};
    if (t.on_device) {
        op_elemwise_unary_gpu(f, t, t);
    } else {
        for (int i = 0; i < t.h; i++) {
            for (int j = 0; j < t.w; j++) {
                Index(t, i, j) = init_val;
            }
        }
    }
}

//This operator initializes tensor with random values
//that are uniformly distributed between min and max
template <typename T>
void op_uniform_init(Tensor<T> &t, T min = 0, T max = 1)
{
    // XXX: Currently, this only works with un-sliced tensor
    assert(t.offset == 0 && t.stride_w == 1);
    if (t.on_device)
    {
        static RandGenGPU g(randgen_seed);
        //curandGenerateUniform generates elements in the range [0,1)
        curandAssert(curandGenerateUniform(g.gen, t.rawp, t.h * t.w));
        //scale the shift the elements to be in the range [min, max)
        op_add<T>(t, min/(max-min), t);
        op_multiply(t, max-min, t);
    } else {
        assert(0);
    }
}


// This operator checks if all elements of two tensors are the "same" (aka close enough) with each other
// For now, let's settle with only CPU implementation of allclose
template <typename T>
bool op_allclose(const Tensor<T> &at, Tensor<T> &bt)
{
    if (at.h != bt.h || at.w != bt.w)
    {
        return false;
    }
    Tensor<T> att;
    if (at.on_device) {
        att =  at.toHost();
    } else {
        att = at;
    }
    Tensor<T> btt;
    if (bt.on_device) {
        btt = bt.toHost();
    } else {
        btt = bt;
    }
    for (int i = 0; i < at.h; i++)
    {
        for (int j = 0; j < at.w; j++)
        {
            // Check if the numbers are close using both relative and absolute tolerances
            T a = Index(att, i, j);
            T b = Index(btt, i, j);
            if (std::abs(a - b) >
                std::max(ISCLOSE_RELTOL * std::max(std::abs(a), std::abs(b)), ISCLOSE_ABSTOL))
            {
                std::cout << "(" << i << "," << j << ") this=" << a << " other=" << b << " diff=" << (a - b) << std::endl;
                return false;
            }
        }
    }
    return true;
}
