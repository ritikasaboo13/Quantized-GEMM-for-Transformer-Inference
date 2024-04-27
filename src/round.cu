#include <getopt.h>

#include "utils/tensor.cuh"

unsigned long long randgen_seed = 1;

static bool on_gpu = true;
#define ELEMWISE_BLOCK_DIM 32 // thread block has 32x32 threads

template <typename T>
class IdentityFunc
{
public:
    __host__ __device__ T operator()(T a)
    {
        return a;
    }
};


template <typename OpFunc, typename AT, typename T>
__global__ void op_elemwise_unary_kernel(OpFunc f, Tensor<AT> t, Tensor<T> out)
{
    int i = threadIdx.y + ELEMWISE_BLOCK_DIM * blockIdx.y; // row
    int j = threadIdx.x + ELEMWISE_BLOCK_DIM * blockIdx.x; // col
    if (i < t.h && j < t.w){
        Index(out, i, j) = f(Index(t, i, j));
    }
}

template <typename OpFunc, typename AT, typename T>
void op_elemwise_unary_gpu(OpFunc f, const Tensor<AT> &t, Tensor<T> &out)
{
    int M_ = (t.w + ELEMWISE_BLOCK_DIM - 1)/ ELEMWISE_BLOCK_DIM;
    int N_ = (t.h + ELEMWISE_BLOCK_DIM - 1)/ ELEMWISE_BLOCK_DIM;
    dim3 gridDim(M_, N_, 1);
    dim3 blockDim(ELEMWISE_BLOCK_DIM, ELEMWISE_BLOCK_DIM, 1);
    op_elemwise_unary_kernel<<<gridDim, blockDim>>>(f, t, out);
}


template <typename AT, typename OutT>
void op_round(const Tensor<AT> &a, Tensor<OutT> &out)
{
    assert(out.h == a.h && out.w == a.w);
    IdentityFunc<AT> f;
    if (a.on_device && out.on_device) {
        op_elemwise_unary_gpu(f, a, out);
    } else {
        assert(0);
    }
}


int main(){

    Tensor<float> X_host{1, 1};
    Index(X_host, 0, 0) = 90.567;
    Tensor<float> X;
    if (on_gpu) {
        X = X_host.toDevice();
    } else {
        X = X_host;
    } 
    Tensor<int8_t> Y{1, 1, on_gpu};
    op_round(X, Y);
    std::cout << "size of 1 element of X: " << CHAR_BIT * sizeof(Index(X, 0, 0)) << " bits "  << "X=" << X.str();
    std::cout << "size of 1 element of Y: " << CHAR_BIT * sizeof(Index(Y, 0, 0)) << " bits "  << "Y=" << Y.str();
    
    return 0;
}