#pragma once
#include "utils/tensor.cuh"
#include "ops/op_elemwise.cuh"
#include "ops/op_reduction.cuh"
#include <cmath>
#define ELEMWISE_BLOCK_DIM 32 

template <typename T>
__global__ void op_scaling_kernel(Tensor<T> logits, Tensor<T> logits_max, Tensor<T> logits_temp){
    // each thread computes 1 row of logits
    int idx = threadIdx.x + (blockDim.x * blockIdx.x); // gives index of row
    for(int i=0; i<logits.w; i++){
        Index(logits_temp, idx, i) = Index(logits, idx, i) - Index(logits_max, idx, 0);
    }
}

template <typename T>
__global__ void op_exp_kernel(Tensor<T> logits_){
    // each thread computes 1 row of logits
    int idx = threadIdx.x + (blockDim.x * blockIdx.x); // gives index of row
    for(int i=0; i<logits_.w; i++){
        Index(logits_, idx, i) = exp(Index(logits_, idx, i));
    }
}

template <typename T>
__global__ void op_prob_kernel(Tensor<T> logits_, Tensor<T> logits_sum, Tensor<T> logits_softmax){
    // each thread computes 1 row of logits
    int idx = threadIdx.x + (blockDim.x * blockIdx.x); // gives index of row
    for(int i=0; i<logits_.w; i++){
        Index(logits_softmax, idx, i) = Index(logits_, idx, i) / Index(logits_sum, idx, 0);
    }
}

template <typename T>
__global__ void op_cel_kernel(Tensor<T> logits_, Tensor<char> targets, Tensor<T> cel){
    // each thread computes 1 row of logits
    int idx = threadIdx.x + (blockDim.x * blockIdx.x); // gives index of row
    
    T logit_at_target_index = Index(logits_, idx, Index(targets, idx, 0));
    Index(cel, idx, 0) = 0.0f - log(logit_at_target_index);
}

template <typename T>
__global__ void op_dlogits_kernel(Tensor<T> logits_softmax, Tensor<char> targets, const int b, Tensor<T> d_logits){
    // each thread computes 1 row of logits
    int idx = threadIdx.x + (blockDim.x * blockIdx.x); // gives index of row
    
    for (int i=0; i<logits_softmax.w; i++){
        int t = Index(targets, idx, 0);
        if (i == t){
            Index(d_logits, idx, i) = (Index(logits_softmax, idx, i) - 1.0f) / b;
        }
        else{
            Index(d_logits, idx, i) = Index(logits_softmax, idx, i) / b;
        }
    }
}


//This function calculates the cross_entropy loss from the "logits" tensor for a batch of training innput
//and the batch's corresponding "target" label tensor and returns the average loss of the batch.
//It also returns the gradient of the logits tensor.
template <typename T>
T op_cross_entropy_loss(const Tensor<T> &logits, const Tensor<char> &targets,
                               Tensor<T> &d_logits)
{
    assert(logits.h == targets.h && logits.h == d_logits.h);
    assert(logits.w == d_logits.w);
    assert(targets.w == 1);

    assert(logits.on_device && targets.on_device && d_logits.on_device); 

    //Lab-2: please add your code here. 
    //You need to define separate GPU kernel function(s) and launch them here
    //In order to calculate d_logits, you should derive what its values should be 
    //symbolically.

    // std::cout << logits.str();
    // std::cout << targets.str();
    const int b = logits.h; // batch_size
    Tensor<float> logits_temp{logits.h, logits.w, logits.on_device}; // shape (32,10)
    Tensor<float> logits_softmax{logits.h, logits.w, logits.on_device}; // shape (32,10)
    Tensor<float> logits_max{logits.h, 1, logits.on_device}; // shape (32, 1)
    Tensor<float> logits_sum{logits.h, 1, logits.on_device}; // shape (32, 1)
    Tensor<float> cel{logits.h, 1, logits.on_device}; // shape (32, 1)
    Tensor<float> cel_batch{1, 1, targets.on_device};

    dim3 dimGrid((b + ELEMWISE_BLOCK_DIM - 1)/ELEMWISE_BLOCK_DIM);
    dim3 dimBlock(ELEMWISE_BLOCK_DIM, 1, 1);
    op_max(logits, logits_max);
    op_scaling_kernel<<<dimGrid, dimBlock>>>(logits, logits_max, logits_temp);
    op_exp_kernel<<<dimGrid, dimBlock>>>(logits_temp);
    // std::cout << logits_temp.str() << std::endl;
    op_sum(logits_temp, logits_sum);
    // std::cout << logits_sum.str() << std::endl;
    op_prob_kernel<<<dimGrid, dimBlock>>>(logits_temp, logits_sum, logits_softmax);
    // std::cout << logits_softmax.str() << std::endl;
    op_dlogits_kernel<<<dimGrid, dimBlock>>>(logits_softmax, targets, b, d_logits);
    // std::cout << d_logits.str() << std::endl;
    op_cel_kernel<<<dimGrid, dimBlock>>>(logits_softmax, targets, cel);
    // std::cout << cel.str() << std::endl;
    op_sum(cel, cel_batch);
    // std::cout << Index(cel_batch.toHost(), 0, 0) << std::endl;
    auto loss = Index(cel_batch.toHost(), 0, 0) / targets.h;

    return loss;

}
