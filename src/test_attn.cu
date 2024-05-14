#include <getopt.h>

#include "utils/tensor.cuh"
#include "ops/op_mm.cuh"
#include "ops/op_elemwise.cuh"
#include "ops/op_reduction.cuh"
#include "ops/op_softmax.cuh"
#include "modules/attention.cuh"
#include "modules/linear.cuh"
#include "modules/mlp.cuh"

unsigned long long randgen_seed = 0;
static bool on_gpu = true;

void test_singleHeadAttn(const Tensor<float> X, Tensor<float> attnOutput, int d_model) {
    // Single Head attention
    // Singlehead Attention parameters 
    // For single headed attention, d_model, d_k, d_v are all same as d_model ~ 512
    int d_k = d_model, d_v = d_model; 
    AttentionLayer<float> attn{d_model, d_k, d_v, on_gpu};
    attn.init_uniform(); // attn.parameters(); ???
    attn.forward(X, attnOutput);
    std::cout << "X= \n" << X.str() << std::endl;
    std::cout << "output= \n" << attnOutput.str() << std::endl;
}

void test_multiHeadAttn(const Tensor<float> X, Tensor<float> attnOutput, int d_model, int heads) {
    // Multihead Attention parameters
    // For multi headed attention, d_model ~ 512 and d_k, d_v are d_model / num_of_heads 
    int d_k = d_model/heads, d_v = d_model/heads; 
    Tensor<float> attnOutputPerHead(X.h, d_model, true);

    for(int i = 0; i < heads; ++i) {
        AttentionLayer<float> attn{d_model, d_k, d_v, on_gpu};
        attn.init_uniform(); // attn.parameters(); ???
        attn.forward(X, attnOutputPerHead);
        std::cout << "X= \n" << X.str() << std::endl;
        std::cout << "output= \n" << attnOutputPerHead.str() << std::endl;
        // concatenate attnOutput to attnOutput

    }

}


int main() {
    // Attention variables 
    int seq_length = 2; 
    int d_model = 512, heads = 8; 
    float scale_factor = 1.0 / std::sqrt(d_model);

    // Feedforward variables
    int n_layers = 2;
    std::vector<int> layer_dims;
    int d_ff = 2048;

    // Initialize the embedding vectors for a sequence 
    Tensor<float> X(seq_length, d_model, true); 
    op_uniform_init(X, -scale_factor, scale_factor);

    // Test single head attention
    Tensor<float> attnOutput(X.h, d_model, true);
    test_singleHeadAttn(X, attnOutput, d_model);
    std::cout << "Single head attention tested! " << std::endl; 

    // Test mutli head attention : incomplete
    //test_multiHeadAttn(X, attnOutput, d_model, heads);
    //std::cout << "Multi head attention tested! " << std::endl; 
    // INCOMPLETE MULTI HEAD ATTENTION, DO NOT USE RIGHT NOW

    // Add & Norm
    op_add(X, attnOutput, X);
    // INCOMPLETE: need to write layer normalization 

    // Feedforward Layer: 2 layer mlp with d_model => d_ff => d_model and relu in between 
    for (int i = 0; i < n_layers - 1; i++)
    {
        layer_dims.push_back(d_ff);
    }
    layer_dims.push_back(d_model); // last layer's out dimension is always 10 (# of digits) // 

    Tensor<float> X_ffd(seq_length, d_model, true); 
    MLP<float> mlp(seq_length, d_model, layer_dims, on_gpu);
    mlp.init();
    mlp.forward(X, X_ffd);
    return 0;
}