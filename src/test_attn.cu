#include <getopt.h>

#include "utils/tensor.cuh"
#include "ops/op_mm.cuh"
#include "ops/op_elemwise.cuh"
#include "ops/op_reduction.cuh"
#include "ops/op_layernorm.cuh"
#include "ops/op_softmax.cuh"
#include "modules/attention.cuh"
#include "modules/linear.cuh"
#include "modules/mlp.cuh"

unsigned long long randgen_seed = 0;
static bool on_gpu = false;


void test_layerNorm() {
    
    Tensor<float> X{2, 4, true};
    Tensor<float> Y{2, 4, true};
    op_uniform_init(X, -3.0f, 3.0f);
    op_layernorm(X, Y); 
    std::cout << "X= \n" << X.str() << std::endl;
    std::cout << "Y= \n" << Y.str() << std::endl;

}

void init_W_for_test(const AttentionLayer<float> attn) {
    // Initialize W_q
    Index(attn.W_q.t, 0, 0) = 0.3816;
    Index(attn.W_q.t, 0, 1) = 0.4737;
    Index(attn.W_q.t, 1, 0) = 0.6913;
    Index(attn.W_q.t, 1, 1) = 0.9397;

    // Initialize W_k
    Index(attn.W_k.t, 0, 0) = 0.0814;
    Index(attn.W_k.t, 0, 1) = 0.1727;
    Index(attn.W_k.t, 1, 0) = 0.0551;
    Index(attn.W_k.t, 1, 1) = 0.6236;

    // Initialize W_v
    Index(attn.W_v.t, 0, 0) = 0.8240;
    Index(attn.W_v.t, 0, 1) = 0.5434;
    Index(attn.W_v.t, 0, 2) = 0.6511;
    Index(attn.W_v.t, 0, 3) = 0.7613;
    Index(attn.W_v.t, 1, 0) = 0.3102;
    Index(attn.W_v.t, 1, 1) = 0.8053;
    Index(attn.W_v.t, 1, 2) = 0.3108;
    Index(attn.W_v.t, 1, 3) = 0.3031;        
    }

void test_singleHeadAttn(const Tensor<float> X, Tensor<float> attnOutput, int d_model) {
    // Single Head attention
    // Singlehead Attention parameters 
    // For single headed attention, d_model, d_k, d_v are all same as d_model
    int d_k = d_model;
    int d_v = d_model; 
    AttentionLayer<float> attn{d_model, d_k, d_v, on_gpu};

    init_W_for_test(attn);
    
    attn.forward(X, attnOutput);
    // std::cout << Index(attn.W_v.t, 1, 3) << std::endl;
    // std::cout << "X= \n" << X.str() << std::endl;
    // std::cout << "output= \n" << attnOutput.str() << std::endl;
}

void test_multiHeadAttn(const Tensor<float> X, Tensor<float> attnOutput, int d_model, int heads) {
    // Multihead Attention parameters
    // For multi headed attention, d_model ~ 512 and d_k, d_v are d_model / num_of_heads 
    int d_k = d_model/heads, d_v = d_model/heads; 
    Tensor<float> attnOutputPerHead(X.h, d_model/heads, true); 
    Tensor<float> attnOutputHost(attnOutput.h, attnOutput.w, false);  
    Tensor<float> attnOutputPerHeadHost(attnOutputPerHead.h, attnOutputPerHead.w, false);  

    // attnOutputPerHead has shape (seq_length, d_model/heads)
    // attnOutput has shape (seq_length, d_model)

    for(int i = 0; i < heads; ++i) { 
        AttentionLayer<float> attn{d_model, d_k, d_v, on_gpu};
        attn.init_uniform(); // attn.parameters(); ???
        attn.forward(X, attnOutputPerHead);
        std::cout << "X= \n" << X.str() << std::endl;
        std::cout << "output= \n" << attnOutputPerHead.str() << std::endl;
        // concatenate attnOutputPerHead to attnOutput
        attnOutputPerHead.toHost(attnOutputPerHeadHost);

        for(int row = 0; row < X.h; row++) {
            for(int col = i*(d_model/heads); col < (i+1)*(d_model/heads); col++) {
                //std::cout << "access: " << Index(attnOutputPerHead, row, col - (i*(d_model/heads))) << std::endl; 
                Index(attnOutputHost, row, col) = Index(attnOutputPerHeadHost, row, col - (i*(d_model/heads)));
            }
        }
    }
    
    attnOutputHost.toDevice(attnOutput);
}


int main() {
    // Attention variables 
    int seq_length = 2; 
    int d_model = 2;
    int heads = 4; 
    float scale_factor = 1.0 / std::sqrt(d_model);

    // Feedforward variables
    int n_layers = 2;
    std::vector<int> layer_dims;
    int d_ff = 2048;

    // Initialize the embedding vectors for a sequence 
    Tensor<float> X_host(seq_length, d_model);
    Index(X_host, 0, 0) = 0.3374;
    Index(X_host, 0, 1) = -0.1778;
    Index(X_host, 1, 0) = -0.3035;
    Index(X_host, 1, 1) = -0.5880;

    Tensor<float> X;
    if (on_gpu) {
        X = X_host.toDevice();
    } else {
        X = X_host;
    } 
    // op_uniform_init(X, -scale_factor, scale_factor);

    // Test single head attention
    Tensor<float> attnOutput(X.h, d_model, true);
    test_singleHeadAttn(X, attnOutput, d_model);

    // Tensor<float> attnOutputTorch_host{X.h, d_model};
    // Index(attnOutputTorch_host, 0, 0) = -0.0088;
    // Index(attnOutputTorch_host, 0, 1) = -0.3661;
    // Index(attnOutputTorch_host, 0, 2) = -0.0837;
    // Index(attnOutputTorch_host, 0, 3) = -0.0548;
    // Index(attnOutputTorch_host, 1, 0) = -0.0122;
    // Index(attnOutputTorch_host, 1, 1) = -0.3768;
    // Index(attnOutputTorch_host, 1, 2) = -0.0932;
    // Index(attnOutputTorch_host, 1, 3) = -0.0682;
    // Tensor<float> attnOutputTorch{X.h, d_model, true};
    // if (on_gpu) {
    //     attnOutputTorch = attnOutputTorch_host.toDevice();
    // } else {
    //     attnOutputTorch = attnOutputTorch_host;
    // } 
    // assert(op_allclose(attnOutputTorch, attnOutput));
    // std::cout << "Single head attention tested! " << std::endl; 

    // // Attention variables 
    // int seq_length = 2; 
    // int d_model = 512, heads = 8; 
    // float scale_factor = 1.0 / std::sqrt(d_model);

    // // Feedforward variables
    // int n_layers = 2;
    // std::vector<int> layer_dims;
    // int d_ff = 2048;

    // // Initialize the embedding vectors for a sequence 
    // Tensor<float> X(seq_length, d_model, true); 
    // op_uniform_init(X, -1.0f, 1.0f);

    // // Test single head attention
    // Tensor<float> attnOutput(X.h, d_model, true);
    // test_singleHeadAttn(X, attnOutput, d_model);
    // std::cout << "Single head attention tested! " << std::endl; 

    // // Test multi head attention
    // test_multiHeadAttn(X, attnOutput, d_model, heads);
    // std::cout << "Multi head attention tested! " << std::endl; 

    // // Add & Norm
    // Tensor<float> addNormOutput(X.h, d_model, true);
    // op_add(X, attnOutput, addNormOutput);
    // op_layernorm(addNormOutput, addNormOutput);
    // // INCOMPLETE: need to write layer normalization 

    // // Feedforward Layer: 2 layer mlp with d_model => d_ff => d_model and relu in between 
    // for (int i = 0; i < n_layers - 1; i++)
    // {
    //     layer_dims.push_back(d_ff);
    // }
    // layer_dims.push_back(d_model); // last layer's out dimension is always 10 (# of digits) // 

    // Tensor<float> outputFfd(seq_length, d_model, true); 
    // MLP<float> mlp(seq_length, d_model, layer_dims, on_gpu);
    // mlp.init();
    // mlp.forward(addNormOutput, outputFfd);

    return 0; 
}
