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
static bool on_gpu = true;


void test_layerNorm() {
    
    Tensor<float> X{2, 4, true};
    Tensor<float> Y{2, 4, true};
    op_uniform_init(X, -3.0f, 3.0f);
    op_layernorm(X, Y); 
    std::cout << "X= \n" << X.str() << std::endl;
    std::cout << "Y= \n" << Y.str() << std::endl;

}

Tensor<float> assign_to_tensor(std::vector<float>& values, int rows, int columns, Tensor<float> mat) {
    Tensor<float> X_host;
    if (mat.on_device) {
        X_host = mat.toHost();
    } else {
        X_host = mat;
    } 
    int index = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            Index(X_host, i, j) = values[index++];
        }
    }
    Tensor<float> X;
    if (mat.on_device) {
        X = X_host.toDevice();
    } else {
        X = X_host;
    }
    return X; 
}

void test_singleHeadAttn(const Tensor<float> X, Tensor<float> attnOutput, int d_model) {
    // Single Head attention
    // Singlehead Attention parameters 
    // For single headed attention, d_model, d_k, d_v are all same as d_model
    int d_k = 2;
    int d_q = d_k;
    int d_v = 4; 
    AttentionLayer<float> attn{d_model, d_k, d_v, on_gpu};
    std::vector<float> values = {0.2482, 0.2993, 0.0759, 0.4924, 0.5452, 0.0620};
    attn.W_q.t = assign_to_tensor(values, attn.W_q.t.h, attn.W_q.t.w, attn.W_q.t);
    values = {0.8953, 0.6716, 0.7721, 0.8670, 0.0594, 0.7789};
    attn.W_k.t = assign_to_tensor(values, attn.W_k.t.h, attn.W_k.t.w, attn.W_k.t);
    values = {0.9583, 0.6320, 0.1432, 0.0040, 0.0569, 0.1546, 0.6056, 0.3590, 0.6802,
        0.0089, 0.1739, 0.7423};
    attn.W_v.t = assign_to_tensor(values, attn.W_v.t.h, attn.W_v.t.w, attn.W_v.t);
    attn.forward(X, attnOutput);
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
    // int embedding_size = 3; 
    int d_model = 3;
    int d_k = 2;
    int d_q = d_k;
    int d_v = 4; 

    // Initialize the embedding vectors for a sequence 
    Tensor<float> X_host(seq_length, d_model);
    std::vector<float> values = {0.3374, -0.1778, -0.3035, -0.5880,  0.3486,  0.6603};
    X_host = assign_to_tensor(values, X_host.h, X_host.w, X_host);

    Tensor<float> X;
    if (on_gpu) {
        X = X_host.toDevice();
    } else {
        X = X_host;
    } 

    // Test single head attention
    Tensor<float> attnOutput(X.h, d_v, true);
    test_singleHeadAttn(X, attnOutput, d_model);
    std::cout << attnOutput.str() << std::endl;

    Tensor<float> attnOutputTorch(X.h, d_v, true);
    values = {0.0661,  0.1509,  0.1394, -0.0064,  0.0743,  0.1694,  0.1561, -0.0105};
    attnOutputTorch = assign_to_tensor(values, attnOutputTorch.h, attnOutputTorch.w, attnOutputTorch);

    assert(op_allclose(attnOutputTorch, attnOutput));
    std::cout << "Single head attention tested! " << std::endl; 

    return 0; 
}
