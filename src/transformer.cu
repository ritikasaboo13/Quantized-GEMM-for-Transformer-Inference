#pragma once

#include "ops/op_elemwise.cuh"
#include "ops/op_reduction.cuh"
#include "ops/op_mm.cuh"
#include "ops/op_softmax.cuh"
#include "ops/op_layernorm.cuh"
#include "modules/attention.cuh"
#include "modules/linear.cuh"
#include "modules/mlp.cuh"

unsigned long long randgen_seed = 0;

void Encoder(const Tensor<float> &X, Tensor<float> &output, int n_heads, int n_blocks, int d_ff) {
    // TODO:: assert dimensions of X & output
    std::cout << "ENCODER\n=======\n";
    std::cout << "X: \n" << X.str() << std::endl;
    std::cout << "output: \n" << output.str() << std::endl;

    int d_model = X.w;
    int d_k = d_model/n_heads;
    int d_v = d_model/n_heads;

    for (int i = 0; i < n_blocks; i++) {

        // MULTI HEAD SELF ATTENTION
        Tensor<float> attnHeadOut{output.h, d_v, true};
        Tensor<float> attnHeadOutHost{output.h, d_v, false};

        Tensor<float> multiHeadOut{output.h, d_model, true};
        Tensor<float> multiHeadOutHost{output.h, d_model, false};

        for (int j = 0; j < n_heads; j++) {
            AttentionLayer<float> attn{d_model, d_k, d_v, true};
            attn.init_uniform(); // load random weights for W_q, W_k, W_v
            if (i == 0) {
                attn.forward(X, X, attnHeadOut);
            } else {
                attn.forward(output, output, attnHeadOut);
            }
            std::cout << "attnHeadOut: \n" << attnHeadOut.str() << std::endl;

            attnHeadOut.toHost(attnHeadOutHost);
            for (int row = 0; row < attnHeadOutHost.h; row++) {
                for (int col = j*d_v; col < (j+1)*d_v; col++) {
                    Index(multiHeadOutHost, row, col) = Index(attnHeadOutHost, row, col - (j*d_v));
                }
            }
        }
        multiHeadOutHost.toDevice(multiHeadOut);
        std::cout << "multiHeadOut: \n" << multiHeadOut.str() << std::endl;
        Tensor<float> W_O{d_model, d_model, true};
        op_uniform_init(W_O, -1.0f, 1.0f);
        op_mm(multiHeadOut, W_O, output);
        std::cout << "output: \n" << output.str() << std::endl;

        // ADD & NORM
        op_add(output, multiHeadOut, output);
        op_layernorm(output, output);

        // FEED FORWARD [FFN(x) = max(0, xW1 + b1 )W2 + b2]
        Tensor<float> ffnOut{X.h, d_model, true};

        LinearLayer<float> ll1{d_model, d_ff, true};
        ll1.init_uniform(); // load random weights for W1, b1
        ll1.forward(output, ffnOut);
        op_relu(ffnOut, ffnOut);

        LinearLayer<float> ll2{d_ff, d_model, true};
        ll2.init_uniform(); // load random weights for W2, b2
        ll2.forward(ffnOut, output);

        // ADD & NORM
        op_add(output, multiHeadOut, output);
        op_layernorm(output, output);
    }
}

void Decoder(const Tensor<float> &X, Tensor<float> &enc_output, Tensor<float> &output, 
             int n_heads, int n_blocks, int d_ff) {
    // TODO:: assert dimensions of X, enc_output & output
    std::cout << "DECODER\n=======\n";
    std::cout << "X: \n" << X.str() << std::endl;
    std::cout << "enc_output: \n" << enc_output.str() << std::endl;
    std::cout << "output: \n" << output.str() << std::endl;

    int d_model = X.w;
    int d_k = d_model/n_heads;
    int d_v = d_model/n_heads;

    for (int i = 0; i < n_blocks; i++) {

        // MULTI HEAD SELF ATTENTION
        Tensor<float> attnHeadOut{output.h, d_v, true};
        Tensor<float> attnHeadOutHost{output.h, d_v, false};

        Tensor<float> multiHeadOut{output.h, d_model, true};
        Tensor<float> multiHeadOutHost{output.h, d_model, false};

        for (int j = 0; j < n_heads; j++) {
            AttentionLayer<float> attn{d_model, d_k, d_v, true};
            attn.init_uniform(); // load random weights for W_q, W_k, W_v
            if (i == 0) {
                attn.forward(X, X, attnHeadOut);
            } else {
                attn.forward(output, output, attnHeadOut);
            }
            std::cout << "attnHeadOut: \n" << attnHeadOut.str() << std::endl;

            attnHeadOut.toHost(attnHeadOutHost);
            for (int row = 0; row < attnHeadOutHost.h; row++) {
                for (int col = j*d_v; col < (j+1)*d_v; col++) {
                    Index(multiHeadOutHost, row, col) = Index(attnHeadOutHost, row, col - (j*d_v));
                }
            }
        }
        multiHeadOutHost.toDevice(multiHeadOut);
        std::cout << "multiHeadOut: \n" << multiHeadOut.str() << std::endl;
        Tensor<float> W_O{d_model, d_model, true};
        op_uniform_init(W_O, -1.0f, 1.0f);
        op_mm(multiHeadOut, W_O, output);
        std::cout << "output: \n" << output.str() << std::endl;

        // ADD & NORM
        op_add(output, multiHeadOut, output);
        op_layernorm(output, output);

        // MULTI HEAD CROSS ATTENTION
        for (int j = 0; j < n_heads; j++) {
            AttentionLayer<float> attn{d_model, d_k, d_v, true};
            attn.init_uniform(); // load random weights for W_q, W_k, W_v
            attn.forward(output, enc_output, attnHeadOut);
            std::cout << "attnHeadOut: \n" << attnHeadOut.str() << std::endl;

            attnHeadOut.toHost(attnHeadOutHost);
            for (int row = 0; row < attnHeadOutHost.h; row++) {
                for (int col = j*d_v; col < (j+1)*d_v; col++) {
                    Index(multiHeadOutHost, row, col) = Index(attnHeadOutHost, row, col - (j*d_v));
                }
            }
        }
        multiHeadOutHost.toDevice(multiHeadOut);
        std::cout << "multiHeadOut: \n" << multiHeadOut.str() << std::endl;
        op_uniform_init(W_O, -1.0f, 1.0f);
        op_mm(multiHeadOut, W_O, output);
        std::cout << "output: \n" << output.str() << std::endl;

        // ADD & NORM
        op_add(output, multiHeadOut, output);
        op_layernorm(output, output);

        // FEED FORWARD [FFN(x) = max(0, xW1 + b1 )W2 + b2]
        Tensor<float> ffnOut{X.h, d_model, true};

        LinearLayer<float> ll1{d_model, d_ff, true};
        ll1.init_uniform(); // load random weights for W1, b1
        ll1.forward(output, ffnOut);
        op_relu(ffnOut, ffnOut);

        LinearLayer<float> ll2{d_ff, d_model, true};
        ll2.init_uniform(); // load random weights for W2, b2
        ll2.forward(ffnOut, output);

        // ADD & NORM
        op_add(output, multiHeadOut, output);
        op_layernorm(output, output);
    }
}

int main() {
    Tensor<float> X{6, 8, true};
    op_uniform_init(X, -1.0f, 1.0f);
    Tensor<float> output{6, 8, true};
    int n_heads = 4;
    int n_blocks = 2;
    Tensor<float> enc_output{X.h, X.w, true};
    Tensor<float> dec_output{X.h, X.w, true};

    Encoder(X, enc_output, n_heads, n_blocks, 8);
    Decoder(X, enc_output, dec_output, n_heads, n_blocks, 8);

    // TODO:: MLP & op_softmax

    return 0;
}
