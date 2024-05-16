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

void Encoder(const Tensor<float> &X, Tensor<float> &output, int n_heads, int n_blocks) {
    // TODO:: assert dimensions of X & output
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

            attnHeadOut.toHost(attnHeadOutHost);
            for (int row = 0; row < attnHeadOutHost.h; row++) {
                for (int col = 0; col < attnHeadOutHost.w; col++) {
                    Index(multiHeadOutHost, row, col) = Index(attnHeadOutHost, row, col - (i*d_v));
                }
            }
            multiHeadOutHost.toDevice(multiHeadOut);
        }

        // ADD & NORM
        op_add(output, multiHeadOut, output);
        op_layernorm(output, output);

        // FEED FORWARD [FFN(x) = max(0, xW1 + b1 )W2 + b2]
        Tensor<float> ffnOut{X.h, d_model, true};

        LinearLayer<float> ll1{d_model, d_model, true};
        ll1.init_uniform(); // load random weights for W1, b1
        ll1.forward(output, ffnOut);
        op_relu(ffnOut, ffnOut);

        LinearLayer<float> ll2{d_model, d_model, true};
        ll2.init_uniform(); // load random weights for W2, b2
        ll2.forward(ffnOut, output);

        // ADD & NORM
        op_add(output, multiHeadOut, output);
        op_layernorm(output, output);
    }
}

void Decoder(const Tensor<float> &X, Tensor<float> &enc_output, Tensor<float> &output, 
             int n_heads, int n_blocks) {
    // TODO:: assert dimensions of X, enc_output & output
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

            attnHeadOut.toHost(attnHeadOutHost);
            for (int row = 0; row < attnHeadOutHost.h; row++) {
                for (int col = 0; col < attnHeadOutHost.w; col++) {
                    Index(multiHeadOutHost, row, col) = Index(attnHeadOutHost, row, col - (i*d_v));
                }
            }
            multiHeadOutHost.toDevice(multiHeadOut);
        }

        // ADD & NORM
        op_add(output, multiHeadOut, output);
        op_layernorm(output, output);

        // MULTI HEAD CROSS ATTENTION
        Tensor<float> attnHeadOut{output.h, d_v, true};
        Tensor<float> attnHeadOutHost{output.h, d_v, false};

        Tensor<float> multiHeadOut{output.h, d_model, true};
        Tensor<float> multiHeadOutHost{output.h, d_model, false};

        for (int j = 0; j < n_heads; j++) {
            AttentionLayer<float> attn{d_model, d_k, d_v, true};
            attn.init_uniform(); // load random weights for W_q, W_k, W_v
            attn.forward(output, enc_output, attnHeadOut);

            attnHeadOut.toHost(attnHeadOutHost);
            for (int row = 0; row < attnHeadOutHost.h; row++) {
                for (int col = 0; col < attnHeadOutHost.w; col++) {
                    Index(multiHeadOutHost, row, col) = Index(attnHeadOutHost, row, col - (i*d_v));
                }
            }
            multiHeadOutHost.toDevice(multiHeadOut);
        }

        // ADD & NORM
        op_add(output, multiHeadOut, output);
        op_layernorm(output, output);

        // FEED FORWARD [FFN(x) = max(0, xW1 + b1 )W2 + b2]
        Tensor<float> ffnOut{X.h, d_model, true};

        LinearLayer<float> ll1{d_model, d_model, true};
        ll1.init_uniform(); // load random weights for W1, b1
        ll1.forward(output, ffnOut);
        op_relu(ffnOut, ffnOut);

        LinearLayer<float> ll2{d_model, d_model, true};
        ll2.init_uniform(); // load random weights for W2, b2
        ll2.forward(ffnOut, output);

        // ADD & NORM
        op_add(output, multiHeadOut, output);
        op_layernorm(output, output);
    }
}

void Transformer(const Tensor<float> &X, Tensor<float> &output, int n_heads, int n_blocks) {
    Tensor<float> enc_output{X.h, X.w, true};
    Tensor<float> dec_output{X.h, X.w, true};

    Encoder(X, enc_output, n_heads, n_blocks);
    Decoder(X, enc_output, dec_output, n_heads, n_blocks);

    // TODO:: MLP & op_softmax
}
