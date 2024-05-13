#include <getopt.h>

#include "utils/tensor.cuh"
#include "ops/op_mm.cuh"
#include "ops/op_elemwise.cuh"
#include "ops/op_reduction.cuh"
#include "ops/op_softmax.cuh"

unsigned long long randgen_seed = 0;

int main() {
    int n = 2;
    int d_model = 8;
    int d_k = 4;
    float scale_factor = 1.0 / std::sqrt(d_k);

    Tensor<float> X(n, d_model, true);
    op_uniform_init(X, -scale_factor, scale_factor);
    std::cout << "X= \n" << X.str() << std::endl;

    Tensor<float> W_q(d_model, d_k, true);
    op_uniform_init(W_q, -scale_factor, scale_factor);
    std::cout << "W_q= \n" << W_q.str() << std::endl;

    Tensor<float> W_k(d_model, d_k, true);
    op_uniform_init(W_k, -scale_factor, scale_factor);
    std::cout << "W_k= \n" << W_k.str() << std::endl;

    Tensor<float> W_v(d_model, d_k, true);
    op_uniform_init(W_v, -scale_factor, scale_factor);
    std::cout << "W_v= \n" << W_v.str() << std::endl;

    Tensor<float> Q(X.h, W_q.w, true);
    op_mm(X, W_q, Q);
    std::cout << "Q= \n" << Q.str() << std::endl;

    Tensor<float> K(X.h, W_k.w, true);
    op_mm(X, W_k, K);
    std::cout << "K= \n" << K.str() << std::endl;

    Tensor<float> V(X.h, W_v.w, true);
    op_mm(X, W_v, V);
    std::cout << "V= \n" << V.str() << std::endl;

    Tensor<float> K_transpose = K.transpose();
    Tensor<float> QK_T(Q.h, K_transpose.w, true);
    op_mm(Q, K_transpose, QK_T);
    std::cout << "QK_T= \n" << QK_T.str() << std::endl;

    Tensor<float> scaled_QK_T(QK_T.h, QK_T.w, true);
    Tensor<float> scale_tensor(QK_T.h, QK_T.w, true);
    op_const_init(scale_tensor, scale_factor);
    op_multiply(QK_T, scale_tensor, scaled_QK_T);
    std::cout << "scaled_QK_T= \n" << scaled_QK_T.str() << std::endl;

    Tensor<float> softmax_QK_T(QK_T.h, QK_T.w, true);
    op_softmax(scaled_QK_T, softmax_QK_T);
    std::cout << "softmax_QK_T= \n" << softmax_QK_T.str() << std::endl;

    Tensor<float> output(Q.h, V.w, true);
    op_mm(softmax_QK_T, V, output);
    std::cout << "output= \n" << output.str() << std::endl;

    return 0;
}