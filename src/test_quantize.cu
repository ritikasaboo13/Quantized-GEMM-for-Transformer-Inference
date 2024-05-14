#include <getopt.h>

#include "modules/mlp.cuh"
#include "modules/linear.cuh"
#include "modules/sgd.cuh"

#include "ops/op_elemwise.cuh"
#include "ops/op_mm.cuh"
#include "ops/op_reduction.cuh"

unsigned long long randgen_seed = 1;

static bool on_gpu = true;

using namespace std;


bool is_close_enough(float a, float b) {
    if (std::abs(a - b) > 0.0001) {
        return false;
    } else {
        return true;
    }
}
void assert_all_close_enough(Tensor<float> t, std::vector<float> v)
{
    for (int i = 0; i < t.h; i++) {
        for (int j = 0; j < t.w; j++) {
            assert(is_close_enough(Index(t, i, j), v[i*t.w+j]));
        }
    }
}

void 
test_quantization(int m, int n, int k, bool on_gpu)
{

    Tensor<float> X_host{m, k};
    Index(X_host, 0, 0) = 2.0;
    Index(X_host, 0, 1) = -1.0;
    Index(X_host, 0, 2) = -1.0;
    Index(X_host, 1, 0) = 0.0;
    Index(X_host, 1, 1) = 3.0;
    Index(X_host, 1, 2) = 2.0;
    Index(X_host, 2, 0) = -1.0;
    Index(X_host, 2, 1) = -1.0;
    Index(X_host, 2, 2) = 0.0;

    Tensor<float> X;
    if (on_gpu) {
        X = X_host.toDevice();
    } else {
        X = X_host;
    } 

    Tensor<float> W_host{k, n};
    Index(W_host, 0, 0) = -1.0;
    Index(W_host, 0, 1) = 0.0;
    Index(W_host, 1, 0) = 0.0;
    Index(W_host, 1, 1) = -2.0;  
    Index(W_host, 2, 0) = -1.0;
    Index(W_host, 2, 1) = 2.0;  
    
    Tensor<float> W;
    if (on_gpu) {
        W = W_host.toDevice();
    } else {
        W = W_host;
    } 

    Tensor<float> uQ_out{m, n, on_gpu};
    op_mm(X, W, uQ_out);
    std::cout << "Unquantized result: " << std::endl;
    std::cout << uQ_out.str() << std::endl;

    float range = 127.0;
    Tensor<float> Q_out{m, n, on_gpu};
    op_quantized_mm(X, W, Q_out, range);

    Tensor<float> Q_error{m, n, on_gpu};
    op_subtract(uQ_out, Q_out, Q_error);
    cout << "Mean quantization error: " << endl;
    std::cout << Q_error.toHost().mean() << std::endl; 

}

int main(int argc, char *argv[])
{
    bool test_gpu = true;
    int test_m = 3, test_n = 2, test_k= 3;

    for (;;)
    {
        switch (getopt(argc, argv, "s:ch:l:b:e:"))
        {
        case 's':
            randgen_seed = atoll(optarg);
            continue;
        case 'c': //cpu testing only
            test_gpu = false;
            continue;
        case 'm':
            test_m = atoi(optarg);
            continue;
        case 'n':
            test_n = atoi(optarg);
            continue;
        case 'k':
            test_k = atoi(optarg);
            continue;
        case -1:
            break;
        }
        break;
    }
    test_quantization(test_m, test_n, test_k, test_gpu);
    std::cout << "All tests completed successfully!" << std::endl;
    return 0;
}

/*
void 
test_llmint8_quantization(int m, int n, int k, bool on_gpu)
{

    Tensor<float> X_host{m, k};
    Index(X_host, 0, 0) = 2.0;
    Index(X_host, 0, 1) = 45.0;
    Index(X_host, 0, 2) = -1.0;
    Index(X_host, 0, 3) = -17.0;
    Index(X_host, 0, 4) = -1.0;
    Index(X_host, 1, 0) = 0.0;
    Index(X_host, 1, 1) = 12.0;
    Index(X_host, 1, 2) = 3.0;
    Index(X_host, 1, 3) = -63.0;
    Index(X_host, 1, 4) = 2.0;
    Index(X_host, 2, 0) = -1.0;
    Index(X_host, 2, 1) = 37.0;
    Index(X_host, 2, 2) = -1.0;
    Index(X_host, 2, 3) = -83.0;
    Index(X_host, 2, 4) = 0.0;

    Tensor<float> X;
    if (on_gpu) {
        X = X_host.toDevice();
    } else {
        X = X_host;
    } 

    Tensor<float> W_host{k, n};
    Index(W_host, 0, 0) = -1.0;
    Index(W_host, 0, 1) = 0.0;
    Index(W_host, 1, 0) = 2.0;
    Index(W_host, 1, 1) = 0.0;
    Index(W_host, 2, 0) = 0.0;
    Index(W_host, 2, 1) = -2.0;
    Index(W_host, 3, 0) = 3.0;
    Index(W_host, 3, 1) = -2.0;   
    Index(W_host, 4, 0) = -1.0;
    Index(W_host, 4, 1) = 2.0;  
    
    Tensor<float> W;
    if (on_gpu) {
        W = W_host.toDevice();
    } else {
        W = W_host;
    } 
    
    std::cout << "size of 1 element of X: " << CHAR_BIT * sizeof(Index(X, 0, 0)) << " bits" << std::endl;
    std::cout << "size of 1 element of W: " << CHAR_BIT * sizeof(Index(W, 0, 0)) << " bits" << std::endl;

    // // Compute time taken for standard MM of X and W and store in O32
    // Tensor<int32_t> O32{m, n};
    
    // Compute time taken for quantized MM of X and W and store in O16
    Tensor<int16_t> O16{m, n, on_gpu};
}*/


/*
void vectorwise_quantized_mm(Tensor<float>& X, Tensor<float>& W, Tensor<float>& O)
{   
    // Step-1 Find vector-wise constants
    // Create Cx from X
    std::cout << "Input values: " << std::endl;
    std::cout << typeid(Index(X, 0,0)).name() <<" " << X.str() << std::endl;
    std::cout << typeid(Index(W, 0,0)).name() <<" " << W.str() << std::endl;

    Tensor<float> Cx{X.h, 1, on_gpu};
    op_absmax(X, Cx);

    // Create Cw from W
    Tensor<float> Cw{1, W.w, on_gpu};
    op_absmax(W, Cw);

    std::cout << "C_X, C_W values: " << std::endl;
    
    std::cout << Cx.str() << std::endl;
    std::cout << Cw.str() << std::endl;

    // Step-2 Quantize
    float range = 127.0;

    Tensor<float> sx{Cx.h, Cx.w, on_gpu};
    op_inv_divide(Cx, range, sx);

    Tensor<float> sw{Cw.h, Cw.w, on_gpu};
    op_inv_divide(Cw, range, sw);

    // std::cout << sx.str() << std::endl;
    // std::cout << sw.str() << std::endl;

    Tensor<int8_t> X_int8{X.h, X.w, on_gpu};
    op_multiply(X, sx, X_int8);
    Tensor<int8_t> W_int8{W.h, W.w, on_gpu};
    op_multiply(W, sw, W_int8);

    // Step-3 Int8 Matmul
    
    std::cout << "X_int8, W_int8 values: " << std::endl;
    std::cout << X_int8.str() << std::endl;
    std::cout << W_int8.str() << std::endl;
    Tensor<int> O_int32{X.h, W.w, on_gpu};
    op_mm(X_int8, W_int8, O_int32);
    std::cout << "Out_i32 values: " << std::endl;
    std::cout << O_int32.str() << std::endl;

    // Step-4 Outer Product

    std::cout << "Outer product values: " << std::endl;
    Tensor<float> Outer_Product{Cx.h, Cw.w, on_gpu};
    op_mm(Cx, Cw, Outer_Product);
    std::cout << Outer_Product.str() << std::endl;

    // Step-4 Out_16 calculation

    std::cout << "Quantized result" << std::endl;
    Tensor<float> O_fp{O_int32.h, O_int32.w, on_gpu};
    op_dequantize(O_int32, Outer_Product, O_fp);
    op_multiply(O_fp, 1/(range*range), O_fp);
    std::cout << O_fp.str() << std::endl;

    // Actual result without quantization

    Tensor<float> O_fp_unquantized{X.h, W.w, on_gpu};
    op_mm(X, W, O_fp_unquantized);
    std::cout << "Unquantized result: " << std::endl;
    std::cout << O_fp_unquantized.str() << std::endl;

}*/