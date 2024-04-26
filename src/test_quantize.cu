#include <getopt.h>

#include "modules/mlp.cuh"
#include "modules/linear.cuh"
#include "modules/sgd.cuh"

#include "ops/op_elemwise.cuh"
#include "ops/op_mm_quantize.cuh"
#include "ops/op_mm.cuh"
#include "ops/op_reduction.cuh"

unsigned long long randgen_seed = 1;

static bool on_gpu = true;


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
vectorwise_quantized_mm(Tensor<float>& X, Tensor<float>& W, Tensor<int16_t>& O)
{   
    // Step-1 Find vector-wise constants
    // Create Cx from X
    Tensor<float> Cx{X.h, 1, on_gpu};
    op_absmax(X, Cx);

    // Create Cw from W
    Tensor<float> Cw{1, W.w, on_gpu};
    op_absmax(W, Cw);
    
    // std::cout << Cx.str() << std::endl;
    // std::cout << Cw.str() << std::endl;

    // Step-2 Quantize
    

    // Step-3 Int8 Matmul


}


void 
llmint8_quantized_mm(Tensor<float>& X, Tensor<float>& W, Tensor<int16_t>& O)
{   
    Tensor<float> outlierIndicesInActivation{1, X.w, on_gpu};
    auto X_row = X.slice(0, 1, 0, X.w);
    float threshold = 6.0;
    std::cout << X_row.str();
    op_outlier_extractor(X_row, threshold, outlierIndicesInActivation);
    std::cout << outlierIndicesInActivation.str();
    
    // // Create X_regular, X_outlier, W_regular, W_outlier
    // Tensor<float> numOutliers{1, 1, on_gpu}; 
    // op_sum(outlierIndicesInActivation, numOutliers);

    // Tensor<float> X_regular{X.h, X.w-Index(numOutliers, 1, 1), on_gpu};
    // Tensor<float> X_outlier{X.h, Index(numOutliers, 1, 1), on_gpu};
    // Tensor<float> W_regular{W.h-Index(numOutliers, 1, 1), W.w, on_gpu};
    // Tensor<float> W_outlier{Index(numOutliers, 1, 1), W.w, on_gpu};


    // Create Cx from X_regular
    // Create Cw from W_regular
    
    // op_mm_quantize(X, W, O);

}

void 
test_quantization(int m, int n, int k, bool on_gpu)
{

    Tensor<float> X_host{m, k};
    Index(X_host, 0, 0) = 2;
    Index(X_host, 0, 1) = -1;
    Index(X_host, 0, 2) = -1;
    Index(X_host, 1, 0) = 0;
    Index(X_host, 1, 1) = 3;
    Index(X_host, 1, 2) = 2;
    Index(X_host, 2, 0) = -1;
    Index(X_host, 2, 1) = -1;
    Index(X_host, 2, 2) = 0;

    Tensor<float> X;
    if (on_gpu) {
        X = X_host.toDevice();
    } else {
        X = X_host;
    } 

    Tensor<float> W_host{k, n};
    Index(W_host, 0, 0) = -1;
    Index(W_host, 0, 1) = 0;
    Index(W_host, 1, 0) = 0;
    Index(W_host, 1, 1) = -2;  
    Index(W_host, 2, 0) = -1;
    Index(W_host, 2, 1) = 2;  
    
    Tensor<float> W;
    if (on_gpu) {
        W = W_host.toDevice();
    } else {
        W = W_host;
    } 
    
    std::cout << "size of 1 element of X: " << CHAR_BIT * sizeof(Index(X, 0, 0)) << " bits" << std::endl;
    std::cout << "size of 1 element of W: " << CHAR_BIT * sizeof(Index(W, 0, 0)) << " bits" << std::endl;

    // Compute time taken for absmax quantized MM of X and W and store in O16
    Tensor<int16_t> O16{m, n, on_gpu};
    vectorwise_quantized_mm(X, W, O16);
    



}




void 
test_llmint8_quantization(int m, int n, int k, bool on_gpu)
{

    Tensor<float> X_host{m, k};
    Index(X_host, 0, 0) = 2;
    Index(X_host, 0, 1) = 45;
    Index(X_host, 0, 2) = -1;
    Index(X_host, 0, 3) = -17;
    Index(X_host, 0, 4) = -1;
    Index(X_host, 1, 0) = 0;
    Index(X_host, 1, 1) = 12;
    Index(X_host, 1, 2) = 3;
    Index(X_host, 1, 3) = -63;
    Index(X_host, 1, 4) = 2;
    Index(X_host, 2, 0) = -1;
    Index(X_host, 2, 1) = 37;
    Index(X_host, 2, 2) = -1;
    Index(X_host, 2, 3) = -83;
    Index(X_host, 2, 4) = 0;

    Tensor<float> X;
    if (on_gpu) {
        X = X_host.toDevice();
    } else {
        X = X_host;
    } 

    Tensor<float> W_host{k, n};
    Index(W_host, 0, 0) = -1;
    Index(W_host, 0, 1) = 0;
    Index(W_host, 1, 0) = 2;
    Index(W_host, 1, 1) = 0;
    Index(W_host, 2, 0) = 0;
    Index(W_host, 2, 1) = -2;
    Index(W_host, 3, 0) = 3;
    Index(W_host, 3, 1) = -2;   
    Index(W_host, 4, 0) = -1;
    Index(W_host, 4, 1) = 2;  
    
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
}

int main(int argc, char *argv[])
{
    bool test_gpu = true;
    int test_m = 3, test_n = 2, test_k= 5;

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