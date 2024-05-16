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
    std::cout << "Quantized result: " << std::endl;
    std::cout << Q_out.str() << std::endl;

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