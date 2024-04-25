#include <getopt.h>
#include <limits.h>

#include "utils/tensor.cuh"
#include "ops/op_mm_quantize.cuh"

unsigned long long randgen_seed = 0;

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

    Tensor<float> X{m, k};
    Index(X, 0, 0) = 2;
    Index(X, 0, 1) = 45;
    Index(X, 0, 2) = -1;
    Index(X, 0, 3) = -17;
    Index(X, 0, 4) = -1;
    Index(X, 1, 0) = 0;
    Index(X, 1, 1) = 12;
    Index(X, 1, 2) = 3;
    Index(X, 1, 3) = -63;
    Index(X, 1, 4) = 2;
    Index(X, 2, 0) = -1;
    Index(X, 2, 1) = 37;
    Index(X, 2, 2) = -1;
    Index(X, 2, 3) = -83;
    Index(X, 2, 4) = 0;

    Tensor<float> W{k, n};
    Index(W, 0, 0) = -1;
    Index(W, 0, 1) = 0;
    Index(W, 1, 0) = 2;
    Index(W, 1, 1) = 0;
    Index(W, 2, 0) = 0;
    Index(W, 2, 1) = -2;
    Index(W, 3, 0) = 3;
    Index(W, 3, 1) = -2;   
    Index(W, 4, 0) = -1;
    Index(W, 4, 1) = 2;  
    
    std::cout << "size of 1 element of X: " << CHAR_BIT * sizeof(Index(X, 0, 0)) << " bits" << std::endl;
    std::cout << "size of 1 element of W: " << CHAR_BIT * sizeof(Index(W, 0, 0)) << " bits" << std::endl;

    Tensor<int32_t> O{m, n};
    op_mm_quantize(X, Y, O);


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