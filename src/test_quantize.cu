#include <getopt.h>
#include <limits.h>

#include "utils/tensor.cuh"
#include "ops/op_mm.cuh"
#include "ops/op_elemwise.cuh"
#include "ops/op_reduction.cuh"

unsigned long long randgen_seed = 0;

void test_elemwise(int m, int n, bool on_gpu)
{
    
    Tensor<float> X{m, n, on_gpu};
    op_const_init(X, 2.0);

    Tensor<float> Y{m, n, on_gpu};
    op_const_init(Y, 3.0);

    Tensor<float> Z{m, n, on_gpu};
    op_add(X, Y, Z);

    Tensor<float> Zref{m, n, false};
    op_const_init(Zref, 5.0);
    assert(op_allclose(Z, Zref));

    Tensor<float> Y2{1, n, on_gpu};
    op_const_init(Y2, 3.0);
    op_add(X, Y2, Z); //test broadcasting
    assert(op_allclose(Z, Zref));

    op_add<float>(X, 3.0, Z);
    assert(op_allclose(Z, Zref));

    std::cout << "op_add passed..." << std::endl;

    op_multiply(X, Y, Z);

    op_const_init(Zref, 6.0);
    assert(op_allclose(Z, Zref));

    op_multiply(X, Y2, Z);
    assert(op_allclose(Z, Zref));

    op_multiply<float>(X, 3.0, Z);
    assert(op_allclose(Z, Zref));

    std::cout << "op_multiply passed..." << std::endl;

    float lr = 0.02;
    Tensor<float> A{m, n, on_gpu};
    op_uniform_init(A);
    Tensor<float> A_host = A.toHost();

    Tensor<float> dA{m, n, on_gpu};
    op_uniform_init(dA);
    Tensor<float> dA_host = dA.toHost();

    Tensor<float> Aref{m, n, false};
    for (int i = 0; i < Aref.h; i++)
    {
        for (int j = 0; j < Aref.w; j++)
        {
          Index(Aref, i, j) = Index(A_host, i, j) - lr * Index(dA_host, i, j);
        }
    }
    op_sgd(A, dA, A, lr);
    assert(op_allclose(A, Aref));

    std::cout << "op_sgd passed..." << std::endl;

}

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

    float threshold = 6.0;

    Tensor<int> X{m, k};
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

    Tensor<int> W{k, n};
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