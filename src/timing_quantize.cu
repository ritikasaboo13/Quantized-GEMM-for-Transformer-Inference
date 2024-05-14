#include <getopt.h>
#include <sys/time.h>
#include <unistd.h>
#include "utils/tensor.cuh"
#include "ops/op_mm.cuh"
#include "ops/op_elemwise.cuh"
#include "ops/op_reduction.cuh"

unsigned long long randgen_seed = 0;

//unsigned long long randgen_seed = 0;
void test_matmul(int m, int n, int k, bool on_gpu) {

    struct timeval start, finish;
    struct timeval qstart, qfinish;
    
    Tensor<float> X{m, k, on_gpu};
    op_uniform_init(X);
    Tensor<float> W{k, n, on_gpu};
    op_uniform_init(W);
    Tensor<float> C{m, n, on_gpu};
    Tensor<float> qC{m, n, on_gpu};
    //Tensor<float> C2{n, m, on_gpu};
    //op_mm(B.transpose(), A.transpose(), C2);
    //assert(op_allclose(C2.transpose(), C)); // test transpose

    gettimeofday(&start, NULL);
    op_mm(X, W, C);
    cudaDeviceSynchronize();
    gettimeofday(&finish, NULL);
    
    double t = (finish.tv_sec - start.tv_sec) * 1000000 + (finish.tv_usec - start.tv_usec);
    std::cout << "Time taken for matmul: " << std::endl;
    std::cout << t / 1000 << std::endl;  // ms


    // quantized matmul steps 
    gettimeofday(&qstart, NULL);
    op_quantized_mm(X, W, qC, 127.0f);
    cudaDeviceSynchronize();
    gettimeofday(&qfinish, NULL);

    double t2 = (qfinish.tv_sec - qstart.tv_sec) * 1000000 + (qfinish.tv_usec - qstart.tv_usec);
    std::cout << "Time taken for quantized matmul: " << std::endl;
    std::cout << t2 / 1000 << std::endl;  // ms

    Tensor<float> Q_error{X.h, W.w, on_gpu};
    op_subtract(C, qC, Q_error); 
    std::cout << "Mean Quantization error: " <<std::endl;
    std::cout << Q_error.toHost().mean() << std::endl;


}

int main(int argc, char *argv[]) {
    bool test_gpu = true;
    int test_m = 2048, test_n = 256, test_k= 256;
    for (;;) {
        switch (getopt(argc, argv, "s:cm:n:k:")) {
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
    test_matmul(test_m, test_n, test_k, test_gpu);
    return 0;
}
