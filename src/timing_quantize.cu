#include <getopt.h>
#include <sys/time.h>
#include <unistd.h>
#include "utils/tensor.cuh"
#include "ops/op_mm.cuh"
#include "ops/op_elemwise.cuh"
#include "ops/op_reduction.cuh"

unsigned long long randgen_seed = 0;

//unsigned long long randgen_seed = 0;
void test_matmul(int m, int n, int k, bool on_gpu, double* times) {

    struct timeval start, finish;
    struct timeval qstart, qfinish; 
    
    Tensor<float> X{m, k, on_gpu};
    op_uniform_init(X, -1.0f, 1.0f);
    Tensor<float> W{k, n, on_gpu};
    op_uniform_init(W, -1.0f, 1.0f);
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
    times[0] = t/1000;

    // quantized matmul steps 
    gettimeofday(&qstart, NULL);
    Tensor<float> Cx{X.h, 1, on_gpu};
    op_absmax(X, Cx);
    Tensor<float> Cw{1, W.w, on_gpu};
    op_absmax(W, Cw);
    float range = 127.0;
    Tensor<float> sx{Cx.h, Cx.w, on_gpu};
    op_inv_divide(Cx, range, sx);
    Tensor<float> sw{Cw.h, Cw.w, on_gpu};
    op_inv_divide(Cw, range, sw);
    Tensor<int8_t> X_int8{X.h, X.w, on_gpu};
    op_multiply(X, sx, X_int8);
    Tensor<int8_t> W_int8{W.h, W.w, on_gpu};
    op_multiply(W, sw, W_int8);
    Tensor<int> O_int32{X.h, W.w, on_gpu};
    op_mm(X_int8, W_int8, O_int32);
    Tensor<float> Outer_Product{Cx.h, Cw.w, on_gpu};
    op_mm(Cx, Cw, Outer_Product);
    Tensor<float> O_fp{O_int32.h, O_int32.w, on_gpu};
    op_dequantize(O_int32, Outer_Product, qC);
    op_multiply(qC, 1/(range*range), qC);
    cudaDeviceSynchronize();
    gettimeofday(&qfinish, NULL);

    double t2 = (qfinish.tv_sec - qstart.tv_sec) * 1000000 + (qfinish.tv_usec - qstart.tv_usec);
    std::cout << "Time taken for quantized matmul: " << std::endl;
    std::cout << t2 / 1000 << std::endl;  // ms
    times[1] = t2/1000;

    Tensor<float> Q_error{X.h, W.w, on_gpu};
    op_subtract(C, qC, Q_error); 
    std::cout << "Mean Quantization error: " <<std::endl;
    std::cout << Q_error.toHost().mean() << std::endl;

}

int main(int argc, char *argv[]) {
    bool test_gpu = true;
    int test_m = 2048, test_n = 2048, test_k = 2048;
    double times[2] = {0, 0};
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
    double time, qtime; 

    for(int i=0; i < 50; ++i) {
        test_matmul(test_m, test_n, test_k, test_gpu, times);
        std::cout << times[0] << " " << times[1] <<std::endl;
        time += times[0];
        qtime += times[1];
    }
    std::cout << "Final times" <<std::endl;
    std::cout << time/50 << " " << qtime/50 <<std::endl;
    return 0;
}

