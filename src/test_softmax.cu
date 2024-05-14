#include <iostream>
#include <cmath>
#include "utils/tensor.cuh"
#include "ops/op_softmax.cuh"

// Simple function to print tensor contents
template <typename T>
void printTensor(const Tensor<T>& tensor, int batch_size, int num_classes) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_classes; j++) {
            std::cout << Index(tensor, i, j) << " ";
        }
        std::cout << std::endl;
    }
}

// Function to check if two tensors are approximately equal
template <typename T>
bool checkApproxEqual(const Tensor<T>& tensor1, const Tensor<T>& tensor2, int batch_size, int num_classes, T epsilon = 1e-5) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_classes; j++) {
            if (std::abs(Index(tensor1, i, j) - Index(tensor2, i, j)) > epsilon) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    // Define the input tensor and expected output tensor
    Tensor<float> input(1, 3, false); // 1 batch, 3 classes
    Tensor<float> output(1, 3, true);
    Tensor<float> expected_output(1, 3, false);

    // Initialize input tensor
    Index(input, 0, 0) = 1.0;
    Index(input, 0, 1) = 2.0;
    Index(input, 0, 2) = 5.0;

    // Expected softmax output calculated manually or using a reliable tool/library
    Index(expected_output, 0, 0) = exp(Index(input, 0, 0)) / (exp(Index(input, 0, 0)) + exp(Index(input, 0, 1)) + exp(Index(input, 0, 2)));
    Index(expected_output, 0, 1) = exp(Index(input, 0, 1)) / (exp(Index(input, 0, 0)) + exp(Index(input, 0, 1)) + exp(Index(input, 0, 2)));
    Index(expected_output, 0, 2) = exp(Index(input, 0, 2)) / (exp(Index(input, 0, 0)) + exp(Index(input, 0, 1)) + exp(Index(input, 0, 2)));

    // Execute softmax operation
    Tensor<float> input_ = input.toDevice();
    op_softmax_new(input_, output);   // CHANGE

    // Print results
    std::cout << "Input:" << std::endl;
    printTensor(input, 1, 3);
    std::cout << "Output:" << std::endl;
    printTensor(output.toHost(), 1, 3);
    std::cout << "Expected Output:" << std::endl;
    printTensor(expected_output, 1, 3);

    // Check and print if the output matches expected output
    if (checkApproxEqual(output.toHost(), expected_output, 1, 3)) {
        std::cout << "Test passed." << std::endl;
    } else {
        std::cout << "Test failed." << std::endl;
    }

    return 0;
}
