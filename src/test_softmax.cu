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
    Tensor<float> input(1, 3, true); // 1 batch, 3 classes
    Tensor<float> output(1, 3, true);
    Tensor<float> expected_output(1, 3, true);

    // Initialize input tensor
    Index(input, 0, 0) = 1.0f;
    Index(input, 0, 1) = 2.0f;
    Index(input, 0, 2) = 3.0f;

    // Expected softmax output calculated manually or using a reliable tool/library
    Index(expected_output, 0, 0) = exp(1.0f) / (exp(1.0f) + exp(2.0f) + exp(3.0f));
    Index(expected_output, 0, 1) = exp(2.0f) / (exp(1.0f) + exp(2.0f) + exp(3.0f));
    Index(expected_output, 0, 2) = exp(3.0f) / (exp(1.0f) + exp(2.0f) + exp(3.0f));

    // Execute softmax operation
    op_softmax(input, output);

    // Print results
    std::cout << "Input:" << std::endl;
    printTensor(input, 1, 3);
    std::cout << "Output:" << std::endl;
    printTensor(output, 1, 3);
    std::cout << "Expected Output:" << std::endl;
    printTensor(expected_output, 1, 3);

    // Check and print if the output matches expected output
    if (checkApproxEqual(output, expected_output, 1, 3)) {
        std::cout << "Test passed." << std::endl;
    } else {
        std::cout << "Test failed." << std::endl;
    }

    return 0;
}
