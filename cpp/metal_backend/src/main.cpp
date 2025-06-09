#include <iostream>
#include "metal/metal_matmul.hpp"
#include <cstdlib>



void print_matrix(const std::vector<float>& mat, size_t rows, size_t cols, const std::string& name) {
    std::cout << name << " (" << rows << "x" << cols << "):\n";
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << mat[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

int main() {
    const char* libPathEnv = std::getenv("METAL_LIB_PATH");
    size_t M = 4, K = 4, N = 4;

    std::vector<float> A = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    std::vector<float> B = {
        17, 18, 19, 20,
        21, 22, 23, 24,
        25, 26, 27, 28,
        29, 30, 31, 32
    };

    std::vector<float> C;


    MatMulKernel matmul(
        libPathEnv,
        "matmul",
        std::max({M * K, K * N, M * N})
    );

     auto start = std::chrono::high_resolution_clock::now();

    matmul.run(A, B, C, M, K, K, N);

     auto end = std::chrono::high_resolution_clock::now();
      auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      std::cout << duration_us << " microseconds\n\n";

        print_matrix(A, M, K, "a");
        print_matrix(B, K, N, "b");
        print_matrix(C, M, N, "c");

  return 0;
}
