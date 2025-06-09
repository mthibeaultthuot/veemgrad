#include "metal_matmul.hpp"
#include "metal_wrapper.h"
#include <vector>
#include <cstring>
#include <iostream>
#include <cstdlib>

extern "C" {

MatMulKernelHandle matmul_factory_create(const char* metallib_path, const char* function_name, size_t max_elements) {
   return static_cast<MatMulKernelHandle>(
        new MatMulKernel(metallib_path, function_name, max_elements));
}

void matmul_destroy(MatMulKernelHandle handle) {
            delete static_cast<MatMulKernel*>(handle);
}

void warmUp(MatMulKernelHandle handle) {
   
        auto* kernel = static_cast<MatMulKernel*>(handle);
        kernel->warmUp();
}

bool matmul_run(MatMulKernelHandle handle,
                const float* matrix_a,
                const float* matrix_b,
                float* result,
                size_t rows_a, size_t cols_a,
                size_t rows_b, size_t cols_b) {
    
    
        auto* kernel = static_cast<MatMulKernel*>(handle);
        
        std::vector<float> vecA(matrix_a, matrix_a + rows_a * cols_a);
        std::vector<float> vecB(matrix_b, matrix_b + rows_b * cols_b);
        std::vector<float> vecC;
        
        bool success = kernel->run(vecA, vecB, vecC, rows_a, cols_a, rows_b, cols_b);
        
        if (success) {
            std::memcpy(result, vecC.data(), vecC.size() * sizeof(float));
        }
        
        return success;
    
}

}
