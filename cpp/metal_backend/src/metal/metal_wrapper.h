
#pragma once

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* MatMulKernelHandle;

MatMulKernelHandle matmul_factory_create(const char* metallib_path, const char* function_name, size_t max_elements);
void matmul_destroy(MatMulKernelHandle handle);


void warmUp(MatMulKernelHandle handle);

bool matmul_run(MatMulKernelHandle handle,
                const float* matrix_a,
                const float* matrix_b,
                float* result,
                size_t rows_a, size_t cols_a,
                size_t rows_b, size_t cols_b);

#ifdef __cplusplus
}
#endif

