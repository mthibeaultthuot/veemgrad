
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "metal_backend.h"

TEST_CASE("D = A * B * C", "[mattul]") {
  MetalRuntime runtime;
  runtime.init();

  constexpr size_t N = 4;
  constexpr size_t SIZE = N * N * sizeof(float);

  float A[N * N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  float B[N * N] = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};

  float C[N * N] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  float D[N * N] = {0};

  void* a_ptr = runtime.allocate(SIZE);
  void* b_ptr = runtime.allocate(SIZE);
  void* c_ptr = runtime.allocate(SIZE);
  void* d_ptr = runtime.allocate(SIZE);

  runtime.copy_to_device(a_ptr, A, SIZE);
  runtime.copy_to_device(b_ptr, B, SIZE);
  runtime.copy_to_device(c_ptr, C, SIZE);

  size_t shape[2] = {N, N};

  BufferInfo A_buf{a_ptr, SIZE, 2, shape, nullptr, 0};
  BufferInfo B_buf{b_ptr, SIZE, 2, shape, nullptr, 0};
  BufferInfo C_buf{c_ptr, SIZE, 2, shape, nullptr, 0};
  BufferInfo D_buf{d_ptr, SIZE, 2, shape, nullptr, 0};

  const std::string fused_kernel = R"(
      using namespace metal;

      kernel void matmul_fused_4x4(
          device const float* A [[buffer(0)]],
          device const float* B [[buffer(1)]],
          device const float* C [[buffer(2)]],
          device float* D [[buffer(3)]],
          uint2 gid [[thread_position_in_grid]]) {

          if (gid.x >= 4 || gid.y >= 4) return;

          float AB = 0.0;
          for (uint j = 0; j < 4; ++j) {
              AB += A[gid.y * 4 + j] * B[j * 4 + gid.x];
          }

          float sum = 0.0;
          for (uint k = 0; k < 4; ++k) {
              float ab = 0.0;
              for (uint j = 0; j < 4; ++j) {
                  ab += A[gid.y * 4 + j] * B[j * 4 + k];
              }
              sum += ab * C[k * 4 + gid.x];
          }

          D[gid.y * 4 + gid.x] = sum;
      })";

  runtime.compile(fused_kernel, "matmul_fused_4x4");

  BufferInfo inputs[] = {A_buf, B_buf, C_buf};
  BufferInfo outputs[] = {D_buf};
  runtime.run_kernel("matmul_fused_4x4", inputs, 3, outputs, 1, 16, 4);

  runtime.copy_from_device(D, d_ptr, SIZE);

  float temp[N * N] = {0};
  float expected[N * N] = {0};

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        temp[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        expected[i * N + j] += temp[i * N + k] * C[k * N + j];
      }
    }
  }

  for (int i = 0; i < N * N; ++i) {
    REQUIRE(D[i] == Catch::Approx(expected[i]));
  }

  std::cout << "Result matrix C = A * B * D (fused):\n";
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      std::cout << D[i * N + j] << " ";
    }
    std::cout << "\n";
  }

  runtime.deallocate(a_ptr);
  runtime.deallocate(b_ptr);
  runtime.deallocate(c_ptr);
  runtime.deallocate(d_ptr);

  REQUIRE(true);
}
