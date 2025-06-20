#pragma once

#include <Metal/Metal.hpp>
#include <cstddef>
#include <vector>

#include "backend.h"

class MetalRuntime : public BackendRuntime {
 public:
  MetalRuntime();
  ~MetalRuntime() override;

  void init() override;
  void* allocate(size_t size) override;
  void deallocate(void* ptr) override;

  bool copy_to_device(void* dst, const void* src, size_t size) override;
  bool copy_from_device(void* dst, const void* src, size_t size) override;

  bool compile(const std::string& kernel_code, const std::string& kernal_name) override;
  bool run_kernel(const std::string& kernel_name, const BufferInfo* inputs, size_t num_inputs,
                  const BufferInfo* outputs, size_t num_outputs, size_t grid_dim,
                  size_t block_dim) override;

  void synchronize() override;
};
