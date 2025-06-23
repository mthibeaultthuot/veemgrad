#pragma once

#include <Metal/Metal.hpp>
#include <cstddef>
#include <memory>
#include <vector>

#include "backend.h"

// Void * not working with CXX instead use uintptr_t
// CXX don't support inheritance
// class MetalRuntime : public backend {
class MetalRuntime {
 public:
  MetalRuntime();
  ~MetalRuntime();

  void init();
  uintptr_t allocate(size_t size);
  void deallocate(uintptr_t ptr);

  bool copy_to_device(uintptr_t dst, uintptr_t src, size_t size);
  bool copy_from_device(uintptr_t dst, uintptr_t src, size_t size);

  bool compile(const std::string& kernel_code, const std::string& kernel_name);
  bool run_kernel(const std::string& kernel_name, const RustBufferInfo* inputs, size_t num_inputs,
                  const RustBufferInfo* outputs, size_t num_outputs, size_t grid_dim,
                  size_t block_dim);

  void synchronize();

 private:
  MTL::Device* device_;
  MTL::CommandQueue* commandQueue_;
  MTL::ComputePipelineState* pipeline_;
  MTL::ComputeCommandEncoder* encoder_;
  MTL::CommandBuffer* cmdBuffer_;
};

std::unique_ptr<MetalRuntime> new_metal_runtime();
