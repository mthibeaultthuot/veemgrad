#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct BufferInfo {
  void* ptr;
  size_t size;
  size_t ndim;
  const size_t* shape;
  const size_t* strides;
  int dtype;
};

// API abstract class backend runtime
// Need to be implemented by Cuda, Metal, RocM or more
class BackendRuntime {
 public:
  virtual ~BackendRuntime() = default;

  virtual void init() = 0;

  // Allocate buffers on the device from the matrix
  virtual void* allocate(size_t size) = 0;
  virtual void deallocate(void* ptr) = 0;

  // Handle memory management between cpu and gpu
  virtual bool copy_to_device(void* dst, void* src, size_t size) = 0;
  virtual bool copy_from_device(void* dst, void* src, size_t size) = 0;

  // Compile JIT codegen code
  virtual bool compile(const std::string& kernel_code, const std::string& kernel_name) = 0;

  // Run compile kernel
  virtual bool run_kernel(const std::string& kernel_name, const BufferInfo* inputs,
                          size_t num_inputs, const BufferInfo* outputs, size_t num_outputs,
                          size_t grid_dim, size_t block_dim) = 0;

  // Synchronize the kernel execution -> wait for all kernel to finish running
  virtual void synchronize() = 0;
};
