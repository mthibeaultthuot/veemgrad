#pragma once

#include <Metal/Metal.hpp>
#include <vector>

#include "backend.h"

class MetalRuntime : public BackendRuntime {
 public:
  MetalRuntime();
  ~MetalRuntime() override;

  void init() override;
  std::vector<void*> get_devices() override;
  bool allocate_buffers(const std::vector<size_t>& sizes) override;
  bool deallocate_buffers() override;
  bool copy_to_device(size_t index, const void* host_data, size_t size) override;
  bool copy_from_device(size_t index, void* host_data, size_t size) override;
  void synchronize() override;
  bool run_kernel(const std::string& kernel_code, const std::vector<void*>& inputs,
                  std::vector<void*>& outputs, const std::vector<size_t>& shapes,
                  const std::vector<int>& metadata) override;

 private:
  std::vector<MTL::Device*> devices;
  std::vector<MTL::Buffer*> buffers;
  MTL::CommandQueue* queue = nullptr;
};
