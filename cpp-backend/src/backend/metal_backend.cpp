#include <cstddef>
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <iostream>

#include "metal_backend.h"

MetalRuntime::MetalRuntime() { std::cout << "MetalRuntime constructed\n"; }
MetalRuntime::~MetalRuntime() { std::cout << "MetalRuntime destroyed\n"; }

void MetalRuntime::init() {
  device_ = MTL::CreateSystemDefaultDevice();
  commandQueue_ = device_->newCommandQueue();
}

uintptr_t MetalRuntime::allocate(size_t size) {
  MTL::Buffer* buffer = device_->newBuffer(size, MTL::StorageModeShared);
  return reinterpret_cast<uintptr_t>(buffer);
}

void MetalRuntime::deallocate(uintptr_t ptr) {}

bool MetalRuntime::copy_to_device(uintptr_t dst, uintptr_t src, size_t size) {
  auto* buffer = reinterpret_cast<MTL::Buffer*>(dst);
  const void* src_ptr = reinterpret_cast<const void*>(src);
  std::memcpy(buffer->contents(), src_ptr, size);
  return true;
}

bool MetalRuntime::copy_from_device(uintptr_t dst, uintptr_t src, size_t size) {
  auto* buffer = reinterpret_cast<MTL::Buffer*>(src);
  void* dst_ptr = reinterpret_cast<void*>(dst);
  std::memcpy(dst_ptr, buffer->contents(), size);
  return true;
}

bool MetalRuntime::compile(const std::string& kernel_code, const std::string& kernal_namne) {
  NS::String* sourceString = NS::String::string(kernel_code.c_str(), NS::UTF8StringEncoding);
  NS::String* functionString = NS::String::string(kernal_namne.c_str(), NS::UTF8StringEncoding);

  NS::Error* error = nullptr;

  MTL::Library* library = device_->newLibrary(sourceString, nullptr, &error);
  MTL::Function* function = library->newFunction(functionString);
  pipeline_ = device_->newComputePipelineState(function, &error);

  cmdBuffer_ = commandQueue_->commandBuffer();
  encoder_ = cmdBuffer_->computeCommandEncoder();
  encoder_->setComputePipelineState(pipeline_);

  return error == nullptr;
};

bool MetalRuntime::run_kernel(const std::string& kernel_name, const RustBufferInfo* inputs,
                              size_t num_inputs, const RustBufferInfo* outputs, size_t num_outputs,
                              size_t grid_dim, size_t block_dim) {
  int buffer_index = 0;
  for (size_t i = 0; i < num_inputs; i++) {
    encoder_->setBuffer(reinterpret_cast<MTL::Buffer*>(inputs[i].ptr), 0, buffer_index++);
  }

  for (size_t i = 0; i < num_outputs; i++) {
    encoder_->setBuffer(reinterpret_cast<MTL::Buffer*>(outputs[i].ptr), 0, buffer_index++);
  }

  MTL::Size gridSize = MTL::Size(grid_dim, grid_dim, 1);
  MTL::Size threadGroupSize = MTL::Size(block_dim, 1, 1);

  encoder_->dispatchThreads(gridSize, threadGroupSize);
  encoder_->endEncoding();
  cmdBuffer_->commit();
  cmdBuffer_->waitUntilCompleted();

  return true;
}

void MetalRuntime::synchronize() {}

std::unique_ptr<MetalRuntime> new_metal_runtime() { return std::make_unique<MetalRuntime>(); }
