#include <__config>
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

void* MetalRuntime::allocate(size_t size) {
  MTL::Buffer* buffer = device_->newBuffer(size, MTL::StorageModeShared);
  return static_cast<void*>(buffer);
}

void MetalRuntime::deallocate(void* ptr) {}

bool MetalRuntime::copy_to_device(void* dst, void* src, size_t size) {
  auto* buffer = static_cast<MTL::Buffer*>(dst);
  std::memcpy(buffer->contents(), src, size);
  return true;
}

bool MetalRuntime::copy_from_device(void* dst, void* src, size_t size) {
  auto* buffer = static_cast<MTL::Buffer*>(src);
  std::memcpy(dst, buffer->contents(), size);
  return true;
}

bool MetalRuntime::compile(const std::string& kernel_code, const std::string& kernal_namne) {
  NS::String* sourceString = NS::String::string(kernel_code.c_str(), NS::UTF8StringEncoding);
  NS::String* functionString = NS::String::string(kernal_namne.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nullptr;
  MTL::Library* library = device_->newLibrary(sourceString, nullptr, &error);
  MTL::Function* function = library->newFunction(functionString);
  pipeline_ = device_->newComputePipelineState(function, &error);
  return error == nullptr;
};

bool MetalRuntime::run_kernel(const std::string& kernel_name, const BufferInfo* inputs,
                              size_t num_inputs, const BufferInfo* outputs, size_t num_outputs,
                              size_t grid_dim, size_t block_dim) {
  MTL::CommandBuffer* cmdBuffer = commandQueue_->commandBuffer();
  MTL::ComputeCommandEncoder* encoder = cmdBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipeline_);

  int buffer_index = 0;
  for (size_t i = 0; i < num_inputs; i++) {
    encoder->setBuffer(static_cast<MTL::Buffer*>(inputs[i].ptr), 0, buffer_index++);
  }

  for (size_t i = 0; i < num_outputs; i++) {
    encoder->setBuffer(static_cast<MTL::Buffer*>(outputs[i].ptr), 0, buffer_index++);
  }

  MTL::Size gridSize = MTL::Size(grid_dim, grid_dim, 1);
  MTL::Size threadGroupSize = MTL::Size(block_dim, 1, 1);

  encoder->dispatchThreads(gridSize, threadGroupSize);
  encoder->endEncoding();
  cmdBuffer->commit();
  cmdBuffer->waitUntilCompleted();

  return true;
}

void MetalRuntime::synchronize() {}
