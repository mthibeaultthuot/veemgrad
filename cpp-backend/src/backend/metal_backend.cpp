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

void MetalRuntime::init() {}

void* MetalRuntime::allocate(size_t size) { return nullptr; }

void MetalRuntime::deallocate(void* ptr) {}

bool MetalRuntime::copy_to_device(void* dst, const void* src, size_t size) { return false; }

bool MetalRuntime::copy_from_device(void* dst, const void* src, size_t size) { return false; }

bool MetalRuntime::compile(const std::string& kernel_code, const std::string& kernal_namne) {
  return false;
};

bool MetalRuntime::run_kernel(const std::string& kernel_name, const BufferInfo* inputs,
                              size_t num_inputs, const BufferInfo* outputs, size_t num_outputs,
                              size_t grid_dim, size_t block_dim) {
  return false;
}

void MetalRuntime::synchronize() {}
