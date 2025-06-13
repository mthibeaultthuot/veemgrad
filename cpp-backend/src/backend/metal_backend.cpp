#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "metal_backend.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <iostream>

MetalRuntime::MetalRuntime() { std::cout << "MetalRuntime constructed\n"; }
MetalRuntime::~MetalRuntime() { std::cout << "MetalRuntime destroyed\n"; }

void MetalRuntime::init() {}
std::vector<void*> MetalRuntime::get_devices() { return {}; }
bool MetalRuntime::allocate_buffers(const std::vector<size_t>& sizes) { return true; }
bool MetalRuntime::deallocate_buffers() { return true; }
bool MetalRuntime::copy_to_device(size_t, const void*, size_t) { return true; }
bool MetalRuntime::copy_from_device(size_t, void*, size_t) { return true; }
void MetalRuntime::synchronize() {}
bool MetalRuntime::run_kernel(const std::string&, const std::vector<void*>&, std::vector<void*>&,
                              const std::vector<size_t>&, const std::vector<int>&) {
  std::cout << "Running Metal kernel (stub)\n";
  return true;
}
