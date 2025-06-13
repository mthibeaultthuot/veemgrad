#pragma once

#include <string>
#include <vector>

class BackendRuntime {
 public:
  virtual ~BackendRuntime() = default;

  virtual void init() = 0;
  virtual std::vector<void*> get_devices() = 0;
  virtual bool allocate_buffers(const std::vector<size_t>& sizes) = 0;
  virtual bool deallocate_buffers() = 0;
  virtual bool copy_to_device(size_t, const void*, size_t) = 0;
  virtual bool copy_from_device(size_t, void*, size_t) = 0;
  virtual void synchronize() = 0;
  virtual bool run_kernel(const std::string&, const std::vector<void*>&, std::vector<void*>&,
                          const std::vector<size_t>&, const std::vector<int>& metadata = {}) = 0;
};
