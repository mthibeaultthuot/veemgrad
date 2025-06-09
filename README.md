# I just want to build a tiny deep learning framework to learn how deep learning works.

# Current benchmark

- matmul : ~250-400 µs

---

# Roadmap

### Tensor

| Task                                 | Statut  |
| ------------------------------------ | ------- |
| `matmul`                             | ✅ Done |
| `broadcast` + `add`                  | ☐ Todo  |
| `ReLU`, `Sigmoid`                    | ☐ Todo  |
| `Softmax`                            | ☐ Todo  |
| `sub`, `mul`, `div`                  | ☐ Todo  |
| `sum`, `mean`, `max`                 | ☐ Todo  |
| `exp`, `log`, `pow`                  | ☐ Todo  |
| `transpose`                          | ☐ Todo  |
| `reshape`, `squeeze`, `broadcast_to` | ☐ Todo  |

### Autograd

| Task                                 | Status  |
| ------------------------------------ | ------- |
| Gradients for `matmul`, `ReLU`       | ✅ Done |
| Gradients for `sigmoid`, `softmax`   | ☐ Todo  |
| Operation tracing (tape-based graph) | ☐ Todo  |
| Reverse-mode backpropagation         | ☐ Todo  |

### Layers

| Task                            | Status |
| ------------------------------- | ------ |
| `Linear` (dense: matmul + bias) | ☐ Todo |
| `Dropout`                       | ☐ Todo |
| `LayerNorm`                     | ☐ Todo |
| `Embedding`                     | ☐ Todo |
| `Conv2d` (optional, for CNNs)   | ☐ Todo |

### 4. Optimizers

| Task                    | Status |
| ----------------------- | ------ |
| `SGD`                   | ☐ Todo |
| `Adam`                  | ☐ Todo |
| Learning rate scheduler | ☐ Todo |

### Training

| Task                          | Status |
| ----------------------------- | ------ |
| Forward + Backward pass       | ☐ Todo |
| Parameter update logic        | ☐ Todo |
| Loss calculation & printing   | ☐ Todo |
| Dataset loading (e.g., MNIST) | ☐ Todo |

### Binding / ffi

| Task                                | Status  |
| ----------------------------------- | ------- |
| `build.rs` for compiling Metal/CUDA | ✅ Done |
| FFI between Rust and C++            | ✅ Done |
| `bindgen` integration in Rust       | ☐ Todo  |

---

# Building for metal

### 1. set the current path for your matal cpp backend

```shell
export MACOSX_DEPLOYMENT_TARGET=15.0
export METAL_LIB_PATH=/Users/mathieuthibeault-thuot/Documents/GitHub/veem-dl/cpp/metal_backend/build/
export CPP_METAL_PATH=/Users/mathieuthibeault-thuot/Documents/GitHub/veem-dl/cpp/metal_backend/src/metal/
```
