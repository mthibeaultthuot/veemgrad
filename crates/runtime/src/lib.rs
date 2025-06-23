#[cxx::bridge]
mod ffi {

    unsafe extern "C++" {
        include!("backend/metal_backend.h");

        type MetalRuntime;
        type RustBufferInfo;

        fn new_metal_runtime() -> UniquePtr<MetalRuntime>;

        fn init(self: Pin<&mut MetalRuntime>);
        unsafe fn allocate(self: Pin<&mut MetalRuntime>, size: usize) -> usize;
        unsafe fn deallocate(self: Pin<&mut MetalRuntime>, ptr: usize);

        unsafe fn copy_to_device(
            self: Pin<&mut MetalRuntime>,
            dst: usize,
            src: usize,
            size: usize,
        ) -> bool;
        unsafe fn copy_from_device(
            self: Pin<&mut MetalRuntime>,
            dst: usize,
            src: usize,
            size: usize,
        ) -> bool;

        unsafe fn compile(
            self: Pin<&mut MetalRuntime>,
            kernel_code: &CxxString,
            kernel_name: &CxxString,
        ) -> bool;

        unsafe fn run_kernel(
            self: Pin<&mut MetalRuntime>,
            kernel_name: &CxxString,
            inputs: *const RustBufferInfo,
            num_inputs: usize,
            outputs: *const RustBufferInfo,
            num_outputs: usize,
            grid_dim: usize,
            block_dim: usize,
        ) -> bool;

        fn synchronize(self: Pin<&mut MetalRuntime>);
    }

    struct RustBufferInfo {
        ptr: usize,
        size: usize,
        ndim: usize,
        shape: usize,
        strides: usize,
        dtype: i32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Instant;

    use cxx::let_cxx_string;
    use ffi::{RustBufferInfo, new_metal_runtime};

    #[test]
    fn test_matmul_fused_4x4() {
        let mut binding = new_metal_runtime();
        let mut runtime = binding.pin_mut();
        runtime.as_mut().init();
        const N: usize = 4;
        const SIZE: usize = N * N * std::mem::size_of::<f32>();

        let A: [f32; N * N] = [
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ];

        let B: [f32; N * N] = [
            17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32.,
        ];

        let C: [f32; N * N] = [1.; N * N];

        let mut D = [0f32; N * N];
        let a_ptr = unsafe { runtime.as_mut().allocate(SIZE) };
        let b_ptr = unsafe { runtime.as_mut().allocate(SIZE) };
        let c_ptr = unsafe { runtime.as_mut().allocate(SIZE) };
        let d_ptr = unsafe { runtime.as_mut().allocate(SIZE) };

        unsafe {
            runtime
                .as_mut()
                .copy_to_device(a_ptr as usize, A.as_ptr() as usize, SIZE);
            runtime
                .as_mut()
                .copy_to_device(b_ptr as usize, B.as_ptr() as usize, SIZE);
            runtime
                .as_mut()
                .copy_to_device(c_ptr as usize, C.as_ptr() as usize, SIZE);
        }

        let shape = [N, N];

        let A_buf = RustBufferInfo {
            ptr: a_ptr,
            size: SIZE,
            ndim: 2,
            shape: shape.as_ptr() as usize,
            strides: 0,
            dtype: 0,
        };
        let B_buf = RustBufferInfo {
            ptr: b_ptr,
            size: SIZE,
            ndim: 2,
            shape: shape.as_ptr() as usize,
            strides: 0,
            dtype: 0,
        };
        let C_buf = RustBufferInfo {
            ptr: c_ptr,
            size: SIZE,
            ndim: 2,
            shape: shape.as_ptr() as usize,
            strides: 0,
            dtype: 0,
        };
        let D_buf = RustBufferInfo {
            ptr: d_ptr,
            size: SIZE,
            ndim: 2,
            shape: shape.as_ptr() as usize,
            strides: 0,
            dtype: 0,
        };

        let fused_kernel = r#"
            using namespace metal;

            kernel void matmul_fused_4x4(
                device const float* A [[buffer(0)]],
                device const float* B [[buffer(1)]],
                device const float* C [[buffer(2)]],
                device float* D [[buffer(3)]],
                uint2 gid [[thread_position_in_grid]]) {

                if (gid.x >= 4 || gid.y >= 4) return;

                float sum = 0.0;
                for (uint k = 0; k < 4; ++k) {
                    float ab = 0.0;
                    for (uint j = 0; j < 4; ++j) {
                        ab += A[gid.y * 4 + j] * B[j * 4 + k];
                    }
                    sum += ab * C[k * 4 + gid.x];
                }

                D[gid.y * 4 + gid.x] = sum;
            }
        "#;
        let_cxx_string!(fused_kernel_cpp_str = fused_kernel);
        let_cxx_string!(kernel_function = "matmul_fused_4x4");

        let compiled = unsafe {
            runtime
                .as_mut()
                .compile(&fused_kernel_cpp_str, &kernel_function)
        };
        assert!(compiled, "kernel compilation failed");

        let inputs = [A_buf, B_buf, C_buf];
        let outputs = [D_buf];

        let start = Instant::now();
        let ran = unsafe {
            runtime.as_mut().run_kernel(
                &kernel_function,
                inputs.as_ptr(),
                inputs.len(),
                outputs.as_ptr(),
                outputs.len(),
                16,
                4,
            )
        };
        assert!(ran, "kernel run failed");
        let duration = start.elapsed();
        println!("kernel run exec : {:?} ", duration);

        unsafe {
            runtime
                .as_mut()
                .copy_from_device(D.as_mut_ptr() as usize, d_ptr, SIZE);
        }

        let mut temp = [0f32; N * N];
        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    temp[i * N + j] += A[i * N + k] * B[k * N + j];
                }
            }
        }

        let mut expected = [0f32; N * N];
        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    expected[i * N + j] += temp[i * N + k] * C[k * N + j];
                }
            }
        }

        for i in 0..N {
            for j in 0..N {
                print!("{} ", D[i * N + j]);
            }
            println!();
        }

        unsafe {
            runtime.as_mut().deallocate(a_ptr);
            runtime.as_mut().deallocate(b_ptr);
            runtime.as_mut().deallocate(c_ptr);
            runtime.as_mut().deallocate(d_ptr);
        }
    }
}
