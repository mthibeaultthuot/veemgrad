#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::ffi::CString;
use std::os::raw::{c_char, c_float, c_void};
use std::path::PathBuf;
use std::time::Instant;

type MatMulKernelHandle = *mut c_void;

#[link(name = "matmul")]
unsafe extern "C" {
    pub fn matmul_factory_create(
        metallib_path: *const c_char,
        function_name: *const c_char,
        max_elements: usize,
    ) -> MatMulKernelHandle;

    pub fn matmul_destroy(handle: MatMulKernelHandle);

    pub fn warmUp(handle: MatMulKernelHandle);

    pub fn matmul_run(
        handle: MatMulKernelHandle,
        matrix_a: *const c_float,
        matrix_b: *const c_float,
        result: *mut c_float,
        rows_a: usize,
        cols_a: usize,
        rows_b: usize,
        cols_b: usize,
    ) -> bool;
}

pub struct MatMulKernel {
    handle: MatMulKernelHandle,
}

impl MatMulKernel {
    pub fn new(lib_path: &str, func: &str, max_elements: usize) -> Result<Self, &'static str> {
        let c_lib = CString::new(lib_path).map_err(|_| "lib not found")?;
        let c_func = CString::new(func).map_err(|_| "metal shader function invalid")?;

        let handle =
            unsafe { matmul_factory_create(c_lib.as_ptr(), c_func.as_ptr(), max_elements) };

        if handle.is_null() {
            Err("Failed to create MatMulKernel")
        } else {
            Ok(MatMulKernel { handle })
        }
    }

    pub fn warm_up(&self) {
        unsafe { warmUp(self.handle) };
    }

    pub fn run(
        &self,
        matrix_a: &[f32],
        matrix_b: &[f32],
        rows_a: usize,
        cols_a: usize,
        rows_b: usize,
        cols_b: usize,
    ) -> Result<Vec<f32>, &'static str> {
        if cols_a != rows_b {
            return Err("matrix dimensions are incompatible");
        }

        let mut result = vec![0.0f32; rows_a * cols_b];
        let success = unsafe {
            matmul_run(
                self.handle,
                matrix_a.as_ptr(),
                matrix_b.as_ptr(),
                result.as_mut_ptr(),
                rows_a,
                cols_a,
                rows_b,
                cols_b,
            )
        };

        if success {
            Ok(result)
        } else {
            Err("matmul_run failed")
        }
    }
}

impl Drop for MatMulKernel {
    fn drop(&mut self) {
        unsafe {
            matmul_destroy(self.handle);
        }
    }
}

#[test]
fn test_matmul_basic() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let compiled_shader_path = out_dir.join("metal_matmul.metallib");

    let kernel = MatMulKernel::new(
        compiled_shader_path
            .to_str()
            .expect("invalid metal shader path"),
        "matmul",
        1024,
    )
    .expect("Failed to create kernel");

    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![5.0f32, 6.0, 7.0, 8.0];

    let start = Instant::now();
    let result = kernel
        .run(&a, &b, 2, 2, 2, 2)
        .expect("Multiplication failed");
    let end = start.elapsed();
    println!("{:?}", end);
    println!("{:?}", result);
}
