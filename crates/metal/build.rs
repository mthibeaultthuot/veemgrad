use std::process::Command;
use std::{env, path::PathBuf};

fn main() {
    let metal_shader_path = std::env::var("METAL_LIB_PATH").unwrap();
    let cpp_metal_path = std::env::var("CPP_METAL_PATH").unwrap();
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    // 1. Build the shader
    let status = Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metal",
            "-c",
            &format!("{}metal_matmul.metal", cpp_metal_path),
            "-o",
        ])
        .arg(out_path.join("metal_matmul.air"))
        .status()
        .expect("failed to compile .metal to .air");

    if !status.success() {
        panic!("Failed to compile Metal shader");
    }

    let status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib"])
        .arg(out_path.join("metal_matmul.air"))
        .args(["-o"])
        .arg(out_path.join("metal_matmul.metallib"))
        .status()
        .expect("Failed to link .air to .metallib");

    if !status.success() {
        panic!("failed to create metallib");
    }

    cc::Build::new()
        .cpp(true)
        .file(format!("{}metal_wrapper.cpp", cpp_metal_path))
        .file(format!("{}metal_matmul.cpp", cpp_metal_path))
        .flag("-std=c++17")
        .flag("-fobjc-arc")
        .flag("-mmacosx-version-min=15.0")
        .flag("-arch")
        .flag("arm64")
        .flag("-framework")
        .flag("Foundation")
        .flag("-framework")
        .flag("Metal")
        .flag("-framework")
        .flag("QuartzCore")
        .include(format!("{}metal-cpp", cpp_metal_path))
        .include(&cpp_metal_path)
        .compile("matmul");

    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=QuartzCore");
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=objc");

    println!(
        "cargo:rerun-if-changed={}",
        format!("{}metal_wrapper.cpp", &cpp_metal_path)
    );
    println!(
        "cargo:rerun-if-changed={}",
        format!("{}metal_wrapper.h", cpp_metal_path)
    );
    println!(
        "cargo:rerun-if-changed={}",
        format!("{}metal_matmul.cpp", cpp_metal_path)
    );
    println!(
        "cargo:rerun-if-changed={}",
        format!("{}metal_matmul.hpp", cpp_metal_path)
    );
}
