fn main() {
    cxx_build::bridge("src/lib.rs")
        .file("../../cpp-backend/src/backend/metal_backend.cpp")
        .include("../../cpp-backend/include")
        .include("../../cpp-backend/src")
        .include("../../cpp-backend/src/backend/metal-cpp")
        .flag("-std=c++17")
        .compile("metal_backend");

    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=QuartzCore");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=../../cpp-backend/backend/src/metal_backend.cpp");
    println!("cargo:rerun-if-changed=../../cpp-backend/backend/src/metal_backend.h");
    println!("cargo:rustc-env=MACOSX_DEPLOYMENT_TARGET=15.2");
}
