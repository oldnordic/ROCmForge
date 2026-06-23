use std::env;
use std::path::PathBuf;

mod gpu_build {
    use super::*;

    pub fn compile_kernels() {
        compile_hip_kernels();
    }

    fn compile_hip_kernels() {
        use std::path::Path;
        use std::process::Command;

        println!("cargo:rerun-if-changed=hip_kernels");

        let hip_path = find_rocm_path();
        let hip_path = hip_path.as_deref().unwrap_or(Path::new("/opt/rocm"));

        // Detect wave size for target architecture
        let target_arch = "gfx1100"; // RX 7900 XT (RDNA3)
        let wave_size = detect_wave_size_for_arch(target_arch);

        println!(
            "cargo:warning=Target arch: {} (wave size: {})",
            target_arch, wave_size
        );

        let kernels = [
            ("norm", "hip_kernels/norm.hip"),
            ("rope", "hip_kernels/rope.hip"),
            ("matmul", "hip_kernels/matmul.hip"),
            ("elementwise", "hip_kernels/elementwise.hip"),
            ("attention", "hip_kernels/attention.hip"),
            ("ssm", "hip_kernels/ssm.hip"),
            ("shortconv", "hip_kernels/shortconv.hip"),
            ("sparse_csr", "hip_kernels/sparse_csr.hip"),
            ("mpo", "hip_kernels/mpo.hip"),
        ];

        let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
        let hipcc = hip_path.join("bin/hipcc");
        let hip_include = hip_path.join("include");

        if !hipcc.exists() {
            println!(
                "cargo:warning=hipcc not found at {:?}, skipping kernel compilation",
                hipcc
            );
            println!("cargo:warning=GPU feature will use Memoria's libgpu.so as fallback");
            return;
        }

        for (name, source) in kernels {
            let source_file = Path::new(source);
            if !source_file.exists() {
                println!("cargo:warning=Kernel source {} not found, skipping", source);
                continue;
            }

            let obj_file = format!("{}/{}.o", out_dir, name);
            let lib_file = format!("{}/lib{}.a", out_dir, name);

            println!("cargo:warning=Compiling HIP kernel: {}", name);

            // Compile to object file with wave size specialization
            let compile_status = Command::new(&hipcc)
                .arg(source_file)
                .arg("-o")
                .arg(&obj_file)
                .arg("-c")
                .arg("-fPIC")
                .arg("-O3")
                .arg("-ffast-math")
                .arg(format!("--offload-arch={}", target_arch))
                .arg(format!("-DWARP_SIZE={}", wave_size))
                .arg(format!("-I{}", hip_include.display()))
                .status();

            match compile_status {
                Ok(s) if s.success() => {
                    // Create static library from object file
                    let ar_status = Command::new("ar")
                        .arg("rcs")
                        .arg(&lib_file)
                        .arg(&obj_file)
                        .status();

                    match ar_status {
                        Ok(_) => {
                            println!("cargo:rustc-link-lib=static={}", name);
                            println!("cargo:rustc-link-search=native={}", out_dir);
                        }
                        Err(e) => {
                            println!("cargo:warning=Failed to archive kernel {}: {:?}", name, e);
                        }
                    }
                }
                Ok(s) => {
                    println!(
                        "cargo:warning=Kernel {} compilation returned non-zero exit code: {:?}",
                        name,
                        s.code()
                    );
                }
                Err(e) => {
                    println!("cargo:warning=Kernel {} compilation failed: {:?}", name, e);
                }
            }
        }
    }

    fn detect_wave_size_for_arch(arch: &str) -> usize {
        match arch {
            "gfx1201" => 32,                                                       // RDNA4
            "gfx1100" | "gfx1101" | "gfx1102" => 32,                               // RDNA3
            "gfx1030" | "gfx1031" | "gfx1032" => 32,                               // RDNA2
            "gfx1010" | "gfx1011" | "gfx1012" => 32,                               // RDNA1
            "gfx900" | "gfx906" | "gfx908" | "gfx90a" | "gfx90c" | "gfx942" => 64, // CDNA
            _ => 32, // Conservative default for RDNA
        }
    }

    pub fn compile_quant_kernels() {
        use std::path::Path;
        use std::process::Command;

        println!("cargo:rerun-if-changed=hip_kernels/quant");

        // Find cmake executable
        let cmake = find_cmake();
        let cmake = cmake.as_deref().unwrap_or(Path::new("cmake"));

        // Quantization kernel paths
        let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
        let lib_dest = Path::new(&out_dir);
        let quant_src = Path::new("hip_kernels/quant");
        let quant_build = PathBuf::from(&out_dir).join("quant-cmake-build");
        let rocm_path = find_rocm_path().unwrap_or_else(|| PathBuf::from("/opt/rocm"));
        let hip_dir = rocm_path.join("lib/cmake/hip");

        // Create build directory if it doesn't exist
        if !quant_build.exists() {
            if let Err(e) = std::fs::create_dir_all(&quant_build) {
                println!(
                    "cargo:warning=Failed to create quant build directory: {}",
                    e
                );
                return;
            }
        }

        println!("cargo:warning=Compiling quantization kernels with CMake");

        // Detect wave size for target architecture
        let target_arch = "gfx1100"; // RX 7900 XT (RDNA3)
        let wave_size = detect_wave_size_for_arch(target_arch);
        println!("cargo:warning=Quant kernels wave size: {}", wave_size);

        // Configure with CMake
        let config_status = Command::new(cmake)
            .arg("-S")
            .arg(quant_src)
            .arg("-B")
            .arg(&quant_build)
            .arg("-DCMAKE_BUILD_TYPE=Release")
            .arg(format!("-DCMAKE_PREFIX_PATH={}", rocm_path.display()))
            .arg(format!("-Dhip_DIR={}", hip_dir.display()))
            .arg(format!("-DCMAKE_CXX_FLAGS=-ffast-math -DWARP_SIZE={}", wave_size))
            .arg(format!("-DCMAKE_HIP_FLAGS=-ffast-math -DWARP_SIZE={}", wave_size))
            .status();

        match config_status {
            Ok(s) if s.success() => {
                // Build the project
                let build_status = Command::new(cmake)
                    .arg("--build")
                    .arg(&quant_build)
                    .arg("--config")
                    .arg("Release")
                    .status();

                match build_status {
                    Ok(_) => {
                        // Copy libraries to output directory for Cargo linking
                        let libs_to_copy = vec![
                            ("libtest_quant.a", "test_quant"),
                            // Q4_K libraries
                            ("libq4_k_quantize.a", "q4_k_quantize"),
                            ("libq4_k_dequantize.a", "q4_k_dequantize"),
                            ("libq4_k_verify.a", "q4_k_verify"),
                            // Q8_0 libraries
                            ("libq8_0_quantize.a", "q8_0_quantize"),
                            ("libq8_0_dequantize.a", "q8_0_dequantize"),
                            ("libq8_0_verify.a", "q8_0_verify"),
                            // Q8_0 GEMV library
                            ("libq8_0_gemv.a", "q8_0_gemv"),
                            // Q4_K GEMV library
                            ("libq4_k_gemv.a", "q4_k_gemv"),
                            ("libq4_k_reference_gemv.a", "q4_k_reference_gemv"),
                            // Q2_K GEMV library
                            ("libq2_k_gemv.a", "q2_k_gemv"),
                            ("libq2_k_reference_gemv.a", "q2_k_reference_gemv"),
                            // Q3_K GEMV library
                            ("libq3_k_gemv.a", "q3_k_gemv"),
                            ("libq3_k_reference_gemv.a", "q3_k_reference_gemv"),
                            // Q5_K libraries
                            ("libq5_k_quantize.a", "q5_k_quantize"),
                            ("libq5_k_dequantize.a", "q5_k_dequantize"),
                            ("libq5_k_verify.a", "q5_k_verify"),
                            ("libq5_k_gemv.a", "q5_k_gemv"),
                            ("libq5_k_reference_gemv.a", "q5_k_reference_gemv"),
                            // Q5_0 libraries
                            ("libq5_0_gemv.a", "q5_0_gemv"),
                            // Q5_1 libraries
                            ("libq5_1_gemv.a", "q5_1_gemv"),
                            // Q6_K libraries
                            ("libq6_k_quantize.a", "q6_k_quantize"),
                            ("libq6_k_dequantize.a", "q6_k_dequantize"),
                            ("libq6_k_verify.a", "q6_k_verify"),
                            ("libq6_k_gemv.a", "q6_k_gemv"),
                            ("libq6_k_reference_gemv.a", "q6_k_reference_gemv"),
                            ("libq6_k_test.a", "q6_k_test"),
                            ("libq6_k_tile.a", "q6_k_tile"),
                            ("libq6_k_vectorized.a", "q6_k_vectorized"),
                            // Q4_0 libraries
                            ("libq4_0_quantize.a", "q4_0_quantize"),
                            ("libq4_0_dequantize.a", "q4_0_dequantize"),
                            ("libq4_0_verify.a", "q4_0_verify"),
                            ("libq4_0_gemv.a", "q4_0_gemv"),
                            ("libq4_0_fused_qkv_rope_gqa.a", "q4_0_fused_qkv_rope_gqa"),
                            (
                                "libq4_0_fused_norm_qkv_rope_dp4a.a",
                                "q4_0_fused_norm_qkv_rope_dp4a",
                            ),
                            (
                                "libq4_0_fused_norm_qkv_rope_wmma.a",
                                "q4_0_fused_norm_qkv_rope_wmma",
                            ),
                            // Q4_1 libraries
                            ("libq4_1_quantize.a", "q4_1_quantize"),
                            ("libq4_1_dequantize.a", "q4_1_dequantize"),
                            ("libq4_1_verify.a", "q4_1_verify"),
                            ("libq4_1_gemv.a", "q4_1_gemv"),
                            ("libq4_1_ffn_experimental.a", "q4_1_ffn_experimental"),
                            // GEMM libraries
                            ("libq4_0_gemm.a", "q4_0_gemm"),
                            ("libbatched_q4_0_gemm.a", "batched_q4_0_gemm"),
                            ("libbatched_q4_1_gemm.a", "batched_q4_1_gemm"),
                            ("libq4_0_fused.a", "q4_0_fused"),
                            ("libq4_0_fused_q8.a", "q4_0_fused_q8"),
                            ("libq4_1_gemm.a", "q4_1_gemm"),
                            ("libq8_0_gemm.a", "q8_0_gemm"),
                            ("libq4_k_gemm.a", "q4_k_gemm"),
                            ("libq5_k_gemm.a", "q5_k_gemm"),
                            ("libq5_0_gemm.a", "q5_0_gemm"),
                            ("libq5_1_gemm.a", "q5_1_gemm"),
                            ("libq6_k_gemm.a", "q6_k_gemm"),
                            // RFM unpack library
                            ("librfm_unpack.a", "rfm_unpack"),
                            // WMMA matrix core library
                            ("libwmma_matmul.a", "wmma_matmul"),
                        ];

                        for (lib_name, link_name) in libs_to_copy {
                            let src_lib = quant_build.join("lib").join(lib_name);
                            let dst_lib = lib_dest.join(lib_name);

                            if src_lib.exists() {
                                if let Err(e) = std::fs::copy(&src_lib, &dst_lib) {
                                    println!("cargo:warning=Failed to copy {}: {}", lib_name, e);
                                    continue;
                                }
                                println!("cargo:rustc-link-lib=static={}", link_name);
                            } else {
                                println!("cargo:warning={} not found (skipping)", lib_name);
                            }
                        }

                        println!("cargo:rustc-link-search=native={}", out_dir);
                    }
                    Err(e) => {
                        println!("cargo:warning=Quantization CMake build failed: {:?}", e);
                    }
                }
            }
            Ok(s) => {
                println!(
                    "cargo:warning=CMake configuration returned non-zero: {:?}",
                    s.code()
                );
            }
            Err(e) => {
                println!("cargo:warning=CMake configuration failed: {:?}", e);
            }
        }
    }

    fn find_cmake() -> Option<PathBuf> {
        // Try common CMake locations
        let standard_paths = [
            PathBuf::from("/usr/bin/cmake"),
            PathBuf::from("/usr/local/bin/cmake"),
            PathBuf::from("/opt/homebrew/bin/cmake"),
        ];

        for path in standard_paths.iter() {
            if path.exists() {
                return Some(path.clone());
            }
        }

        None
    }

    fn find_rocm_path() -> Option<PathBuf> {
        if let Ok(hip_path) = env::var("HIP_PATH") {
            let path = PathBuf::from(&hip_path);
            if path.exists() {
                return Some(path);
            }
        }

        if let Ok(rocm_path) = env::var("ROCM_PATH") {
            let path = PathBuf::from(&rocm_path);
            if path.exists() {
                return Some(path);
            }
        }

        let standard_paths = [
            PathBuf::from("/opt/rocm"),
            PathBuf::from("/usr/lib/x86_64-linux-gnu"),
        ];

        for path in standard_paths.iter() {
            if path.exists() && path.join("bin/hipcc").exists() {
                return Some(path.clone());
            }
        }

        None
    }
}

fn main() {
    if env::var("CARGO_FEATURE_GPU").is_ok() {
        // Compile HIP kernels
        gpu_build::compile_kernels();

        // Compile quantization kernels via CMake
        gpu_build::compile_quant_kernels();

        // Link HIP runtime
        let hip_path = env::var("HIP_PATH")
            .ok()
            .and_then(|p| {
                let path = PathBuf::from(&p);
                if path.exists() {
                    Some(path)
                } else {
                    None
                }
            })
            .or_else(|| {
                if PathBuf::from("/opt/rocm/lib").exists() {
                    Some(PathBuf::from("/opt/rocm"))
                } else {
                    None
                }
            });

        if let Some(path) = hip_path {
            println!("cargo:rustc-link-search=native={}/lib", path.display());
        }

        println!("cargo:rustc-link-lib=hiprtc");
        println!("cargo:rustc-link-lib=amdhip64");
        println!("cargo:rustc-link-lib=rocblas");
        println!("cargo:rustc-link-lib=rocsolver");
        println!("cargo:rerun-if-env-changed=HIP_PATH");
        println!("cargo:rerun-if-env-changed=ROCM_PATH");
        println!("cargo:rerun-if-changed=hip_kernels");
    }
}
