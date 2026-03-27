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

        let hip_path = find_rocm_path();
        let hip_path = hip_path.as_ref().map(|p| p.as_path()).unwrap_or(Path::new("/opt/rocm"));

        let kernels = [
            ("norm", "hip_kernels/norm.hip"),
            ("rope", "hip_kernels/rope.hip"),
            ("matmul", "hip_kernels/matmul.hip"),
            ("elementwise", "hip_kernels/elementwise.hip"),
            ("attention", "hip_kernels/attention.hip"),
        ];

        let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
        let hipcc = hip_path.join("bin/hipcc");

        if !hipcc.exists() {
            println!("cargo:warning=hipcc not found at {:?}, skipping kernel compilation", hipcc);
            println!("cargo:warning=GPU feature will use Memoria's libgpu.so as fallback");
            return;
        }

        for (name, source) in kernels {
            let source_file = Path::new(source);
            if !source_file.exists() {
                println!("cargo:warning=Kernel source {} not found, skipping", source);
                continue;
            }

            let output_file = format!("{}/{}.o", out_dir, name);

            println!("cargo:warning=Compiling HIP kernel: {}", name);

            let status = Command::new(&hipcc)
                .arg(source_file)
                .arg("-o")
                .arg(&output_file)
                .arg("-c")
                .arg("-fPIC")
                .arg("-O3")
                .arg("--amdgpu-target=gfx1100")
                .status();

            match status {
                Ok(s) if s.success() => {
                    println!("cargo:rustc-link-lib=static={}", name);
                    println!("cargo:rustc-link-search=native={}", out_dir);
                }
                _ => {
                    println!("cargo:warning=Failed to compile kernel {}, will use fallback", name);
                }
            }
        }
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

        // Link HIP runtime
        let hip_path = env::var("HIP_PATH").ok()
            .and_then(|p| {
                let path = PathBuf::from(&p);
                if path.exists() { Some(path) } else { None }
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
        println!("cargo:rerun-if-env-changed=HIP_PATH");
        println!("cargo:rerun-if-env-changed=ROCM_PATH");
        println!("cargo:rerun-if-changed=hip_kernels");
    }
}
