use std::env;
use std::path::PathBuf;

fn main() {
    // Only link HIP when gpu feature is enabled
    if env::var("CARGO_FEATURE_GPU").is_ok() {
        // Try to find HIP installation
        let hip_path = env::var("HIP_PATH").ok().and_then(|p| {
            if PathBuf::from(&p).exists() {
                Some(p)
            } else {
                None
            }
        }).or_else(|| {
            // Default ROCm installation path
            if PathBuf::from("/opt/rocm/lib").exists() {
                Some("/opt/rocm".to_string())
            } else {
                None
            }
        });

        if let Some(path) = hip_path {
            println!("cargo:rustc-link-search=native={}/lib", path);
        }

        // Link against HIP runtime
        println!("cargo:rustc-link-lib=hiprtc");
        println!("cargo:rustc-link-lib=amdhip64");

        // Tell cargo to rerun build.rs if HIP_PATH changes
        println!("cargo:rerun-if-env-changed=HIP_PATH");
    }
}
