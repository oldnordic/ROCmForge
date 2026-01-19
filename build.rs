use std::env;

fn main() {
    println!("cargo:rerun-if-changed=kernels");

    // Only link with ROCm HIP library if rocm feature is enabled
    let rocm_root = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    println!("cargo:rustc-link-search=native={}/lib", rocm_root);
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=hipblas");
    println!("cargo:rustc-link-lib=dylib=hiprtc");

    #[cfg(feature = "rocm")]
    {
        compile_hip_kernels();
    }
}

#[cfg(feature = "rocm")]
fn compile_hip_kernels() {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    let hipcc = env::var("HIPCC").unwrap_or_else(|_| {
        let rocm_root = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
        format!("{}/bin/hipcc", rocm_root)
    });

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Verify hipcc exists
    if !Path::new(&hipcc).exists() {
        println!(
            "cargo:warning=hipcc not found at {}. Skipping kernel compilation.",
            hipcc
        );
        return;
    }

    // Target architecture for AMD Radeon RX 7900 XT (gfx1100, RDNA3)
    let target_arch = env::var("ROCm_ARCH").unwrap_or_else(|_| "gfx1100".to_string());

    // Kernels to compile: (source_file, env_var_name, kernel_name)
    let kernels = [
        ("kernels/scale.hip", "SCALE_HSACO", "scale_kernel"),
        ("kernels/mask.hip", "MASK_HSACO", "mask_kernel"),
        ("kernels/softmax.hip", "SOFTMAX_HSACO", "softmax_kernel"),
        ("kernels/rope.hip", "ROPE_HSACO", "rope_kernel"),
        (
            "kernels/position_embeddings.hip",
            "POSITION_EMBEDDINGS_HSACO",
            "position_embeddings_kernel",
        ),
        (
            "kernels/qkt_matmul.hip",
            "QKT_MATMUL_HSACO",
            "qkt_matmul_kernel",
        ),
        (
            "kernels/weighted_matmul.hip",
            "WEIGHTED_MATMUL_HSACO",
            "weighted_matmul_kernel",
        ),
        (
            "kernels/flash_attention_nocausal.hip",
            "FLASH_ATTENTION_NCAUSAL_HSACO",
            "flash_attention_nocausal_kernel",
        ),
        (
            "kernels/causal_mask.hip",
            "CAUSAL_MASK_HSACO",
            "causal_mask_kernel",
        ),
        (
            "kernels/flash_attention_causal.hip",
            "FLASH_ATTENTION_CAUSAL_HSACO",
            "flash_attention_causal_kernel",
        ),
        (
            "kernels/flash_attention.hip",
            "FLASH_ATTENTION_HSACO",
            "flash_attention_kernel",
        ),
        ("kernels/swiglu.hip", "SWIGLU_HSACO", "swiglu_kernel"),
        ("kernels/rms_norm.hip", "RMS_NORM_HSACO", "rms_norm_kernel"),
        (
            "kernels/mxfp_dequant.hip",
            "MXFP_DEQUANT_HSACO",
            "mxfp4_to_fp32_kernel",
        ),
        (
            "kernels/q8_0_dequant.hip",
            "Q8_0_DEQUANT_HSACO",
            "q8_0_to_fp32_kernel",
        ),
        (
            "kernels/q4_k_dequant.hip",
            "Q4_K_DEQUANT_HSACO",
            "q4_k_to_fp32_kernel",
        ),
        (
            "kernels/q6_k_dequant.hip",
            "Q6_K_DEQUANT_HSACO",
            "q6_k_to_fp32_kernel",
        ),
        (
            "kernels/mqa_kv_replicate.hip",
            "MQA_KV_REPLICATE_HSACO",
            "mqa_kv_replicate_kernel",
        ),
        (
            "kernels/q4_0_dequant.hip",
            "Q4_0_DEQUANT_HSACO",
            "q4_0_to_fp32_kernel",
        ),
        (
            "kernels/q4_0_matmul.hip",
            "Q4_0_MATMUL_HSACO",
            "q4_0_matmul_kernel",
        ),
        (
            "kernels/q4_k_matmul.hip",
            "Q4_K_MATMUL_HSACO",
            "q4_k_matmul_kernel",
        ),
        (
            "kernels/q6_k_matmul.hip",
            "Q6_K_MATMUL_HSACO",
            "q6_k_matmul_kernel",
        ),
        (
            "kernels/q5_k_dequant.hip",
            "Q5_K_DEQUANT_HSACO",
            "q5_k_to_fp32_kernel",
        ),
        (
            "kernels/q3_k_dequant.hip",
            "Q3_K_DEQUANT_HSACO",
            "q3_k_to_fp32_kernel",
        ),
        (
            "kernels/q2_k_dequant.hip",
            "Q2_K_DEQUANT_HSACO",
            "q2_k_to_fp32_kernel",
        ),
        (
            "kernels/fused_dequant_rmsnorm.hip",
            "FUSED_DEQUANT_RMSNORM_HSACO",
            "fused_q4_0_rmsnorm_kernel",
        ),
        (
            "kernels/fused_rope_kvappend.hip",
            "FUSED_ROPE_KVAPPEND_HSACO",
            "fused_rope_kv_cache_append_kernel",
        ),
        (
            "kernels/sampling_utils.hip",
            "SAMPLING_UTILS_HSACO",
            "softmax_kernel",
        ),
        (
            "kernels/sampling_utils.hip",
            "TEMPERATURE_SCALE_HSACO",
            "temperature_scale_kernel",
        ),
        (
            "kernels/topk_sampling.hip",
            "TOPK_SAMPLING_HSACO",
            "topk_sampling_kernel",
        ),
    ];

    for (src_file, env_name, kernel_name) in &kernels {
        let src_path = PathBuf::from(src_file);

        if !src_path.exists() {
            println!("cargo:warning=Kernel source not found: {}", src_file);
            continue;
        }

        let hsaco_path = out_dir.join(format!("{}.hsaco", kernel_name));

        // Compile HIP kernel to HSACO
        let status = Command::new(&hipcc)
            .arg("-c")
            .arg("--genco")
            .arg(format!("--offload-arch={}", target_arch))
            .arg("-O3")
            .arg(src_file)
            .arg("-o")
            .arg(&hsaco_path)
            .status();

        match status {
            Ok(status_code) if status_code.success() => {
                println!("cargo:rustc-env={}={}", env_name, hsaco_path.display());
                println!("cargo:rustc-env={}_PATH={}", env_name, hsaco_path.display());
                println!("Compiled {} -> {}", src_file, hsaco_path.display());
            }
            Ok(status_code) => {
                println!(
                    "cargo:warning=Failed to compile {}: exit code {:?}",
                    src_file,
                    status_code.code()
                );
            }
            Err(e) => {
                println!(
                    "cargo:warning=Failed to execute hipcc for {}: {:?}",
                    src_file, e
                );
            }
        }
    }
}
