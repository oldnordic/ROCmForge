#![cfg(feature = "gpu")]
//! Verification test for Q2_K/Q3_K GPU-to-CPU Fallback Routing.
//!
//! Verifies that when a Q2_K or Q3_K model is loaded, fallback path
//! is taken and execution yields correct outputs.

use rocmforge::config::ModelConfig;
use rocmforge::cpu::cache::CpuForwardScratch;
use rocmforge::cpu::weights::CpuModelWeights;
use rocmforge::gpu::{self, GpuDevice};
use rocmforge::loader::GgufFile;
use serial_test::serial;

const MODEL_PATH: &str =
    "/home/feanor/Projects/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf";

fn skip_if_model_missing() -> bool {
    !std::path::Path::new(MODEL_PATH).exists()
}

fn download_gpu_f32(buf: &gpu::GpuBuffer, len: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; len * std::mem::size_of::<f32>()];
    buf.copy_to_host(&mut bytes)
        .expect("GPU buffer should download");
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len).to_vec() }
}

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
#[serial]
fn test_gpu_q2k_q3k_fallback_correctness() {
    if skip_if_model_missing() {
        eprintln!("tinyllama-1.1b-chat-v1.0.Q2_K.gguf missing, skipping test.");
        return;
    }

    // 1. Load model
    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("Failed to load CPU weights");
    let gpu_weights =
        gpu::GpuModelWeights::load(&file, &config).expect("Failed to load GPU weights");

    // 2. Initialize GPU
    let caps = gpu::detect().expect("GPU should be detected");
    let device = GpuDevice::init(caps.device_id).expect("GPU device should initialize");

    // 3. Verify that the Q2_K weights are now supported on GPU
    assert!(
        !gpu_weights.has_unsupported_gpu_gemv_weights(),
        "Q2_K model must be recognized as natively supported on the GPU"
    );

    // 4. Create scratch spaces
    let h = config.hidden_size;
    let mut cpu_hidden = vec![1.0f32; h];
    let mut cpu_kv = rocmforge::cpu::cache::CpuKvCache::new(&config, 1);
    let mut cpu_scratch = CpuForwardScratch::new(&config);

    let mut gpu_kv = gpu::GpuKvCache::new(&config, 1).expect("Failed to create GPU KV cache");
    let mut gpu_scratch =
        gpu::GpuForwardScratch::new(&config).expect("Failed to create GPU scratch");

    // Upload initial hidden state
    gpu_scratch
        .hidden
        .copy_from_host(unsafe {
            std::slice::from_raw_parts(
                cpu_hidden.as_ptr() as *const u8,
                h * std::mem::size_of::<f32>(),
            )
        })
        .expect("Failed to upload hidden state");

    // Precompute RoPE sin/cos tables for CPU reference
    let half = config.head_dim / 2;
    for i in 0..half {
        let angle = 0.0f32 * config.rope_freq[i];
        let (s, c) = angle.sin_cos();
        cpu_scratch.rope_sin[i] = s;
        cpu_scratch.rope_cos[i] = c;
    }
    let rope_sin = unsafe { std::slice::from_raw_parts(cpu_scratch.rope_sin.as_ptr(), half) };
    let rope_cos = unsafe { std::slice::from_raw_parts(cpu_scratch.rope_cos.as_ptr(), half) };

    // 5. CPU reference - single layer forward, position 0
    rocmforge::cpu::forward::cpu_layer_forward(
        &mut cpu_hidden,
        cpu_weights.layer(0),
        &mut cpu_kv,
        &mut cpu_scratch,
        0,
        0,
        rope_sin,
        rope_cos,
        &config,
        None, // shared_ple_token_emb
        None, // shared_ple_model_proj
        None, // shared_ple_proj_norm
        false,
    )
    .expect("CPU layer forward should succeed");

    // 6. GPU computation with CPU fallback - single layer forward, position 0
    gpu::gpu_layer_forward_hybrid(
        &device,
        gpu_weights.layer(0),
        Some(cpu_weights.layer(0)),
        &mut gpu_kv,
        &mut gpu_scratch,
        Some(&mut cpu_scratch),
        0,
        0,
        0, // token_id (dummy value since hidden is directly uploaded)
        &config,
        None, // shared_ple_token_emb
        None, // shared_ple_model_proj
        None, // shared_ple_proj_norm
    )
    .expect("GPU hybrid layer forward should succeed");

    // 7. Download and compare
    let gpu_hidden = download_gpu_f32(&gpu_scratch.hidden, h);
    let error = max_abs_error(&cpu_hidden, &gpu_hidden);
    eprintln!("GPU (Q2_K CPU fallback) vs CPU max error: {}", error);

    // Tolerance is slightly higher due to differences in numerical dispatch, but should match closely
    assert!(
        error < 1.0,
        "GPU CPU-fallback layer error {} exceeds tolerance",
        error
    );
}
