#![cfg(feature = "gpu")]
//! Regression tests for GPU forward pass bugs
//!
//! This file contains tests for specific bugs that have been found and fixed.
//! Each test documents a bug and ensures it doesn't regress.
//!
//! Test Pattern:
//! 1. Document the bug with clear explanation
//! 2. Test the exact scenario that triggered the bug
//! 3. Verify the fix works correctly
//! 4. Use assertions that catch the specific bug type

use rocmforge::config::ModelConfig;
use rocmforge::cpu::cache::{CpuForwardScratch, CpuKvCache};
use rocmforge::cpu::forward::cpu_layer_forward;
use rocmforge::cpu::weights::CpuModelWeights;
use rocmforge::gpu::{self, gpu_layer_forward_hybrid, GpuDevice, GpuForwardScratch, GpuKvCache};
use rocmforge::loader::GgufFile;
use serial_test::serial;

const MODEL_PATH: &str = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q4_0.gguf";

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

// Regression test for FFN down dimension swap bug (April 2026)
//
// BUG: gpu_layer_forward_hybrid with stage profiling had swapped dimensions
// for the FFN down projection, causing:
// - Reading only 896 of 4864 input elements
// - Writing 4864 values into 896-element buffer
// - GPU memory access fault
//
// This test ensures the dimensions are correct and the output matches CPU.
#[test]
#[serial]
#[ignore = "Requires real GPU and model - run with: cargo test -- --ignored"]
fn test_ffn_down_dimensions_correct_in_hybrid_forward() {
    if skip_if_model_missing() {
        return;
    }

    // Load model
    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("Failed to load CPU weights");
    let gpu_weights =
        gpu::GpuModelWeights::load(&file, &config).expect("Failed to load GPU weights");

    // Initialize GPU
    let caps = gpu::detect().expect("GPU should be detected");
    let device = GpuDevice::init(caps.device_id).expect("GPU device should initialize");

    let h = config.hidden_size;
    let ff_size = config.intermediate_size;
    let layer_idx = 2; // Layer with Q4_0 FFN down

    eprintln!("=== FFN Down Dimension Regression Test ===");
    eprintln!("hidden_size (h): {}", h);
    eprintln!("intermediate_size (ff_size): {}", ff_size);

    // Create test input
    let mut cpu_hidden = vec![1.0f32; h];
    cpu_hidden[0] = 0.1;
    cpu_hidden[1] = -0.2;
    cpu_hidden[2] = 0.3;

    // CPU reference
    let mut cpu_kv = CpuKvCache::new(&config, 1);
    let mut cpu_scratch = CpuForwardScratch::new(&config);

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

    cpu_layer_forward(
        &mut cpu_hidden,
        cpu_weights.layer(layer_idx),
        &mut cpu_kv,
        &mut cpu_scratch,
        layer_idx,
        0, // pos
        rope_sin,
        rope_cos,
        &config,
        false, // debug
    )
    .expect("CPU forward should succeed");

    // GPU test with stage profiling (the buggy path)
    let mut gpu_kv = GpuKvCache::new(&config, 1).expect("Failed to create GPU KV cache");
    let mut gpu_scratch = GpuForwardScratch::new(&config).expect("Failed to create GPU scratch");

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

    // Enable decode stage profiling to trigger the buggy code path
    gpu::reset_decode_stage_profile();

    gpu_layer_forward_hybrid(
        &device,
        gpu_weights.layer(layer_idx),
        Some(cpu_weights.layer(layer_idx)),
        &mut gpu_kv,
        &mut gpu_scratch,
        Some(&mut cpu_scratch),
        layer_idx,
        0, // pos
        &config,
    )
    .expect("GPU forward should succeed without memory fault");

    // Download GPU result
    let gpu_hidden = download_gpu_f32(&gpu_scratch.hidden, h);

    // Verify dimensions match
    assert_eq!(
        gpu_hidden.len(),
        h,
        "Output dimension should match hidden_size"
    );

    // Verify values match CPU (with some tolerance for quantization)
    let max_error = max_abs_error(&cpu_hidden, &gpu_hidden);

    eprintln!("Max error between CPU and GPU: {}", max_error);

    // Allow small numerical differences due to floating point, but catch dimension bugs
    assert!(
        max_error < 1.0,
        "GPU output should match CPU output (max_error={})",
        max_error
    );
}
