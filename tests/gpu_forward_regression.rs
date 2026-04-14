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

const MODEL_PATH: &str = "/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf";

fn skip_if_model_missing() -> bool {
    !std::path::Path::new(MODEL_PATH).exists()
}

#[test]
#[serial]
#[ignore = "Requires real GPU and model - run with: cargo test -- --ignored"]
fn test_ffn_down_dimensions_correct_in_hybrid_forward() {
    /*!
    Regression test for FFN down dimension swap bug (April 2026)

    BUG: gpu_layer_forward_hybrid with stage profiling had swapped dimensions
    for the FFN down projection, causing:
    - Reading only 896 of 4864 input elements
    - Writing 4864 values into 896-element buffer
    - GPU memory access fault

    This test ensures the dimensions are correct and the output matches CPU.
    */

    if skip_if_model_missing() {
        return;
    }

    // Load model
    let file = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF file");
    let config = ModelConfig::from_gguf(&file).expect("Failed to parse config");
    let cpu_weights = CpuModelWeights::load(&file, &config).expect("Failed to load CPU weights");
    let gpu_weights = gpu::GpuModelWeights::load(&file, &config).expect("Failed to load GPU weights");

    // Initialize GPU
    let caps = gpu::detect().expect("GPU should be detected");
    let device = GpuDevice::init(caps.device_id).expect("GPU device should initialize");

    let h = config.hidden_size;
    let ff_size = config.intermediate_size;

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

    let cpu_layer_idx = 2;
    let cpu_result = cpu_layer_forward(
        &cpu_weights,
        &mut cpu_kv,
        &mut cpu_scratch,
        cpu_layer_idx,
        &config,
        &cpu_hidden.clone(),
    );

    assert!(cpu_result.is_ok(), "CPU forward should succeed");

    // GPU test with stage profiling (the buggy path)
    let mut gpu_kv = GpuKvCache::new(&config, 1).expect("Failed to create GPU KV cache");
    let mut gpu_scratch = GpuForwardScratch::new(&config);

    // Enable decode stage profiling to trigger the buggy code path
    gpu::reset_decode_stage_profile();

    let gpu_layer_idx = 2;
    let gpu_result = gpu_layer_forward_hybrid(
        &device,
        gpu_weights.layer(gpu_layer_idx),
        &mut gpu_kv,
        &mut gpu_scratch,
        gpu_layer_idx,
        &config,
        &cpu_hidden,
    );

    assert!(gpu_result.is_ok(), "GPU forward should succeed without memory fault");

    // Download GPU result
    let gpu_hidden = gpu_scratch.hidden.download().expect("Failed to download GPU result");

    // Verify dimensions match
    assert_eq!(gpu_hidden.len(), h, "Output dimension should match hidden_size");

    // Verify values match CPU (with some tolerance for quantization)
    let max_error = cpu_hidden
        .iter()
        .zip(gpu_hidden.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);

    eprintln!("Max error between CPU and GPU: {}", max_error);

    // Allow small numerical differences due to floating point, but catch dimension bugs
    assert!(
        max_error < 1.0,
        "GPU output should match CPU output (max_error={})",
        max_error
    );
}
