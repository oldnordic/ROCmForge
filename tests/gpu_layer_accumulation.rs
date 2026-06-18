#![cfg(feature = "gpu")]
//! Test for proper state accumulation across GPU layers
//!
//! This tests two scenarios:
//! 1. Individual layer correctness (each layer with same input)
//! 2. Accumulated correctness (output of layer N-1 becomes input of layer N)

use rocmforge::config::ModelConfig;
use rocmforge::cpu::cache::{CpuForwardScratch, CpuKvCache};
use rocmforge::cpu::forward::cpu_layer_forward;
use rocmforge::cpu::weights::CpuModelWeights;
use rocmforge::gpu::{
    self, gpu_layer_forward_hybrid, GpuBuffer, GpuDevice, GpuForwardScratch, GpuKvCache,
};
use rocmforge::loader::GgufFile;
use serial_test::serial;

const MODEL_PATH: &str = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q4_0.gguf";

fn skip_if_model_missing() -> bool {
    !std::path::Path::new(MODEL_PATH).exists()
}

fn download_gpu_f32(buf: &GpuBuffer, len: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; len * std::mem::size_of::<f32>()];
    buf.copy_to_host(&mut bytes)
        .expect("GPU buffer should download");
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len).to_vec() }
}

#[allow(dead_code)]
fn upload_gpu_f32(data: &[f32]) -> GpuBuffer {
    let size_in_bytes = std::mem::size_of_val(data);
    let mut buf = GpuBuffer::alloc(size_in_bytes).expect("Failed to allocate GPU buffer");
    buf.copy_from_host(unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, size_in_bytes)
    })
    .expect("Failed to upload to GPU");
    buf
}

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
#[serial]
fn test_gpu_state_accumulation_across_layers() {
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
    let test_layers = 5; // Test first 5 layers

    eprintln!(
        "Testing GPU state accumulation across {} layers",
        test_layers
    );

    // Test 1: Individual layer correctness (same input to each layer)
    eprintln!("\n=== Test 1: Individual Layer Correctness ===");
    for layer_idx in 0..test_layers {
        let mut cpu_hidden = vec![1.0f32; h];
        let mut cpu_kv = CpuKvCache::new(&config, 1);
        let mut cpu_scratch = CpuForwardScratch::new(&config);

        let mut gpu_kv = GpuKvCache::new(&config, 1).expect("Failed to create GPU KV cache");
        let mut gpu_scratch =
            GpuForwardScratch::new(&config).expect("Failed to create GPU scratch");

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

        // CPU reference
        cpu_layer_forward(
            &mut cpu_hidden,
            cpu_weights.layer(layer_idx),
            &mut cpu_kv,
            &mut cpu_scratch,
            0,
            0,
            rope_sin,
            rope_cos,
            &config,
            false,
        )
        .expect("CPU layer forward should succeed");

        // GPU computation
        gpu_layer_forward_hybrid(
            &device,
            gpu_weights.layer(layer_idx),
            Some(cpu_weights.layer(layer_idx)),
            &mut gpu_kv,
            &mut gpu_scratch,
            Some(&mut cpu_scratch),
            0,
            0,
            0, // token_id (dummy value since hidden is directly uploaded)
            &config,
        )
        .expect("GPU layer forward should succeed");

        // Download and compare
        let gpu_hidden = download_gpu_f32(&gpu_scratch.hidden, h);
        let error = max_abs_error(&cpu_hidden, &gpu_hidden);

        eprintln!("Layer {} (individual): max error = {:.6}", layer_idx, error);
        if error >= 1.0 {
            eprintln!("  CPU[0..3]: {:?}", &cpu_hidden[0..3]);
            eprintln!("  GPU[0..3]: {:?}", &gpu_hidden[0..3]);
        }

        // Allow reasonable tolerance for quantization
        assert!(
            error < 1.0,
            "GPU layer {} individual error {} exceeds tolerance",
            layer_idx,
            error
        );
    }

    // Test 2: Accumulated correctness (output of layer N-1 -> input of layer N)
    // This is still a single-token decode path, so sequence position stays fixed.
    eprintln!("\n=== Test 2: Accumulated State Correctness ===");

    // Initialize state for accumulated test
    let mut cpu_hidden_accum = vec![1.0f32; h];
    let mut cpu_kv_accum = CpuKvCache::new(&config, 1);
    let mut cpu_scratch_accum = CpuForwardScratch::new(&config);

    let mut gpu_kv_accum = GpuKvCache::new(&config, 1).expect("Failed to create GPU KV cache");
    let mut gpu_scratch_accum =
        GpuForwardScratch::new(&config).expect("Failed to create GPU scratch");

    // Upload initial hidden state for GPU
    gpu_scratch_accum
        .hidden
        .copy_from_host(unsafe {
            std::slice::from_raw_parts(
                cpu_hidden_accum.as_ptr() as *const u8,
                h * std::mem::size_of::<f32>(),
            )
        })
        .expect("Failed to upload hidden state");

    let decode_pos = 0usize;
    for layer_idx in 0..test_layers {
        // CPU accumulated computation
        let cpu_input = cpu_hidden_accum.clone();

        // Precompute RoPE sin/cos tables for the single decode position.
        let half = config.head_dim / 2;
        for i in 0..half {
            let angle = decode_pos as f32 * config.rope_freq[i];
            let (s, c) = angle.sin_cos();
            cpu_scratch_accum.rope_sin[i] = s;
            cpu_scratch_accum.rope_cos[i] = c;
        }
        let rope_sin =
            unsafe { std::slice::from_raw_parts(cpu_scratch_accum.rope_sin.as_ptr(), half) };
        let rope_cos =
            unsafe { std::slice::from_raw_parts(cpu_scratch_accum.rope_cos.as_ptr(), half) };

        cpu_layer_forward(
            &mut cpu_hidden_accum,
            cpu_weights.layer(layer_idx),
            &mut cpu_kv_accum,
            &mut cpu_scratch_accum,
            layer_idx,
            decode_pos,
            rope_sin,
            rope_cos,
            &config,
            false,
        )
        .expect("CPU accumulated layer forward should succeed");

        // GPU accumulated computation
        gpu_layer_forward_hybrid(
            &device,
            gpu_weights.layer(layer_idx),
            Some(cpu_weights.layer(layer_idx)),
            &mut gpu_kv_accum,
            &mut gpu_scratch_accum,
            Some(&mut cpu_scratch_accum),
            layer_idx,
            decode_pos,
            0, // token_id (dummy value)
            &config,
        )
        .expect("GPU accumulated layer forward should succeed");

        // Download and compare accumulated state
        let gpu_hidden_accum = download_gpu_f32(&gpu_scratch_accum.hidden, h);
        let error = max_abs_error(&cpu_hidden_accum, &gpu_hidden_accum);

        eprintln!(
            "Layer {} (accumulated): max error = {:.6}",
            layer_idx, error
        );
        if error >= 1.0 {
            eprintln!("  CPU input[0..3]:  {:?}", &cpu_input[0..3]);
            eprintln!("  CPU output[0..3]: {:?}", &cpu_hidden_accum[0..3]);
            eprintln!("  GPU output[0..3]: {:?}", &gpu_hidden_accum[0..3]);
        }

        // Check for divergence
        assert!(
            error < 2.0,
            "GPU layer {} accumulated error {} exceeds tolerance",
            layer_idx,
            error
        );
    }

    eprintln!("\nAccumulated state test passed - GPU state propagation is correct");
}
