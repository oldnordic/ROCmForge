#![cfg(feature = "gpu")]
//! Multi-layer test to isolate where GPU/CPU divergence begins

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

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
#[serial]
fn test_multi_layer_divergence_isolation() {
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
    let n_layers = config.num_layers.min(5); // Test first 5 layers to save time

    eprintln!(
        "Testing multi-layer divergence isolation with {} layers",
        n_layers
    );

    for layer_idx in 0..n_layers {
        // Create scratch spaces for this layer
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

        // CPU reference - single layer
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

        // GPU computation - single layer
        gpu_layer_forward_hybrid(
            &device,
            gpu_weights.layer(layer_idx),
            Some(cpu_weights.layer(layer_idx)),
            &mut gpu_kv,
            &mut gpu_scratch,
            Some(&mut cpu_scratch),
            0,
            0,
            0, // token_id (dummy value)
            &config,
        )
        .expect("GPU layer forward should succeed");

        // Download GPU hidden state
        let gpu_hidden = download_gpu_f32(&gpu_scratch.hidden, h);

        // Compare
        let error = max_abs_error(&cpu_hidden, &gpu_hidden);
        eprintln!("Layer {} max error: {}", layer_idx, error);
        eprintln!("  CPU hidden[0..3]: {:?}", &cpu_hidden[0..3]);
        eprintln!("  GPU hidden[0..3]: {:?}", &gpu_hidden[0..3]);

        // Allow reasonable tolerance for quantization
        assert!(
            error < 1.0,
            "GPU layer {} error {} exceeds tolerance",
            layer_idx,
            error
        );
    }

    eprintln!("Multi-layer test passed - all individual layers are correct");
}
