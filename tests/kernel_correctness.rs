//! Kernel correctness tests.
//!
//! Verifies that GPU kernels produce the same output as CPU reference.

#![cfg(feature = "gpu")]

mod common;

use rocmforge::config::ModelConfig;
use rocmforge::cpu::{
    cache::{CpuForwardScratch, CpuKvCache},
    forward::{cpu_embed_token, cpu_full_forward},
    weights::CpuModelWeights,
};
use rocmforge::gpu::{GpuDevice, GpuForwardScratch, GpuKvCache};
use rocmforge::loader::GgufFile;
use rocmforge::tokenizer::BpeTokenizer;
use serial_test::serial;
use std::path::Path;

const MODEL_PATH: &str = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q4_0.gguf";
const TOLERANCE: f32 = 1e-3; // Allow small floating-point differences

fn skip_if_model_missing() -> bool {
    !Path::new(MODEL_PATH).exists()
}

#[test]
#[serial]
fn test_q4_0_dequantization_correctness() {
    if skip_if_model_missing() {
        println!("Skipping: test model not found at {}", MODEL_PATH);
        return;
    }

    require_real_model_gpu_tests!();

    // Load model
    let gguf = GgufFile::open(MODEL_PATH).unwrap();
    let config = ModelConfig::from_gguf(&gguf).unwrap();
    let cpu_weights = CpuModelWeights::load(&gguf, &config).unwrap();
    let tok = BpeTokenizer::from_gguf(gguf.tokenizer_data());

    // Create sample input
    let prompt = "Hello, world!";
    let tokens = tok.encode(prompt, false);

    // Run CPU reference
    let mut cpu_kv = CpuKvCache::new(&config, tokens.len());
    let mut cpu_scratch = CpuForwardScratch::new(&config);
    let mut cpu_hidden = vec![0.0f32; config.hidden_size];

    for (pos, &token_id) in tokens.iter().enumerate() {
        cpu_embed_token(token_id, &cpu_weights, &mut cpu_hidden, &config, None);
        cpu_full_forward(
            &mut cpu_hidden,
            &cpu_weights,
            &mut cpu_kv,
            &mut cpu_scratch,
            pos,
            &config,
        )
        .expect("CPU forward should succeed");
    }

    let cpu_output = cpu_scratch.logits.to_vec();

    // Run GPU kernel
    let device = GpuDevice::init(0).expect("GPU device should be available");
    let gpu_weights =
        rocmforge::gpu::GpuModelWeights::load(&gguf, &config).expect("GPU weights should load");

    let mut gpu_kv = GpuKvCache::new(&config, tokens.len()).expect("GPU KV cache should allocate");
    let mut gpu_scratch = GpuForwardScratch::new(&config).expect("GPU scratch should allocate");
    let mut host_scratch = CpuForwardScratch::new(&config);

    for (pos, &token_id) in tokens.iter().enumerate() {
        rocmforge::gpu::gpu_embed_token_hybrid(
            &device,
            token_id,
            &gpu_weights,
            &cpu_weights,
            &mut gpu_scratch,
            &mut host_scratch,
            &config,
        )
        .expect("GPU embed should succeed");

        rocmforge::gpu::gpu_full_forward_hybrid(
            &device,
            &gpu_weights,
            &cpu_weights,
            &mut gpu_kv,
            &mut gpu_scratch,
            &mut host_scratch,
            pos,
            &config,
            rocmforge::gpu::GpuLogitsMode::DownloadToHost,
        )
        .expect("GPU forward should succeed");
    }

    // Download GPU output
    let gpu_output = gpu_scratch
        .logits
        .copy_to_host_vec()
        .expect("GPU logits should download");

    // Compare outputs
    assert_eq!(
        cpu_output.len(),
        gpu_output.len(),
        "Output lengths must match"
    );

    let mut max_diff = 0.0f32;
    for (i, (cpu_val, gpu_val)) in cpu_output.iter().zip(gpu_output.iter()).enumerate() {
        let diff = (cpu_val - gpu_val).abs();
        max_diff = max_diff.max(diff);

        if diff > TOLERANCE {
            panic!(
                "Output mismatch at index {}: CPU={}, GPU={}, diff={} (tolerance={})",
                i, cpu_val, gpu_val, diff, TOLERANCE
            );
        }
    }

    println!(
        "✅ Q4_0 dequantization correctness test passed (max_diff={:.6})",
        max_diff
    );
}

#[test]
#[serial]
fn test_fusion_kernel_coherence() {
    if skip_if_model_missing() {
        println!("Skipping: test model not found at {}", MODEL_PATH);
        return;
    }

    require_real_model_gpu_tests!();

    // Load model
    let gguf = GgufFile::open(MODEL_PATH).unwrap();
    let config = ModelConfig::from_gguf(&gguf).unwrap();
    let cpu_weights = CpuModelWeights::load(&gguf, &config).unwrap();
    let tok = BpeTokenizer::from_gguf(gguf.tokenizer_data());

    let device = GpuDevice::init(0).expect("GPU device should be available");
    let gpu_weights =
        rocmforge::gpu::GpuModelWeights::load(&gguf, &config).expect("GPU weights should load");

    // Create GPU forward pass
    let mut gpu_kv = GpuKvCache::new(&config, 512).expect("GPU KV cache should allocate");
    let mut gpu_scratch = GpuForwardScratch::new(&config).expect("GPU scratch should allocate");
    let mut host_scratch = CpuForwardScratch::new(&config);

    // Generate tokens for a simple prompt
    let prompt = "The capital of France is";
    let tokens = tok.encode(prompt, false);

    // Run generation through GPU
    let mut output_tokens = Vec::new();

    // Process prompt tokens
    for (pos, &token_id) in tokens.iter().enumerate() {
        rocmforge::gpu::gpu_embed_token_hybrid(
            &device,
            token_id,
            &gpu_weights,
            &cpu_weights,
            &mut gpu_scratch,
            &mut host_scratch,
            &config,
        )
        .expect("GPU embed should succeed");

        rocmforge::gpu::gpu_full_forward_hybrid(
            &device,
            &gpu_weights,
            &cpu_weights,
            &mut gpu_kv,
            &mut gpu_scratch,
            &mut host_scratch,
            pos,
            &config,
            rocmforge::gpu::GpuLogitsMode::DownloadToHost,
        )
        .expect("GPU forward should succeed");
    }

    // Generate a few tokens
    #[allow(unused_assignments)]
    let mut current_token = *tokens.last().unwrap();
    for pos in tokens.len()..tokens.len() + 10 {
        // Sample from logits (greedy)
        let logits = gpu_scratch
            .logits
            .copy_to_host_vec()
            .expect("GPU logits should download");
        current_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();

        output_tokens.push(current_token);

        // Embed and forward
        rocmforge::gpu::gpu_embed_token_hybrid(
            &device,
            current_token,
            &gpu_weights,
            &cpu_weights,
            &mut gpu_scratch,
            &mut host_scratch,
            &config,
        )
        .expect("GPU embed should succeed");

        rocmforge::gpu::gpu_full_forward_hybrid(
            &device,
            &gpu_weights,
            &cpu_weights,
            &mut gpu_kv,
            &mut gpu_scratch,
            &mut host_scratch,
            pos,
            &config,
            rocmforge::gpu::GpuLogitsMode::DownloadToHost,
        )
        .expect("GPU forward should succeed");
    }

    // Verify output is coherent (not repetitive loops or garbage)
    let output_text = tok.decode(&output_tokens, false);

    // Simple heuristic: output should not contain 3+ consecutive repeated tokens
    let mut repeat_count = 0;
    for i in 2..output_tokens.len() {
        if output_tokens[i] == output_tokens[i - 1] && output_tokens[i] == output_tokens[i - 2] {
            repeat_count += 1;
        }
    }

    assert!(
        repeat_count < 3,
        "Output contains repetitive loops (count: {}): \"{}\"",
        repeat_count,
        output_text
    );

    // Check for garbage (mixed Chinese/English is a common symptom)
    let has_chinese = output_text.chars().any(|c| {
        let cp = c as u32;
        (0x4E00..=0x9FFF).contains(&cp)
    });

    assert!(
        !has_chinese,
        "Output has mixed Chinese/English (garbage symptom): \"{}\"",
        output_text
    );

    println!(
        "✅ Fusion kernel coherence test passed: \"{}\"",
        output_text
    );
}

#[test]
#[serial]
fn test_single_layer_correctness() {
    if skip_if_model_missing() {
        println!("Skipping: test model not found at {}", MODEL_PATH);
        return;
    }

    require_real_model_gpu_tests!();

    // Load model
    let gguf = GgufFile::open(MODEL_PATH).unwrap();
    let config = ModelConfig::from_gguf(&gguf).unwrap();
    let cpu_weights = CpuModelWeights::load(&gguf, &config).unwrap();

    let device = GpuDevice::init(0).expect("GPU device should be available");
    let gpu_weights =
        rocmforge::gpu::GpuModelWeights::load(&gguf, &config).expect("GPU weights should load");

    // Test first layer only
    let layer_idx = 0;
    let layer_weights = &cpu_weights.layers[layer_idx];
    let gpu_layer_weights = &gpu_weights.layers[layer_idx];

    // Create input
    let mut cpu_hidden: Vec<f32> = (1..=config.hidden_size as i32)
        .map(|i| i as f32 / 1000.0)
        .collect();

    // CPU reference
    let mut cpu_kv = CpuKvCache::new(&config, 1);
    let mut cpu_scratch = CpuForwardScratch::new(&config);
    let half = config.head_dim / 2;
    for i in 0..half {
        let angle = 0.0f32 * config.rope_freq[i];
        let (s, c) = angle.sin_cos();
        cpu_scratch.rope_sin[i] = s;
        cpu_scratch.rope_cos[i] = c;
    }
    let rope_sin = unsafe { std::slice::from_raw_parts(cpu_scratch.rope_sin.as_ptr(), half) };
    let rope_cos = unsafe { std::slice::from_raw_parts(cpu_scratch.rope_cos.as_ptr(), half) };

    rocmforge::cpu::forward::cpu_layer_forward(
        &mut cpu_hidden,
        layer_weights,
        &mut cpu_kv,
        &mut cpu_scratch,
        layer_idx,
        0,
        rope_sin,
        rope_cos,
        &config,
        false,
    )
    .expect("CPU layer forward should succeed");

    let cpu_output = cpu_hidden.clone();

    // GPU kernel
    let mut gpu_kv = GpuKvCache::new(&config, 1).expect("GPU KV cache should allocate");
    let mut gpu_scratch = GpuForwardScratch::new(&config).expect("GPU scratch should allocate");

    let cpu_hidden_bytes = unsafe {
        std::slice::from_raw_parts(
            cpu_hidden.as_ptr() as *const u8,
            cpu_hidden.len() * std::mem::size_of::<f32>(),
        )
    };
    gpu_scratch
        .hidden
        .copy_from_host(cpu_hidden_bytes)
        .expect("GPU hidden buffer copy failed");

    rocmforge::gpu::gpu_layer_forward_hybrid(
        &device,
        gpu_layer_weights,
        Some(cpu_weights.layer(layer_idx)),
        &mut gpu_kv,
        &mut gpu_scratch,
        Some(&mut cpu_scratch),
        layer_idx,
        0,
        &config,
    )
    .expect("GPU layer forward should succeed");

    let gpu_output = gpu_scratch
        .hidden
        .copy_to_host_vec()
        .expect("GPU output should download");

    // Compare
    assert_eq!(
        cpu_output.len(),
        gpu_output.len(),
        "Output lengths must match"
    );

    let mut max_diff = 0.0f32;
    for (i, (cpu_val, gpu_val)) in cpu_output.iter().zip(gpu_output.iter()).enumerate() {
        let diff = (cpu_val - gpu_val).abs();
        max_diff = max_diff.max(diff);

        if diff > TOLERANCE {
            panic!(
                "Layer {} output mismatch at index {}: CPU={}, GPU={}, diff={} (tolerance={})",
                layer_idx, i, cpu_val, gpu_val, diff, TOLERANCE
            );
        }
    }

    println!(
        "✅ Single layer correctness test passed (layer={}, max_diff={:.6})",
        layer_idx, max_diff
    );
}
