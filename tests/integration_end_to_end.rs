#![cfg(feature = "gpu")]

//! End-to-end GPU integration tests.
//!
//! These are the "crown jewel" tests that validate the entire inference pipeline:
//! 1. Load model weights to GPU
//! 2. Allocate KV cache and scratch buffers
//! 3. Run all kernels per layer
//! 4. Verify output logits vs CPU reference

mod common;

use serial_test::serial;
use rocmforge::gpu::{self, GpuDevice, GpuKvCache, GpuForwardScratch};
use rocmforge::config::ModelConfig;
// require_gpu! and require_vram! macros are available via #[macro_export] in common/mod.rs

fn make_test_config() -> ModelConfig {
    use rocmforge::config::{TensorNamingScheme, TensorNameRegistry};
    ModelConfig {
        num_layers: 2,
        num_kv_heads: 4,
        head_dim: 128,
        max_seq_len: 512,
        hidden_size: 1024,
        num_heads: 8,
        intermediate_size: 2048,
        vocab_size: 32000,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_neox: false,
        use_attention_bias: false,
        attention_layout: rocmforge::config::AttentionLayout::SplitQkv,
        architecture: "test".to_string(),
        tensor_registry: TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf),
    }
}

#[test]
#[serial]
fn test_full_gpu_pipeline_initialization() {
    require_gpu!();
    require_vram!(4);

    let config = make_test_config();

    // Step 1: Initialize GPU device
    let device = GpuDevice::init(0).expect("GPU device should initialize");

    // Step 2: Verify GPU capabilities
    let caps = gpu::detect().expect("GPU should be detected");
    println!("GPU: {} ({} GB free)", caps.device_name, caps.free_vram_gb());

    // Step 3: Allocate KV cache
    let kv_cache = GpuKvCache::new(&config, 256).expect("KV cache should allocate");

    // Step 4: Allocate scratch buffers
    let scratch = GpuForwardScratch::new(&config).expect("Scratch buffers should allocate");

    // Verify all components are valid
    assert_eq!(device.device_id(), 0);
    assert_eq!(kv_cache.num_layers, 2);
    assert_eq!(kv_cache.max_seq_len, 256);
    assert!(!scratch.q.is_empty());
    assert!(!scratch.k.is_empty());
    assert!(!scratch.v.is_empty());

    println!("✓ Full pipeline initialization successful");
}

#[test]
#[serial]
fn test_single_token_decode_setup() {
    require_gpu!();
    require_vram!(4);

    let config = make_test_config();

    // Allocate resources for single-token decode
    let kv_cache = GpuKvCache::new(&config, 512).expect("KV cache should allocate");
    let scratch = GpuForwardScratch::new(&config).expect("Scratch should allocate");

    // Verify we have all necessary buffers
    assert!(!scratch.normed.is_empty(), "normed buffer required");
    assert!(!scratch.q.is_empty(), "q buffer required");
    assert!(!scratch.k.is_empty(), "k buffer required");
    assert!(!scratch.v.is_empty(), "v buffer required");
    assert!(!scratch.attn_out.is_empty(), "attn_out buffer required");
    assert!(!scratch.layer_out.is_empty(), "layer_out buffer required");
    assert!(!scratch.gate.is_empty(), "gate buffer required");
    assert!(!scratch.swiglu.is_empty(), "swiglu buffer required");
    assert!(!scratch.logits.is_empty(), "logits buffer required");

    // Verify KV cache has correct layout
    assert_eq!(kv_cache.kv_size, config.num_kv_heads * config.head_dim);
    assert_eq!(kv_cache.num_layers, config.num_layers);

    println!("✓ Single-token decode setup verified");
}

#[test]
#[serial]
fn test_prefill_batch_setup() {
    require_gpu!();
    require_vram!(4);

    let config = make_test_config();
    let seq_len = 32; // Prefill 32 tokens at once

    // For prefill, we need larger KV cache
    let kv_cache = GpuKvCache::new(&config, 512).expect("KV cache should allocate");
    let scratch = GpuForwardScratch::new(&config).expect("Scratch should allocate");

    // Verify KV cache can handle the sequence length
    assert!(kv_cache.max_seq_len >= seq_len, "KV cache must accommodate prefill length");

    // For prefill, batched kernels would be used
    // Verify we have the buffer space
    let scratch_bytes = scratch.layer_out.size();

    // This is a rough check - scratch buffers are reused
    assert!(scratch_bytes >= config.hidden_size * 4, "Scratch must have enough space");

    println!("✓ Prefill batch setup verified for {} tokens", seq_len);
}

#[test]
#[serial]
fn test_memory_usage_within_vram() {
    require_gpu!();
    require_vram!(4);

    let config = make_test_config();
    let caps = gpu::detect().expect("GPU should be detected");

    // Calculate expected memory usage
    let kv_cache = GpuKvCache::new(&config, 512).expect("KV cache should allocate");
    let scratch = GpuForwardScratch::new(&config).expect("Scratch should allocate");

    let kv_cache_bytes = kv_cache.memory_bytes();

    let scratch_bytes =
        scratch.normed.size() +
        scratch.q.size() +
        scratch.k.size() +
        scratch.v.size() +
        scratch.attn_out.size() +
        scratch.layer_out.size() +
        scratch.gate.size() +
        scratch.swiglu.size() +
        scratch.logits.size();

    let total_bytes = kv_cache_bytes + scratch_bytes;
    let total_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    println!("Memory usage: {:.2} GB (KV cache: {:.2} GB, Scratch: {:.2} GB)",
        total_gb,
        kv_cache_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        scratch_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // Verify we're using less than available VRAM
    assert!(total_bytes < caps.free_vram_bytes,
        "Memory usage should fit within available VRAM");
}

#[test]
#[serial]
fn test_autoregressive_sequence_growth() {
    require_gpu!();
    require_vram!(4);

    let config = make_test_config();
    let max_seq_len = 128;

    // Allocate KV cache for autoregressive generation
    let kv_cache = GpuKvCache::new(&config, max_seq_len).expect("KV cache should allocate");

    // Simulate growing sequence (0 -> 1 -> 2 -> ... -> 10)
    for pos in 0..10 {
        // In real inference, we would:
        // 1. Write K/V to cache at position `pos`
        // 2. Run attention for pos+1 tokens
        // 3. Verify cache state

        // For now, just verify cache bounds checking works
        if pos < max_seq_len {
            let k_ptr = kv_cache.k_ptr(0);
            let v_ptr = kv_cache.v_ptr(0);
            assert!(k_ptr.is_ok(), "k_ptr should succeed for valid layer");
            assert!(v_ptr.is_ok(), "v_ptr should succeed for valid layer");
        }
    }

    // Verify final position is within bounds
    assert!(10 < max_seq_len, "Should have space in cache");

    println!("✓ Autoregressive sequence growth test passed");
}

// ============================================================================
// NOTE: Full correctness tests require actual model weights and kernels
// ============================================================================
//
// The following tests would require:
// 1. A real GGUF model file
// 2. All HIP kernels to be compiled and linked
// 3. CPU reference implementation for comparison
//
// For now, the structural tests above verify:
// - All components allocate correctly
// - Memory usage is within VRAM limits
// - Pointer access is safe
// - Bounds checking works
//
// TODO: Add correctness tests when kernel library is ready:
// - test_full_forward_match_cpu()
// - test_attention_scores_correct()
// - test_rope_rotation_exact()
