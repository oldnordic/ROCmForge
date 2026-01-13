//! Phase C TDD Tests: Execution Plan + Fused QKV + GPU Attention Integration
//!
//! These tests must be created BEFORE implementation and should initially fail.
//! They verify ExecutionPlan construction, fused QKV correctness, attention correctness,
//! KV cache operations, and full decode_step() functionality.

use rocmforge::backend::{DeviceTensor, HipBackend, HipError, HipResult};
use rocmforge::loader::TensorShape;
use rocmforge::model::{
    config::{ModelConfig, ModelType},
    execution_plan::{ExecutionPlan, LayerPlan},
    kv_cache::KVCache,
};
use serial_test::serial;
use std::sync::Arc;

// REMOVED: Duplicate test_execution_plan_construction
// This test is already in execution_plan_construction_tests.rs:14

/// Test (B): Fused QKV correctness (CPU reference)
///
/// Given a weight matrix W_qkv and input X, fused QKV output must match CPU-split Q, K, V results.
#[test]
#[serial]
fn test_fused_qkv_correctness() {
    // Create tiny test data
    let hidden_size = 4;
    let head_dim = 2;
    let num_heads = 2;
    let seq_len = 3;

    // Input tensor X: [seq_len, hidden_size]
    let x_data = vec![
        1.0, 2.0, 3.0, 4.0, // token 0
        5.0, 6.0, 7.0, 8.0, // token 1
        9.0, 10.0, 11.0, 12.0, // token 2
    ];

    // QKV weight matrix: [3 * hidden_size, hidden_size] = [12, 4]
    let w_qkv_data = vec![
        // Q projection weights
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, // K projection weights
        0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, // V projection weights
        1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
    ];

    // Compute CPU reference: X @ W_qkv^T
    let mut cpu_fused_output = vec![0.0f32; seq_len * 3 * hidden_size];
    for i in 0..seq_len {
        for j in 0..(3 * hidden_size) {
            for k in 0..hidden_size {
                cpu_fused_output[i * 3 * hidden_size + j] +=
                    x_data[i * hidden_size + k] * w_qkv_data[j * hidden_size + k];
            }
        }
    }

    // Split fused output into Q, K, V on CPU
    let mut cpu_q = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut cpu_k = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut cpu_v = vec![0.0f32; seq_len * num_heads * head_dim];

    for i in 0..seq_len {
        for h in 0..num_heads {
            for d in 0..head_dim {
                let base_idx = i * 3 * hidden_size + h * head_dim + d;
                cpu_q[i * num_heads * head_dim + h * head_dim + d] = cpu_fused_output[base_idx];
                cpu_k[i * num_heads * head_dim + h * head_dim + d] =
                    cpu_fused_output[base_idx + num_heads * head_dim];
                cpu_v[i * num_heads * head_dim + h * head_dim + d] =
                    cpu_fused_output[base_idx + 2 * num_heads * head_dim];
            }
        }
    }

    // Test with actual implementation (this will fail until implemented)
    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();

    // Create device tensors
    let x_tensor = DeviceTensor::from_host_vec(
        backend,
        x_data,
        TensorShape::from_dims(&[seq_len, hidden_size]),
    )
    .expect("Failed to create X tensor");
    let w_qkv_tensor = DeviceTensor::from_host_vec(
        backend,
        w_qkv_data,
        TensorShape::from_dims(&[3 * hidden_size, hidden_size]),
    )
    .expect("Failed to create W_qkv tensor");

    // Perform fused QKV projection
    let fused_output = backend
        .fused_qkv(&x_tensor, &w_qkv_tensor, None)
        .expect("Fused QKV not implemented yet");

    // Split into Q, K, V
    let (q_tensor, k_tensor, v_tensor) = backend
        .split_qkv(&fused_output, num_heads, head_dim)
        .expect("QKV split not implemented yet");

    // Copy results back to host
    let gpu_q = q_tensor.to_host_vec().expect("Failed to copy Q to host");
    let gpu_k = k_tensor.to_host_vec().expect("Failed to copy K to host");
    let gpu_v = v_tensor.to_host_vec().expect("Failed to copy V to host");

    // Compare with CPU reference (allowing for small floating point differences)
    for i in 0..cpu_q.len() {
        assert!(
            (cpu_q[i] - gpu_q[i]).abs() < 1e-6,
            "Q mismatch at index {}: {} vs {}",
            i,
            cpu_q[i],
            gpu_q[i]
        );
    }
    for i in 0..cpu_k.len() {
        assert!(
            (cpu_k[i] - gpu_k[i]).abs() < 1e-6,
            "K mismatch at index {}: {} vs {}",
            i,
            cpu_k[i],
            gpu_k[i]
        );
    }
    for i in 0..cpu_v.len() {
        assert!(
            (cpu_v[i] - gpu_v[i]).abs() < 1e-6,
            "V mismatch at index {}: {} vs {}",
            i,
            cpu_v[i],
            gpu_v[i]
        );
    }
    // Check for memory leaks
    fixture.assert_no_leak(5);
}

/// Test (C): Attention correctness (CPU fallback reference)
///
/// GPU attention output must match CPU reference for a 1-layer tiny model
/// using scratch buffers + KV cache.
#[test]
#[serial]
fn test_attention_correctness() {
    // Tiny model parameters
    let num_heads = 2;
    let head_dim = 4;
    let seq_len = 3;
    let hidden_size = num_heads * head_dim;

    // Create backend and config
    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();
    let config = ModelConfig::new(
        1, // num_hidden_layers
        num_heads,
        head_dim,
        128, // max_position_embeddings
        hidden_size,
        32,    // intermediate_size (tiny)
        32000, // vocab_size
        ModelType::Llama,
    );

    // Create scratch buffers and KV cache
    let mut scratch = backend
        .create_scratch_buffers(&config)
        .expect("Failed to create scratch buffers");
    let mut kv_cache = KVCache::new(
        backend,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.head_dim,
        config.max_position_embeddings,
    )
    .expect("Failed to create KV cache");

    // Create test Q, K, V tensors
    let q_data = vec![1.0; seq_len * num_heads * head_dim];
    let k_data = vec![2.0; seq_len * num_heads * head_dim];
    let v_data = vec![3.0; seq_len * num_heads * head_dim];

    let q_tensor = DeviceTensor::from_host_vec(
        backend,
        q_data.clone(),
        TensorShape::from_dims(&[seq_len, num_heads, head_dim]),
    )
    .expect("Failed to create Q tensor");
    let k_tensor = DeviceTensor::from_host_vec(
        backend,
        k_data.clone(),
        TensorShape::from_dims(&[seq_len, num_heads, head_dim]),
    )
    .expect("Failed to create K tensor");
    let v_tensor = DeviceTensor::from_host_vec(
        backend,
        v_data.clone(),
        TensorShape::from_dims(&[seq_len, num_heads, head_dim]),
    )
    .expect("Failed to create V tensor");

    // Append to KV cache
    kv_cache
        .append(0, &k_tensor, &v_tensor)
        .expect("Failed to append to KV cache");

    // Compute CPU reference attention
    let cpu_attention_output =
        compute_cpu_attention_reference(&q_data, &k_data, &v_data, num_heads, head_dim, seq_len);

    // Compute GPU attention using scratch buffers
    // Note: This test will fail until compute_attention is properly implemented
    // For now, we'll create a placeholder output
    let output_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
    let gpu_attention_output =
        DeviceTensor::empty(&backend, output_shape).expect("Failed to create output tensor");

    // Compare results
    let gpu_output_host = gpu_attention_output
        .to_host_vec()
        .expect("Failed to copy GPU output");

    assert_eq!(
        cpu_attention_output.len(),
        gpu_output_host.len(),
        "Output length mismatch"
    );

    for i in 0..cpu_attention_output.len() {
        assert!(
            (cpu_attention_output[i] - gpu_output_host[i]).abs() < 1e-4,
            "Attention mismatch at index {}: {} vs {}",
            i,
            cpu_attention_output[i],
            gpu_output_host[i]
        );
    }
    // Check for memory leaks
    fixture.assert_no_leak(5);
}

/// Test (D): KV cache update + retrieval
///
/// decode_step() must append K,V, retrieve full history, and produce attention output with correct sequence length.
#[test]
#[serial]
fn test_kv_cache_update_and_retrieval() {
    let num_layers = 1;
    let num_heads = 2;
    let head_dim = 4;
    let hidden_size = num_heads * head_dim;

    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();
    let config = ModelConfig::new(
        num_layers,
        num_heads,
        head_dim,
        128, // max_position_embeddings
        hidden_size,
        32,    // intermediate_size
        32000, // vocab_size
        ModelType::Llama,
    );

    // Create ModelRuntime
    let mut runtime = backend
        .create_model_runtime(&config)
        .expect("Failed to create ModelRuntime");

    // Initial token embedding (token 0)
    let token_embedding = vec![1.0f32; hidden_size];
    let embedding_tensor = DeviceTensor::from_host_vec(
        backend,
        token_embedding,
        TensorShape::from_dims(&[hidden_size]),
    )
    .expect("Failed to create embedding tensor");

    // First decode step
    let logits1 = runtime
        .decode_step(&embedding_tensor)
        .expect("First decode_step failed");

    // Verify KV cache has 1 token
    assert_eq!(
        runtime.kv_cache().get_current_length(0).unwrap(),
        1,
        "KV cache should have 1 token after first step"
    );

    // Second token embedding (token 1)
    let token_embedding2 = vec![2.0f32; hidden_size];
    let embedding_tensor2 = DeviceTensor::from_host_vec(
        backend,
        token_embedding2,
        TensorShape::from_dims(&[hidden_size]),
    )
    .expect("Failed to create second embedding tensor");

    // Second decode step
    let logits2 = runtime
        .decode_step(&embedding_tensor2)
        .expect("Second decode_step failed");

    // Verify KV cache has 2 tokens
    assert_eq!(
        runtime.kv_cache().get_current_length(0).unwrap(),
        2,
        "KV cache should have 2 tokens after second step"
    );

    // Verify logits are different (indicating different attention patterns)
    let logits1_host = logits1.to_host_vec().expect("Failed to copy first logits");
    let logits2_host = logits2.to_host_vec().expect("Failed to copy second logits");

    let mut different = false;
    for i in 0..logits1_host.len().min(logits2_host.len()) {
        if (logits1_host[i] - logits2_host[i]).abs() > 1e-6 {
            different = true;
            break;
        }
    }
    assert!(different, "Logits should be different between steps");
    // Check for memory leaks
    fixture.assert_no_leak(5);
}

/// Test (E): Full decode_step() with micro-model
///
/// Build a 1-layer micro-model (hidden size 4 or 8), run decode_step(),
/// and output logits must match CPU reference path.
#[test]
#[serial]
fn test_full_decode_step_micro_model() {
    // Micro-model parameters
    let num_layers = 1;
    let num_heads = 2;
    let head_dim = 2;
    let hidden_size = num_heads * head_dim; // 4
    let intermediate_size = 8;
    let vocab_size = 100;

    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();
    let config = ModelConfig::new(
        num_layers,
        num_heads,
        head_dim,
        64, // max_position_embeddings
        hidden_size,
        intermediate_size,
        vocab_size,
        ModelType::Llama,
    );

    // Create ModelRuntime with synthetic weights
    let mut runtime = create_micro_model_runtime(&backend, &config, vocab_size)
        .expect("Failed to create micro-model runtime");

    // Input token embedding
    let token_embedding = vec![0.5f32; hidden_size];
    let embedding_tensor = DeviceTensor::from_host_vec(
        backend,
        token_embedding.clone(),
        TensorShape::from_dims(&[hidden_size]),
    )
    .expect("Failed to create embedding tensor");

    // Run decode_step
    let logits = runtime
        .decode_step(&embedding_tensor)
        .expect("decode_step failed");

    // Verify logits shape
    assert_eq!(
        logits.shape().dims(),
        &[vocab_size],
        "Logits shape should be [vocab_size]"
    );

    // Compute CPU reference for comparison
    let cpu_logits = compute_cpu_decode_step_reference(&token_embedding, &config, vocab_size);

    // Compare results
    let gpu_logits_host = logits.to_host_vec().expect("Failed to copy logits to host");

    assert_eq!(
        cpu_logits.len(),
        gpu_logits_host.len(),
        "Logits length mismatch"
    );

    for i in 0..cpu_logits.len() {
        assert!(
            (cpu_logits[i] - gpu_logits_host[i]).abs() < 1e-3,
            "Logit mismatch at index {}: {} vs {}",
            i,
            cpu_logits[i],
            gpu_logits_host[i]
        );
    }
    // Check for memory leaks
    fixture.assert_no_leak(5);
}

// Helper functions for CPU reference computations

fn compute_cpu_attention_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * num_heads * head_dim];

    for h in 0..num_heads {
        for i in 0..seq_len {
            let mut attention_weights = vec![0.0f32; seq_len];

            // Compute QK^T for this head and position
            for j in 0..seq_len {
                let mut dot_product = 0.0f32;
                for d in 0..head_dim {
                    let q_idx = i * num_heads * head_dim + h * head_dim + d;
                    let k_idx = j * num_heads * head_dim + h * head_dim + d;
                    dot_product += q[q_idx] * k[k_idx];
                }
                attention_weights[j] = dot_product / (head_dim as f32).sqrt();
            }

            // Apply softmax
            let max_weight = attention_weights
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_sum: f32 = attention_weights
                .iter()
                .map(|&w| (w - max_weight).exp())
                .sum();
            for j in 0..seq_len {
                attention_weights[j] = (attention_weights[j] - max_weight).exp() / exp_sum;
            }

            // Compute weighted sum of V
            for d in 0..head_dim {
                let mut weighted_sum = 0.0f32;
                for j in 0..seq_len {
                    let v_idx = j * num_heads * head_dim + h * head_dim + d;
                    weighted_sum += attention_weights[j] * v[v_idx];
                }
                let out_idx = i * num_heads * head_dim + h * head_dim + d;
                output[out_idx] = weighted_sum;
            }
        }
    }

    output
}

fn compute_cpu_decode_step_reference(
    token_embedding: &[f32],
    config: &ModelConfig,
    vocab_size: usize,
) -> Vec<f32> {
    // Simplified CPU reference for micro-model
    // This would implement the full transformer layer computation
    let hidden_size = config.hidden_size;
    let intermediate_size = config.intermediate_size;

    // For now, return a simple linear projection as reference
    // In a real implementation, this would be the full forward pass
    let mut logits = vec![0.0f32; vocab_size];
    for i in 0..vocab_size {
        for j in 0..hidden_size {
            // Simple linear combination with synthetic weights
            logits[i] += token_embedding[j] * ((i as f32 + j as f32) * 0.01);
        }
    }

    logits
}

fn create_micro_model_runtime(
    backend: &HipBackend,
    config: &ModelConfig,
    vocab_size: usize,
) -> Result<rocmforge::backend::hip_backend::ModelRuntime, HipError> {
    // This would create a ModelRuntime with synthetic weights for testing
    // For now, this will fail until the actual implementation exists
    Err(HipError::GenericError(
        "create_micro_model_runtime not implemented yet".to_string(),
    ))
}
