//! TDD tests for KV Cache and Scratch Buffer Manager
//! Tests must fail initially, then pass after implementation

#[cfg(feature = "rocm")]
use rocmforge::backend::scratch::ScratchBufferManager;
#[cfg(feature = "rocm")]
use rocmforge::backend::ScratchBufferManager;
#[cfg(feature = "rocm")]
use rocmforge::backend::{DeviceTensor, HipBackend, ModelRuntime};
#[cfg(feature = "rocm")]
use rocmforge::loader::mmap_loader::TensorShape;
#[cfg(feature = "rocm")]
use rocmforge::model::KVCache;
#[cfg(feature = "rocm")]
use rocmforge::model::ModelConfig;
#[cfg(feature = "rocm")]
use serial_test::serial;

#[cfg(feature = "rocm")]
#[test]
#[serial]
fn scratch_buffer_reuse_invariant() {
    // Test that scratch buffers are reused without reallocation
    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();

    // Create scratch buffer manager with typical LLaMA-7B config
    let mut scratch = ScratchBufferManager::new(
        backend, 32,   // num_heads
        4096, // hidden_size
        128,  // head_dim
        2048, // max_seq_len
    )
    .expect("Failed to create scratch buffer manager");

    // Get initial buffer sizes
    let initial_attention_size = scratch.attention_scores().size();
    let initial_softmax_size = scratch.softmax_temp().size();
    let initial_mlp_size = scratch.mlp_intermediate().size();
    let initial_layernorm_size = scratch.layernorm_temp().size();

    // Simulate first attention call - modify buffers
    let attention_scores = scratch.attention_scores();
    let softmax_temp = scratch.softmax_temp();

    // Verify buffers have correct shapes
    assert_eq!(
        attention_scores.len(),
        32 * 2048 * 2048,
        "Attention scores should have correct shape"
    );
    assert_eq!(
        softmax_temp.len(),
        32 * 2048,
        "Softmax temp should have correct shape"
    );

    // Simulate second attention call - buffers should be reused
    let attention_scores_2 = scratch.attention_scores();
    let softmax_temp_2 = scratch.softmax_temp();

    // Assert no buffer shapes/sizes changed (reuse invariants)
    assert_eq!(
        attention_scores_2.size(),
        initial_attention_size,
        "Attention scores buffer should be reused"
    );
    assert_eq!(
        softmax_temp_2.size(),
        initial_softmax_size,
        "Softmax temp buffer should be reused"
    );
    assert_eq!(
        scratch.mlp_intermediate().size(),
        initial_mlp_size,
        "MLP intermediate buffer should be reused"
    );
    assert_eq!(
        scratch.layernorm_temp().size(),
        initial_layernorm_size,
        "Layernorm temp buffer should be reused"
    );

    // Assert no new allocations occurred (same memory addresses)
    assert_eq!(
        attention_scores.as_ptr() as usize,
        attention_scores_2.as_ptr() as usize,
        "Attention scores should use same memory address"
    );
    assert_eq!(
        softmax_temp.as_ptr() as usize,
        softmax_temp_2.as_ptr() as usize,
        "Softmax temp should use same memory address"
    );

    // Check for memory leaks
    fixture.assert_no_leak(5);
}

#[cfg(feature = "rocm")]
#[test]
#[serial]
fn kv_cache_append_and_retrieve_consistency() {
    // Test KV cache append and retrieve operations
    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();

    // Create KV cache with typical LLaMA-7B config
    let mut kv_cache = KVCache::new(
        backend, 32,   // num_layers
        32,   // num_heads
        128,  // head_dim
        2048, // max_seq_len
    )
    .expect("Failed to create KV cache");

    // Create test key and value tensors for layer 0, head 0
    let key_shape = TensorShape::from_dims(&[1, 32, 128]); // [seq_len, num_heads, head_dim]
    let value_shape = TensorShape::from_dims(&[1, 32, 128]); // [seq_len, num_heads, head_dim]

    let key_data = vec![1.0f32; 1 * 32 * 128];
    let value_data = vec![2.0f32; 1 * 32 * 128];

    let key_tensor = DeviceTensor::from_host_vec(backend, key_data.clone(), key_shape)
        .expect("Failed to create key tensor");
    let value_tensor = DeviceTensor::from_host_vec(backend, value_data.clone(), value_shape)
        .expect("Failed to create value tensor");

    // Append to KV cache
    kv_cache
        .append(0, &key_tensor, &value_tensor)
        .expect("Failed to append to KV cache");

    // Retrieve from KV cache
    let (retrieved_key, retrieved_value) = kv_cache
        .get(0, 0, 0)
        .expect("Failed to retrieve from KV cache");

    // Verify GPU buffer size + shape consistency
    assert_eq!(
        retrieved_key.len(),
        128,
        "Retrieved key should have head_dim elements"
    );
    assert_eq!(
        retrieved_value.len(),
        128,
        "Retrieved value should have head_dim elements"
    );
    assert_eq!(
        retrieved_key.size(),
        128 * 4,
        "Retrieved key should be 128 bytes"
    );
    assert_eq!(
        retrieved_value.size(),
        128 * 4,
        "Retrieved value should be 128 bytes"
    );

    // Verify data integrity by copying back to host
    let retrieved_key_host = retrieved_key
        .to_host_vec()
        .expect("Failed to copy retrieved key to host");
    let retrieved_value_host = retrieved_value
        .to_host_vec()
        .expect("Failed to copy retrieved value to host");

    // First element should match what we stored (head 0, seq 0)
    assert_eq!(
        retrieved_key_host[0], 1.0,
        "Retrieved key should match stored key"
    );
    assert_eq!(
        retrieved_value_host[0], 2.0,
        "Retrieved value should match stored value"
    );

    // Check for memory leaks
    fixture.assert_no_leak(5);
}

#[cfg(feature = "rocm")]
#[test]
#[serial]
fn kv_cache_capacity_boundary() {
    // Test KV cache capacity limits
    let fixture = rocmforge::GPU_FIXTURE
        .as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();

    // Create KV cache with small max_seq_len for testing
    let mut kv_cache = KVCache::new(
        backend, 2,  // num_layers
        4,  // num_heads
        64, // head_dim
        2,  // max_seq_len (small for testing)
    )
    .expect("Failed to create KV cache");

    // Create test tensors
    let key_shape = TensorShape::from_dims(&[1, 4, 64]);
    let value_shape = TensorShape::from_dims(&[1, 4, 64]);

    let key_data = vec![1.0f32; 1 * 4 * 64];
    let value_data = vec![2.0f32; 1 * 4 * 64];

    let key_tensor = DeviceTensor::from_host_vec(backend, key_data, key_shape)
        .expect("Failed to create key tensor");
    let value_tensor = DeviceTensor::from_host_vec(backend, value_data, value_shape)
        .expect("Failed to create value tensor");

    // Append up to capacity (should succeed)
    kv_cache
        .append(0, &key_tensor, &value_tensor)
        .expect("First append should succeed");
    kv_cache
        .append(0, &key_tensor, &value_tensor)
        .expect("Second append should succeed");

    // Attempt to append beyond max_seq_len (should fail with meaningful error)
    let result = kv_cache.append(0, &key_tensor, &value_tensor);
    assert!(result.is_err(), "Append beyond capacity should fail");

    let error = result.unwrap_err();
    let error_string = format!("{}", error);
    assert!(
        error_string.contains("capacity") || error_string.contains("exceeded"),
        "Error should mention capacity: {}",
        error_string
    );

    // Check for memory leaks
    fixture.assert_no_leak(5);
}

#[cfg(feature = "rocm")]
#[test]
#[serial]
fn model_runtime_initialization_consistency() {
    // Test ModelRuntime creates both KV cache and scratch buffers
    let model_config = ModelConfig {
        num_hidden_layers: 32,
        num_attention_heads: 32,
        head_dim: Some(128),
        hidden_size: 4096,
        intermediate_size: 11008,
        max_position_embeddings: 2048,
        vocab_size: 32000,
        model_type: "llama".to_string(),
    };

    let mut runtime = ModelRuntime::new_with_config(model_config)
        .expect("Failed to create ModelRuntime with config");

    // Test that scratch buffers exist and have correct shapes
    let scratch = runtime.scratch_buffers();
    assert!(
        scratch.attention_scores().len() > 0,
        "Attention scores buffer should exist"
    );
    assert!(
        scratch.softmax_temp().len() > 0,
        "Softmax temp buffer should exist"
    );
    assert!(
        scratch.mlp_intermediate().len() > 0,
        "MLP intermediate buffer should exist"
    );
    assert!(
        scratch.layernorm_temp().len() > 0,
        "Layernorm temp buffer should exist"
    );

    // Test scratch buffer shapes match config
    let expected_attention_size = 32 * 2048 * 2048; // num_heads * max_seq_len * max_seq_len
    let expected_softmax_size = 32 * 2048; // num_heads * max_seq_len
    let expected_mlp_size = 4096 * 4; // hidden_size * 4 (for SwiGLU)
    let expected_layernorm_size = 4096; // hidden_size

    assert_eq!(
        scratch.attention_scores().len(),
        expected_attention_size,
        "Attention scores buffer should have correct size"
    );
    assert_eq!(
        scratch.softmax_temp().len(),
        expected_softmax_size,
        "Softmax temp buffer should have correct size"
    );
    assert_eq!(
        scratch.mlp_intermediate().len(),
        expected_mlp_size,
        "MLP intermediate buffer should have correct size"
    );
    assert_eq!(
        scratch.layernorm_temp().len(),
        expected_layernorm_size,
        "Layernorm temp buffer should have correct size"
    );

    // Test that KV cache exists and has correct shape
    let kv_cache = runtime.kv_cache();
    let (key, value) = kv_cache
        .get(0, 0, 0) // layer 0, head 0, seq 0
        .expect("KV cache should allow retrieval");

    assert_eq!(key.len(), 128, "KV cache key should have head_dim elements");
    assert_eq!(
        value.len(),
        128,
        "KV cache value should have head_dim elements"
    );

    // Test memory invariants - all buffers should be GPU-resident
    assert_eq!(
        scratch.attention_scores().buffer().size(),
        expected_attention_size * 4,
        "Attention scores should be GPU-resident"
    );
    assert_eq!(
        scratch.softmax_temp().buffer().size(),
        expected_softmax_size * 4,
        "Softmax temp should be GPU-resident"
    );
    assert_eq!(
        scratch.mlp_intermediate().buffer().size(),
        expected_mlp_size * 4,
        "MLP intermediate should be GPU-resident"
    );
    assert_eq!(
        scratch.layernorm_temp().buffer().size(),
        expected_layernorm_size * 4,
        "Layernorm temp should be GPU-resident"
    );

    // Note: ModelRuntime creates its own backend, so we can't check for leaks here
}
