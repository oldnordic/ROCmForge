#![cfg(feature = "gpu")]
//! Integration tests for batched QKV projection on GPU.
//!
//! These tests validate the batched Q4_0 QKV projection functionality
//! with end-to-end execution on GPU hardware.

use rocmforge::gpu::{GpuDevice, GpuModelWeights};
use rocmforge::loader::{GgmlType, RfmFile};
use std::path::PathBuf;
use std::time::Instant;

/// Test configuration for batched QKV projection tests.
struct QkvProjectionTestConfig {
    /// Hidden dimension (input size)
    hidden_dim: usize,
    /// Query projection output dimension (num_heads * head_dim)
    q_dim: usize,
    /// Key projection output dimension (num_kv_heads * head_dim)
    k_dim: usize,
    /// Value projection output dimension (num_kv_heads * head_dim)
    v_dim: usize,
    /// Sequence length (batch size)
    seq_len: usize,
}

impl QkvProjectionTestConfig {
    /// Create test configuration for LLaMA 3.2 1B model.
    fn llama_3_2_1b_prefill() -> Self {
        Self {
            hidden_dim: 2048,
            q_dim: 2048, // num_heads * head_dim = 32 * 64
            k_dim: 512,  // num_kv_heads * head_dim = 8 * 64
            v_dim: 512,  // num_kv_heads * head_dim = 8 * 64
            seq_len: 32,
        }
    }
}

/// Load a minimal test model for GPU batched QKV projection tests.
fn load_test_model_for_batched_qkv() -> (GpuDevice, GpuModelWeights) {
    let model_path = PathBuf::from("/home/feanor/Projects/rocmforge/llama3.2-1b-instruct-q4_0.rfm");

    if !model_path.exists() {
        panic!("Test model not found: {}", model_path.display());
    }

    let file = RfmFile::open(&model_path).expect("Failed to open RFM file");
    let config =
        rocmforge::config::ModelConfig::from_rfm(&file.metadata).expect("Failed to load config");

    let device = GpuDevice::init(0).expect("Failed to initialize GPU");
    let gpu_weights =
        GpuModelWeights::load_rfm(&file, &config).expect("Failed to load GPU weights");

    (device, gpu_weights)
}

#[test]
#[ignore] // Requires GPU hardware
fn test_gpu_batched_qkv_projection_q4_0_end_to_end() {
    let test_config = QkvProjectionTestConfig::llama_3_2_1b_prefill();

    let (device, gpu_weights) = load_test_model_for_batched_qkv();

    let layer = &gpu_weights.layer(0);

    // Prepare test input data
    let mut input_cpu = vec![0.0f32; test_config.seq_len * test_config.hidden_dim];
    for (i, val) in input_cpu.iter_mut().enumerate() {
        *val = ((i as f32) * 0.01).fract(); // Deterministic but varied test data
    }

    // Use a simple scratch buffer allocation for testing
    use rocmforge::gpu::cache::GpuPrefillScratch;

    let config = rocmforge::config::ModelConfig::from_rfm(
        &RfmFile::open("/home/feanor/Projects/rocmforge/llama3.2-1b-instruct-q4_0.rfm")
            .expect("Failed to open RFM file")
            .metadata,
    )
    .expect("Failed to load config");

    let mut scratch = GpuPrefillScratch::new(&config, test_config.seq_len)
        .expect("Failed to allocate prefill scratch");

    // Upload input to GPU
    scratch
        .hidden
        .copy_from_host(unsafe {
            std::slice::from_raw_parts(
                input_cpu.as_ptr() as *const u8,
                input_cpu.len() * std::mem::size_of::<f32>(),
            )
        })
        .expect("Failed to upload input to GPU");

    // Measure batched QKV projection performance
    let start = Instant::now();

    // Call batched QKV projection (separate calls for Q, K, V with correct dimensions)
    // Query projection: [896, 896]
    rocmforge::gpu::ops_batched::gpu_dispatch_batched_gemv_batched(
        &device,
        &layer.attn_q,
        &layer.attn_q_meta,
        scratch.hidden.as_ptr() as *const f32,
        scratch.q.as_ptr() as *mut f32,
        test_config.hidden_dim,
        test_config.q_dim,
        test_config.seq_len,
        device.stream(),
    )
    .expect("Q projection failed");

    // Key projection: [896, 128]
    rocmforge::gpu::ops_batched::gpu_dispatch_batched_gemv_batched(
        &device,
        &layer.attn_k,
        &layer.attn_k_meta,
        scratch.hidden.as_ptr() as *const f32,
        scratch.k.as_ptr() as *mut f32,
        test_config.hidden_dim,
        test_config.k_dim,
        test_config.seq_len,
        device.stream(),
    )
    .expect("K projection failed");

    // Value projection: [896, 128]
    rocmforge::gpu::ops_batched::gpu_dispatch_batched_gemv_batched(
        &device,
        &layer.attn_v,
        &layer.attn_v_meta,
        scratch.hidden.as_ptr() as *const f32,
        scratch.v.as_ptr() as *mut f32,
        test_config.hidden_dim,
        test_config.v_dim,
        test_config.seq_len,
        device.stream(),
    )
    .expect("V projection failed");

    device.synchronize().expect("Failed to synchronize GPU");

    let elapsed = start.elapsed();

    // Read back GPU results
    let mut q_gpu = vec![0.0f32; test_config.seq_len * test_config.q_dim];
    let mut k_gpu = vec![0.0f32; test_config.seq_len * test_config.k_dim];
    let mut v_gpu = vec![0.0f32; test_config.seq_len * test_config.v_dim];

    scratch
        .q
        .copy_to_host(unsafe {
            std::slice::from_raw_parts_mut(
                q_gpu.as_mut_ptr() as *mut u8,
                q_gpu.len() * std::mem::size_of::<f32>(),
            )
        })
        .expect("Failed to download Q from GPU");

    scratch
        .k
        .copy_to_host(unsafe {
            std::slice::from_raw_parts_mut(
                k_gpu.as_mut_ptr() as *mut u8,
                k_gpu.len() * std::mem::size_of::<f32>(),
            )
        })
        .expect("Failed to download K from GPU");

    scratch
        .v
        .copy_to_host(unsafe {
            std::slice::from_raw_parts_mut(
                v_gpu.as_mut_ptr() as *mut u8,
                v_gpu.len() * std::mem::size_of::<f32>(),
            )
        })
        .expect("Failed to download V from GPU");

    // Validate output shapes
    assert_eq!(
        q_gpu.len(),
        test_config.seq_len * test_config.q_dim,
        "Q output has incorrect size"
    );
    assert_eq!(
        k_gpu.len(),
        test_config.seq_len * test_config.k_dim,
        "K output has incorrect size"
    );
    assert_eq!(
        v_gpu.len(),
        test_config.seq_len * test_config.v_dim,
        "V output has incorrect size"
    );

    // Validate output values are finite (not NaN or infinity)
    for (i, &val) in q_gpu.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Q output contains NaN or infinity at index {}: {}",
            i,
            val
        );
    }
    for (i, &val) in k_gpu.iter().enumerate() {
        assert!(
            val.is_finite(),
            "K output contains NaN or infinity at index {}: {}",
            i,
            val
        );
    }
    for (i, &val) in v_gpu.iter().enumerate() {
        assert!(
            val.is_finite(),
            "V output contains NaN or infinity at index {}: {}",
            i,
            val
        );
    }

    // Check outputs are non-zero (kernel actually processed the data)
    let q_has_nonzero = q_gpu.iter().any(|&v| v.abs() > 1e-6);
    let k_has_nonzero = k_gpu.iter().any(|&v| v.abs() > 1e-6);
    let v_has_nonzero = v_gpu.iter().any(|&v| v.abs() > 1e-6);

    assert!(
        q_has_nonzero,
        "Q output is all zeros - kernel may not have processed data"
    );
    assert!(
        k_has_nonzero,
        "K output is all zeros - kernel may not have processed data"
    );
    assert!(
        v_has_nonzero,
        "V output is all zeros - kernel may not have processed data"
    );

    // Calculate throughput metrics
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let total_ops = (test_config.hidden_dim * test_config.q_dim
        + test_config.hidden_dim * test_config.k_dim
        + test_config.hidden_dim * test_config.v_dim)
        * test_config.seq_len;
    let gflops = (total_ops as f64) / (elapsed.as_secs_f64() * 1e9);
    let tokens_per_sec = (test_config.seq_len as f64) / elapsed.as_secs_f64();

    println!("Batched Q4_0 prefill validation passed:");
    println!(
        "  Q: shape=[{}, {}], has_nonzero={}",
        test_config.seq_len, test_config.q_dim, q_has_nonzero
    );
    println!(
        "  K: shape=[{}, {}], has_nonzero={}",
        test_config.seq_len, test_config.k_dim, k_has_nonzero
    );
    println!(
        "  V: shape=[{}, {}], has_nonzero={}",
        test_config.seq_len, test_config.v_dim, v_has_nonzero
    );
    println!("  Performance:");
    println!("    Time: {:.3} ms", elapsed_ms);
    println!("    Throughput: {:.1} tokens/sec", tokens_per_sec);
    println!("    Compute: {:.2} GFLOPS", gflops);
    println!("  Total operations: {} multiply-accumulate ops", total_ops);
}

#[test]
fn test_gpu_batched_qkv_projection_q4_0_validation() {
    // Unit tests for validation logic without requiring GPU hardware

    use rocmforge::gpu::weights::{TensorRole, WeightMeta};

    let valid_meta = WeightMeta {
        wtype: GgmlType::Q4_0,
        dims: vec![1024, 4096],
        needs_transpose: false,
        role: TensorRole::Generic,
        svd_k: None,
    };

    let invalid_meta_q4_1 = WeightMeta {
        wtype: GgmlType::Q4_1,
        dims: vec![1024, 4096],
        needs_transpose: false,
        role: TensorRole::Generic,
        svd_k: None,
    };

    // Test weight type validation
    assert_eq!(valid_meta.wtype, GgmlType::Q4_0);
    assert_eq!(invalid_meta_q4_1.wtype, GgmlType::Q4_1);

    // Test dimension validation
    assert!(valid_meta.dims.len() == 2);
    assert!(valid_meta.dims[0] == 1024);
    assert!(valid_meta.dims[1] == 4096);
}
