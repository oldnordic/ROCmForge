// Tests for GPU causal mask implementation
// Test-Driven Development approach:
// 1. Write tests first
// 2. Verify they fail (proving tests work)
// 3. Implement the feature
// 4. Verify tests pass

// Note: Types are already in scope from parent module

// Use shared GPU fixture to avoid creating multiple backends (prevents GPU resets)
#[cfg(test)]
use crate::backend::gpu_test_common::GPU_FIXTURE;
#[cfg(test)]
use serial_test::serial;

/// Test helper: Create CPU causal mask reference
fn create_cpu_causal_mask(seq_len: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    mask
}

/// Test 1: Correct mask application (upper triangle set to -inf)
#[cfg(feature = "rocm")]
#[test]
#[serial]
fn test_gpu_causal_mask_upper_triangle() {
    use crate::ops::attention_gpu::HipAttentionKernels;

    // Test parameters
    let seq_len = 8;
    let num_heads = 4;
    let batch_size = 2;

    // Use shared GPU fixture to avoid creating multiple backends (prevents GPU resets)
    let fixture = match GPU_FIXTURE.as_ref() {
        Some(f) => f,
        None => {
            println!("Warning: No GPU available, skipping GPU causal mask test");
            return;
        }
    };
    let backend = fixture.backend();

    // Create attention tensor [batch, num_heads, seq_len, seq_len]
    let mut attention_host = vec![1.0f32; batch_size * num_heads * seq_len * seq_len];
    let attention_shape = TensorShape::from_dims(&[batch_size, num_heads, seq_len, seq_len]);
    let mut attention = DeviceTensor::from_host_vec(&backend, attention_host, attention_shape)
        .expect("Failed to create attention tensor");

    // Apply causal mask
    let kernels = HipAttentionKernels::new(&backend).expect("Failed to create kernels");
    kernels
        .apply_causal_mask(&mut attention, seq_len, seq_len)
        .expect("Failed to apply causal mask");

    // Verify result
    let result = attention.to_host_vec().expect("Failed to copy result");
    let cpu_reference = create_cpu_causal_mask(seq_len);

    // Check each batch and head
    for batch in 0..batch_size {
        for head in 0..num_heads {
            let base_offset = batch * num_heads * seq_len * seq_len + head * seq_len * seq_len;

            for i in 0..seq_len {
                for j in 0..seq_len {
                    let idx = base_offset + i * seq_len + j;
                    let expected = cpu_reference[i * seq_len + j];

                    if j > i {
                        // Upper triangle should be -inf
                        assert!(
                            result[idx].is_finite() == false || result[idx] < -1e9,
                            "Upper triangle at batch={}, head={}, [{}, {}] should be -inf, got {}",
                            batch, head, i, j, result[idx]
                        );
                    } else {
                        // Lower triangle should be unchanged (1.0)
                        assert!(
                            (result[idx] - 1.0).abs() < 1e-5,
                            "Lower triangle at batch={}, head={}, [{}, {}] should be 1.0, got {}",
                            batch, head, i, j, result[idx]
                        );
                    }
                }
            }
        }
    }
}

/// Test 2: Lower triangle preserved
#[cfg(feature = "rocm")]
#[test]
#[serial]
fn test_gpu_causal_mask_lower_triangle_preserved() {
    use crate::ops::attention_gpu::HipAttentionKernels;

    let seq_len = 16;
    let num_heads = 8;
    let batch_size = 1;

    // Use shared GPU fixture to avoid creating multiple backends (prevents GPU resets)
    let fixture = match GPU_FIXTURE.as_ref() {
        Some(f) => f,
        None => {
            println!("Warning: No GPU available, skipping GPU causal mask test");
            return;
        }
    };
    let backend = fixture.backend();

    // Create attention with varying values
    let mut attention_host = Vec::with_capacity(batch_size * num_heads * seq_len * seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            attention_host.push((i * seq_len + j) as f32);
        }
    }
    // Repeat for all heads and batches
    let base_values = attention_host.clone();
    for _ in 1..num_heads * batch_size {
        attention_host.extend(&base_values);
    }

    let attention_shape = TensorShape::from_dims(&[batch_size, num_heads, seq_len, seq_len]);
    let mut attention = DeviceTensor::from_host_vec(&backend, attention_host, attention_shape)
        .expect("Failed to create attention tensor");

    // Apply causal mask
    let kernels = HipAttentionKernels::new(&backend).expect("Failed to create kernels");
    kernels
        .apply_causal_mask(&mut attention, seq_len, seq_len)
        .expect("Failed to apply causal mask");

    // Verify lower triangle preserved
    let result = attention.to_host_vec().expect("Failed to copy result");

    for head in 0..num_heads {
        let base_offset = head * seq_len * seq_len;

        for i in 0..seq_len {
            for j in 0..=i {
                let idx = base_offset + i * seq_len + j;
                let expected = (i * seq_len + j) as f32;

                assert!(
                    (result[idx] - expected).abs() < 1e-5,
                    "Lower triangle at head={}, [{}, {}] should be preserved as {}, got {}",
                    head, i, j, expected, result[idx]
                );
            }
        }
    }
}

/// Test 3: Batch dimension handled correctly
#[cfg(feature = "rocm")]
#[test]
#[serial]
fn test_gpu_causal_mask_batch_dimension() {
    use crate::ops::attention_gpu::HipAttentionKernels;

    let seq_len = 8;
    let num_heads = 4;
    let batch_sizes = vec![1, 2, 4, 8];

    // Use shared GPU fixture to avoid creating multiple backends (prevents GPU resets)
    let fixture = match GPU_FIXTURE.as_ref() {
        Some(f) => f,
        None => {
            println!("Warning: No GPU available, skipping GPU causal mask test");
            return;
        }
    };
    let backend = fixture.backend();

    for batch_size in batch_sizes {
        let attention_host = vec![1.0f32; batch_size * num_heads * seq_len * seq_len];
        let attention_shape = TensorShape::from_dims(&[batch_size, num_heads, seq_len, seq_len]);
        let mut attention = DeviceTensor::from_host_vec(&backend, attention_host, attention_shape)
            .expect("Failed to create attention tensor");

        let kernels = HipAttentionKernels::new(&backend).expect("Failed to create kernels");
        kernels
            .apply_causal_mask(&mut attention, seq_len, seq_len)
            .expect("Failed to apply causal mask");

        let result = attention.to_host_vec().expect("Failed to copy result");

        // Verify all batches processed correctly
        for batch in 0..batch_size {
            for head in 0..num_heads {
                let base_offset = batch * num_heads * seq_len * seq_len + head * seq_len * seq_len;

                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let idx = base_offset + i * seq_len + j;

                        if j > i {
                            assert!(
                                result[idx].is_finite() == false || result[idx] < -1e9,
                                "Batch {}: Upper triangle at head={}, [{}, {}] should be -inf",
                                batch, head, i, j
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Test 4: Multiple heads handled correctly
#[cfg(feature = "rocm")]
#[test]
#[serial]
fn test_gpu_causal_mask_multiple_heads() {
    use crate::ops::attention_gpu::HipAttentionKernels;

    let seq_len = 8;
    let head_counts = vec![1, 2, 4, 8, 16, 32];
    let batch_size = 2;

    // Use shared GPU fixture to avoid creating multiple backends (prevents GPU resets)
    let fixture = match GPU_FIXTURE.as_ref() {
        Some(f) => f,
        None => {
            println!("Warning: No GPU available, skipping GPU causal mask test");
            return;
        }
    };
    let backend = fixture.backend();

    for num_heads in head_counts {
        let attention_host = vec![1.0f32; batch_size * num_heads * seq_len * seq_len];
        let attention_shape = TensorShape::from_dims(&[batch_size, num_heads, seq_len, seq_len]);
        let mut attention = DeviceTensor::from_host_vec(&backend, attention_host, attention_shape)
            .expect("Failed to create attention tensor");

        let kernels = HipAttentionKernels::new(&backend).expect("Failed to create kernels");
        kernels
            .apply_causal_mask(&mut attention, seq_len, seq_len)
            .expect("Failed to apply causal mask");

        let result = attention.to_host_vec().expect("Failed to copy result");

        // Verify all heads processed correctly
        for batch in 0..batch_size {
            for head in 0..num_heads {
                let base_offset = batch * num_heads * seq_len * seq_len + head * seq_len * seq_len;

                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let idx = base_offset + i * seq_len + j;

                        if j > i {
                            assert!(
                                result[idx].is_finite() == false || result[idx] < -1e9,
                                "Head {} of {}: Upper triangle at [{}, {}] should be -inf",
                                head, num_heads, i, j
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Test 5: Comparison with CPU implementation (accuracy)
#[cfg(feature = "rocm")]
#[test]
#[serial]
fn test_gpu_causal_mask_matches_cpu() {
    use crate::ops::attention_gpu::HipAttentionKernels;

    let seq_len = 32;
    let num_heads = 8;
    let batch_size = 4;

    // Use shared GPU fixture to avoid creating multiple backends (prevents GPU resets)
    let fixture = match GPU_FIXTURE.as_ref() {
        Some(f) => f,
        None => {
            println!("Warning: No GPU available, skipping GPU causal mask test");
            return;
        }
    };
    let backend = fixture.backend();

    // Create CPU reference
    let cpu_mask = create_cpu_causal_mask(seq_len);

    // Apply GPU mask
    let attention_host = vec![1.0f32; batch_size * num_heads * seq_len * seq_len];
    let attention_shape = TensorShape::from_dims(&[batch_size, num_heads, seq_len, seq_len]);
    let mut attention = DeviceTensor::from_host_vec(&backend, attention_host, attention_shape)
        .expect("Failed to create attention tensor");

    let kernels = HipAttentionKernels::new(&backend).expect("Failed to create kernels");
    kernels
        .apply_causal_mask(&mut attention, seq_len, seq_len)
        .expect("Failed to apply causal mask");

    let gpu_result = attention.to_host_vec().expect("Failed to copy result");

    // Compare GPU result with CPU reference
    for batch in 0..batch_size {
        for head in 0..num_heads {
            let base_offset = batch * num_heads * seq_len * seq_len + head * seq_len * seq_len;

            for i in 0..seq_len {
                for j in 0..seq_len {
                    let gpu_idx = base_offset + i * seq_len + j;
                    let cpu_idx = i * seq_len + j;

                    let gpu_val = gpu_result[gpu_idx];
                    let cpu_val = cpu_mask[cpu_idx];

                    // For non-masked positions, GPU should preserve original value (1.0)
                    // For masked positions, both should be -inf
                    if j > i {
                        // Masked position
                        assert!(
                            gpu_val.is_finite() == false || gpu_val < -1e9,
                            "GPU masked value mismatch at batch={}, head={}, [{}, {}]: {}",
                            batch, head, i, j, gpu_val
                        );
                        assert!(
                            cpu_val.is_finite() == false || cpu_val < -1e9,
                            "CPU masked value mismatch at [{}, {}]: {}",
                            i, j, cpu_val
                        );
                    } else {
                        // Non-masked position
                        assert!(
                            (gpu_val - 1.0).abs() < 1e-5,
                            "GPU unmasked value at batch={}, head={}, [{}, {}]: {} (expected 1.0)",
                            batch, head, i, j, gpu_val
                        );
                    }
                }
            }
        }
    }
}

/// Test 6: Edge case - seq_len = 1
#[cfg(feature = "rocm")]
#[test]
#[serial]
fn test_gpu_causal_mask_single_element() {
    use crate::ops::attention_gpu::HipAttentionKernels;

    let seq_len = 1;
    let num_heads = 4;
    let batch_size = 2;

    // Use shared GPU fixture to avoid creating multiple backends (prevents GPU resets)
    let fixture = match GPU_FIXTURE.as_ref() {
        Some(f) => f,
        None => {
            println!("Warning: No GPU available, skipping GPU causal mask test");
            return;
        }
    };
    let backend = fixture.backend();

    let attention_host = vec![1.0f32; batch_size * num_heads * seq_len * seq_len];
    let attention_shape = TensorShape::from_dims(&[batch_size, num_heads, seq_len, seq_len]);
    let mut attention = DeviceTensor::from_host_vec(&backend, attention_host, attention_shape)
        .expect("Failed to create attention tensor");

    let kernels = HipAttentionKernels::new(&backend).expect("Failed to create kernels");
    kernels
        .apply_causal_mask(&mut attention, seq_len, seq_len)
        .expect("Failed to apply causal mask");

    let result = attention.to_host_vec().expect("Failed to copy result");

    // Single element should remain unchanged (not masked)
    for i in 0..result.len() {
        assert!(
            (result[i] - 1.0).abs() < 1e-5,
            "Single element at index {} should be 1.0, got {}",
            i, result[i]
        );
    }
}

/// Test 7: Large sequence performance test
#[cfg(feature = "rocm")]
#[test]
#[serial]
fn test_gpu_causal_mask_large_sequence() {
    use crate::ops::attention_gpu::HipAttentionKernels;
    use std::time::Instant;

    let seq_len = 2048;
    let num_heads = 32;
    let batch_size = 4;

    // Use shared GPU fixture to avoid creating multiple backends (prevents GPU resets)
    let fixture = match GPU_FIXTURE.as_ref() {
        Some(f) => f,
        None => {
            println!("Warning: No GPU available, skipping GPU causal mask test");
            return;
        }
    };
    let backend = fixture.backend();

    let attention_host = vec![1.0f32; batch_size * num_heads * seq_len * seq_len];
    let attention_shape = TensorShape::from_dims(&[batch_size, num_heads, seq_len, seq_len]);
    let mut attention = DeviceTensor::from_host_vec(&backend, attention_host, attention_shape)
        .expect("Failed to create attention tensor");

    let kernels = HipAttentionKernels::new(&backend).expect("Failed to create kernels");

    // Measure execution time
    let start = Instant::now();
    kernels
        .apply_causal_mask(&mut attention, seq_len, seq_len)
        .expect("Failed to apply causal mask");
    let duration = start.elapsed();

    println!("GPU causal mask for seq_len={}, heads={}, batch={} took: {:?}", seq_len, num_heads, batch_size, duration);

    // Verify a few elements
    let result = attention.to_host_vec().expect("Failed to copy result");

    // Check upper triangle is masked
    let batch_idx = 0;
    let head_idx = 0;
    let base_offset = batch_idx * num_heads * seq_len * seq_len + head_idx * seq_len * seq_len;

    // Sample some positions
    for i in 0..seq_len.min(100) {
        for j in (i + 1)..(i + 2).min(seq_len) {
            let idx = base_offset + i * seq_len + j;
            assert!(
                result[idx].is_finite() == false || result[idx] < -1e9,
                "Large seq: Upper triangle at [{}, {}] should be -inf",
                i, j
            );
        }
    }

    // Performance assertion: should complete in reasonable time
    // (This is a soft assertion - adjust based on hardware)
    assert!(
        duration.as_secs_f64() < 1.0,
        "GPU causal mask took too long: {:?} (expected < 1s)",
        duration
    );
}
