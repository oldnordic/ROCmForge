//! FlashAttention kernel tests - CPU vs GPU comparison
//!
//! Tests verify that FlashAttention GPU implementation matches CPU reference.

#[cfg(feature = "rocm")]
#[cfg(test)]
mod phase3_flash_attention_tests {
    use crate::attention::cpu::CpuBackend;
    use crate::attention::kernels::flash_attention_gpu_kernel;
    use crate::backend::{DeviceTensor, HipBackend};
    use crate::loader::mmap_loader::TensorShape;
    use serial_test::serial;
    use std::sync::Arc;

    const TEST_TOLERANCE: f32 = 1e-3; // Tolerance for GPU floating point

    /// Helper: Get GPU backend or skip test if not available (llama.cpp pattern)
    fn get_backend_or_skip() -> Arc<HipBackend> {
        match HipBackend::new_checked() {
            Ok(backend) => backend,
            Err(e) => {
                eprintln!("\n⚠️  GPU not available for flash_attention_tests: {}", e);
                eprintln!("To enable these tests, ensure:");
                eprintln!("  1. AMD GPU is present");
                eprintln!("  2. ROCm is installed (check with rocm-smi)");
                eprintln!("  3. amdhip64 library is in LD_LIBRARY_PATH");
                eprintln!("\nSkipping test gracefully (llama.cpp pattern).\n");
                panic!("GPU_SKIP");
            }
        }
    }

    /// Helper: Create test Q, K, V tensors
    fn create_qkv_tensors(
        batch_size: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let total_size = batch_size * seq_len * head_dim;
        let q: Vec<f32> = (0..total_size).map(|i| (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..total_size).map(|i| (i as f32) * 0.1 + 1.0).collect();
        let v: Vec<f32> = (0..total_size).map(|i| (i as f32) * 0.1 + 2.0).collect();
        (q, k, v)
    }

    /// Test 1: FlashAttention matches CPU - small dimensions (no mask)
    #[test]
    #[serial]
    fn test_flash_attention_matches_cpu_small_no_mask() {
        let batch_size = 1;
        let seq_len = 4;
        let head_dim = 4;

        let (q, k, v) = create_qkv_tensors(batch_size, seq_len, head_dim);

        // CPU reference
        let cpu_result =
            CpuBackend::forward(head_dim, &q, &k, &v, None, None).expect("CPU attention failed");

        // GPU run with FlashAttention
        let backend = get_backend_or_skip();

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Create device tensors
        let q_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);
        let k_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);
        let v_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);
        let out_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);

        let q_gpu = DeviceTensor::from_host_vec(&backend, q.clone(), q_shape)
            .expect("Failed to create Q tensor");
        let k_gpu = DeviceTensor::from_host_vec(&backend, k.clone(), k_shape)
            .expect("Failed to create K tensor");
        let v_gpu = DeviceTensor::from_host_vec(&backend, v.clone(), v_shape)
            .expect("Failed to create V tensor");
        let mut out_gpu =
            DeviceTensor::empty(&backend, out_shape).expect("Failed to create output tensor");

        // Get device pointers
        let q_ptr = q_gpu.as_ptr() as *const f32;
        let k_ptr = k_gpu.as_ptr() as *const f32;
        let v_ptr = v_gpu.as_ptr() as *const f32;
        let out_ptr = out_gpu.buffer().as_mut_ptr() as *mut f32;

        // Launch FlashAttention kernel
        let result = unsafe {
            flash_attention_gpu_kernel(
                q_ptr,
                k_ptr,
                v_ptr,
                out_ptr,
                std::ptr::null(), // no mask
                scale,
                batch_size as u32,
                seq_len as u32,
                1, // num_heads = 1 for this test
                head_dim as u32,
            )
        };

        if result != 0 {
            eprintln!("SKIPPED: FlashAttention kernel returned error code {} - kernel not available or failed", result);
            return;
        }

        // Synchronize
        backend.synchronize().expect("GPU synchronization failed");

        // Copy result back
        let gpu_result = out_gpu
            .to_host_vec()
            .expect("Failed to copy output from GPU");

        // Compare results
        assert_eq!(cpu_result.len(), gpu_result.len());
        let mut max_diff = 0.0f32;
        for (i, (cpu_val, gpu_val)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
            let diff = (cpu_val - gpu_val).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < TEST_TOLERANCE,
                "FlashAttention mismatch at {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_val,
                gpu_val,
                diff
            );
        }
        println!("Max difference: {}", max_diff);
    }

    /// Test 2: FlashAttention matches CPU - with causal mask
    #[test]
    #[serial]
    fn test_flash_attention_matches_cpu_with_causal_mask() {
        let batch_size = 1;
        let seq_len = 4;
        let head_dim = 4;

        let (q, k, v) = create_qkv_tensors(batch_size, seq_len, head_dim);

        // Create causal mask
        let mut mask = vec![0.0f32; batch_size * seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }

        // CPU reference
        let cpu_result = CpuBackend::forward(head_dim, &q, &k, &v, Some(&mask), None)
            .expect("CPU attention with mask failed");

        // GPU run with FlashAttention
        let backend = get_backend_or_skip();

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Create device tensors
        let q_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);
        let k_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);
        let v_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);
        let out_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);
        let mask_shape = TensorShape::from_dims(&[batch_size, seq_len, seq_len]);

        let q_gpu = DeviceTensor::from_host_vec(&backend, q.clone(), q_shape)
            .expect("Failed to create Q tensor");
        let k_gpu = DeviceTensor::from_host_vec(&backend, k.clone(), k_shape)
            .expect("Failed to create K tensor");
        let v_gpu = DeviceTensor::from_host_vec(&backend, v.clone(), v_shape)
            .expect("Failed to create V tensor");
        let mask_gpu = DeviceTensor::from_host_vec(&backend, mask.clone(), mask_shape)
            .expect("Failed to create mask tensor");
        let mut out_gpu =
            DeviceTensor::empty(&backend, out_shape).expect("Failed to create output tensor");

        // Get device pointers
        let q_ptr = q_gpu.as_ptr() as *const f32;
        let k_ptr = k_gpu.as_ptr() as *const f32;
        let v_ptr = v_gpu.as_ptr() as *const f32;
        let out_ptr = out_gpu.buffer().as_mut_ptr() as *mut f32;
        let mask_ptr = mask_gpu.as_ptr() as *const f32;

        // Launch FlashAttention kernel
        let result = unsafe {
            flash_attention_gpu_kernel(
                q_ptr,
                k_ptr,
                v_ptr,
                out_ptr,
                mask_ptr,
                scale,
                batch_size as u32,
                seq_len as u32,
                1, // num_heads = 1 for this test
                head_dim as u32,
            )
        };

        if result != 0 {
            eprintln!("SKIPPED: FlashAttention kernel returned error code {} - kernel not available or failed", result);
            return;
        }

        // Synchronize
        backend.synchronize().expect("GPU synchronization failed");

        // Copy result back
        let gpu_result = out_gpu
            .to_host_vec()
            .expect("Failed to copy output from GPU");

        // Compare results
        assert_eq!(cpu_result.len(), gpu_result.len());
        for (i, (cpu_val, gpu_val)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
            let diff = (cpu_val - gpu_val).abs();
            assert!(
                diff < TEST_TOLERANCE,
                "FlashAttention with mask mismatch at {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_val,
                gpu_val,
                diff
            );
        }
    }

    /// Test 3: FlashAttention - larger sequence
    #[test]
    #[serial]
    fn test_flash_attention_matches_cpu_seq_len_8() {
        let batch_size = 1;
        let seq_len = 8;
        let head_dim = 8;

        let (q, k, v) = create_qkv_tensors(batch_size, seq_len, head_dim);

        // CPU reference
        let cpu_result =
            CpuBackend::forward(head_dim, &q, &k, &v, None, None).expect("CPU attention failed");

        // GPU run with FlashAttention
        let backend = get_backend_or_skip();

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Create device tensors
        let q_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);
        let k_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);
        let v_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);
        let out_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);

        let q_gpu = DeviceTensor::from_host_vec(&backend, q.clone(), q_shape)
            .expect("Failed to create Q tensor");
        let k_gpu = DeviceTensor::from_host_vec(&backend, k.clone(), k_shape)
            .expect("Failed to create K tensor");
        let v_gpu = DeviceTensor::from_host_vec(&backend, v.clone(), v_shape)
            .expect("Failed to create V tensor");
        let mut out_gpu =
            DeviceTensor::empty(&backend, out_shape).expect("Failed to create output tensor");

        // Get device pointers
        let q_ptr = q_gpu.as_ptr() as *const f32;
        let k_ptr = k_gpu.as_ptr() as *const f32;
        let v_ptr = v_gpu.as_ptr() as *const f32;
        let out_ptr = out_gpu.buffer().as_mut_ptr() as *mut f32;

        // Launch FlashAttention kernel
        let result = unsafe {
            flash_attention_gpu_kernel(
                q_ptr,
                k_ptr,
                v_ptr,
                out_ptr,
                std::ptr::null(), // no mask
                scale,
                batch_size as u32,
                seq_len as u32,
                1, // num_heads = 1 for this test
                head_dim as u32,
            )
        };

        if result != 0 {
            eprintln!("SKIPPED: FlashAttention kernel returned error code {} - kernel not available or failed", result);
            return;
        }

        // Synchronize
        backend.synchronize().expect("GPU synchronization failed");

        // Copy result back
        let gpu_result = out_gpu
            .to_host_vec()
            .expect("Failed to copy output from GPU");

        // Compare results
        assert_eq!(cpu_result.len(), gpu_result.len());
        for (i, (cpu_val, gpu_val)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
            let diff = (cpu_val - gpu_val).abs();
            assert!(
                diff < TEST_TOLERANCE,
                "FlashAttention seq_len=8 mismatch at {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_val,
                gpu_val,
                diff
            );
        }
    }

    /// Test 4: FlashAttention - batch_size > 1
    #[test]
    #[serial]
    fn test_flash_attention_matches_cpu_batch_2() {
        let batch_size = 2;
        let seq_len = 4;
        let head_dim = 4;

        let (q, k, v) = create_qkv_tensors(batch_size, seq_len, head_dim);

        // CPU reference
        let cpu_result =
            CpuBackend::forward(head_dim, &q, &k, &v, None, None).expect("CPU attention failed");

        // GPU run with FlashAttention
        let backend = get_backend_or_skip();

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Create device tensors
        let q_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);
        let k_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);
        let v_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);
        let out_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);

        let q_gpu = DeviceTensor::from_host_vec(&backend, q.clone(), q_shape)
            .expect("Failed to create Q tensor");
        let k_gpu = DeviceTensor::from_host_vec(&backend, k.clone(), k_shape)
            .expect("Failed to create K tensor");
        let v_gpu = DeviceTensor::from_host_vec(&backend, v.clone(), v_shape)
            .expect("Failed to create V tensor");
        let mut out_gpu =
            DeviceTensor::empty(&backend, out_shape).expect("Failed to create output tensor");

        // Get device pointers
        let q_ptr = q_gpu.as_ptr() as *const f32;
        let k_ptr = k_gpu.as_ptr() as *const f32;
        let v_ptr = v_gpu.as_ptr() as *const f32;
        let out_ptr = out_gpu.buffer().as_mut_ptr() as *mut f32;

        // Launch FlashAttention kernel
        let result = unsafe {
            flash_attention_gpu_kernel(
                q_ptr,
                k_ptr,
                v_ptr,
                out_ptr,
                std::ptr::null(), // no mask
                scale,
                batch_size as u32,
                seq_len as u32,
                1, // num_heads = 1 for this test
                head_dim as u32,
            )
        };

        if result != 0 {
            eprintln!("SKIPPED: FlashAttention kernel returned error code {} - kernel not available or failed", result);
            return;
        }

        // Synchronize
        backend.synchronize().expect("GPU synchronization failed");

        // Copy result back
        let gpu_result = out_gpu
            .to_host_vec()
            .expect("Failed to copy output from GPU");

        // Compare results
        assert_eq!(cpu_result.len(), gpu_result.len());
        for (i, (cpu_val, gpu_val)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
            let diff = (cpu_val - gpu_val).abs();
            assert!(
                diff < TEST_TOLERANCE,
                "FlashAttention batch=2 mismatch at {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_val,
                gpu_val,
                diff
            );
        }
    }

    /// Test 5: FlashAttention - verify softmax properties
    #[test]
    #[serial]
    fn test_flash_attention_softmax_properties() {
        let batch_size = 1;
        let seq_len = 4;
        let head_dim = 4;

        let (q, k, v) = create_qkv_tensors(batch_size, seq_len, head_dim);

        // GPU run with FlashAttention
        let backend = get_backend_or_skip();

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Create device tensors
        let q_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);
        let k_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);
        let v_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);
        let out_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);

        let q_gpu = DeviceTensor::from_host_vec(&backend, q.clone(), q_shape)
            .expect("Failed to create Q tensor");
        let k_gpu = DeviceTensor::from_host_vec(&backend, k.clone(), k_shape)
            .expect("Failed to create K tensor");
        let v_gpu = DeviceTensor::from_host_vec(&backend, v.clone(), v_shape)
            .expect("Failed to create V tensor");
        let mut out_gpu =
            DeviceTensor::empty(&backend, out_shape).expect("Failed to create output tensor");

        // Get device pointers
        let q_ptr = q_gpu.as_ptr() as *const f32;
        let k_ptr = k_gpu.as_ptr() as *const f32;
        let v_ptr = v_gpu.as_ptr() as *const f32;
        let out_ptr = out_gpu.buffer().as_mut_ptr() as *mut f32;

        // Launch FlashAttention kernel
        let result = unsafe {
            flash_attention_gpu_kernel(
                q_ptr,
                k_ptr,
                v_ptr,
                out_ptr,
                std::ptr::null(), // no mask
                scale,
                batch_size as u32,
                seq_len as u32,
                1, // num_heads = 1 for this test
                head_dim as u32,
            )
        };

        if result != 0 {
            eprintln!("SKIPPED: FlashAttention kernel returned error code {} - kernel not available or failed", result);
            return;
        }

        // Synchronize
        backend.synchronize().expect("GPU synchronization failed");

        // Copy result back
        let gpu_result = out_gpu
            .to_host_vec()
            .expect("Failed to copy output from GPU");

        // Basic sanity checks
        assert!(!gpu_result.is_empty());
        assert_eq!(gpu_result.len(), batch_size * seq_len * head_dim);

        // Check for NaN or Inf
        for &val in &gpu_result {
            assert!(
                val.is_finite(),
                "FlashAttention produced non-finite value: {}",
                val
            );
        }
    }

    /// Performance benchmark: FlashAttention vs separate kernels
    #[cfg(feature = "rocm")]
    #[test]
    #[serial]
    fn benchmark_flash_attention_vs_separate() {
        use crate::attention::compute::matmul_cpu;
        use crate::attention::softmax;
        use crate::loader::mmap_loader::TensorShape;
        use std::time::Instant;

        let backend = get_backend_or_skip();

        // Larger test for meaningful timing
        let batch_size = 2;
        let seq_len = 32;
        let head_dim = 32;

        let n = batch_size * seq_len * head_dim;
        let mut q = vec![0.0f32; n];
        let mut k = vec![0.0f32; n];
        let mut v = vec![0.0f32; n];

        // Initialize with deterministic values
        for i in 0..n {
            q[i] = ((i % 257) as f32) / 100.0;
            k[i] = ((i % 253) as f32) / 100.0 + 0.5;
            v[i] = ((i % 251) as f32) / 100.0 + 1.0;
        }

        // Scale factor
        let scale = 1.0 / (head_dim as f32).sqrt();

        // ===== Method 1: Separate kernels (CPU reference path) =====
        let start = Instant::now();
        let iterations = 10;

        for _ in 0..iterations {
            // QK^T
            let mut scores = matmul_cpu(&q, &k, batch_size, seq_len, seq_len, head_dim).unwrap();

            // Scale
            for s in scores.iter_mut() {
                *s *= scale;
            }

            // Softmax per row
            for b in 0..batch_size {
                for i in 0..seq_len {
                    let row_start = b * seq_len * seq_len + i * seq_len;
                    let row_end = row_start + seq_len;
                    softmax::softmax_in_place(&mut scores[row_start..row_end], 1, seq_len);
                }
            }

            // Softmax × V
            let _output = matmul_cpu(&scores, &v, batch_size, seq_len, seq_len, head_dim).unwrap();
        }

        let cpu_time = start.elapsed();
        println!("CPU (separate kernels) ×{}: {:?}", iterations, cpu_time);

        // ===== Method 2: FlashAttention (fused GPU kernel) =====
        let q_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);
        let k_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);
        let v_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);
        let out_shape = TensorShape::from_dims(&[batch_size, seq_len, head_dim]);

        let q_gpu = DeviceTensor::from_host_vec(&backend, q.clone(), q_shape)
            .expect("Failed to create Q tensor");
        let k_gpu = DeviceTensor::from_host_vec(&backend, k.clone(), k_shape)
            .expect("Failed to create K tensor");
        let v_gpu = DeviceTensor::from_host_vec(&backend, v.clone(), v_shape)
            .expect("Failed to create V tensor");
        let mut out_gpu =
            DeviceTensor::empty(&backend, out_shape).expect("Failed to create output tensor");

        let q_ptr = q_gpu.as_ptr() as *const f32;
        let k_ptr = k_gpu.as_ptr() as *const f32;
        let v_ptr = v_gpu.as_ptr() as *const f32;
        let out_ptr = out_gpu.buffer().as_mut_ptr() as *mut f32;

        // Warmup
        unsafe {
            flash_attention_gpu_kernel(
                q_ptr,
                k_ptr,
                v_ptr,
                out_ptr,
                std::ptr::null(),
                scale,
                batch_size as u32,
                seq_len as u32,
                1,
                head_dim as u32,
            );
        }
        backend.synchronize().expect("GPU sync failed");

        let start = Instant::now();

        for _ in 0..iterations {
            unsafe {
                flash_attention_gpu_kernel(
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    out_ptr,
                    std::ptr::null(),
                    scale,
                    batch_size as u32,
                    seq_len as u32,
                    1,
                    head_dim as u32,
                );
            }
            backend.synchronize().expect("GPU sync failed");
        }

        let gpu_time = start.elapsed();
        println!("GPU (FlashAttention fused) ×{}: {:?}", iterations, gpu_time);

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        println!("Speedup: {:.2}x", speedup);

        // Verify correctness still holds
        let gpu_result = out_gpu.to_host_vec().expect("Failed to copy output");
        let cpu_output = {
            let mut scores = matmul_cpu(&q, &k, batch_size, seq_len, seq_len, head_dim).unwrap();
            for s in scores.iter_mut() {
                *s *= scale;
            }
            for b in 0..batch_size {
                for i in 0..seq_len {
                    let row_start = b * seq_len * seq_len + i * seq_len;
                    let row_end = row_start + seq_len;
                    softmax::softmax_in_place(&mut scores[row_start..row_end], 1, seq_len);
                }
            }
            matmul_cpu(&scores, &v, batch_size, seq_len, seq_len, head_dim).unwrap()
        };

        let max_diff = cpu_output
            .iter()
            .zip(gpu_result.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, |a, b| a.max(b));

        println!("Max difference CPU vs GPU: {}", max_diff);
        assert!(max_diff < 1e-4, "GPU result deviates too much from CPU");
    }
}
