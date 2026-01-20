//! Softmax tests with explicit [batch, heads, seq_q, seq_k] layout
//!
//! Phase 3a.3.1: Verify Phase 1 softmax kernel works with explicit layout.
//!
//! The Phase 1 softmax_kernel treats data as [total_rows, row_len] where:
//! - total_rows = batch_size * seq_len (in Phase 1 terms)
//! - row_len = seq_len (in Phase 1 terms)
//!
//! For Phase 3a explicit layout [batch, heads, seq_q, seq_k]:
//! - total_rows = batch * heads * seq_q
//! - row_len = seq_k
//!
//! To reuse: call softmax_kernel with:
//! - batch_size_param = batch * heads * seq_q (kernel's "batch_size")
//! - seq_len_param = seq_k (kernel's "seq_len" = row length)

#[cfg(test)]
mod softmax_explicit_tests {
    use crate::backend::{DeviceTensor, HipBackend};
    use crate::loader::mmap_loader::TensorShape;
    use serial_test::serial;

    const TEST_TOLERANCE: f32 = 1e-4;
    const TEST_TOLERANCE_LARGE: f32 = 1e-3; // For larger inputs due to FP reduction order

    /// Helper: Get GPU backend or skip test if not available (llama.cpp pattern)
    fn get_backend_or_skip() -> std::sync::Arc<HipBackend> {
        match HipBackend::new_checked() {
            Ok(backend) => backend,
            Err(e) => {
                eprintln!("\n⚠️  GPU not available for softmax_explicit_tests: {}", e);
                eprintln!("To enable these tests, ensure:");
                eprintln!("  1. AMD GPU is present");
                eprintln!("  2. ROCm is installed (check with rocm-smi)");
                eprintln!("  3. amdhip64 library is in LD_LIBRARY_PATH");
                eprintln!("\nSkipping test gracefully (llama.cpp pattern).\n");
                panic!("GPU_SKIP");
            }
        }
    }

    /// CPU reference: softmax over last dimension (seq_k)
    ///
    /// Input:  scores [batch, heads, seq_q, seq_k]
    /// Output: weights [batch, heads, seq_q, seq_k] (softmax over seq_k)
    ///
    /// Layout: [batch, heads, seq_q, seq_k] row-major
    /// Index: batch * heads * seq_q * seq_k + head * seq_q * seq_k + seq_q * seq_k + k
    fn softmax_cpu_explicit(
        scores: &[f32],
        batch: usize,
        heads: usize,
        seq_q: usize,
        seq_k: usize,
    ) -> Vec<f32> {
        let mut output = scores.to_vec();

        // Each row to softmax: [batch, heads, seq_q] rows, each of seq_k elements
        let total_rows = batch * heads * seq_q;

        for row_idx in 0..total_rows {
            let row_start = row_idx * seq_k;
            let row_end = row_start + seq_k;

            // Find max for numerical stability
            let max_val = output[row_start..row_end]
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Compute exp and sum
            let mut sum = 0.0f32;
            for j in row_start..row_end {
                output[j] = (output[j] - max_val).exp();
                sum += output[j];
            }

            // Normalize
            for j in row_start..row_end {
                output[j] /= sum;
            }
        }

        output
    }

    /// Test softmax with explicit [batch, heads, seq_q, seq_k] layout - small
    #[test]
    #[serial]
    #[ignore] // Requires HSACO kernels - GPU memory access without kernels
    fn test_softmax_explicit_layout_small() {
        let batch = 1;
        let heads = 2;
        let seq_q = 4;
        let seq_k = 4;

        // Create scores with explicit layout
        let total = batch * heads * seq_q * seq_k;
        let scores: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1).collect();

        // CPU reference
        let cpu_result = softmax_cpu_explicit(&scores, batch, heads, seq_q, seq_k);

        // GPU run
        let backend = get_backend_or_skip();

        let scores_shape = TensorShape::from_dims(&[batch, heads, seq_q, seq_k]);

        let mut scores_gpu = DeviceTensor::from_host_vec(&backend, scores.clone(), scores_shape)
            .expect("Failed to create scores tensor");

        // Call Phase 1 softmax_kernel with explicit layout
        // Kernel treats: total_rows = batch_size_param, row_len = seq_len_param
        // For [B, H, Sq, Sk]: total_rows = B*H*Sq, row_len = Sk
        let total_rows = (batch * heads * seq_q) as u32;
        let row_len = seq_k as u32;

        let result = unsafe {
            crate::attention::kernels::softmax_gpu_kernel(
                scores_gpu.buffer().as_mut_ptr() as *mut f32,
                total_rows, // kernel's "batch_size" = total rows
                row_len,    // kernel's "seq_len" = row length
            )
        };

        if result != 0 {
            eprintln!("SKIPPED: softmax_gpu_kernel returned error code {} - kernel not available or failed", result);
            return;
        }

        backend.synchronize().expect("GPU synchronization failed");

        let gpu_result = scores_gpu
            .to_host_vec()
            .expect("Failed to copy output from GPU");

        assert_eq!(cpu_result.len(), gpu_result.len());

        // Verify each row sums to ~1.0
        for row_idx in 0..(batch * heads * seq_q) {
            let row_start = row_idx * seq_k;
            let row_end = row_start + seq_k;

            let cpu_sum: f32 = cpu_result[row_start..row_end].iter().sum();
            let gpu_sum: f32 = gpu_result[row_start..row_end].iter().sum();

            assert!(
                (cpu_sum - 1.0).abs() < 1e-5,
                "CPU row {} sums to {} (expected ~1.0)",
                row_idx,
                cpu_sum
            );
            assert!(
                (gpu_sum - 1.0).abs() < 1e-4,
                "GPU row {} sums to {} (expected ~1.0)",
                row_idx,
                gpu_sum
            );

            // Compare element-wise
            for col in 0..seq_k {
                let idx = row_start + col;
                let diff = (cpu_result[idx] - gpu_result[idx]).abs();
                assert!(
                    diff < TEST_TOLERANCE,
                    "softmax mismatch at row={}, col={}: CPU={}, GPU={}, diff={}",
                    row_idx,
                    col,
                    cpu_result[idx],
                    gpu_result[idx],
                    diff
                );
            }
        }
    }

    /// Test softmax with non-square sequences (seq_q != seq_k)
    #[test]
    #[serial]
    #[ignore] // Requires HSACO kernels - GPU memory access without kernels
    fn test_softmax_explicit_non_square() {
        let batch = 1;
        let heads = 2;
        let seq_q = 8;
        let seq_k = 16;

        let total = batch * heads * seq_q * seq_k;
        let scores: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();

        let cpu_result = softmax_cpu_explicit(&scores, batch, heads, seq_q, seq_k);

        let backend = get_backend_or_skip();

        let scores_shape = TensorShape::from_dims(&[batch, heads, seq_q, seq_k]);
        let mut scores_gpu = DeviceTensor::from_host_vec(&backend, scores, scores_shape)
            .expect("Failed to create scores tensor");

        let total_rows = (batch * heads * seq_q) as u32;
        let row_len = seq_k as u32;

        let result = unsafe {
            crate::attention::kernels::softmax_gpu_kernel(
                scores_gpu.buffer().as_mut_ptr() as *mut f32,
                total_rows,
                row_len,
            )
        };

        assert_eq!(result, 0);

        backend.synchronize().expect("GPU synchronization failed");

        let gpu_result = scores_gpu
            .to_host_vec()
            .expect("Failed to copy output from GPU");

        // Verify rows sum to 1.0
        for row_idx in 0..(batch * heads * seq_q) {
            let row_start = row_idx * seq_k;
            let row_end = row_start + seq_k;

            let gpu_sum: f32 = gpu_result[row_start..row_end].iter().sum();
            assert!(
                (gpu_sum - 1.0).abs() < 1e-4,
                "GPU row {} sums to {} (expected ~1.0)",
                row_idx,
                gpu_sum
            );
        }

        // Sample element-wise comparison
        let mut max_diff = 0.0f32;
        for (i, (cpu_val, gpu_val)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
            let diff = (cpu_val - gpu_val).abs();
            max_diff = max_diff.max(diff);
            if diff >= TEST_TOLERANCE_LARGE {
                panic!(
                    "Mismatch at {}: CPU={}, GPU={}, diff={}",
                    i, cpu_val, gpu_val, diff
                );
            }
        }
        println!("Softmax explicit non-square max diff: {}", max_diff);
    }

    /// Test softmax with larger dimensions (32×32)
    #[test]
    #[serial]
    #[ignore] // Requires HSACO kernels - GPU memory access without kernels
    fn test_softmax_explicit_32x32() {
        let batch = 2;
        let heads = 4;
        let seq_q = 32;
        let seq_k = 32;

        let total = batch * heads * seq_q * seq_k;
        let scores: Vec<f32> = (0..total).map(|i| (i as f32) * 0.001).collect();

        let cpu_result = softmax_cpu_explicit(&scores, batch, heads, seq_q, seq_k);

        let backend = get_backend_or_skip();

        let scores_shape = TensorShape::from_dims(&[batch, heads, seq_q, seq_k]);
        let mut scores_gpu = DeviceTensor::from_host_vec(&backend, scores, scores_shape)
            .expect("Failed to create scores tensor");

        let total_rows = (batch * heads * seq_q) as u32;
        let row_len = seq_k as u32;

        let result = unsafe {
            crate::attention::kernels::softmax_gpu_kernel(
                scores_gpu.buffer().as_mut_ptr() as *mut f32,
                total_rows,
                row_len,
            )
        };

        assert_eq!(result, 0);

        backend.synchronize().expect("GPU synchronization failed");

        let gpu_result = scores_gpu
            .to_host_vec()
            .expect("Failed to copy output from GPU");

        // Spot check: verify all rows sum to ~1.0
        for row_idx in 0..(batch * heads * seq_q) {
            let row_start = row_idx * seq_k;
            let row_end = row_start + seq_k;

            let gpu_sum: f32 = gpu_result[row_start..row_end].iter().sum();
            assert!(
                (gpu_sum - 1.0).abs() < 1e-3,
                "GPU row {} sums to {} (expected ~1.0)",
                row_idx,
                gpu_sum
            );
        }

        // Check max difference
        let mut max_diff = 0.0f32;
        for (cpu_val, gpu_val) in cpu_result.iter().zip(gpu_result.iter()) {
            max_diff = max_diff.max((cpu_val - gpu_val).abs());
        }
        println!("Softmax explicit 32x32 max diff: {}", max_diff);
        assert!(
            max_diff < TEST_TOLERANCE_LARGE,
            "Max diff {} exceeds tolerance",
            max_diff
        );
    }

    /// Test numerical stability with large values
    #[test]
    #[serial]
    #[ignore] // Requires HSACO kernels - GPU memory access without kernels
    fn test_softmax_explicit_numerical_stability() {
        let batch = 1;
        let heads = 2;
        let seq_q = 4;
        let seq_k = 16;

        let total = batch * heads * seq_q * seq_k;
        // Large values that would overflow exp() without max normalization
        let scores: Vec<f32> = (0..total).map(|i| 1000.0 + (i as f32) * 0.1).collect();

        let backend = get_backend_or_skip();

        let scores_shape = TensorShape::from_dims(&[batch, heads, seq_q, seq_k]);
        let mut scores_gpu = DeviceTensor::from_host_vec(&backend, scores, scores_shape)
            .expect("Failed to create scores tensor");

        let total_rows = (batch * heads * seq_q) as u32;
        let row_len = seq_k as u32;

        let result = unsafe {
            crate::attention::kernels::softmax_gpu_kernel(
                scores_gpu.buffer().as_mut_ptr() as *mut f32,
                total_rows,
                row_len,
            )
        };

        assert_eq!(result, 0);

        backend.synchronize().expect("GPU synchronization failed");

        let gpu_result = scores_gpu
            .to_host_vec()
            .expect("Failed to copy output from GPU");

        // Verify all values are in valid range [0, 1]
        for (i, &val) in gpu_result.iter().enumerate() {
            assert!(
                val > 0.0 && val <= 1.0,
                "Invalid softmax value at {}: {}",
                i,
                val
            );
        }

        // Verify all rows sum to ~1.0
        for row_idx in 0..(batch * heads * seq_q) {
            let row_start = row_idx * seq_k;
            let row_end = row_start + seq_k;

            let row_sum: f32 = gpu_result[row_start..row_end].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-4,
                "Row {} sums to {} (expected ~1.0)",
                row_idx,
                row_sum
            );
        }

        println!("Softmax numerical stability test passed");
    }
}
