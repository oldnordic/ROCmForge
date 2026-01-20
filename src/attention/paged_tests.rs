//! Paged Attention GPU Kernel Tests (Phase 3)
//!
//! Tests for paged attention computation with non-contiguous KV blocks.

#[cfg(test)]
mod tests {
    use super::super::paged_kernel::{PagedAttentionConfig, PagedAttentionKernels};
use serial_test::serial;
    use crate::backend::{DeviceTensor, HipBackend, HipError};
    use crate::loader::TensorShape;
    use std::sync::Arc;

    /// Helper: Get GPU backend or skip test if not available (llama.cpp pattern)
    fn get_backend_or_skip() -> Arc<HipBackend> {
        match HipBackend::new_checked() {
            Ok(backend) => backend,
            Err(e) => {
                eprintln!("\n⚠️  GPU not available for paged_tests: {}", e);
                eprintln!("To enable these tests, ensure:");
                eprintln!("  1. AMD GPU is present");
                eprintln!("  2. ROCm is installed (check with rocm-smi)");
                eprintln!("  3. amdhip64 library is in LD_LIBRARY_PATH");
                eprintln!("\nSkipping test gracefully (llama.cpp pattern).\n");
                panic!("GPU_SKIP");
            }
        }
    }

    /// Helper function to create test tensors
    fn create_test_qkv(
        backend: &HipBackend,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<(DeviceTensor, DeviceTensor, DeviceTensor), HipError> {
        let q_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        let k_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        let v_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);

        let total_elements = seq_len * num_heads * head_dim;

        // Create test data
        let mut q_data = Vec::with_capacity(total_elements);
        let mut k_data = Vec::with_capacity(total_elements);
        let mut v_data = Vec::with_capacity(total_elements);

        for i in 0..total_elements {
            q_data.push((i as f32) * 0.1);
            k_data.push((i as f32) * 0.2 + 1.0);
            v_data.push((i as f32) * 0.3 + 2.0);
        }

        let q = DeviceTensor::from_host_vec(backend, q_data, q_shape)?;
        let k = DeviceTensor::from_host_vec(backend, k_data, k_shape)?;
        let v = DeviceTensor::from_host_vec(backend, v_data, v_shape)?;

        Ok((q, k, v))
    }

    // Test 1: Kernel compilation test
    #[test]
    #[serial]
    fn test_paged_attention_kernel_compilation() {
        let backend = get_backend_or_skip();
        let config = PagedAttentionConfig {
            block_size: 16,
            num_heads: 4,
            head_dim: 32,
        };

        let result = PagedAttentionKernels::new(&backend, &config);
        assert!(
            result.is_ok(),
            "PagedAttentionKernels creation should succeed: {:?}",
            result.err()
        );

        let kernels = result.unwrap();
        assert!(
            kernels.is_compiled(),
            "Kernel should be compiled after creation"
        );
    }

    // Test 2: Single block paged attention
    #[test]
    #[serial]
    fn test_paged_attention_single_block() {
        let backend = get_backend_or_skip();
        let config = PagedAttentionConfig {
            block_size: 16,
            num_heads: 4,
            head_dim: 32,
        };

        let kernels =
            PagedAttentionKernels::new(&backend, &config).expect("Failed to create kernels");

        let seq_len = 8;
        let num_heads = 4;
        let head_dim = 32;

        let (q, k, v) = create_test_qkv(&backend, seq_len, num_heads, head_dim)
            .expect("Failed to create test tensors");

        // Single block mapping: all positions in block 0
        let block_indices = vec![0u32; seq_len];
        let block_offsets: Vec<usize> = (0..seq_len).collect();

        let output_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        let mut output =
            DeviceTensor::empty(&backend, output_shape).expect("Failed to create output tensor");

        let result = kernels.compute_paged_attention(
            &q,
            &[k.clone()],
            &[v.clone()],
            &block_indices,
            &block_offsets,
            &mut output,
        );

        assert!(
            result.is_ok(),
            "Paged attention computation should succeed: {:?}",
            result.err()
        );

        // Verify output is not all zeros (kernel actually executed)
        let output_host = output.to_host_vec().expect("Failed to copy output to host");

        let has_nonzero = output_host.iter().any(|&x| x.abs() > 1e-6);
        assert!(has_nonzero, "Output should contain non-zero values");
    }

    // Test 3: Multiple non-contiguous blocks
    #[test]
    #[serial]
    fn test_paged_attention_multiple_blocks() {
        let backend = get_backend_or_skip();
        let config = PagedAttentionConfig {
            block_size: 4, // Small block size for testing
            num_heads: 2,
            head_dim: 16,
        };

        let kernels =
            PagedAttentionKernels::new(&backend, &config).expect("Failed to create kernels");

        let seq_len = 12; // 3 blocks of size 4
        let num_heads = 2;
        let head_dim = 16;

        let (q, k, v) = create_test_qkv(&backend, seq_len, num_heads, head_dim)
            .expect("Failed to create test tensors");

        // Non-contiguous block mapping: blocks [2, 0, 1] instead of [0, 1, 2]
        let block_indices = vec![
            2, 2, 2, 2, // Block 2: positions 0-3
            0, 0, 0, 0, // Block 0: positions 4-7
            1, 1, 1, 1, // Block 1: positions 8-11
        ];
        let block_offsets: Vec<usize> = (0..seq_len).map(|i| i % 4).collect();

        // Create 3 separate K/V blocks
        let k_data = k.to_host_vec().expect("Failed to copy K to host");
        let v_data = v.to_host_vec().expect("Failed to copy V to host");

        let block_size = config.block_size;
        let num_kv_elements = block_size * num_heads * head_dim;

        let mut k_blocks = Vec::new();
        let mut v_blocks = Vec::new();

        for block_id in 0..3 {
            let k_block_shape = TensorShape::from_dims(&[block_size, num_heads, head_dim]);
            let v_block_shape = TensorShape::from_dims(&[block_size, num_heads, head_dim]);

            let k_start = block_id * num_kv_elements;
            let k_end = k_start + num_kv_elements;

            let k_block_data = k_data[k_start..k_end].to_vec();
            let v_block_data = v_data[k_start..k_end].to_vec();

            let k_block = DeviceTensor::from_host_vec(&backend, k_block_data, k_block_shape)
                .expect("Failed to create K block");
            let v_block = DeviceTensor::from_host_vec(&backend, v_block_data, v_block_shape)
                .expect("Failed to create V block");

            k_blocks.push(k_block);
            v_blocks.push(v_block);
        }

        let output_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        let mut output =
            DeviceTensor::empty(&backend, output_shape).expect("Failed to create output tensor");

        let result = kernels.compute_paged_attention(
            &q,
            &k_blocks,
            &v_blocks,
            &block_indices,
            &block_offsets,
            &mut output,
        );

        assert!(
            result.is_ok(),
            "Multi-block paged attention should succeed: {:?}",
            result.err()
        );

        // Verify output
        let output_host = output.to_host_vec().expect("Failed to copy output to host");

        let has_nonzero = output_host.iter().any(|&x| x.abs() > 1e-6);
        assert!(has_nonzero, "Output should contain non-zero values");
    }

    // Test 4: MQA (Multi-Query Attention) support
    #[test]
    #[serial]
    fn test_paged_attention_mqa() {
        let backend = get_backend_or_skip();
        let config = PagedAttentionConfig {
            block_size: 8,
            num_heads: 8, // 8 query heads
            head_dim: 16,
        };

        let kernels =
            PagedAttentionKernels::new(&backend, &config).expect("Failed to create kernels");

        let seq_len = 8;
        let num_q_heads = 8; // 8 query heads
        let num_kv_heads = 2; // 2 KV heads (MQA: 4 query heads per KV head)
        let head_dim = 16;

        // Create Q with 8 heads, K/V with 2 heads
        let q_shape = TensorShape::from_dims(&[seq_len, num_q_heads, head_dim]);
        let k_shape = TensorShape::from_dims(&[seq_len, num_kv_heads, head_dim]);
        let v_shape = TensorShape::from_dims(&[seq_len, num_kv_heads, head_dim]);

        let q_elements = seq_len * num_q_heads * head_dim;
        let kv_elements = seq_len * num_kv_heads * head_dim;

        let q_data: Vec<f32> = (0..q_elements).map(|i| i as f32 * 0.1).collect();
        let k_data: Vec<f32> = (0..kv_elements).map(|i| i as f32 * 0.2 + 1.0).collect();
        let v_data: Vec<f32> = (0..kv_elements).map(|i| i as f32 * 0.3 + 2.0).collect();

        let q = DeviceTensor::from_host_vec(&backend, q_data, q_shape)
            .expect("Failed to create Q tensor");
        let k = DeviceTensor::from_host_vec(&backend, k_data, k_shape)
            .expect("Failed to create K tensor");
        let v = DeviceTensor::from_host_vec(&backend, v_data, v_shape)
            .expect("Failed to create V tensor");

        // Single block
        let block_indices = vec![0u32; seq_len];
        let block_offsets: Vec<usize> = (0..seq_len).collect();

        let output_shape = TensorShape::from_dims(&[seq_len, num_q_heads, head_dim]);
        let mut output =
            DeviceTensor::empty(&backend, output_shape).expect("Failed to create output tensor");

        let result = kernels.compute_paged_attention_mqa(
            &q,
            &[k],
            &[v],
            &block_indices,
            &block_offsets,
            num_kv_heads,
            &mut output,
        );

        assert!(
            result.is_ok(),
            "MQA paged attention should succeed: {:?}",
            result.err()
        );

        // Verify output
        let output_host = output.to_host_vec().expect("Failed to copy output to host");

        let has_nonzero = output_host.iter().any(|&x| x.abs() > 1e-6);
        assert!(has_nonzero, "MQA output should contain non-zero values");
    }

    // Test 5: Block boundary crossing
    #[test]
    #[serial]
    fn test_paged_attention_block_boundary() {
        let backend = get_backend_or_skip();
        let config = PagedAttentionConfig {
            block_size: 4,
            num_heads: 2,
            head_dim: 16,
        };

        let kernels =
            PagedAttentionKernels::new(&backend, &config).expect("Failed to create kernels");

        let seq_len = 8; // Exactly 2 blocks
        let num_heads = 2;
        let head_dim = 16;

        let (q, k, v) = create_test_qkv(&backend, seq_len, num_heads, head_dim)
            .expect("Failed to create test tensors");

        // Split K/V into 2 separate blocks (each of size 4)
        let k_data = k.to_host_vec().expect("Failed to copy K to host");
        let v_data = v.to_host_vec().expect("Failed to copy V to host");

        let block_size = config.block_size;
        let num_kv_elements = block_size * num_heads * head_dim;

        let mut k_blocks = Vec::new();
        let mut v_blocks = Vec::new();

        // Create 2 blocks of size 4 each
        for block_id in 0..2 {
            let k_block_shape = TensorShape::from_dims(&[block_size, num_heads, head_dim]);
            let v_block_shape = TensorShape::from_dims(&[block_size, num_heads, head_dim]);

            let k_start = block_id * num_kv_elements;
            let k_end = k_start + num_kv_elements;

            let k_block_data = k_data[k_start..k_end].to_vec();
            let v_block_data = v_data[k_start..k_end].to_vec();

            let k_block = DeviceTensor::from_host_vec(&backend, k_block_data, k_block_shape)
                .expect("Failed to create K block");
            let v_block = DeviceTensor::from_host_vec(&backend, v_block_data, v_block_shape)
                .expect("Failed to create V block");

            k_blocks.push(k_block);
            v_blocks.push(v_block);
        }

        // Block mapping with explicit boundary
        let block_indices = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let block_offsets: Vec<usize> = (0..seq_len).map(|i| i % 4).collect();

        let output_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        let mut output =
            DeviceTensor::empty(&backend, output_shape).expect("Failed to create output tensor");

        let result = kernels.compute_paged_attention(
            &q,
            &k_blocks,
            &v_blocks,
            &block_indices,
            &block_offsets,
            &mut output,
        );

        assert!(
            result.is_ok(),
            "Block boundary paged attention should succeed: {:?}",
            result.err()
        );
    }

    // Test 6: Invalid input handling
    #[test]
    #[serial]
    fn test_paged_attention_invalid_input() {
        let backend = get_backend_or_skip();
        let config = PagedAttentionConfig {
            block_size: 16,
            num_heads: 4,
            head_dim: 32,
        };

        let kernels =
            PagedAttentionKernels::new(&backend, &config).expect("Failed to create kernels");

        let seq_len = 8;
        let num_heads = 4;
        let head_dim = 32;

        let (q, k, v) = create_test_qkv(&backend, seq_len, num_heads, head_dim)
            .expect("Failed to create test tensors");

        let output_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        let mut output =
            DeviceTensor::empty(&backend, output_shape).expect("Failed to create output tensor");

        // Mismatched block_indices length (too short)
        let block_indices = vec![0u32; seq_len / 2];
        let block_offsets: Vec<usize> = (0..seq_len).collect();

        let result = kernels.compute_paged_attention(
            &q,
            &[k],
            &[v],
            &block_indices,
            &block_offsets,
            &mut output,
        );

        assert!(
            result.is_err(),
            "Should return error for mismatched block_indices length"
        );
    }
}
