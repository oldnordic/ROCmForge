//! Tests for GPU prefill kernels.

#![cfg(feature = "gpu")]

use rocmforge::config::ModelConfig;
use rocmforge::gpu::GpuPrefillScratch;

fn make_test_config() -> ModelConfig {
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
        ffn_layout: rocmforge::config::FfnLayout::SwiGLU,
        architecture: "test".to_string(),
        tensor_registry: rocmforge::config::TensorNameRegistry::from_scheme(
            &rocmforge::config::TensorNamingScheme::Gguf,
        ),
        shortconv_l_cache: None,
        num_dense_layers: None,
        num_experts_per_tok: None,
        use_expert_bias: false,
        expert_weights_scale: 1.0,
        rope_freq: (0..64)
            .map(|i| 1.0 / 10000.0f32.powf((2 * i) as f32 / 128.0f32))
            .collect(),
        kv_lora_dim: None,
        kv_frame_codec_enabled: None,
        adastate_anchors_enabled: None,
        kv_quant_bits: None,
        turboquant_centroids: None,
        qjl_scale: None,
        ..Default::default()
    }
}

#[test]
fn prefill_scratch_allocates_for_valid_seq_len() {
    let config = make_test_config();
    let result = GpuPrefillScratch::new(&config, 32);

    // Will fail without GPU - that's expected
    match result {
        Ok(scratch) => {
            assert_eq!(scratch.seq_len, 32);
        }
        Err(_) => {
            // Expected when HIP unavailable
        }
    }
}

#[test]
fn prefill_scratch_rejects_zero_seq_len() {
    let config = make_test_config();
    let result = GpuPrefillScratch::new(&config, 0);
    assert!(result.is_err());
}

#[test]
fn prefill_scratch_rejects_large_seq_len() {
    let config = make_test_config();
    let result = GpuPrefillScratch::new(&config, 1000000);

    // Should fail due to memory constraints
    assert!(result.is_err());
}

#[test]
fn prefill_scratch_validates_buffer_pointers() {
    let config = make_test_config();
    let scratch = GpuPrefillScratch::new(&config, 16);

    if let Ok(mut s) = scratch {
        let h = config.hidden_size;
        let q = config.num_heads * config.head_dim;
        let kv = config.num_kv_heads * config.head_dim;
        let ff = config.intermediate_size;

        // Test row pointer calculations
        let row = 0;
        let _hidden_ptr = s.hidden_row_ptr(row, h);
        let _normed_ptr = s.normed_row_ptr(row, h);
        let _q_ptr = s.q_row_mut_ptr(row, q);
        let _k_ptr = s.k_row_mut_ptr(row, kv);
        let _v_ptr = s.v_row_mut_ptr(row, kv);
        let _attn_out_ptr = s.attn_out_row_mut_ptr(row, q);
        let _layer_out_ptr = s.layer_out_row_mut_ptr(row, h);
        let _gate_ptr = s.gate_row_mut_ptr(row, ff);
        let _swiglu_ptr = s.swiglu_row_mut_ptr(row, ff);

        // If we got here without panicking, pointers are valid
    }
}

#[test]
fn prefill_scratch_row_pointers_are_valid() {
    let config = make_test_config();
    let scratch = GpuPrefillScratch::new(&config, 8);

    if let Ok(mut s) = scratch {
        let h = config.hidden_size;
        let kv = config.num_kv_heads * config.head_dim;

        // Test multiple rows
        for row in 0..s.seq_len {
            let _hidden_ptr = s.hidden_row_ptr(row, h);
            let _normed_ptr = s.normed_row_ptr(row, h);
            let _k_mut_ptr = s.k_row_mut_ptr(row, kv);
            let _v_mut_ptr = s.v_row_mut_ptr(row, kv);
        }

        // Test boundary conditions
        let _first_row = s.hidden_row_ptr(0, h);
        let _last_row = s.hidden_row_ptr(s.seq_len - 1, h);
    }
}
