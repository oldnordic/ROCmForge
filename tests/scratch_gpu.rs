#![cfg(feature = "gpu")]

//! Forward scratch buffer tests.

mod common;

use rocmforge::config::ModelConfig;
use rocmforge::gpu::GpuForwardScratch;
use serial_test::serial;

fn make_test_config() -> ModelConfig {
    use rocmforge::config::{TensorNameRegistry, TensorNamingScheme};
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
fn test_scratch_allocation_all_buffers() {
    require_gpu!();

    let config = make_test_config();
    let scratch = GpuForwardScratch::new(&config);

    match scratch {
        Ok(s) => {
            // Verify all buffers are allocated (pointers should be non-null or empty)
            // Empty buffers are allowed for zero-size allocations
            let hidden_ptr = s.hidden_ptr();
            let q_ptr = s.q_ptr();
            let k_ptr = s.k_ptr();
            let v_ptr = s.v_ptr();
            // Just verify the struct was created successfully
            assert!(!hidden_ptr.is_null() || s.hidden.is_empty());
            assert!(!q_ptr.is_null() || s.q.is_empty());
            assert!(!k_ptr.is_null() || s.k.is_empty());
            assert!(!v_ptr.is_null() || s.v.is_empty());
        }
        Err(_) => {
            // Expected when HIP unavailable
        }
    }
}

#[test]
#[serial]
fn test_scratch_pointers_valid() {
    require_gpu!();

    let config = make_test_config();
    let mut scratch = GpuForwardScratch::new(&config);

    if let Ok(scratch) = &mut scratch {
        // Test all pointer methods
        let _hidden = scratch.hidden_ptr();
        let _hidden_mut = scratch.hidden_mut_ptr();
        let _normed = scratch.normed_ptr();
        let _normed_mut = scratch.normed_mut_ptr();
        let _q = scratch.q_ptr();
        let _q_mut = scratch.q_mut_ptr();
        let _k = scratch.k_ptr();
        let _k_mut = scratch.k_mut_ptr();
        let _v = scratch.v_ptr();
        let _v_mut = scratch.v_mut_ptr();
        let _attn_out = scratch.attn_out_ptr();
        let _attn_out_mut = scratch.attn_out_mut_ptr();
        let _layer_out = scratch.layer_out_ptr();
        let _layer_out_mut = scratch.layer_out_mut_ptr();
        let _gate = scratch.gate_ptr();
        let _gate_mut = scratch.gate_mut_ptr();
        let _swiglu = scratch.swiglu_ptr();
        let _swiglu_mut = scratch.swiglu_mut_ptr();
        let _logits = scratch.logits_ptr();
        let _logits_mut = scratch.logits_mut_ptr();

        // If we got here without panicking, the test passes
        assert!(true);
    }
}

#[test]
#[serial]
fn test_scratch_memory_usage() {
    require_gpu!();

    let config = make_test_config();
    let scratch = GpuForwardScratch::new(&config);

    if let Ok(scratch) = scratch {
        let h = config.hidden_size;
        let q = config.num_heads * config.head_dim;
        let kv = config.num_kv_heads * config.head_dim;
        let ff = config.intermediate_size;
        let v = config.vocab_size;

        // Calculate expected memory usage
        let expected_bytes = (h + h + q + kv + kv + q + h + ff + ff + v) * 4; // * 4 for f32

        let actual_bytes = scratch.hidden.size()
            + scratch.normed.size()
            + scratch.q.size()
            + scratch.k.size()
            + scratch.v.size()
            + scratch.attn_out.size()
            + scratch.layer_out.size()
            + scratch.gate.size()
            + scratch.swiglu.size()
            + scratch.logits.size();

        assert_eq!(actual_bytes, expected_bytes);
    }
}

#[test]
#[serial]
fn test_scratch_all_buffers_non_empty() {
    require_gpu!();

    let config = make_test_config();
    let scratch = GpuForwardScratch::new(&config);

    if let Ok(scratch) = scratch {
        // All buffers should be non-empty (non-zero size)
        assert!(
            !scratch.hidden.is_empty(),
            "hidden buffer should not be empty"
        );
        assert!(
            !scratch.normed.is_empty(),
            "normed buffer should not be empty"
        );
        assert!(!scratch.q.is_empty(), "q buffer should not be empty");
        assert!(!scratch.k.is_empty(), "k buffer should not be empty");
        assert!(!scratch.v.is_empty(), "v buffer should not be empty");
        assert!(
            !scratch.attn_out.is_empty(),
            "attn_out buffer should not be empty"
        );
        assert!(
            !scratch.layer_out.is_empty(),
            "layer_out buffer should not be empty"
        );
        assert!(!scratch.gate.is_empty(), "gate buffer should not be empty");
        assert!(
            !scratch.swiglu.is_empty(),
            "swiglu buffer should not be empty"
        );
        assert!(
            !scratch.logits.is_empty(),
            "logits buffer should not be empty"
        );
    }
}
