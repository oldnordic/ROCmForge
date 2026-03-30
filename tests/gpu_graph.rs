#![cfg(feature = "gpu")]

use rocmforge::config::{AttentionLayout, ModelConfig, TensorNameRegistry, TensorNamingScheme};
use rocmforge::gpu::{DecodeGraphKey, GpuLogitsMode, TensorRole};
use rocmforge::loader::GgmlType;

fn make_test_config() -> ModelConfig {
    ModelConfig {
        num_layers: 24,
        num_kv_heads: 2,
        head_dim: 64,
        max_seq_len: 2048,
        hidden_size: 896,
        num_heads: 14,
        intermediate_size: 4864,
        vocab_size: 151936,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_neox: false,
        use_attention_bias: true,
        attention_layout: AttentionLayout::SplitQkv,
        architecture: "test".to_string(),
        tensor_registry: TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf),
    }
}

#[test]
fn decode_graph_key_depends_on_logits_mode() {
    let config = make_test_config();
    let greedy = DecodeGraphKey::from_parts(0, 32, &config, GpuLogitsMode::GreedyArgmax);
    let host = DecodeGraphKey::from_parts(0, 32, &config, GpuLogitsMode::DownloadToHost);

    assert_ne!(greedy, host);
}

#[test]
fn decode_graph_key_depends_on_wavefront_size() {
    let config = make_test_config();
    let wave32 = DecodeGraphKey::from_parts(0, 32, &config, GpuLogitsMode::GreedyArgmax);
    let wave64 = DecodeGraphKey::from_parts(0, 64, &config, GpuLogitsMode::GreedyArgmax);

    assert_ne!(wave32, wave64);
}

#[test]
fn decode_graph_key_depends_on_bound_tensors() {
    let config = make_test_config();
    let base = DecodeGraphKey::from_parts_with_bindings(
        0,
        32,
        &config,
        GpuLogitsMode::GreedyArgmax,
        0x1000,
        0x2000,
        GgmlType::Q8_0,
        TensorRole::LmHead,
    );
    let retied = DecodeGraphKey::from_parts_with_bindings(
        0,
        32,
        &config,
        GpuLogitsMode::GreedyArgmax,
        0x1000,
        0x2001,
        GgmlType::Q8_0,
        TensorRole::TiedLmHead,
    );

    assert_ne!(base, retied);
}
