#![cfg(feature = "gpu")]

mod common;

use rocmforge::config::{
    AttentionLayout, FfnLayout, ModelConfig, TensorNameRegistry, TensorNamingScheme,
};
use rocmforge::gpu::{
    CapturedDecodeGraph, DecodeGraphKey, GpuBuffer, GpuDevice, GpuForwardScratch, GpuLogitsMode,
    HipGraph, TensorRole,
};
use rocmforge::loader::GgmlType;
use serial_test::serial;

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
        ffn_layout: FfnLayout::SwiGLU,
        architecture: "test".to_string(),
        tensor_registry: TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf),
        shortconv_l_cache: None,
        num_dense_layers: None,
        num_experts_per_tok: None,
        use_expert_bias: false,
        expert_weights_scale: 1.0,
        rope_freq: (0..32)
            .map(|i| 1.0 / 10000.0f32.powf((2 * i) as f32 / 64.0f32))
            .collect(),
        kv_lora_dim: None,
        kv_frame_codec_enabled: None,
        adastate_anchors_enabled: None,
        kv_quant_bits: None,
        turboquant_centroids: None,
        qjl_scale: None,
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

#[test]
#[serial]
fn test_decode_graph_exec_update() {
    require_gpu!();

    let caps = rocmforge::gpu::detect().expect("GPU should be detected"); // test
    let device = GpuDevice::init(caps.device_id).expect("GPU device should initialize"); // test

    let config = make_test_config();
    let mut scratch = GpuForwardScratch::new(&config).expect("Failed to create scratch"); // test

    // Create initial key
    let key1 = DecodeGraphKey::from_parts_with_bindings(
        device.device_id(),
        device.warp_size(),
        &config,
        GpuLogitsMode::GreedyArgmax,
        0x1000,
        0x2000,
        GgmlType::F32,
        TensorRole::Generic,
    );

    // Allocate all buffers upfront before starting capture
    let dummy_src = GpuBuffer::alloc(4).unwrap(); // test
    let mut dummy_dst = GpuBuffer::alloc(4).unwrap(); // test
    let dummy_src2 = GpuBuffer::alloc(4).unwrap(); // test
    let mut dummy_dst2 = GpuBuffer::alloc(4).unwrap(); // test

    // Capture graph 1
    device
        .begin_capture(rocmforge::gpu::ffi::hipStreamCaptureMode::hipStreamCaptureModeGlobal)
        .unwrap(); // test
                   // Do a dummy copy to have some node in the graph
    dummy_dst
        .copy_from_buffer_async(&dummy_src, 4, device.stream())
        .unwrap(); // test
    let raw_graph1 = device.end_capture().unwrap(); // test
    let graph1 = HipGraph::from_raw(raw_graph1);

    // Instantiate CapturedDecodeGraph
    let captured = CapturedDecodeGraph::from_captured_graph(graph1, key1).unwrap(); // test
    scratch.replace_decode_graph(captured);

    assert!(scratch.has_decode_graph_for(key1));

    // Create a new key (different pointer binding)
    let key2 = DecodeGraphKey::from_parts_with_bindings(
        device.device_id(),
        device.warp_size(),
        &config,
        GpuLogitsMode::GreedyArgmax,
        0x1008, // changed pointer
        0x2000,
        GgmlType::F32,
        TensorRole::Generic,
    );

    assert!(!scratch.has_decode_graph_for(key2));

    // Capture graph 2 (same structure, different pointers)
    device
        .begin_capture(rocmforge::gpu::ffi::hipStreamCaptureMode::hipStreamCaptureModeGlobal)
        .unwrap(); // test
    dummy_dst2
        .copy_from_buffer_async(&dummy_src2, 4, device.stream())
        .unwrap(); // test
    let raw_graph2 = device.end_capture().unwrap(); // test
    let graph2 = HipGraph::from_raw(raw_graph2);

    // Try fast update
    let update_res = scratch.try_update_decode_graph(graph2, key2).unwrap(); // test

    // Since topology is identical (just different buffers/pointers), update should succeed
    assert!(update_res.is_ok(), "Fast update of graph should succeed");

    // Verify key has been updated and key1 is no longer matched
    assert!(scratch.has_decode_graph_for(key2));
    assert!(!scratch.has_decode_graph_for(key1));
}
