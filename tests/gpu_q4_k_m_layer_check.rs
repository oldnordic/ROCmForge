//! Layer-by-layer diagnostic for Q4_K_M model: compare GPU layer output to CPU reference.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use rocmforge::config::ModelConfig;
use rocmforge::cpu::{
    cache::{CpuForwardScratch, CpuKvCache},
    forward::{cpu_embed_token, cpu_layer_forward},
    weights::CpuModelWeights,
};
use rocmforge::gpu::{self, GpuDevice, GpuForwardScratch, GpuKvCache};
use rocmforge::loader::GgufFile;
use std::path::Path;

const Q4K_MODEL: &str = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn mean_abs_err(a: &[f32], b: &[f32]) -> f32 {
    let sum: f32 = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum();
    sum / a.len() as f32
}

fn download_gpu_f32(buf: &rocmforge::gpu::GpuBuffer, len: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; len * std::mem::size_of::<f32>()];
    buf.copy_to_host(&mut bytes).expect("download gpu buffer");
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len).to_vec() }
}

#[test]
#[serial_test::serial]
fn test_q4_k_m_layer_0_matches_cpu() {
    if !Path::new(Q4K_MODEL).exists() {
        eprintln!("Skipping test: model file not found at {}", Q4K_MODEL);
        return;
    }

    require_gpu!();

    let token_id: u32 = 1000;

    let gguf = GgufFile::open(Q4K_MODEL).expect("open model");
    let config = ModelConfig::from_gguf(&gguf).expect("config from gguf");

    // CPU reference
    let cpu_weights = CpuModelWeights::load(&gguf, &config).expect("cpu weights");
    let mut cpu_kv = CpuKvCache::new(&config, 1);
    let mut cpu_scratch = CpuForwardScratch::new(&config);
    let mut cpu_hidden = vec![0.0f32; config.hidden_size];

    cpu_embed_token(token_id, &cpu_weights, &mut cpu_hidden, &config, None);
    let cpu_embed_hidden = cpu_hidden.clone();

    // Precompute RoPE sin/cos for position 0 (sin=0, cos=1 for all pairs).
    let head_dim = config.head_dim;
    let rotated_half = head_dim / 2;
    let rope_sin = vec![0.0f32; rotated_half];
    let rope_cos = vec![1.0f32; rotated_half];
    cpu_layer_forward(
        &mut cpu_hidden,
        cpu_weights.layer(0),
        &mut cpu_kv,
        &mut cpu_scratch,
        0,
        0,
        &rope_sin,
        &rope_cos,
        &config,
        false,
    )
    .expect("CPU layer 0 forward");

    // GPU
    let caps = gpu::detect().expect("detect gpu");
    let device = GpuDevice::init(caps.device_id).expect("init gpu");
    let gpu_weights = gpu::GpuModelWeights::load(&gguf, &config).expect("load gpu weights");
    let gpu_layer0 = gpu_weights.layer(0);
    eprintln!("[diag] GPU layer0 type={:?}", gpu_layer0.layer_type);
    eprintln!(
        "[diag] layer0 wtypes: attn_q={:?} attn_k={:?} attn_v={:?} attn_o={:?} gate={:?} up={:?} down={:?}",
        gpu_layer0.attn_q_meta.wtype,
        gpu_layer0.attn_k_meta.wtype,
        gpu_layer0.attn_v_meta.wtype,
        gpu_layer0.attn_o_meta.wtype,
        gpu_layer0.ffn_gate_meta.as_ref().map(|m| m.wtype),
        gpu_layer0.ffn_up_meta.wtype,
        gpu_layer0.ffn_down_meta.wtype,
    );
    let mut gpu_kv = GpuKvCache::new(&config, 1).expect("alloc kv cache");
    let mut gpu_scratch = GpuForwardScratch::new(&config).expect("alloc scratch");
    let mut host_scratch = CpuForwardScratch::new(&config);

    rocmforge::gpu::gpu_embed_token_hybrid(
        &device,
        token_id,
        &gpu_weights,
        &cpu_weights,
        &mut gpu_scratch,
        &mut host_scratch,
        &config,
    )
    .expect("gpu embed");

    device.synchronize().expect("sync after embed");
    let gpu_embed_hidden = download_gpu_f32(&gpu_scratch.hidden, config.hidden_size);
    let embed_err = max_abs_err(&gpu_embed_hidden, &cpu_embed_hidden);
    println!("After embed max_err={:.6}", embed_err);
    assert!(
        embed_err < 1e-5,
        "Embedding mismatch too large: {}",
        embed_err
    );

    gpu::gpu_layer_forward_hybrid(
        &device,
        gpu_weights.layer(0),
        Some(cpu_weights.layer(0)),
        &mut gpu_kv,
        &mut gpu_scratch,
        Some(&mut host_scratch),
        0,
        0,
        &config,
    )
    .expect("gpu layer 0 forward");

    device.synchronize().expect("sync");
    let gpu_hidden = download_gpu_f32(&gpu_scratch.hidden, config.hidden_size);

    let max_err = max_abs_err(&gpu_hidden, &cpu_hidden);
    let mean_err = mean_abs_err(&gpu_hidden, &cpu_hidden);
    println!(
        "Layer 0 hidden max_err={:.6} mean_err={:.6}",
        max_err, mean_err
    );
    println!(
        "  first 10 CPU: {:?}",
        &cpu_hidden[..10.min(cpu_hidden.len())]
    );
    println!(
        "  first 10 GPU: {:?}",
        &gpu_hidden[..10.min(gpu_hidden.len())]
    );
    assert!(
        max_err < 1e-2,
        "Layer 0 hidden mismatch too large: {}",
        max_err
    );
}
