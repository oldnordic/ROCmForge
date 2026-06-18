#![cfg(feature = "gpu")]
#![allow(warnings)]

//! Correctness test for GPU shortconv against CPU reference.

mod common;

use common::helpers::*;
use rocmforge::config::{
    AttentionLayout, FfnLayout, ModelConfig, TensorNameRegistry, TensorNamingScheme, TensorRole,
};
use rocmforge::cpu::cache::{CpuForwardScratch, CpuKvCache};
use rocmforge::cpu::forward::cpu_layer_forward;
use rocmforge::cpu::weights::{CpuLayerWeights, CpuShortconvWeights};
use rocmforge::gpu::weights::{GpuLayerType, GpuShortconvWeights};
use rocmforge::gpu::{
    GpuBuffer, GpuDevice, GpuForwardScratch, GpuKvCache, GpuLayerWeights, GpuPrefillScratch,
    WeightMeta,
};
use rocmforge::loader::GgmlType;
use serial_test::serial;
use std::sync::Arc;

fn make_lfm2_config() -> ModelConfig {
    let mut config = mock_model_config();
    config.architecture = "lfm2".to_string();
    config.shortconv_l_cache = Some(3);
    config
}

#[test]
#[serial]
fn test_gpu_shortconv_decode_parity() {
    require_gpu!();
    let device = GpuDevice::init(0).expect("GPU init failed");
    let dev_id = device.device_id();
    let config = make_lfm2_config();
    let h = config.hidden_size;
    let ff = config.intermediate_size;
    let l_cache = config.shortconv_l_cache.expect("shortconv_l_cache missing");

    // 1. Setup CPU weights
    let mut cpu_layer = CpuLayerWeights {
        is_attention_layer: false,
        attn_norm: vec![1.0f32; h],
        attn_q: vec![],
        attn_q_meta: mock_cpu_meta(h, h, GgmlType::F32, TensorRole::Generic),
        attn_k: vec![],
        attn_k_meta: mock_cpu_meta(h, h, GgmlType::F32, TensorRole::Generic),
        attn_v: vec![],
        attn_v_meta: mock_cpu_meta(h, h, GgmlType::F32, TensorRole::Generic),
        attn_o: vec![],
        attn_o_meta: mock_cpu_meta(h, h, GgmlType::F32, TensorRole::Generic),
        attn_qkv: None,
        attn_qkv_meta: None,
        attn_gate: None,
        attn_gate_meta: None,
        attn_q_bias: None,
        attn_k_bias: None,
        attn_v_bias: None,
        attn_q_norm: None,
        attn_k_norm: None,
        ffn_norm: vec![1.0f32; h],
        ffn_gate: None,
        ffn_gate_meta: None,
        ffn_up: vec![0u8; ff * h * 4],
        ffn_up_meta: mock_cpu_meta(ff, h, GgmlType::F32, TensorRole::Generic),
        ffn_down: vec![0u8; h * ff * 4],
        ffn_down_meta: mock_cpu_meta(h, ff, GgmlType::F32, TensorRole::Generic),
        ssm: None,
        shortconv: None,
        moe: None,
        weight_type: GgmlType::F32,
        inp_gate: None,
        proj: None,
        post_attention_norm: None,
        post_ffw_norm: None,
        post_norm: None,
        layer_output_scale: None,
    };

    let in_proj_data = (0..3 * h * h)
        .map(|i| (i as f32).sin() * 0.1)
        .collect::<Vec<f32>>();
    let conv_data = (0..l_cache * h)
        .map(|i| (i as f32).cos() * 0.5)
        .collect::<Vec<f32>>();
    let out_proj_data = (0..h * h)
        .map(|i| (i as f32).tan() * 0.2)
        .collect::<Vec<f32>>();

    cpu_layer.shortconv = Some(CpuShortconvWeights {
        in_proj: in_proj_data
            .iter()
            .flat_map(|&f| f.to_le_bytes().to_vec())
            .collect(),
        in_proj_meta: mock_cpu_meta(3 * h, h, GgmlType::F32, TensorRole::Generic),
        conv: conv_data
            .iter()
            .flat_map(|&f| f.to_le_bytes().to_vec())
            .collect(),
        conv_meta: mock_cpu_meta(l_cache, h, GgmlType::F32, TensorRole::Generic),
        out_proj: out_proj_data
            .iter()
            .flat_map(|&f| f.to_le_bytes().to_vec())
            .collect(),
        out_proj_meta: mock_cpu_meta(h, h, GgmlType::F32, TensorRole::Generic),
    });

    // 2. Setup GPU weights
    let gpu_layer = GpuLayerWeights {
        attn_norm: upload_f32(dev_id, &cpu_layer.attn_norm),
        attn_q: GpuBuffer::empty(),
        attn_q_meta: mock_gpu_meta(h, h, GgmlType::F32, TensorRole::Generic),
        attn_q_svd: None,
        attn_q_norm: None,
        attn_q_bias: None,
        attn_k: GpuBuffer::empty(),
        attn_k_meta: mock_gpu_meta(h, h, GgmlType::F32, TensorRole::Generic),
        attn_k_svd: None,
        attn_k_norm: None,
        attn_k_bias: None,
        attn_v: GpuBuffer::empty(),
        attn_v_meta: mock_gpu_meta(h, h, GgmlType::F32, TensorRole::Generic),
        attn_v_svd: None,
        attn_v_bias: None,
        attn_qkv: None,
        attn_qkv_meta: None,
        attn_qkv_svd: None,
        attn_gate: None,
        attn_gate_meta: None,
        attn_gate_svd: None,
        ssm: None,
        is_attention_layer: false,
        layer_type: GpuLayerType::Shortconv,
        shortconv: Some(GpuShortconvWeights {
            in_proj: upload_f32(dev_id, &in_proj_data),
            in_proj_meta: mock_gpu_meta(3 * h, h, GgmlType::F32, TensorRole::Generic),
            conv: upload_f32(dev_id, &conv_data),
            conv_meta: mock_gpu_meta(l_cache, h, GgmlType::F32, TensorRole::Generic),
            out_proj: upload_f32(dev_id, &out_proj_data),
            out_proj_meta: mock_gpu_meta(h, h, GgmlType::F32, TensorRole::Generic),
        }),
        attn_o: GpuBuffer::empty(),
        attn_o_meta: mock_gpu_meta(h, h, GgmlType::F32, TensorRole::Generic),
        attn_o_svd: None,
        ffn_norm: upload_f32(dev_id, &cpu_layer.ffn_norm),
        ffn_gate: None,
        ffn_gate_meta: None,
        ffn_gate_svd: None,
        ffn_gate_sparse: None,
        ffn_gate_mpo: None,
        ffn_up: upload_f32(dev_id, &vec![0.0f32; ff * h]),
        ffn_up_meta: mock_gpu_meta(ff, h, GgmlType::F32, TensorRole::Generic),
        ffn_up_svd: None,
        ffn_up_sparse: None,
        ffn_up_mpo: None,
        ffn_down: upload_f32(dev_id, &vec![0.0f32; h * ff]),
        ffn_down_meta: mock_gpu_meta(h, ff, GgmlType::F32, TensorRole::Generic),
        ffn_down_svd: None,
        ffn_down_sparse: None,
        ffn_down_mpo: None,
        ffn_gate_up_interleaved: None,
        ffn_gate_up_interleaved_tile4: None,
        moe: None,
        ffn_gate_mpo_experts: None,
        ffn_up_mpo_experts: None,
        ffn_down_mpo_experts: None,
        ffn_gate_compressed: None,
        ffn_up_compressed: None,
        ffn_down_compressed: None,
        inp_gate: None,
        inp_gate_meta: None,
        proj: None,
        proj_meta: None,
        per_layer_token_emb: None,
        per_layer_model_proj: None,
        per_layer_proj_norm: None,
    };

    // 3. Setup input
    let mut x_cpu = vec![0.5f32; h];
    let mut gpu_scratch = GpuForwardScratch::new(&config).expect("alloc scratch");
    upload_f32_to_gpu(&mut gpu_scratch.hidden, &x_cpu).expect("upload hidden");

    let mut cpu_kv = CpuKvCache::new(&config, 1);
    let mut gpu_kv = GpuKvCache::new(&config, 128).expect("alloc kv");

    let mut cpu_scratch = CpuForwardScratch::new(&config);

    // 4. Run CPU
    let rope_sin = Vec::new();
    let rope_cos = Vec::new();
    cpu_layer_forward(
        &mut x_cpu,
        &cpu_layer,
        &mut cpu_kv,
        &mut cpu_scratch,
        0,
        0,
        &rope_sin,
        &rope_cos,
        &config,
        false,
    )
    .expect("cpu forward");

    // 5. Run GPU
    rocmforge::gpu::forward::gpu_layer_forward_hybrid(
        &device,
        &gpu_layer,
        Some(&cpu_layer),
        &mut gpu_kv,
        &mut gpu_scratch,
        Some(&mut cpu_scratch),
        0,
        0,
        0, // token_id (dummy value)
        &config,
        None, // ple_input
        None, // ple_gate
        None, // ple_down
    )
    .expect("gpu forward");
    device.synchronize().expect("sync");

    // 6. Compare
    let gpu_out = download_f32(&gpu_scratch.hidden, h);

    for i in 0..h {
        let diff = (x_cpu[i] - gpu_out[i]).abs();
        assert!(
            diff < 1e-3,
            "Mismatch at index {}: CPU={}, GPU={}, diff={}",
            i,
            x_cpu[i],
            gpu_out[i],
            diff
        );
    }
    println!("Decode parity test passed!");
}

#[test]
#[serial]
fn test_gpu_shortconv_prefill_parity() {
    require_gpu!();
    let device = GpuDevice::init(0).expect("GPU init failed");
    let dev_id = device.device_id();
    let config = make_lfm2_config();
    let h = config.hidden_size;
    let ff = config.intermediate_size;
    let seq_len = 16;
    let l_cache = config.shortconv_l_cache.expect("shortconv_l_cache missing");

    // 1. Setup CPU weights (Q4_0 for prefill compatibility)
    let mut cpu_layer = CpuLayerWeights {
        is_attention_layer: false,
        attn_norm: vec![1.0f32; h],
        attn_q: vec![],
        attn_q_meta: mock_cpu_meta(h, h, GgmlType::Q4_0, TensorRole::Generic),
        attn_k: vec![],
        attn_k_meta: mock_cpu_meta(h, h, GgmlType::Q4_0, TensorRole::Generic),
        attn_v: vec![],
        attn_v_meta: mock_cpu_meta(h, h, GgmlType::Q4_0, TensorRole::Generic),
        attn_o: vec![],
        attn_o_meta: mock_cpu_meta(h, h, GgmlType::Q4_0, TensorRole::Generic),
        attn_qkv: None,
        attn_qkv_meta: None,
        attn_gate: None,
        attn_gate_meta: None,
        attn_q_bias: None,
        attn_k_bias: None,
        attn_v_bias: None,
        attn_q_norm: None,
        attn_k_norm: None,
        ffn_norm: vec![1.0f32; h],
        ffn_gate: None,
        ffn_gate_meta: None,
        ffn_up: q4_0_zero_bytes_local(ff * h),
        ffn_up_meta: mock_cpu_meta(ff, h, GgmlType::Q4_0, TensorRole::Generic),
        ffn_down: q4_0_zero_bytes_local(h * ff),
        ffn_down_meta: mock_cpu_meta(h, ff, GgmlType::Q4_0, TensorRole::Generic),
        ssm: None,
        shortconv: None,
        moe: None,
        weight_type: GgmlType::Q4_0,
        inp_gate: None,
        proj: None,
        post_attention_norm: None,
        post_ffw_norm: None,
        post_norm: None,
        layer_output_scale: None,
    };

    cpu_layer.shortconv = Some(CpuShortconvWeights {
        in_proj: q4_0_zero_bytes_local(3 * h * h),
        in_proj_meta: mock_cpu_meta(3 * h, h, GgmlType::Q4_0, TensorRole::Generic),
        conv: vec![0u8; l_cache * h * 4],
        conv_meta: mock_cpu_meta(l_cache, h, GgmlType::F32, TensorRole::Generic),
        out_proj: q4_0_zero_bytes_local(h * h),
        out_proj_meta: mock_cpu_meta(h, h, GgmlType::Q4_0, TensorRole::Generic),
    });

    // 2. Setup GPU weights
    let gpu_layer = GpuLayerWeights {
        attn_norm: upload_f32(dev_id, &cpu_layer.attn_norm),
        attn_q: GpuBuffer::empty(),
        attn_q_meta: mock_gpu_meta(h, h, GgmlType::Q4_0, TensorRole::Generic),
        attn_q_svd: None,
        attn_q_norm: None,
        attn_q_bias: None,
        attn_k: GpuBuffer::empty(),
        attn_k_meta: mock_gpu_meta(h, h, GgmlType::Q4_0, TensorRole::Generic),
        attn_k_svd: None,
        attn_k_norm: None,
        attn_k_bias: None,
        attn_v: GpuBuffer::empty(),
        attn_v_meta: mock_gpu_meta(h, h, GgmlType::Q4_0, TensorRole::Generic),
        attn_v_svd: None,
        attn_v_bias: None,
        attn_qkv: None,
        attn_qkv_meta: None,
        attn_qkv_svd: None,
        attn_gate: None,
        attn_gate_meta: None,
        attn_gate_svd: None,
        ssm: None,
        is_attention_layer: false,
        layer_type: GpuLayerType::Shortconv,
        shortconv: Some(GpuShortconvWeights {
            in_proj: upload_raw(
                dev_id,
                &cpu_layer.shortconv.as_ref().expect("sc missing").in_proj,
            ),
            in_proj_meta: mock_gpu_meta(3 * h, h, GgmlType::Q4_0, TensorRole::Generic),
            conv: upload_raw(
                dev_id,
                &cpu_layer.shortconv.as_ref().expect("sc missing").conv,
            ),
            conv_meta: mock_gpu_meta(l_cache, h, GgmlType::F32, TensorRole::Generic),
            out_proj: upload_raw(
                dev_id,
                &cpu_layer.shortconv.as_ref().expect("sc missing").out_proj,
            ),
            out_proj_meta: mock_gpu_meta(h, h, GgmlType::Q4_0, TensorRole::Generic),
        }),
        attn_o: GpuBuffer::empty(),
        attn_o_meta: mock_gpu_meta(h, h, GgmlType::Q4_0, TensorRole::Generic),
        attn_o_svd: None,
        ffn_norm: upload_f32(dev_id, &cpu_layer.ffn_norm),
        ffn_gate: None,
        ffn_gate_meta: None,
        ffn_gate_svd: None,
        ffn_gate_sparse: None,
        ffn_gate_mpo: None,
        ffn_up: upload_raw(dev_id, &cpu_layer.ffn_up),
        ffn_up_meta: mock_gpu_meta(ff, h, GgmlType::Q4_0, TensorRole::Generic),
        ffn_up_svd: None,
        ffn_up_sparse: None,
        ffn_up_mpo: None,
        ffn_down: upload_raw(dev_id, &cpu_layer.ffn_down),
        ffn_down_meta: mock_gpu_meta(h, ff, GgmlType::Q4_0, TensorRole::Generic),
        ffn_down_svd: None,
        ffn_down_sparse: None,
        ffn_down_mpo: None,
        ffn_gate_up_interleaved: None,
        ffn_gate_up_interleaved_tile4: None,
        moe: None,
        ffn_gate_mpo_experts: None,
        ffn_up_mpo_experts: None,
        ffn_down_mpo_experts: None,
        ffn_gate_compressed: None,
        ffn_up_compressed: None,
        ffn_down_compressed: None,
        inp_gate: None,
        inp_gate_meta: None,
        proj: None,
        proj_meta: None,
        per_layer_token_emb: None,
        per_layer_model_proj: None,
        per_layer_proj_norm: None,
    };

    // 3. Setup input sequence
    let x_cpu_seq = (0..seq_len * h)
        .map(|i| (i as f32).cos() * 0.5)
        .collect::<Vec<f32>>();
    let mut gpu_scratch = GpuPrefillScratch::new(&config, seq_len).expect("alloc prefill scratch");

    let x_bytes: Vec<u8> = x_cpu_seq
        .iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();
    gpu_scratch
        .hidden
        .copy_from_host(&x_bytes)
        .expect("upload seq");

    let mut cpu_kv = CpuKvCache::new(&config, seq_len);
    let mut gpu_kv = GpuKvCache::new(&config, seq_len).expect("alloc gpu kv");

    let mut cpu_scratch = CpuForwardScratch::new(&config);

    // 4. Run CPU (token by token)
    let mut x_cpu_ref = x_cpu_seq.clone();
    let rope_sin = Vec::new();
    let rope_cos = Vec::new();
    for t in 0..seq_len {
        let mut x_token = x_cpu_ref[t * h..(t + 1) * h].to_vec();
        cpu_layer_forward(
            &mut x_token,
            &cpu_layer,
            &mut cpu_kv,
            &mut cpu_scratch,
            0,
            t,
            &rope_sin,
            &rope_cos,
            &config,
            false,
        )
        .expect("cpu forward");
        x_cpu_ref[t * h..(t + 1) * h].copy_from_slice(&x_token);
    }

    // 5. Run GPU (whole sequence)
    rocmforge::gpu::prefill_layer::gpu_prefill_shortconv_layer_on_stream(
        &device,
        &gpu_layer,
        &mut gpu_kv,
        &mut gpu_scratch,
        0,
        0,
        &config,
    )
    .expect("gpu prefill");
    device.synchronize().expect("sync");

    // 6. Compare
    let gpu_out = download_f32(&gpu_scratch.hidden, seq_len * h);

    for t in 0..seq_len {
        for i in 0..h {
            let idx = t * h + i;
            let diff = (x_cpu_ref[idx] - gpu_out[idx]).abs();
            assert!(
                diff < 1e-3,
                "Mismatch at token {}, index {}: CPU={}, GPU={}, diff={}",
                t,
                i,
                x_cpu_ref[idx],
                gpu_out[idx],
                diff
            );
        }
    }
    println!("Prefill parity test passed!");
}

fn mock_cpu_meta(
    rows: usize,
    cols: usize,
    wtype: GgmlType,
    role: TensorRole,
) -> rocmforge::cpu::weights::WeightMeta {
    rocmforge::cpu::weights::WeightMeta {
        wtype,
        dims: vec![rows as u64, cols as u64],
        needs_transpose: false,
        role,
        svd_k: None,
    }
}

fn q4_0_zero_bytes_local(elements: usize) -> Vec<u8> {
    let num_blocks = elements / 32;
    let mut bytes = vec![0u8; num_blocks * 18];
    for i in 0..num_blocks {
        bytes[i * 18] = 0x00;
        bytes[i * 18 + 1] = 0x3C;
        for j in 0..16 {
            bytes[i * 18 + 2 + j] = 0x88;
        }
    }
    bytes
}

fn upload_f32_to_gpu(
    buf: &mut GpuBuffer,
    data: &[f32],
) -> Result<(), rocmforge::gpu::error::GpuError> {
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();
    buf.copy_from_host(&bytes)
}
