#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use rocmforge::cpu::forward::cpu_embed_token;
use rocmforge::cpu::ops::{
    dispatch_gemv as cpu_dispatch_gemv, gemv_q4_0_transposed, rms_norm, silu_fuse,
};
use rocmforge::cpu::quant::{embed_q5_0, embed_q8_0};
use rocmforge::cpu::{
    cache::{CpuForwardScratch, CpuKvCache},
    forward::cpu_layer_forward,
    weights::WeightMeta as CpuWeightMeta,
};
use rocmforge::gpu::{
    detect, gpu_dispatch_fused_qkv, GpuBuffer, GpuDevice, GpuModelWeights, GpuQuant, TensorRole,
    WeightMeta, Q4_0_BLOCK_SIZE, QK4_0,
};
use rocmforge::loader::GgmlType;
use rocmforge::{cpu::weights::CpuModelWeights, loader::GgufFile};
use serial_test::serial;

const REAL_Q4K_MODEL: &str = "/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

fn upload_f32(data: &[f32]) -> rocmforge::gpu::GpuResult<GpuBuffer> {
    let mut buf = GpuBuffer::alloc(std::mem::size_of_val(data))?;
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    buf.copy_from_host(bytes)?;
    Ok(buf)
}

fn download_f32(buf: &GpuBuffer, len: usize) -> rocmforge::gpu::GpuResult<Vec<f32>> {
    let mut bytes = vec![0u8; len * std::mem::size_of::<f32>()];
    buf.copy_to_host(&mut bytes)?;
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len).to_vec() })
}

fn download_u8(buf: &GpuBuffer, len: usize) -> rocmforge::gpu::GpuResult<Vec<u8>> {
    let mut bytes = vec![0u8; len];
    buf.copy_to_host(&mut bytes)?;
    Ok(bytes)
}

fn quantize_q4_0_columns(
    gpu_quant: &GpuQuant,
    weights: &[f32],
    n_rows: usize,
    n_cols: usize,
) -> rocmforge::gpu::GpuResult<GpuBuffer> {
    let d_weights = upload_f32(weights)?;
    let d_quantized = GpuBuffer::alloc((n_rows / QK4_0) * n_cols * Q4_0_BLOCK_SIZE)?;

    for col in 0..n_cols {
        let col_weights_ptr = unsafe {
            d_weights
                .as_ptr()
                .add(col * n_rows * std::mem::size_of::<f32>())
        };
        let col_quantized_ptr = unsafe {
            d_quantized
                .as_ptr()
                .add(col * (n_rows / QK4_0) * Q4_0_BLOCK_SIZE)
        };
        gpu_quant.quantize_q4_0(col_weights_ptr as *const f32, col_quantized_ptr, n_rows)?;
    }

    Ok(d_quantized)
}

fn max_abs_error(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn cpu_q4_0_projection(
    quantized: &[u8],
    bias: &[f32],
    input: &[f32],
    n_rows: usize,
    n_cols: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_cols];
    gemv_q4_0_transposed(quantized, input, &mut out, n_cols, n_rows);
    for (dst, b) in out.iter_mut().zip(bias) {
        *dst += *b;
    }
    out
}

fn cpu_quant_projection_with_bias(
    quantized: &[u8],
    wtype: GgmlType,
    bias: &[f32],
    input: &[f32],
    n_rows: usize,
    n_cols: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_cols];
    for (row, out_row) in out.iter_mut().enumerate().take(n_cols) {
        let mut deq = vec![0.0f32; n_rows];
        match wtype {
            GgmlType::Q5_0 => embed_q5_0(row, quantized, &mut deq, n_rows),
            GgmlType::Q8_0 => embed_q8_0(row, quantized, &mut deq, n_rows),
            other => panic!("unsupported projection type: {:?}", other),
        }
        *out_row = deq.iter().zip(input).map(|(w, x)| w * x).sum::<f32>() + bias[row];
    }
    out
}

fn cpu_quant_projection(
    quantized: &[u8],
    wtype: GgmlType,
    input: &[f32],
    n_rows: usize,
    n_cols: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_cols];
    for (row, out_row) in out.iter_mut().enumerate().take(n_cols) {
        let mut deq = vec![0.0f32; n_rows];
        match wtype {
            GgmlType::Q5_0 => embed_q5_0(row, quantized, &mut deq, n_rows),
            GgmlType::Q8_0 => embed_q8_0(row, quantized, &mut deq, n_rows),
            other => panic!("unsupported projection type: {:?}", other),
        }
        *out_row = deq.iter().zip(input).map(|(w, x)| w * x).sum::<f32>();
    }
    out
}

fn cpu_gate_up_reference(
    gate_weights: &[u8],
    gate_wtype: GgmlType,
    up_weights: &[u8],
    up_wtype: GgmlType,
    input: &[f32],
    ff_size: usize,
    h: usize,
) -> (Vec<f32>, Vec<f32>) {
    let gate = cpu_quant_projection(gate_weights, gate_wtype, input, h, ff_size);
    let mut swiglu = cpu_quant_projection(up_weights, up_wtype, input, h, ff_size);
    silu_fuse(&gate, &mut swiglu);
    (gate, swiglu)
}

fn cpu_dispatch_projection(
    quantized: &[u8],
    meta: &CpuWeightMeta,
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    cpu_dispatch_gemv(quantized, meta, input, &mut out, out_dim, in_dim, None)
        .expect("cpu dispatch gemv");
    out
}

#[test]
#[serial]
fn test_gpu_dispatch_fused_qkv_q4_0_matches_cpu_reference() {
    require_gpu!();

    let caps = detect().expect("GPU required for fused QKV test");
    let gpu_quant =
        GpuQuant::new(GpuDevice::init(caps.device_id).expect("Failed to initialize GPU"))
            .expect("Failed to initialize GPU quantization");
    let device = gpu_quant.device();

    let n_rows = 64usize;
    let q_size = 64usize;
    let kv_size = 32usize;

    let input: Vec<f32> = (0..n_rows)
        .map(|i| ((i as f32) * 0.17).sin() * 0.75 + ((i as f32) * 0.07).cos() * 0.15)
        .collect();
    let q_bias: Vec<f32> = (0..q_size)
        .map(|i| ((i as f32) * 0.09).sin() * 0.03)
        .collect();
    let k_bias: Vec<f32> = (0..kv_size)
        .map(|i| ((i as f32) * 0.13).cos() * 0.02)
        .collect();
    let v_bias: Vec<f32> = (0..kv_size)
        .map(|i| ((i as f32) * 0.11).sin() * -0.025)
        .collect();

    let q_weights: Vec<f32> = (0..q_size)
        .flat_map(|col| {
            (0..n_rows).map(move |row| {
                let phase = (col as f32) * 0.031 + (row as f32) * 0.017;
                phase.sin() * 0.55 + phase.cos() * 0.20
            })
        })
        .collect();
    let k_weights: Vec<f32> = (0..kv_size)
        .flat_map(|col| {
            (0..n_rows).map(move |row| {
                let phase = (col as f32) * 0.041 + (row as f32) * 0.019;
                phase.cos() * 0.50 - phase.sin() * 0.18
            })
        })
        .collect();
    let v_weights: Vec<f32> = (0..kv_size)
        .flat_map(|col| {
            (0..n_rows).map(move |row| {
                let phase = (col as f32) * 0.027 + (row as f32) * 0.023;
                phase.sin() * 0.45 + ((col + row) as f32 * 0.015).cos() * 0.22
            })
        })
        .collect();

    let q_meta = WeightMeta {
        wtype: GgmlType::Q4_0,
        dims: vec![q_size as u64, n_rows as u64],
        needs_transpose: false,
        role: TensorRole::Generic,
        svd_k: None,
    };
    let kv_meta = WeightMeta {
        wtype: GgmlType::Q4_0,
        dims: vec![kv_size as u64, n_rows as u64],
        needs_transpose: false,
        role: TensorRole::Generic,
        svd_k: None,
    };

    let d_input = upload_f32(&input).expect("Upload input");
    let d_q_bias = upload_f32(&q_bias).expect("Upload q bias");
    let d_k_bias = upload_f32(&k_bias).expect("Upload k bias");
    let d_v_bias = upload_f32(&v_bias).expect("Upload v bias");
    let d_q_weights =
        quantize_q4_0_columns(&gpu_quant, &q_weights, n_rows, q_size).expect("Quantize q weights");
    let d_k_weights =
        quantize_q4_0_columns(&gpu_quant, &k_weights, n_rows, kv_size).expect("Quantize k weights");
    let d_v_weights =
        quantize_q4_0_columns(&gpu_quant, &v_weights, n_rows, kv_size).expect("Quantize v weights");
    let d_out_q = GpuBuffer::alloc(q_size * std::mem::size_of::<f32>()).expect("Alloc q output");
    let d_out_k = GpuBuffer::alloc(kv_size * std::mem::size_of::<f32>()).expect("Alloc k output");
    let d_out_v = GpuBuffer::alloc(kv_size * std::mem::size_of::<f32>()).expect("Alloc v output");

    gpu_dispatch_fused_qkv(
        device,
        &d_q_weights,
        &q_meta,
        Some(&d_q_bias),
        &d_k_weights,
        &kv_meta,
        Some(&d_k_bias),
        &d_v_weights,
        &kv_meta,
        Some(&d_v_bias),
        d_input.as_ptr() as *const f32,
        d_out_q.as_ptr() as *mut f32,
        d_out_k.as_ptr() as *mut f32,
        d_out_v.as_ptr() as *mut f32,
        q_size,
        kv_size,
        n_rows,
    )
    .expect("Dispatch fused QKV");
    device.synchronize().expect("Synchronize fused QKV");

    let q_quantized = download_u8(&d_q_weights, (n_rows / QK4_0) * q_size * Q4_0_BLOCK_SIZE)
        .expect("Download q weights");
    let k_quantized = download_u8(&d_k_weights, (n_rows / QK4_0) * kv_size * Q4_0_BLOCK_SIZE)
        .expect("Download k weights");
    let v_quantized = download_u8(&d_v_weights, (n_rows / QK4_0) * kv_size * Q4_0_BLOCK_SIZE)
        .expect("Download v weights");

    let expected_q = cpu_q4_0_projection(&q_quantized, &q_bias, &input, n_rows, q_size);
    let expected_k = cpu_q4_0_projection(&k_quantized, &k_bias, &input, n_rows, kv_size);
    let expected_v = cpu_q4_0_projection(&v_quantized, &v_bias, &input, n_rows, kv_size);

    let actual_q = download_f32(&d_out_q, q_size).expect("Download q output");
    let actual_k = download_f32(&d_out_k, kv_size).expect("Download k output");
    let actual_v = download_f32(&d_out_v, kv_size).expect("Download v output");

    let q_err = max_abs_error(&expected_q, &actual_q);
    let k_err = max_abs_error(&expected_k, &actual_k);
    let v_err = max_abs_error(&expected_v, &actual_v);

    assert!(
        q_err <= 1e-3,
        "Q projection mismatch: max_abs_error={}",
        q_err
    );
    assert!(
        k_err <= 1e-3,
        "K projection mismatch: max_abs_error={}",
        k_err
    );
    assert!(
        v_err <= 1e-3,
        "V projection mismatch: max_abs_error={}",
        v_err
    );
}

#[test]
#[serial]
fn test_gpu_dispatch_fused_qkv_real_layer0_matches_cpu_reference() {
    if !std::path::Path::new(REAL_Q4K_MODEL).exists() {
        eprintln!("Skipping test: model file not found at {}", REAL_Q4K_MODEL);
        return;
    }

    require_gpu!();

    let gguf = GgufFile::open(REAL_Q4K_MODEL).expect("open real model");
    let config = rocmforge::config::ModelConfig::from_gguf(&gguf).expect("config");
    let cpu_weights = CpuModelWeights::load(&gguf, &config).expect("cpu weights");
    let layer = cpu_weights.layer(0);

    let q_weight = gguf
        .tensor("blk.0.attn_q.weight")
        .expect("lookup q weight")
        .expect("q weight");
    let k_weight = gguf
        .tensor("blk.0.attn_k.weight")
        .expect("lookup k weight")
        .expect("k weight");
    let v_weight = gguf
        .tensor("blk.0.attn_v.weight")
        .expect("lookup v weight")
        .expect("v weight");

    assert_eq!(q_weight.ggml_type, GgmlType::Q5_0);
    assert_eq!(k_weight.ggml_type, GgmlType::Q5_0);
    assert_eq!(v_weight.ggml_type, GgmlType::Q8_0);

    let n_rows = q_weight.dims[0] as usize;
    let q_size = q_weight.dims[1] as usize;
    let kv_size = k_weight.dims[1] as usize;

    let token_id = 1000u32;
    let mut hidden = vec![0.0f32; config.hidden_size];
    cpu_embed_token(token_id, &cpu_weights, &mut hidden, &config, None);
    let mut input = vec![0.0f32; config.hidden_size];
    rms_norm(&hidden, &layer.attn_norm, &mut input, config.rms_norm_eps);

    let q_meta = WeightMeta {
        wtype: GgmlType::Q5_0,
        dims: q_weight.dims.to_vec(),
        needs_transpose: false,
        role: TensorRole::Generic,
        svd_k: None,
    };
    let k_meta = WeightMeta {
        wtype: GgmlType::Q5_0,
        dims: k_weight.dims.to_vec(),
        needs_transpose: false,
        role: TensorRole::Generic,
        svd_k: None,
    };
    let v_meta = WeightMeta {
        wtype: GgmlType::Q8_0,
        dims: v_weight.dims.to_vec(),
        needs_transpose: false,
        role: TensorRole::Generic,
        svd_k: None,
    };

    let caps = detect().expect("GPU required for real fused QKV test");
    let device = GpuDevice::init(caps.device_id).expect("init GPU");

    let d_input = upload_f32(&input).expect("upload input");
    let mut d_q_weight =
        GpuBuffer::alloc_for_device(q_weight.data.len(), caps.device_id).expect("alloc q");
    d_q_weight.copy_from_host(q_weight.data).expect("upload q");
    let mut d_k_weight =
        GpuBuffer::alloc_for_device(k_weight.data.len(), caps.device_id).expect("alloc k");
    d_k_weight.copy_from_host(k_weight.data).expect("upload k");
    let mut d_v_weight =
        GpuBuffer::alloc_for_device(v_weight.data.len(), caps.device_id).expect("alloc v");
    d_v_weight.copy_from_host(v_weight.data).expect("upload v");

    let d_q_bias = upload_f32(layer.attn_q_bias.as_ref().expect("q bias")).expect("upload q bias");
    let d_k_bias = upload_f32(layer.attn_k_bias.as_ref().expect("k bias")).expect("upload k bias");
    let d_v_bias = upload_f32(layer.attn_v_bias.as_ref().expect("v bias")).expect("upload v bias");

    let d_out_q = GpuBuffer::alloc(q_size * std::mem::size_of::<f32>()).expect("alloc q out");
    let d_out_k = GpuBuffer::alloc(kv_size * std::mem::size_of::<f32>()).expect("alloc k out");
    let d_out_v = GpuBuffer::alloc(kv_size * std::mem::size_of::<f32>()).expect("alloc v out");

    rocmforge::gpu::ops::gpu_dispatch_fused_qkv_on_stream(
        &device,
        &d_q_weight,
        &q_meta,
        None,
        Some(&d_q_bias),
        &d_k_weight,
        &k_meta,
        None,
        Some(&d_k_bias),
        &d_v_weight,
        &v_meta,
        None,
        Some(&d_v_bias),
        d_input.as_ptr() as *const f32,
        d_out_q.as_ptr() as *mut f32,
        d_out_k.as_ptr() as *mut f32,
        d_out_v.as_ptr() as *mut f32,
        q_size,
        kv_size,
        n_rows,
        std::ptr::null_mut(),
        device.stream(),
    )
    .expect("dispatch fused qkv");
    device.synchronize().expect("sync");

    let actual_q = download_f32(&d_out_q, q_size).expect("download q");
    let actual_k = download_f32(&d_out_k, kv_size).expect("download k");
    let actual_v = download_f32(&d_out_v, kv_size).expect("download v");

    let expected_q = cpu_quant_projection_with_bias(
        q_weight.data,
        GgmlType::Q5_0,
        layer.attn_q_bias.as_ref().expect("q bias"),
        &input,
        n_rows,
        q_size,
    );
    let expected_k = cpu_quant_projection_with_bias(
        k_weight.data,
        GgmlType::Q5_0,
        layer.attn_k_bias.as_ref().expect("k bias"),
        &input,
        n_rows,
        kv_size,
    );
    let expected_v = cpu_quant_projection_with_bias(
        v_weight.data,
        GgmlType::Q8_0,
        layer.attn_v_bias.as_ref().expect("v bias"),
        &input,
        n_rows,
        kv_size,
    );

    let q_err = max_abs_error(&expected_q, &actual_q);
    let k_err = max_abs_error(&expected_k, &actual_k);
    let v_err = max_abs_error(&expected_v, &actual_v);

    assert!(q_err <= 1e-3, "real fused Q err={}", q_err);
    assert!(k_err <= 1e-3, "real fused K err={}", k_err);
    assert!(v_err <= 1e-3, "real fused V err={}", v_err);
}

#[test]
#[serial]
fn test_gpu_dispatch_fused_gate_up_real_layer0_matches_cpu_reference() {
    if !std::path::Path::new(REAL_Q4K_MODEL).exists() {
        eprintln!("Skipping test: model file not found at {}", REAL_Q4K_MODEL);
        return;
    }

    require_gpu!();

    let gguf = GgufFile::open(REAL_Q4K_MODEL).expect("open real model");
    let config = rocmforge::config::ModelConfig::from_gguf(&gguf).expect("config");
    let cpu_weights = CpuModelWeights::load(&gguf, &config).expect("cpu weights");
    let layer = cpu_weights.layer(0);

    let token_id = 1000u32;
    let mut hidden = vec![0.0f32; config.hidden_size];
    cpu_embed_token(token_id, &cpu_weights, &mut hidden, &config, None);

    let half = config.head_dim / 2;
    let mut cpu_scratch = CpuForwardScratch::new(&config);
    for i in 0..half {
        let angle = 0.0f32 * config.rope_freq[i];
        let (s, c) = angle.sin_cos();
        cpu_scratch.rope_sin[i] = s;
        cpu_scratch.rope_cos[i] = c;
    }
    let rope_sin = unsafe { std::slice::from_raw_parts(cpu_scratch.rope_sin.as_ptr(), half) };
    let rope_cos = unsafe { std::slice::from_raw_parts(cpu_scratch.rope_cos.as_ptr(), half) };
    let mut cpu_kv = CpuKvCache::new(&config, 1);
    cpu_layer_forward(
        &mut hidden,
        layer,
        &mut cpu_kv,
        &mut cpu_scratch,
        0,
        0,
        rope_sin,
        rope_cos,
        &config,
        false,
    )
    .expect("cpu layer forward");

    let gate_tensor = gguf
        .tensor("blk.0.ffn_gate.weight")
        .expect("lookup gate weight")
        .expect("gate weight");
    let up_tensor = gguf
        .tensor("blk.0.ffn_up.weight")
        .expect("lookup up weight")
        .expect("up weight");
    assert_eq!(gate_tensor.ggml_type, GgmlType::Q5_0);
    assert_eq!(up_tensor.ggml_type, GgmlType::Q5_0);
    let ff_size = config.intermediate_size;
    let h = config.hidden_size;
    let ffn_input = &cpu_scratch.normed[..h];

    let (expected_gate, expected_swiglu) = cpu_gate_up_reference(
        gate_tensor.data,
        gate_tensor.ggml_type,
        up_tensor.data,
        up_tensor.ggml_type,
        ffn_input,
        ff_size,
        h,
    );

    let caps = detect().expect("GPU required for real fused gate/up test");
    let device = GpuDevice::init(caps.device_id).expect("init GPU");
    let gate_gpu_meta = WeightMeta {
        wtype: gate_tensor.ggml_type,
        dims: gate_tensor.dims.to_vec(),
        needs_transpose: false,
        role: TensorRole::Generic,
        svd_k: None,
    };
    let up_gpu_meta = WeightMeta {
        wtype: up_tensor.ggml_type,
        dims: up_tensor.dims.to_vec(),
        needs_transpose: false,
        role: TensorRole::Generic,
        svd_k: None,
    };

    let d_input = upload_f32(ffn_input).expect("upload ffn input");
    let mut d_gate_weight =
        GpuBuffer::alloc_for_device(gate_tensor.data.len(), caps.device_id).expect("alloc gate");
    d_gate_weight
        .copy_from_host(gate_tensor.data)
        .expect("upload gate");
    let mut d_up_weight =
        GpuBuffer::alloc_for_device(up_tensor.data.len(), caps.device_id).expect("alloc up");
    d_up_weight
        .copy_from_host(up_tensor.data)
        .expect("upload up");
    let d_gate = GpuBuffer::alloc(ff_size * std::mem::size_of::<f32>()).expect("alloc gate out");
    let d_swiglu =
        GpuBuffer::alloc(ff_size * std::mem::size_of::<f32>()).expect("alloc swiglu out");

    rocmforge::gpu::ops::gpu_dispatch_fused_gate_up_on_stream(
        &device,
        &d_gate_weight,
        &gate_gpu_meta,
        &d_up_weight,
        &up_gpu_meta,
        None,
        None,
        d_input.as_ptr() as *const f32,
        d_gate.as_ptr() as *mut f32,
        d_swiglu.as_ptr() as *mut f32,
        ff_size,
        h,
        device.stream(),
        None, // config
    )
    .expect("dispatch fused gate/up");
    device.synchronize().expect("sync");

    let actual_gate = download_f32(&d_gate, ff_size).expect("download gate");
    let actual_swiglu = download_f32(&d_swiglu, ff_size).expect("download swiglu");
    let expected_gate_silu: Vec<f32> = expected_gate
        .iter()
        .map(|&x| x / (1.0 + (-x).exp()))
        .collect();
    let gate_err = max_abs_error(&expected_gate_silu, &actual_gate);
    let swiglu_err = max_abs_error(&expected_swiglu, &actual_swiglu);

    assert!(gate_err <= 1e-5, "real fused gate scratch err={}", gate_err);
    assert!(swiglu_err <= 1e-3, "real fused swiglu err={}", swiglu_err);
}

#[test]
#[serial]
fn test_gpu_dispatch_ffn_down_real_layer0_matches_cpu_reference() {
    if !std::path::Path::new(REAL_Q4K_MODEL).exists() {
        eprintln!("Skipping test: model file not found at {}", REAL_Q4K_MODEL);
        return;
    }

    require_gpu!();

    let gguf = GgufFile::open(REAL_Q4K_MODEL).expect("open real model");
    let config = rocmforge::config::ModelConfig::from_gguf(&gguf).expect("config");
    let cpu_weights = CpuModelWeights::load(&gguf, &config).expect("cpu weights");
    let layer = cpu_weights.layer(0);

    let token_id = 1000u32;
    let mut hidden = vec![0.0f32; config.hidden_size];
    cpu_embed_token(token_id, &cpu_weights, &mut hidden, &config, None);

    let half = config.head_dim / 2;
    let mut cpu_scratch = CpuForwardScratch::new(&config);
    for i in 0..half {
        let angle = 0.0f32 * config.rope_freq[i];
        let (s, c) = angle.sin_cos();
        cpu_scratch.rope_sin[i] = s;
        cpu_scratch.rope_cos[i] = c;
    }
    let rope_sin = unsafe { std::slice::from_raw_parts(cpu_scratch.rope_sin.as_ptr(), half) };
    let rope_cos = unsafe { std::slice::from_raw_parts(cpu_scratch.rope_cos.as_ptr(), half) };
    let mut cpu_kv = CpuKvCache::new(&config, 1);
    cpu_layer_forward(
        &mut hidden,
        layer,
        &mut cpu_kv,
        &mut cpu_scratch,
        0,
        0,
        rope_sin,
        rope_cos,
        &config,
        false,
    )
    .expect("cpu layer forward");

    let down_tensor = gguf
        .tensor("blk.0.ffn_down.weight")
        .expect("lookup down weight")
        .expect("down weight");
    let ff_size = config.intermediate_size;
    let h = config.hidden_size;
    let swiglu = &cpu_scratch.swiglu[..ff_size];
    let down_cpu_meta = CpuWeightMeta::from_view(&down_tensor, false);
    let expected = cpu_dispatch_projection(down_tensor.data, &down_cpu_meta, swiglu, h, ff_size);

    let caps = detect().expect("GPU required for real ffn_down test");
    let device = GpuDevice::init(caps.device_id).expect("init GPU");
    let down_gpu_meta = WeightMeta {
        wtype: down_tensor.ggml_type,
        dims: down_tensor.dims.to_vec(),
        needs_transpose: false,
        role: TensorRole::Generic,
        svd_k: None,
    };

    let d_input = upload_f32(swiglu).expect("upload swiglu input");
    let mut d_down_weight =
        GpuBuffer::alloc_for_device(down_tensor.data.len(), caps.device_id).expect("alloc down");
    d_down_weight
        .copy_from_host(down_tensor.data)
        .expect("upload down");
    let d_output = GpuBuffer::alloc(h * std::mem::size_of::<f32>()).expect("alloc down out");

    rocmforge::gpu::ops::gpu_dispatch_gemv_on_stream(
        &device,
        &d_down_weight,
        &down_gpu_meta,
        d_input.as_ptr() as *const f32,
        d_output.as_ptr() as *mut f32,
        h,
        ff_size,
        device.stream(),
    )
    .expect("dispatch ffn_down");
    device.synchronize().expect("sync");

    let actual = download_f32(&d_output, h).expect("download down");
    let err = max_abs_error(&expected, &actual);
    assert!(err <= 1e-3, "real ffn_down err={}", err);
}

#[test]
#[serial]
fn test_gpu_loaded_layer0_qkv_matches_cpu_reference() {
    if !std::path::Path::new(REAL_Q4K_MODEL).exists() {
        eprintln!("Skipping test: model file not found at {}", REAL_Q4K_MODEL);
        return;
    }

    require_gpu!();

    let gguf = GgufFile::open(REAL_Q4K_MODEL).expect("open real model");
    let config = rocmforge::config::ModelConfig::from_gguf(&gguf).expect("config");
    let cpu_weights = CpuModelWeights::load(&gguf, &config).expect("cpu weights");
    let gpu_weights = GpuModelWeights::load(&gguf, &config).expect("gpu weights");
    let cpu_layer = cpu_weights.layer(0);
    let gpu_layer = gpu_weights.layer(0);

    let q_weight = gguf
        .tensor("blk.0.attn_q.weight")
        .expect("lookup q weight")
        .expect("q weight");
    let k_weight = gguf
        .tensor("blk.0.attn_k.weight")
        .expect("lookup k weight")
        .expect("k weight");
    let v_weight = gguf
        .tensor("blk.0.attn_v.weight")
        .expect("lookup v weight")
        .expect("v weight");

    let n_rows = q_weight.dims[0] as usize;
    let q_size = q_weight.dims[1] as usize;
    let kv_size = k_weight.dims[1] as usize;

    let token_id = 1000u32;
    let mut hidden = vec![0.0f32; config.hidden_size];
    cpu_embed_token(token_id, &cpu_weights, &mut hidden, &config, None);
    let mut input = vec![0.0f32; config.hidden_size];
    rms_norm(
        &hidden,
        &cpu_layer.attn_norm,
        &mut input,
        config.rms_norm_eps,
    );

    let caps = detect().expect("GPU required for loaded layer0 qkv test");
    let device = GpuDevice::init(caps.device_id).expect("init GPU");
    let d_input = upload_f32(&input).expect("upload input");
    let d_out_q = GpuBuffer::alloc(q_size * std::mem::size_of::<f32>()).expect("alloc q out");
    let d_out_k = GpuBuffer::alloc(kv_size * std::mem::size_of::<f32>()).expect("alloc k out");
    let d_out_v = GpuBuffer::alloc(kv_size * std::mem::size_of::<f32>()).expect("alloc v out");

    rocmforge::gpu::ops::gpu_dispatch_fused_qkv_on_stream(
        &device,
        &gpu_layer.attn_q,
        &gpu_layer.attn_q_meta,
        gpu_layer.attn_q_svd.as_ref(),
        gpu_layer.attn_q_bias.as_ref(),
        &gpu_layer.attn_k,
        &gpu_layer.attn_k_meta,
        gpu_layer.attn_k_svd.as_ref(),
        gpu_layer.attn_k_bias.as_ref(),
        &gpu_layer.attn_v,
        &gpu_layer.attn_v_meta,
        gpu_layer.attn_v_svd.as_ref(),
        gpu_layer.attn_v_bias.as_ref(),
        d_input.as_ptr() as *const f32,
        d_out_q.as_ptr() as *mut f32,
        d_out_k.as_ptr() as *mut f32,
        d_out_v.as_ptr() as *mut f32,
        q_size,
        kv_size,
        n_rows,
        std::ptr::null_mut(),
        device.stream(),
    )
    .expect("dispatch loaded fused qkv");
    device.synchronize().expect("sync");

    let actual_q = download_f32(&d_out_q, q_size).expect("download q");
    let actual_k = download_f32(&d_out_k, kv_size).expect("download k");
    let actual_v = download_f32(&d_out_v, kv_size).expect("download v");

    let expected_q = cpu_quant_projection_with_bias(
        q_weight.data,
        GgmlType::Q5_0,
        cpu_layer.attn_q_bias.as_ref().expect("q bias"),
        &input,
        n_rows,
        q_size,
    );
    let expected_k = cpu_quant_projection_with_bias(
        k_weight.data,
        GgmlType::Q5_0,
        cpu_layer.attn_k_bias.as_ref().expect("k bias"),
        &input,
        n_rows,
        kv_size,
    );
    let expected_v = cpu_quant_projection_with_bias(
        v_weight.data,
        GgmlType::Q8_0,
        cpu_layer.attn_v_bias.as_ref().expect("v bias"),
        &input,
        n_rows,
        kv_size,
    );

    let q_err = max_abs_error(&expected_q, &actual_q);
    let k_err = max_abs_error(&expected_k, &actual_k);
    let v_err = max_abs_error(&expected_v, &actual_v);

    assert!(q_err <= 1e-3, "loaded layer0 Q err={}", q_err);
    assert!(k_err <= 1e-3, "loaded layer0 K err={}", k_err);
    assert!(v_err <= 1e-3, "loaded layer0 V err={}", v_err);
}

#[test]
#[serial]
fn test_gpu_loaded_layer0_attn_output_matches_cpu_reference() {
    if !std::path::Path::new(REAL_Q4K_MODEL).exists() {
        eprintln!("Skipping test: model file not found at {}", REAL_Q4K_MODEL);
        return;
    }

    require_gpu!();

    let gguf = GgufFile::open(REAL_Q4K_MODEL).expect("open real model");
    let config = rocmforge::config::ModelConfig::from_gguf(&gguf).expect("config");
    let cpu_weights = CpuModelWeights::load(&gguf, &config).expect("cpu weights");
    let gpu_weights = GpuModelWeights::load(&gguf, &config).expect("gpu weights");
    let layer = cpu_weights.layer(0);
    let gpu_layer = gpu_weights.layer(0);

    let token_id = 1000u32;
    let mut embed_hidden = vec![0.0f32; config.hidden_size];
    cpu_embed_token(token_id, &cpu_weights, &mut embed_hidden, &config, None);
    let mut final_hidden = embed_hidden.clone();

    let half = config.head_dim / 2;
    let mut cpu_scratch = CpuForwardScratch::new(&config);
    for i in 0..half {
        let angle = 0.0f32 * config.rope_freq[i];
        let (s, c) = angle.sin_cos();
        cpu_scratch.rope_sin[i] = s;
        cpu_scratch.rope_cos[i] = c;
    }
    let rope_sin = unsafe { std::slice::from_raw_parts(cpu_scratch.rope_sin.as_ptr(), half) };
    let rope_cos = unsafe { std::slice::from_raw_parts(cpu_scratch.rope_cos.as_ptr(), half) };
    let mut cpu_kv = CpuKvCache::new(&config, 1);
    cpu_layer_forward(
        &mut final_hidden,
        layer,
        &mut cpu_kv,
        &mut cpu_scratch,
        0,
        0,
        rope_sin,
        rope_cos,
        &config,
        false,
    )
    .expect("cpu layer forward");

    let attn_o_tensor = gguf
        .tensor("blk.0.attn_output.weight")
        .expect("lookup attn_o weight")
        .expect("attn_o weight");
    let h = config.hidden_size;
    let q_size = config.num_heads * config.head_dim;
    let expected = cpu_quant_projection(
        attn_o_tensor.data,
        GgmlType::Q5_0,
        &cpu_scratch.attn_out,
        q_size,
        h,
    );

    let caps = detect().expect("GPU required for loaded layer0 attn_o test");
    let device = GpuDevice::init(caps.device_id).expect("init GPU");
    let d_input = upload_f32(&cpu_scratch.attn_out).expect("upload attn_out");
    let d_output = GpuBuffer::alloc(h * std::mem::size_of::<f32>()).expect("alloc attn_o out");

    rocmforge::gpu::ops::gpu_dispatch_gemv_on_stream(
        &device,
        &gpu_layer.attn_o,
        &gpu_layer.attn_o_meta,
        d_input.as_ptr() as *const f32,
        d_output.as_ptr() as *mut f32,
        h,
        q_size,
        device.stream(),
    )
    .expect("dispatch attn_o");
    device.synchronize().expect("sync");

    let actual = download_f32(&d_output, h).expect("download attn_o");
    let err = max_abs_error(&expected, &actual);
    assert!(err <= 1e-3, "loaded layer0 attn_o err={}", err);
}

#[test]
#[serial]
fn test_gpu_loaded_layer0_full_forward_intermediates_match_cpu() {
    if !std::path::Path::new(REAL_Q4K_MODEL).exists() {
        eprintln!("Skipping test: model file not found at {}", REAL_Q4K_MODEL);
        return;
    }

    require_gpu!();

    let gguf = GgufFile::open(REAL_Q4K_MODEL).expect("open real model");
    let config = rocmforge::config::ModelConfig::from_gguf(&gguf).expect("config");
    let cpu_weights = CpuModelWeights::load(&gguf, &config).expect("cpu weights");
    let gpu_weights = GpuModelWeights::load(&gguf, &config).expect("gpu weights");

    let token_id = 1000u32;
    let mut cpu_hidden = vec![0.0f32; config.hidden_size];
    cpu_embed_token(token_id, &cpu_weights, &mut cpu_hidden, &config, None);

    let half = config.head_dim / 2;
    let mut cpu_kv = CpuKvCache::new(&config, 1);
    let mut cpu_scratch = CpuForwardScratch::new(&config);
    for i in 0..half {
        let angle = 0.0f32 * config.rope_freq[i];
        let (s, c) = angle.sin_cos();
        cpu_scratch.rope_sin[i] = s;
        cpu_scratch.rope_cos[i] = c;
    }
    let rope_sin = unsafe { std::slice::from_raw_parts(cpu_scratch.rope_sin.as_ptr(), half) };
    let rope_cos = unsafe { std::slice::from_raw_parts(cpu_scratch.rope_cos.as_ptr(), half) };
    cpu_layer_forward(
        &mut cpu_hidden,
        cpu_weights.layer(0),
        &mut cpu_kv,
        &mut cpu_scratch,
        0,
        0,
        rope_sin,
        rope_cos,
        &config,
        false,
    )
    .expect("cpu layer0 forward");

    let caps = detect().expect("GPU required for full intermediate test");
    let device = GpuDevice::init(caps.device_id).expect("init GPU");
    let mut gpu_kv = rocmforge::gpu::GpuKvCache::new(&config, 1).expect("gpu kv");
    let mut gpu_scratch = rocmforge::gpu::GpuForwardScratch::new(&config).expect("gpu scratch");
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
    rocmforge::gpu::gpu_layer_forward_hybrid(
        &device,
        gpu_weights.layer(0),
        Some(cpu_weights.layer(0)),
        &mut gpu_kv,
        &mut gpu_scratch,
        Some(&mut host_scratch),
        0,
        0,
        token_id,
        &config,
        None, // shared_ple_token_emb
        None, // shared_ple_model_proj
        None, // shared_ple_proj_norm
    )
    .expect("gpu layer0 forward");
    device.synchronize().expect("sync");

    let q_size = config.num_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;
    let gpu_q = download_f32(&gpu_scratch.q, q_size).expect("download q");
    let gpu_k = download_f32(&gpu_scratch.k, kv_size).expect("download k");
    let gpu_v = download_f32(&gpu_scratch.v, kv_size).expect("download v");
    let gpu_attn_out = download_f32(&gpu_scratch.attn_out, q_size).expect("download attn_out");
    let gpu_swiglu =
        download_f32(&gpu_scratch.swiglu, config.intermediate_size).expect("download swiglu");
    let gpu_hidden =
        download_f32(&gpu_scratch.hidden, config.hidden_size).expect("download hidden");

    let q_err = max_abs_error(&cpu_scratch.q[..q_size], &gpu_q);
    let k_err = max_abs_error(&cpu_scratch.k[..kv_size], &gpu_k);
    let v_err = max_abs_error(&cpu_scratch.v[..kv_size], &gpu_v);
    let attn_out_err = max_abs_error(&cpu_scratch.attn_out[..q_size], &gpu_attn_out);
    let swiglu_err = max_abs_error(&cpu_scratch.swiglu[..config.intermediate_size], &gpu_swiglu);
    let hidden_err = max_abs_error(&cpu_hidden, &gpu_hidden);

    eprintln!(
        "q_err={:.6} k_err={:.6} v_err={:.6} attn_out_err={:.6} swiglu_err={:.6} hidden_err={:.6}",
        q_err, k_err, v_err, attn_out_err, swiglu_err, hidden_err
    );

    assert!(q_err <= 1e-3, "layer0 q err={}", q_err);
    assert!(k_err <= 1e-3, "layer0 k err={}", k_err);
    assert!(v_err <= 1e-3, "layer0 v err={}", v_err);
    assert!(attn_out_err <= 1e-3, "layer0 attn_out err={}", attn_out_err);
    assert!(swiglu_err <= 1e-3, "layer0 swiglu err={}", swiglu_err);
    assert!(hidden_err <= 1e-3, "layer0 hidden err={}", hidden_err);
}
