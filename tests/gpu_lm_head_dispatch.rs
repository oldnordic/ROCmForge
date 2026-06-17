#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use rocmforge::cpu::ops::{gemv_q8_0, gemv_q8_0_transposed};
use rocmforge::cpu::quant::quantize_f32_to_q8_0;
use rocmforge::gpu::{
    detect, gpu_dispatch_gemv, GpuBuffer, GpuDevice, GpuQuant, TensorRole, WeightMeta,
    Q8_0_BLOCK_SIZE, QK8_0,
};
use rocmforge::loader::GgmlType;
use serial_test::serial;

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

fn quantize_q8_0_columns(
    gpu_quant: &GpuQuant,
    weights: &[f32],
    n_rows: usize,
    n_cols: usize,
) -> rocmforge::gpu::GpuResult<GpuBuffer> {
    let d_weights = upload_f32(weights)?;
    let d_quantized = GpuBuffer::alloc((n_rows / QK8_0) * n_cols * Q8_0_BLOCK_SIZE)?;

    for col in 0..n_cols {
        let col_weights_ptr = unsafe {
            d_weights
                .as_ptr()
                .add(col * n_rows * std::mem::size_of::<f32>())
        };
        let col_quantized_ptr = unsafe {
            d_quantized
                .as_ptr()
                .add(col * (n_rows / QK8_0) * Q8_0_BLOCK_SIZE)
        };
        gpu_quant.quantize_q8_0(col_weights_ptr as *const f32, col_quantized_ptr, n_rows)?;
    }

    Ok(d_quantized)
}

fn max_abs_error(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn build_quantized_q8_lm_head(out_dim: usize, in_dim: usize) -> Vec<u8> {
    assert_eq!(in_dim % QK8_0, 0, "in_dim must be block-aligned");
    let n_blocks = in_dim / QK8_0;
    let mut weights = vec![0u8; out_dim * n_blocks * Q8_0_BLOCK_SIZE];
    let mut block_src = [0.0f32; QK8_0];

    for col in 0..out_dim {
        for block_idx in 0..n_blocks {
            for (lane, block_src_value) in block_src.iter_mut().enumerate().take(QK8_0) {
                let row = block_idx * QK8_0 + lane;
                let phase = (col as f32) * 0.000013 + (row as f32) * 0.0017;
                *block_src_value = phase.cos() * 0.45 + phase.sin() * 0.14;
            }

            let dst_offset = (col * n_blocks + block_idx) * Q8_0_BLOCK_SIZE;
            let dst_block = &mut weights[dst_offset..dst_offset + Q8_0_BLOCK_SIZE];
            let scale = quantize_f32_to_q8_0(&block_src, &mut dst_block[2..34]);
            dst_block[0..2].copy_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
        }
    }

    weights
}

#[test]
#[serial]
fn test_gpu_dispatch_q8_0_lm_head_multirow_outputs_finite_and_matches_cpu() {
    require_gpu!();

    let caps = detect().expect("GPU required for LM-head multi-row dispatch test");
    let gpu_quant =
        GpuQuant::new(GpuDevice::init(caps.device_id).expect("Failed to initialize GPU"))
            .expect("Failed to initialize GPU quantization");
    let device = gpu_quant.device();

    let in_dim = 2048usize;
    let out_dim = 8192usize;

    let weights: Vec<f32> = (0..out_dim)
        .flat_map(|col| {
            (0..in_dim).map(move |row| {
                let phase = (col as f32) * 0.00031 + (row as f32) * 0.0017;
                phase.cos() * 0.45 + phase.sin() * 0.14
            })
        })
        .collect();
    let input: Vec<f32> = (0..in_dim)
        .map(|row| {
            let phase = (row as f32) * 0.013;
            phase.sin() * 0.38 - phase.cos() * 0.11
        })
        .collect();

    let meta = WeightMeta {
        wtype: GgmlType::Q8_0,
        dims: vec![out_dim as u64, in_dim as u64],
        needs_transpose: false,
        role: TensorRole::LmHead,
        svd_k: None,
    };

    let d_weights =
        quantize_q8_0_columns(&gpu_quant, &weights, in_dim, out_dim).expect("Quantize weights");
    let d_input = upload_f32(&input).expect("Upload input");
    let d_output = GpuBuffer::alloc(out_dim * std::mem::size_of::<f32>()).expect("Alloc output");

    gpu_dispatch_gemv(
        device,
        &d_weights,
        &meta,
        d_input.as_ptr() as *const f32,
        d_output.as_ptr() as *mut f32,
        out_dim,
        in_dim,
    )
    .expect("Dispatch LM-head GEMV");
    device.synchronize().expect("Synchronize LM-head GEMV");

    let quantized =
        download_u8(&d_weights, (in_dim / QK8_0) * out_dim * Q8_0_BLOCK_SIZE).expect("Weights");
    let actual = download_f32(&d_output, out_dim).expect("Outputs");

    assert!(
        actual.iter().all(|value| value.is_finite()),
        "LM-head multi-row dispatch produced non-finite outputs"
    );

    let mut expected = vec![0.0f32; out_dim];
    gemv_q8_0(&quantized, &input, &mut expected, out_dim, in_dim);

    let err = max_abs_error(&expected, &actual);
    assert!(
        err <= 1e-3,
        "LM-head multi-row dispatch mismatch: max_abs_error={}",
        err
    );
}

#[test]
#[serial]
fn test_gpu_dispatch_q8_0_lm_head_llama_vocab_outputs_finite_and_matches_cpu() {
    require_gpu!();

    let caps = detect().expect("GPU required for wide LM-head dispatch test");
    let device = GpuDevice::init(caps.device_id).expect("Failed to initialize GPU");

    let in_dim = 2048usize;
    let out_dim = 128_256usize;

    let input: Vec<f32> = (0..in_dim)
        .map(|row| {
            let phase = (row as f32) * 0.013;
            phase.sin() * 0.38 - phase.cos() * 0.11
        })
        .collect();

    let meta = WeightMeta {
        wtype: GgmlType::Q8_0,
        dims: vec![out_dim as u64, in_dim as u64],
        needs_transpose: false,
        role: TensorRole::LmHead,
        svd_k: None,
    };

    let quantized = build_quantized_q8_lm_head(out_dim, in_dim);
    let mut d_weights = GpuBuffer::alloc(quantized.len()).expect("Alloc weights");
    d_weights
        .copy_from_host(&quantized)
        .expect("Upload weights");
    let d_input = upload_f32(&input).expect("Upload input");
    let d_output = GpuBuffer::alloc(out_dim * std::mem::size_of::<f32>()).expect("Alloc output");

    gpu_dispatch_gemv(
        &device,
        &d_weights,
        &meta,
        d_input.as_ptr() as *const f32,
        d_output.as_ptr() as *mut f32,
        out_dim,
        in_dim,
    )
    .expect("Dispatch wide LM-head GEMV");
    device.synchronize().expect("Synchronize wide LM-head GEMV");

    let actual = download_f32(&d_output, out_dim).expect("Outputs");
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "Wide LM-head dispatch produced non-finite outputs"
    );

    let mut expected = vec![0.0f32; out_dim];
    gemv_q8_0(&quantized, &input, &mut expected, out_dim, in_dim);

    let err = max_abs_error(&expected, &actual);
    assert!(
        err <= 1e-3,
        "Wide LM-head dispatch mismatch: max_abs_error={}",
        err
    );
}

#[test]
#[ignore = "Transposed Tied LM Head not supported on GPU"]
#[serial]
fn test_gpu_dispatch_gemv_tied_q8_0_lm_head_matches_cpu_reference() {
    require_gpu!();

    let caps = detect().expect("GPU required for tied LM-head dispatch test");
    let gpu_quant =
        GpuQuant::new(GpuDevice::init(caps.device_id).expect("Failed to initialize GPU"))
            .expect("Failed to initialize GPU quantization");
    let device = gpu_quant.device();

    let in_dim = 128usize;
    let out_dim = 96usize;

    let weights: Vec<f32> = (0..out_dim)
        .flat_map(|col| {
            (0..in_dim).map(move |row| {
                let phase = (col as f32) * 0.031 + (row as f32) * 0.017;
                phase.cos() * 0.45 + phase.sin() * 0.14
            })
        })
        .collect();
    let input: Vec<f32> = (0..in_dim)
        .map(|row| {
            let phase = (row as f32) * 0.029;
            phase.sin() * 0.38 - phase.cos() * 0.11
        })
        .collect();

    let meta = WeightMeta {
        wtype: GgmlType::Q8_0,
        dims: vec![in_dim as u64, out_dim as u64],
        needs_transpose: true,
        role: TensorRole::TiedLmHead,
        svd_k: None,
    };

    let d_weights =
        quantize_q8_0_columns(&gpu_quant, &weights, in_dim, out_dim).expect("Quantize weights");
    let d_input = upload_f32(&input).expect("Upload input");
    let d_output = GpuBuffer::alloc(out_dim * std::mem::size_of::<f32>()).expect("Alloc output");

    gpu_dispatch_gemv(
        device,
        &d_weights,
        &meta,
        d_input.as_ptr() as *const f32,
        d_output.as_ptr() as *mut f32,
        out_dim,
        in_dim,
    )
    .expect("Dispatch tied LM-head GEMV");
    device.synchronize().expect("Synchronize tied LM-head GEMV");

    let quantized =
        download_u8(&d_weights, (in_dim / QK8_0) * out_dim * Q8_0_BLOCK_SIZE).expect("Weights");
    let actual = download_f32(&d_output, out_dim).expect("Outputs");

    let mut expected = vec![0.0f32; out_dim];
    gemv_q8_0_transposed(&quantized, &input, &mut expected, out_dim, in_dim);

    let err = max_abs_error(&expected, &actual);
    assert!(
        err <= 1e-3,
        "Tied LM-head dispatch mismatch: max_abs_error={}",
        err
    );
}
