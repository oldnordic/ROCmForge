#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use rocmforge::cpu::ops::gemv_q8_0_transposed;
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

#[test]
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
