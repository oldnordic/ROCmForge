#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use rocmforge::cpu::ops::gemv_q4_0_transposed;
use rocmforge::gpu::{
    detect, gpu_dispatch_gemm, GpuBuffer, GpuDevice, GpuQuant, TensorRole, WeightMeta,
    Q4_0_BLOCK_SIZE, QK4_0,
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

#[test]
#[serial]
fn test_gpu_dispatch_gemm_q4_0_batched_matches_cpu_reference() {
    require_gpu!();

    let caps = detect().expect("GPU required for batched GEMM test");
    let gpu_quant =
        GpuQuant::new(GpuDevice::init(caps.device_id).expect("Failed to initialize GPU"))
            .expect("Failed to initialize GPU quantization");
    let device = gpu_quant.device();

    let n_rows = 64usize;
    let out_dim = 24usize;
    let batch_size = 5usize;

    let weights: Vec<f32> = (0..out_dim)
        .flat_map(|col| {
            (0..n_rows).map(move |row| {
                let phase = (col as f32) * 0.043 + (row as f32) * 0.019;
                phase.sin() * 0.60 + phase.cos() * 0.18
            })
        })
        .collect();
    let input: Vec<f32> = (0..batch_size)
        .flat_map(|batch| {
            (0..n_rows).map(move |row| {
                let phase = (batch as f32) * 0.071 + (row as f32) * 0.013;
                phase.cos() * 0.52 - phase.sin() * 0.16
            })
        })
        .collect();

    let meta = WeightMeta {
        wtype: GgmlType::Q4_0,
        dims: vec![out_dim as u64, n_rows as u64],
        needs_transpose: false,
        role: TensorRole::Generic,
    };

    let d_weights =
        quantize_q4_0_columns(&gpu_quant, &weights, n_rows, out_dim).expect("Quantize weights");
    let d_input = upload_f32(&input).expect("Upload batched inputs");
    let d_output =
        GpuBuffer::alloc(batch_size * out_dim * std::mem::size_of::<f32>()).expect("Alloc output");

    gpu_dispatch_gemm(
        device,
        &d_weights,
        &meta,
        d_input.as_ptr() as *const f32,
        d_output.as_ptr() as *mut f32,
        out_dim,
        n_rows,
        batch_size,
    )
    .expect("Dispatch batched GEMM");
    device.synchronize().expect("Synchronize batched GEMM");

    let quantized =
        download_u8(&d_weights, (n_rows / QK4_0) * out_dim * Q4_0_BLOCK_SIZE).expect("Weights");
    let actual = download_f32(&d_output, batch_size * out_dim).expect("Outputs");

    let mut expected = vec![0.0f32; batch_size * out_dim];
    for batch in 0..batch_size {
        let input_row = &input[batch * n_rows..(batch + 1) * n_rows];
        let output_row = &mut expected[batch * out_dim..(batch + 1) * out_dim];
        gemv_q4_0_transposed(&quantized, input_row, output_row, out_dim, n_rows);
    }

    let err = max_abs_error(&expected, &actual);
    assert!(err <= 1e-3, "Batched GEMM mismatch: max_abs_error={}", err);
}
