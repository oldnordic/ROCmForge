#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use rocmforge::cpu::ops::gemv_q4_0_transposed;
use rocmforge::gpu::{
    detect, gpu_dispatch_fused_qkv, GpuBuffer, GpuDevice, GpuQuant, TensorRole, WeightMeta,
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
    };
    let kv_meta = WeightMeta {
        wtype: GgmlType::Q4_0,
        dims: vec![kv_size as u64, n_rows as u64],
        needs_transpose: false,
        role: TensorRole::Generic,
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
