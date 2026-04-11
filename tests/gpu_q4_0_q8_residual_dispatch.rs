#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use rocmforge::cpu::quant::{load_f16_scale, Q4_BLOCK_BYTES, Q4_BLOCK_ELEMS, Q8_BLOCK_BYTES};
use rocmforge::gpu::{detect, GpuBuffer, GpuDevice, GpuQuant, WeightMeta};
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
    let d_quantized = GpuBuffer::alloc((n_rows / Q4_BLOCK_ELEMS) * n_cols * Q4_BLOCK_BYTES)?;

    for col in 0..n_cols {
        let col_weights_ptr = unsafe {
            d_weights
                .as_ptr()
                .add(col * n_rows * std::mem::size_of::<f32>())
        };
        let col_quantized_ptr = unsafe {
            d_quantized
                .as_ptr()
                .add(col * (n_rows / Q4_BLOCK_ELEMS) * Q4_BLOCK_BYTES)
        };
        gpu_quant.quantize_q4_0(col_weights_ptr as *const f32, col_quantized_ptr, n_rows)?;
    }

    Ok(d_quantized)
}

fn q4_0_q8_0_residual_cpu_oracle(
    weights: &[u8],
    input_q8: &[u8],
    residual: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let num_blocks = in_dim / Q4_BLOCK_ELEMS;
    let col_bytes = num_blocks * Q4_BLOCK_BYTES;
    assert_eq!(
        input_q8.len(),
        num_blocks * Q8_BLOCK_BYTES,
        "unexpected quantized activation size"
    );

    let mut out = vec![0.0f32; out_dim];
    for col in 0..out_dim {
        let mut acc = residual[col];
        let col_offset = col * col_bytes;

        for block_idx in 0..num_blocks {
            let w_block = &weights[col_offset + block_idx * Q4_BLOCK_BYTES
                ..col_offset + (block_idx + 1) * Q4_BLOCK_BYTES];
            let x_block = &input_q8[block_idx * Q8_BLOCK_BYTES..(block_idx + 1) * Q8_BLOCK_BYTES];
            let scale = load_f16_scale(&w_block[..2]) * load_f16_scale(&x_block[..2]);
            let qs = &w_block[2..18];
            let x_qs = &x_block[2..];

            let mut block_sum = 0i32;
            for i in 0..16 {
                let packed = qs[i];
                block_sum += (((packed & 0x0F) as i32) - 8) * ((x_qs[i] as i8) as i32);
                block_sum += (((packed >> 4) as i32) - 8) * ((x_qs[i + 16] as i8) as i32);
            }

            acc += scale * block_sum as f32;
        }

        out[col] = acc;
    }

    out
}

fn max_abs_error(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

#[test]
#[serial]
fn test_gpu_dispatch_q4_0_residual_uses_q8_activation_fastpath_and_matches_cpu_oracle() {
    require_gpu!();

    unsafe {
        std::env::set_var("ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH", "1");
    }
    rocmforge::gpu::refresh_runtime_env_flags();

    let caps = detect().expect("GPU required for Q4_0 residual dispatch test");
    let device = GpuDevice::init(caps.device_id).expect("Failed to initialize GPU device");
    let gpu_quant = GpuQuant::new(
        GpuDevice::init(caps.device_id).expect("Failed to initialize quantization device"),
    )
    .expect("Failed to initialize GpuQuant");

    let in_dim = 4096usize;
    let out_dim = 96usize;

    let weights: Vec<f32> = (0..out_dim)
        .flat_map(|col| {
            (0..in_dim).map(move |row| {
                let phase = (col as f32) * 0.013 + (row as f32) * 0.007;
                phase.sin() * 0.62 - phase.cos() * 0.18
            })
        })
        .collect();
    let input: Vec<f32> = (0..in_dim)
        .map(|row| {
            let phase = (row as f32) * 0.011;
            phase.cos() * 0.41 + phase.sin() * 0.09
        })
        .collect();
    let residual: Vec<f32> = (0..out_dim)
        .map(|col| ((col as f32) * 0.023).sin() * 0.37 - ((col as f32) * 0.017).cos() * 0.12)
        .collect();

    let meta = WeightMeta {
        wtype: GgmlType::Q4_0,
        dims: vec![in_dim as u64, out_dim as u64],
        needs_transpose: false,
        role: rocmforge::gpu::TensorRole::Generic,
    };

    let d_weights =
        quantize_q4_0_columns(&gpu_quant, &weights, in_dim, out_dim).expect("Quantize weights");
    let d_input = upload_f32(&input).expect("Upload input");
    let d_residual = upload_f32(&residual).expect("Upload residual");
    let d_output_direct =
        GpuBuffer::alloc(out_dim * std::mem::size_of::<f32>()).expect("Alloc direct output");
    let d_output_inline =
        GpuBuffer::alloc(out_dim * std::mem::size_of::<f32>()).expect("Alloc inline output");
    let d_output_dispatch =
        GpuBuffer::alloc(out_dim * std::mem::size_of::<f32>()).expect("Alloc dispatch output");
    let d_input_q8 = GpuBuffer::alloc(rocmforge::gpu::kernels::q8_0_workspace_bytes(in_dim))
        .expect("Alloc Q8 activation buffer");

    rocmforge::gpu::kernels::quantize_q8_0_on_stream(
        d_input.as_ptr() as *const f32,
        d_input_q8.as_ptr(),
        in_dim,
        device.stream(),
    )
    .expect("Quantize input to Q8_0 on stream");
    rocmforge::gpu::kernels::gemv_q4_0_q8_0_residual_on_stream(
        d_weights.as_ptr() as *const u8,
        d_input_q8.as_ptr() as *const u8,
        d_residual.as_ptr() as *const f32,
        d_output_direct.as_ptr() as *mut f32,
        in_dim,
        out_dim,
        device.stream(),
    )
    .expect("Direct Q4_0 x Q8_0 residual GEMV");
    rocmforge::gpu::kernels::gemv_q4_0_f32_q8_inline_residual_on_stream(
        d_weights.as_ptr() as *const u8,
        d_input.as_ptr() as *const f32,
        d_residual.as_ptr() as *const f32,
        d_output_inline.as_ptr() as *mut f32,
        in_dim,
        out_dim,
        device.stream(),
    )
    .expect("Direct inline-quantized Q4_0 residual GEMV");

    let fused = rocmforge::gpu::ops::gpu_dispatch_gemv_residual_on_stream(
        &device,
        &d_weights,
        &meta,
        d_input.as_ptr() as *const f32,
        d_residual.as_ptr() as *const f32,
        d_output_dispatch.as_ptr() as *mut f32,
        in_dim,
        out_dim,
        device.stream(),
    )
    .expect("Dispatch Q4_0 residual GEMV");
    assert!(
        fused,
        "Q4_0 residual dispatch should stay on the fused GPU path"
    );

    device
        .synchronize()
        .expect("Synchronize Q4_0 residual GEMV");

    let quantized = download_u8(
        &d_weights,
        (in_dim / Q4_BLOCK_ELEMS) * out_dim * Q4_BLOCK_BYTES,
    )
    .expect("Download weights");
    let input_q8 = download_u8(
        &d_input_q8,
        rocmforge::gpu::kernels::q8_0_workspace_bytes(in_dim),
    )
    .expect("Download quantized input");
    let actual_direct = download_f32(&d_output_direct, out_dim).expect("Download direct outputs");
    let actual_inline = download_f32(&d_output_inline, out_dim).expect("Download inline outputs");
    let actual_dispatch =
        download_f32(&d_output_dispatch, out_dim).expect("Download dispatch outputs");
    let expected = q4_0_q8_0_residual_cpu_oracle(&quantized, &input_q8, &residual, out_dim, in_dim);

    let direct_err = max_abs_error(&expected, &actual_direct);
    assert!(
        direct_err <= 1e-3,
        "Direct Q4_0 residual Q8 fastpath mismatch: max_abs_error={}",
        direct_err
    );

    let inline_err = max_abs_error(&actual_direct, &actual_inline);
    assert!(
        inline_err <= 1e-6,
        "Inline residual output diverged from direct Q8 fastpath: max_abs_error={}",
        inline_err
    );

    let dispatch_err = max_abs_error(&actual_inline, &actual_dispatch);
    assert!(
        dispatch_err <= 1e-6,
        "Dispatch output diverged from inline residual fastpath: max_abs_error={}",
        dispatch_err
    );

    unsafe {
        std::env::remove_var("ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH");
    }
    rocmforge::gpu::refresh_runtime_env_flags();
}
