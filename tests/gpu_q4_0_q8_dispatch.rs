#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use rocmforge::cpu::quant::{load_f16_scale, Q4_BLOCK_BYTES, Q4_BLOCK_ELEMS, Q8_BLOCK_BYTES};
use rocmforge::gpu::{detect, GpuBuffer, GpuDevice, GpuQuant, TensorRole, WeightMeta};
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

fn q4_0_q8_0_cpu_oracle(
    weights: &[u8],
    input_q8: &[u8],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let num_blocks = in_dim / Q4_BLOCK_ELEMS;
    let col_bytes = num_blocks * Q4_BLOCK_BYTES;
    assert_eq!(input_q8.len(), num_blocks * Q8_BLOCK_BYTES);

    let mut out = vec![0.0f32; out_dim];
    for col in 0..out_dim {
        let mut acc = 0.0f32;
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

fn silu_scalar(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[test]
#[serial]
fn test_gpu_dispatch_q4_0_uses_q8_activation_fastpath_and_matches_cpu_oracle() {
    require_gpu!();
    unsafe {
        std::env::set_var("ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH", "1");
    }
    rocmforge::gpu::refresh_runtime_env_flags();

    let caps = detect().expect("GPU required for Q4_0 dispatch test");
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
                let phase = (col as f32) * 0.011 + (row as f32) * 0.005;
                phase.sin() * 0.59 - phase.cos() * 0.17
            })
        })
        .collect();
    let input: Vec<f32> = (0..in_dim)
        .map(|row| {
            let phase = (row as f32) * 0.009;
            phase.cos() * 0.43 + phase.sin() * 0.08
        })
        .collect();

    let meta = WeightMeta {
        wtype: GgmlType::Q4_0,
        dims: vec![in_dim as u64, out_dim as u64],
        needs_transpose: false,
        role: TensorRole::Generic,
    };

    let d_weights =
        quantize_q4_0_columns(&gpu_quant, &weights, in_dim, out_dim).expect("Quantize weights");
    let d_input = upload_f32(&input).expect("Upload input");
    let d_output_direct =
        GpuBuffer::alloc(out_dim * std::mem::size_of::<f32>()).expect("Alloc direct output");
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
    .expect("Quantize input to Q8_0");
    rocmforge::gpu::kernels::gemv_q4_0_q8_0_on_stream(
        d_weights.as_ptr() as *const u8,
        d_input_q8.as_ptr() as *const u8,
        d_output_direct.as_ptr() as *mut f32,
        in_dim,
        out_dim,
        device.stream(),
    )
    .expect("Direct Q4_0 x Q8_0 GEMV");

    rocmforge::gpu::ops::gpu_dispatch_gemv_on_stream(
        &device,
        &d_weights,
        &meta,
        d_input.as_ptr() as *const f32,
        d_output_dispatch.as_ptr() as *mut f32,
        out_dim,
        in_dim,
        device.stream(),
    )
    .expect("Dispatch Q4_0 GEMV");

    device.synchronize().expect("Synchronize Q4_0 GEMV");

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
    let actual_direct = download_f32(&d_output_direct, out_dim).expect("Download direct output");
    let actual_dispatch =
        download_f32(&d_output_dispatch, out_dim).expect("Download dispatch output");
    let expected = q4_0_q8_0_cpu_oracle(&quantized, &input_q8, out_dim, in_dim);

    let direct_err = max_abs_error(&expected, &actual_direct);
    assert!(
        direct_err <= 1e-3,
        "Direct Q4_0 Q8 fastpath mismatch: max_abs_error={}",
        direct_err
    );

    let dispatch_err = max_abs_error(&actual_direct, &actual_dispatch);
    assert!(
        dispatch_err <= 1e-6,
        "Dispatch output diverged from direct Q8 fastpath: max_abs_error={}",
        dispatch_err
    );

    unsafe {
        std::env::remove_var("ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH");
    }
    rocmforge::gpu::refresh_runtime_env_flags();
}

#[test]
#[serial]
fn test_gpu_q4_0_q8_gate_up_raw_matches_cpu_oracle() {
    require_gpu!();

    let caps = detect().expect("GPU required for Q4_0 gate/up test");
    let device = GpuDevice::init(caps.device_id).expect("Failed to initialize GPU device");
    let gpu_quant = GpuQuant::new(
        GpuDevice::init(caps.device_id).expect("Failed to initialize quantization device"),
    )
    .expect("Failed to initialize GpuQuant");

    let in_dim = 896usize;
    let ff_dim = 256usize;

    let gate_weights: Vec<f32> = (0..ff_dim)
        .flat_map(|col| {
            (0..in_dim).map(move |row| {
                let phase = (col as f32) * 0.031 + (row as f32) * 0.010;
                phase.sin() * 0.58 + phase.cos() * 0.21
            })
        })
        .collect();
    let up_weights: Vec<f32> = (0..ff_dim)
        .flat_map(|col| {
            (0..in_dim).map(move |row| {
                let phase = (col as f32) * 0.017 + (row as f32) * 0.013;
                phase.cos() * 0.54 - phase.sin() * 0.19
            })
        })
        .collect();
    let input: Vec<f32> = (0..in_dim)
        .map(|row| ((row as f32) * 0.014).sin() * 1.08 - ((row as f32) * 0.003).cos() * 0.12)
        .collect();

    let d_gate =
        quantize_q4_0_columns(&gpu_quant, &gate_weights, in_dim, ff_dim).expect("Quantize gate");
    let d_up = quantize_q4_0_columns(&gpu_quant, &up_weights, in_dim, ff_dim).expect("Quantize up");
    let d_input = upload_f32(&input).expect("Upload input");
    let d_input_q8 = GpuBuffer::alloc(rocmforge::gpu::kernels::q8_0_workspace_bytes(in_dim))
        .expect("Alloc Q8 activation buffer");
    let d_gate_output =
        GpuBuffer::alloc(ff_dim * std::mem::size_of::<f32>()).expect("Alloc gate output");
    let d_up_output =
        GpuBuffer::alloc(ff_dim * std::mem::size_of::<f32>()).expect("Alloc up output");

    rocmforge::gpu::kernels::quantize_q8_0_on_stream(
        d_input.as_ptr() as *const f32,
        d_input_q8.as_ptr(),
        in_dim,
        device.stream(),
    )
    .expect("Quantize input to Q8_0");
    rocmforge::gpu::kernels::gemv_gate_up_q4_0_q8_0_on_stream(
        d_gate.as_ptr() as *const u8,
        d_up.as_ptr() as *const u8,
        d_input_q8.as_ptr() as *const u8,
        d_gate_output.as_ptr() as *mut f32,
        d_up_output.as_ptr() as *mut f32,
        in_dim,
        ff_dim,
        device.stream(),
    )
    .expect("Direct Q4_0 x Q8_0 gate/up");

    device.synchronize().expect("Synchronize Q4_0 gate/up");

    let gate_quantized = download_u8(&d_gate, (in_dim / Q4_BLOCK_ELEMS) * ff_dim * Q4_BLOCK_BYTES)
        .expect("Download gate weights");
    let up_quantized = download_u8(&d_up, (in_dim / Q4_BLOCK_ELEMS) * ff_dim * Q4_BLOCK_BYTES)
        .expect("Download up weights");
    let input_q8 = download_u8(
        &d_input_q8,
        rocmforge::gpu::kernels::q8_0_workspace_bytes(in_dim),
    )
    .expect("Download quantized input");
    let actual_gate = download_f32(&d_gate_output, ff_dim).expect("Download gate output");
    let actual_up = download_f32(&d_up_output, ff_dim).expect("Download up output");
    let expected_gate = q4_0_q8_0_cpu_oracle(&gate_quantized, &input_q8, ff_dim, in_dim);
    let expected_up = q4_0_q8_0_cpu_oracle(&up_quantized, &input_q8, ff_dim, in_dim);

    let gate_err = max_abs_error(&expected_gate, &actual_gate);
    assert!(
        gate_err <= 1e-3,
        "Direct Q4_0 gate Q8 fastpath mismatch: max_abs_error={}",
        gate_err
    );

    let up_err = max_abs_error(&expected_up, &actual_up);
    assert!(
        up_err <= 1e-3,
        "Direct Q4_0 up Q8 fastpath mismatch: max_abs_error={}",
        up_err
    );
}

#[test]
#[serial]
fn test_gpu_q4_0_q8_gate_up_swiglu_matches_cpu_oracle() {
    require_gpu!();

    let caps = detect().expect("GPU required for Q4_0 fused gate/up test");
    let device = GpuDevice::init(caps.device_id).expect("Failed to initialize GPU device");
    let gpu_quant = GpuQuant::new(
        GpuDevice::init(caps.device_id).expect("Failed to initialize quantization device"),
    )
    .expect("Failed to initialize GpuQuant");

    let in_dim = 896usize;
    let ff_dim = 256usize;

    let gate_weights: Vec<f32> = (0..ff_dim)
        .flat_map(|col| {
            (0..in_dim).map(move |row| {
                let phase = (col as f32) * 0.029 + (row as f32) * 0.008;
                phase.sin() * 0.63 + phase.cos() * 0.14
            })
        })
        .collect();
    let up_weights: Vec<f32> = (0..ff_dim)
        .flat_map(|col| {
            (0..in_dim).map(move |row| {
                let phase = (col as f32) * 0.021 + (row as f32) * 0.011;
                phase.cos() * 0.49 - phase.sin() * 0.22
            })
        })
        .collect();
    let input: Vec<f32> = (0..in_dim)
        .map(|row| ((row as f32) * 0.016).sin() * 0.97 - ((row as f32) * 0.004).cos() * 0.15)
        .collect();

    let d_gate =
        quantize_q4_0_columns(&gpu_quant, &gate_weights, in_dim, ff_dim).expect("Quantize gate");
    let d_up = quantize_q4_0_columns(&gpu_quant, &up_weights, in_dim, ff_dim).expect("Quantize up");
    let d_input = upload_f32(&input).expect("Upload input");
    let d_input_q8 = GpuBuffer::alloc(rocmforge::gpu::kernels::q8_0_workspace_bytes(in_dim))
        .expect("Alloc Q8 activation buffer");
    let d_output =
        GpuBuffer::alloc(ff_dim * std::mem::size_of::<f32>()).expect("Alloc fused output");

    rocmforge::gpu::kernels::quantize_q8_0_on_stream(
        d_input.as_ptr() as *const f32,
        d_input_q8.as_ptr(),
        in_dim,
        device.stream(),
    )
    .expect("Quantize input to Q8_0");
    rocmforge::gpu::kernels::gemv_gate_up_swiglu_q4_0_q8_0_on_stream(
        d_gate.as_ptr() as *const u8,
        d_up.as_ptr() as *const u8,
        d_input_q8.as_ptr() as *const u8,
        d_output.as_ptr() as *mut f32,
        in_dim,
        ff_dim,
        device.stream(),
    )
    .expect("Direct fused Q4_0 x Q8_0 gate/up");

    device
        .synchronize()
        .expect("Synchronize fused Q4_0 gate/up");

    let gate_quantized = download_u8(&d_gate, (in_dim / Q4_BLOCK_ELEMS) * ff_dim * Q4_BLOCK_BYTES)
        .expect("Download gate weights");
    let up_quantized = download_u8(&d_up, (in_dim / Q4_BLOCK_ELEMS) * ff_dim * Q4_BLOCK_BYTES)
        .expect("Download up weights");
    let input_q8 = download_u8(
        &d_input_q8,
        rocmforge::gpu::kernels::q8_0_workspace_bytes(in_dim),
    )
    .expect("Download quantized input");
    let actual = download_f32(&d_output, ff_dim).expect("Download fused output");
    let expected_gate = q4_0_q8_0_cpu_oracle(&gate_quantized, &input_q8, ff_dim, in_dim);
    let expected_up = q4_0_q8_0_cpu_oracle(&up_quantized, &input_q8, ff_dim, in_dim);
    let expected: Vec<f32> = expected_gate
        .iter()
        .zip(&expected_up)
        .map(|(gate, up)| silu_scalar(*gate) * *up)
        .collect();

    let err = max_abs_error(&expected, &actual);
    assert!(
        err <= 1e-3,
        "Direct fused Q4_0 Q8 fastpath mismatch: max_abs_error={}",
        err
    );
}

#[test]
#[serial]
fn test_gpu_dispatch_fused_gate_up_uses_q8_fastpath_when_enabled() {
    require_gpu!();
    unsafe {
        std::env::set_var("ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH", "1");
    }
    rocmforge::gpu::refresh_runtime_env_flags();

    let caps = detect().expect("GPU required for fused Q4_0 dispatch test");
    let device = GpuDevice::init(caps.device_id).expect("Failed to initialize GPU device");
    let gpu_quant = GpuQuant::new(
        GpuDevice::init(caps.device_id).expect("Failed to initialize quantization device"),
    )
    .expect("Failed to initialize GpuQuant");

    let in_dim = 896usize;
    let ff_dim = 256usize;

    let gate_weights: Vec<f32> = (0..ff_dim)
        .flat_map(|col| {
            (0..in_dim).map(move |row| {
                let phase = (col as f32) * 0.023 + (row as f32) * 0.007;
                phase.sin() * 0.52 + phase.cos() * 0.18
            })
        })
        .collect();
    let up_weights: Vec<f32> = (0..ff_dim)
        .flat_map(|col| {
            (0..in_dim).map(move |row| {
                let phase = (col as f32) * 0.019 + (row as f32) * 0.012;
                phase.cos() * 0.56 - phase.sin() * 0.16
            })
        })
        .collect();
    let input: Vec<f32> = (0..in_dim)
        .map(|row| ((row as f32) * 0.012).sin() * 0.88 - ((row as f32) * 0.005).cos() * 0.11)
        .collect();

    let meta = WeightMeta {
        wtype: GgmlType::Q4_0,
        dims: vec![in_dim as u64, ff_dim as u64],
        needs_transpose: false,
        role: TensorRole::Generic,
    };

    let d_gate =
        quantize_q4_0_columns(&gpu_quant, &gate_weights, in_dim, ff_dim).expect("Quantize gate");
    let d_up = quantize_q4_0_columns(&gpu_quant, &up_weights, in_dim, ff_dim).expect("Quantize up");
    let d_input = upload_f32(&input).expect("Upload input");
    let d_input_q8 = GpuBuffer::alloc(rocmforge::gpu::kernels::q8_0_workspace_bytes(in_dim))
        .expect("Alloc Q8 activation buffer");
    let d_output_direct =
        GpuBuffer::alloc(ff_dim * std::mem::size_of::<f32>()).expect("Alloc direct output");
    let d_output_inline =
        GpuBuffer::alloc(ff_dim * std::mem::size_of::<f32>()).expect("Alloc inline output");
    let d_output_dispatch =
        GpuBuffer::alloc(ff_dim * std::mem::size_of::<f32>()).expect("Alloc dispatch output");

    rocmforge::gpu::kernels::quantize_q8_0_on_stream(
        d_input.as_ptr() as *const f32,
        d_input_q8.as_ptr(),
        in_dim,
        device.stream(),
    )
    .expect("Quantize input to Q8_0");
    rocmforge::gpu::kernels::gemv_gate_up_swiglu_q4_0_q8_0_on_stream(
        d_gate.as_ptr() as *const u8,
        d_up.as_ptr() as *const u8,
        d_input_q8.as_ptr() as *const u8,
        d_output_direct.as_ptr() as *mut f32,
        in_dim,
        ff_dim,
        device.stream(),
    )
    .expect("Direct fused Q4_0 x Q8_0 gate/up");
    rocmforge::gpu::kernels::gemv_gate_up_swiglu_q4_0_f32_q8_inline_on_stream(
        d_gate.as_ptr() as *const u8,
        d_up.as_ptr() as *const u8,
        d_input.as_ptr() as *const f32,
        d_output_inline.as_ptr() as *mut f32,
        in_dim,
        ff_dim,
        device.stream(),
    )
    .expect("Direct inline-quantized fused Q4_0 gate/up");

    rocmforge::gpu::ops::gpu_dispatch_fused_gate_up_on_stream(
        &device,
        &d_gate,
        &meta,
        &d_up,
        &meta,
        d_input.as_ptr() as *const f32,
        d_output_dispatch.as_ptr() as *mut f32,
        ff_dim,
        in_dim,
        device.stream(),
    )
    .expect("Dispatch fused Q4_0 gate/up");

    device
        .synchronize()
        .expect("Synchronize fused Q4_0 dispatch");

    let actual_direct = download_f32(&d_output_direct, ff_dim).expect("Download direct output");
    let actual_inline = download_f32(&d_output_inline, ff_dim).expect("Download inline output");
    let actual_dispatch =
        download_f32(&d_output_dispatch, ff_dim).expect("Download dispatch output");
    let inline_err = max_abs_error(&actual_direct, &actual_inline);
    assert!(
        inline_err <= 1e-6,
        "Inline fused output diverged from direct fused Q8 fastpath: max_abs_error={}",
        inline_err
    );

    let dispatch_err = max_abs_error(&actual_inline, &actual_dispatch);
    assert!(
        dispatch_err <= 1e-6,
        "Dispatch output diverged from inline fused Q8 fastpath: max_abs_error={}",
        dispatch_err
    );

    unsafe {
        std::env::remove_var("ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH");
    }
    rocmforge::gpu::refresh_runtime_env_flags();
}
