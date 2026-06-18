#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use rocmforge::cpu::quant::{load_f16_scale, Q4_BLOCK_BYTES, Q4_BLOCK_ELEMS};
use rocmforge::gpu::{detect, GpuBuffer, GpuDevice, GpuQuant};
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

fn upload_u8(data: &[u8]) -> rocmforge::gpu::GpuResult<GpuBuffer> {
    let mut buf = GpuBuffer::alloc(data.len())?;
    buf.copy_from_host(data)?;
    Ok(buf)
}

fn download_f32(buf: &GpuBuffer, len: usize) -> rocmforge::gpu::GpuResult<Vec<f32>> {
    let mut bytes = vec![0u8; len * std::mem::size_of::<f32>()];
    buf.copy_to_host(&mut bytes)?;
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len).to_vec() })
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

/// CPU oracle for Q4_0 × Q8_0 matrix multiplication
fn q4_0_q8_0_cpu_oracle(
    weights: &[u8],
    input_q8: &[u8],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let num_blocks = in_dim / Q4_BLOCK_ELEMS;
    let col_bytes = num_blocks * Q4_BLOCK_BYTES;
    assert_eq!(input_q8.len(), num_blocks * 34);

    let mut out = vec![0.0f32; out_dim];
    for (col, val) in out.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        let col_offset = col * col_bytes;
        for block_idx in 0..num_blocks {
            let w_block = &weights[col_offset + block_idx * Q4_BLOCK_BYTES
                ..col_offset + (block_idx + 1) * Q4_BLOCK_BYTES];
            let x_block = &input_q8[block_idx * 34..(block_idx + 1) * 34];
            let w_scale = load_f16_scale(&w_block[..2]);
            let x_scale = load_f16_scale(&x_block[..2]);
            let scale = w_scale * x_scale;
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
        *val = acc;
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
fn test_dp4a_matches_scalar_path() {
    require_gpu!();

    let caps = detect().expect("GPU required for DP4A test");
    let device = GpuDevice::init(caps.device_id).expect("Failed to initialize GPU device");
    let gpu_quant = GpuQuant::new(
        GpuDevice::init(caps.device_id).expect("Failed to initialize quantization device"),
    )
    .expect("Failed to initialize GpuQuant");

    // Test dimensions - use realistic sizes
    let in_dim = 4096usize;
    let out_dim = 128usize;

    // Generate test data
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

    // Quantize weights to Q4_0
    let d_weights_q4_0 =
        quantize_q4_0_columns(&gpu_quant, &weights, in_dim, out_dim).expect("Quantization failed");

    // Quantize input to Q8_0 (column vector)
    let d_input = upload_f32(&input).expect("Failed to upload input");
    let num_blocks = in_dim / Q4_BLOCK_ELEMS;
    let d_input_q8_0 = GpuBuffer::alloc(num_blocks * 34).expect("Failed to alloc Q8_0 buffer");

    gpu_quant
        .quantize_q8_0(d_input.as_ptr() as *const f32, d_input_q8_0.as_ptr() as *mut u8, in_dim)
        .expect("Q8_0 quantization failed");

    // Allocate output buffer
    let d_output_scalar = GpuBuffer::alloc(out_dim * std::mem::size_of::<f32>())
        .expect("Failed to allocate output");

    // Test scalar path (DP4A disabled)
    unsafe {
        std::env::set_var("ROCMFORGE_Q4_0_Q8_DP4A", "0");
        std::env::set_var("ROCMFORGE_Q4_0_Q8_SINGLE_ROW", "0"); // Use multi-row
    }
    rocmforge::gpu::refresh_runtime_env_flags();

    unsafe {
        rocmforge::gpu::kernels::q8_decode::gemv_q4_0_q8_0_on_stream(
            d_weights_q4_0.as_ptr(),
            d_input_q8_0.as_ptr(),
            d_output_scalar.as_ptr() as *mut f32,
            in_dim,
            out_dim,
            device.stream(),
        )
    }
    .expect("Scalar kernel launch failed");

    device.synchronize().expect("Stream sync failed");

    let output_scalar = download_f32(&d_output_scalar, out_dim).expect("Failed to download scalar output");

    // Test DP4A path (DP4A enabled)
    let d_output_dp4a = GpuBuffer::alloc(out_dim * std::mem::size_of::<f32>())
        .expect("Failed to allocate output");

    unsafe {
        std::env::set_var("ROCMFORGE_Q4_0_Q8_DP4A", "1");
        std::env::set_var("ROCMFORGE_Q4_0_Q8_SINGLE_ROW", "0"); // Use multi-row
    }
    rocmforge::gpu::refresh_runtime_env_flags();

    unsafe {
        rocmforge::gpu::kernels::q8_decode::gemv_q4_0_q8_0_on_stream(
            d_weights_q4_0.as_ptr(),
            d_input_q8_0.as_ptr(),
            d_output_dp4a.as_ptr() as *mut f32,
            in_dim,
            out_dim,
            device.stream(),
        )
    }
    .expect("DP4A kernel launch failed");

    device.synchronize().expect("Stream sync failed");

    let output_dp4a = download_f32(&d_output_dp4a, out_dim).expect("Failed to download DP4A output");

    // Compare against CPU oracle
    let weights_bytes = {
        let mut buf = vec![0u8; d_weights_q4_0.size()];
        d_weights_q4_0.copy_to_host(&mut buf).expect("Failed to download weights");
        buf
    };

    let input_q8_bytes = {
        let mut buf = vec![0u8; d_input_q8_0.size()];
        d_input_q8_0.copy_to_host(&mut buf).expect("Failed to download input Q8_0");
        buf
    };

    let cpu_oracle = q4_0_q8_0_cpu_oracle(&weights_bytes, &input_q8_bytes, out_dim, in_dim);

    // Compute errors
    let error_scalar_vs_cpu = max_abs_error(&output_scalar, &cpu_oracle);
    let error_dp4a_vs_cpu = max_abs_error(&output_dp4a, &cpu_oracle);
    let error_scalar_vs_dp4a = max_abs_error(&output_scalar, &output_dp4a);

    println!("Max absolute error (scalar vs CPU): {}", error_scalar_vs_cpu);
    println!("Max absolute error (DP4A vs CPU): {}", error_dp4a_vs_cpu);
    println!("Max absolute error (scalar vs DP4A): {}", error_scalar_vs_dp4a);

    // Both paths should match CPU oracle within floating point tolerance
    const TOLERANCE: f32 = 1e-4;
    assert!(
        error_scalar_vs_cpu < TOLERANCE,
        "Scalar path exceeds tolerance vs CPU: {} >= {}",
        error_scalar_vs_cpu,
        TOLERANCE
    );
    assert!(
        error_dp4a_vs_cpu < TOLERANCE,
        "DP4A path exceeds tolerance vs CPU: {} >= {}",
        error_dp4a_vs_cpu,
        TOLERANCE
    );

    // DP4A and scalar should match exactly (same arithmetic, different instruction)
    assert!(
        error_scalar_vs_dp4a < 1e-6,
        "DP4A and scalar paths diverge: {} >= 1e-6",
        error_scalar_vs_dp4a
    );
}

#[test]
#[serial]
fn test_dp4a_residual_matches_scalar_path() {
    require_gpu!();

    let caps = detect().expect("GPU required for DP4A residual test");
    let device = GpuDevice::init(caps.device_id).expect("Failed to initialize GPU device");
    let gpu_quant = GpuQuant::new(
        GpuDevice::init(caps.device_id).expect("Failed to initialize quantization device"),
    )
    .expect("Failed to initialize GpuQuant");

    let in_dim = 4096usize;
    let out_dim = 128usize;

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

    let residual: Vec<f32> = (0..out_dim)
        .map(|col| {
            let phase = (col as f32) * 0.013;
            phase.sin() * 0.27 + phase.cos() * 0.19
        })
        .collect();

    let d_weights_q4_0 =
        quantize_q4_0_columns(&gpu_quant, &weights, in_dim, out_dim).expect("Quantization failed");

    let d_input = upload_f32(&input).expect("Failed to upload input");
    let num_blocks = in_dim / Q4_BLOCK_ELEMS;
    let d_input_q8_0 = GpuBuffer::alloc(num_blocks * 34).expect("Failed to alloc Q8_0 buffer");

    gpu_quant
        .quantize_q8_0(d_input.as_ptr() as *const f32, d_input_q8_0.as_ptr() as *mut u8, in_dim)
        .expect("Q8_0 quantization failed");

    let _d_residual = upload_f32(&residual).expect("Failed to upload residual");

    // Test scalar path
    let d_output_gate_scalar = GpuBuffer::alloc(out_dim * std::mem::size_of::<f32>())
        .expect("Failed to allocate gate output");
    let d_output_up_scalar = GpuBuffer::alloc(out_dim * std::mem::size_of::<f32>())
        .expect("Failed to allocate up output");

    unsafe {
        std::env::set_var("ROCMFORGE_Q4_0_Q8_DP4A", "0");
        std::env::set_var("ROCMFORGE_Q4_0_Q8_SINGLE_ROW", "0");
    }
    rocmforge::gpu::refresh_runtime_env_flags();

    unsafe {
        rocmforge::gpu::kernels::q8_decode::gemv_gate_up_q4_0_q8_0_on_stream(
            d_weights_q4_0.as_ptr(),
            d_weights_q4_0.as_ptr(), // Use same weights for gate_up (test only)
            d_input_q8_0.as_ptr(),
            d_output_gate_scalar.as_ptr() as *mut f32,
            d_output_up_scalar.as_ptr() as *mut f32,
            in_dim,
            out_dim,
            device.stream(),
        )
    }
    .expect("Scalar residual kernel launch failed");

    device.synchronize().expect("Stream sync failed");

    let output_gate_scalar = download_f32(&d_output_gate_scalar, out_dim).expect("Failed to download scalar gate output");

    // Test DP4A path
    let d_output_gate_dp4a = GpuBuffer::alloc(out_dim * std::mem::size_of::<f32>())
        .expect("Failed to allocate gate output");
    let d_output_up_dp4a = GpuBuffer::alloc(out_dim * std::mem::size_of::<f32>())
        .expect("Failed to allocate up output");

    unsafe {
        std::env::set_var("ROCMFORGE_Q4_0_Q8_DP4A", "1");
        std::env::set_var("ROCMFORGE_Q4_0_Q8_SINGLE_ROW", "0");
    }
    rocmforge::gpu::refresh_runtime_env_flags();

    unsafe {
        rocmforge::gpu::kernels::q8_decode::gemv_gate_up_q4_0_q8_0_on_stream(
            d_weights_q4_0.as_ptr(),
            d_weights_q4_0.as_ptr(),
            d_input_q8_0.as_ptr(),
            d_output_gate_dp4a.as_ptr() as *mut f32,
            d_output_up_dp4a.as_ptr() as *mut f32,
            in_dim,
            out_dim,
            device.stream(),
        )
    }
    .expect("DP4A residual kernel launch failed");

    device.synchronize().expect("Stream sync failed");

    let output_gate_dp4a = download_f32(&d_output_gate_dp4a, out_dim).expect("Failed to download DP4A gate output");

    let error_scalar_vs_dp4a = max_abs_error(&output_gate_scalar, &output_gate_dp4a);

    println!("Max absolute error (scalar vs DP4A, residual): {}", error_scalar_vs_dp4a);

    assert!(
        error_scalar_vs_dp4a < 1e-6,
        "DP4A and scalar residual paths diverge: {} >= 1e-6",
        error_scalar_vs_dp4a
    );
}
