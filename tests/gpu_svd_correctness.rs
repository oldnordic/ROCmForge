#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use rocmforge::gpu::ops::gpu_dispatch_gemv_svd_on_stream;
use rocmforge::gpu::{detect, GpuBuffer, GpuDevice, SvdCorrection, TensorRole, WeightMeta};
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

// ── SVD-Quant CPU Reference Math ───────────────────────────────────────────

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += aip * b[p * n + j];
            }
        }
    }
    c
}

fn transpose(a: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; n * m];
    for i in 0..m {
        for j in 0..n {
            t[j * m + i] = a[i * n + j];
        }
    }
    t
}

pub fn svd_thin(a: &[f32], m: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    if m < n {
        let at = transpose(a, m, n);
        let (v, sigma, ut) = svd_thin_tall(&at, n, m);
        let u = transpose(&ut, m, m);
        let vt = transpose(&v, n, m);
        return (u, sigma, vt);
    }
    svd_thin_tall(a, m, n)
}

fn svd_thin_tall(a: &[f32], m: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let at = transpose(a, m, n);
    let mut c = matmul(&at, a, n, m, n);

    let mut v = vec![0.0f32; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    for _ in 0..20 {
        let mut converged = true;
        for p in 0..n {
            for q in (p + 1)..n {
                let cpq = c[p * n + q];
                if cpq.abs() > 1e-15 {
                    converged = false;
                    let cpp = c[p * n + p];
                    let cqq = c[q * n + q];
                    let theta = (cqq - cpp) / (2.0 * cpq);
                    let t = 1.0 / (theta.abs() + (1.0 + theta * theta).sqrt());
                    let t = if theta < 0.0 { -t } else { t };
                    let cos = 1.0 / (1.0 + t * t).sqrt();
                    let sin = t * cos;
                    let tau = sin / (1.0 + cos);

                    c[p * n + p] = cpp - t * cpq;
                    c[q * n + q] = cqq + t * cpq;
                    c[p * n + q] = 0.0;
                    c[q * n + p] = 0.0;

                    for r in 0..n {
                        if r != p && r != q {
                            let crp = c[r * n + p];
                            let crq = c[r * n + q];
                            c[r * n + p] = crp - sin * (crq + tau * crp);
                            c[r * n + q] = crq + sin * (crp - tau * crq);
                            c[p * n + r] = c[r * n + p];
                            c[q * n + r] = c[r * n + q];
                        }
                    }

                    for i in 0..n {
                        let vip = v[i * n + p];
                        let viq = v[i * n + q];
                        v[i * n + p] = vip - sin * (viq + tau * vip);
                        v[i * n + q] = viq + sin * (vip - tau * viq);
                    }
                }
            }
        }
        if converged {
            break;
        }
    }

    let mut sigma: Vec<f32> = (0..n).map(|i| c[i * n + i].max(0.0).sqrt()).collect();

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        sigma[b]
            .partial_cmp(&sigma[a])
            .expect("GPU operation should succeed")
    });
    sigma = order.iter().map(|&i| sigma[i]).collect();
    let mut v_sorted = vec![0.0f32; n * n];
    for (new_col, &old_col) in order.iter().enumerate() {
        for row in 0..n {
            v_sorted[row * n + new_col] = v[row * n + old_col];
        }
    }

    let av = matmul(a, &v_sorted, m, n, n);
    let mut u = vec![0.0f32; m * n];
    for col in 0..n {
        let inv_s = if sigma[col] > 1e-10 {
            1.0 / sigma[col]
        } else {
            0.0
        };
        for row in 0..m {
            u[row * n + col] = av[row * n + col] * inv_s;
        }
    }

    let vt = transpose(&v_sorted, n, n);
    (u, sigma, vt)
}

fn dequantize_q4_0_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / 32;
    let mut out = vec![0.0f32; num_elements];
    for i in 0..num_blocks {
        let block_offset = i * 18;
        let scale = half::f16::from_bits(u16::from_le_bytes([
            data[block_offset],
            data[block_offset + 1],
        ]))
        .to_f32();
        for j in 0..32 {
            let byte_idx = j / 2;
            let nibble_idx = j % 2;
            let val_byte = data[block_offset + 2 + byte_idx];
            let val_nibble = if nibble_idx == 0 {
                val_byte & 0x0F
            } else {
                (val_byte >> 4) & 0x0F
            };
            let qval = (val_nibble as i8) - 8;
            out[i * 32 + j] = qval as f32 * scale;
        }
    }
    out
}

fn quantize_q4_0_block(block: &[f32]) -> [u8; 18] {
    let mut max_abs = 0.0f32;
    for &x in block {
        if x.abs() > max_abs {
            max_abs = x.abs();
        }
    }
    let scale = max_abs / 8.0;
    let scale_f16 = half::f16::from_f32(scale);
    let scale_f32 = scale_f16.to_f32();
    let inv_scale = if scale_f32 > 1e-10 {
        1.0 / scale_f32
    } else {
        0.0
    };

    let mut q = [0i8; 32];
    for j in 0..32 {
        let val = block[j] * inv_scale;
        q[j] = val.round().clamp(-8.0, 7.0) as i8;
    }

    let mut out = [0u8; 18];
    let scale_bytes = scale_f16.to_bits().to_le_bytes();
    out[0] = scale_bytes[0];
    out[1] = scale_bytes[1];

    for i in 0..16 {
        let low = (q[2 * i] + 8) as u8 & 0x0F;
        let high = (q[2 * i + 1] + 8) as u8 & 0x0F;
        out[2 + i] = low | (high << 4);
    }
    out
}

fn quantize_matrix_q4_0(data: &[f32]) -> Vec<u8> {
    let num_blocks = data.len() / 32;
    let mut out = Vec::with_capacity(num_blocks * 18);
    for i in 0..num_blocks {
        let block = &data[i * 32..(i + 1) * 32];
        let q_block = quantize_q4_0_block(block);
        out.extend_from_slice(&q_block);
    }
    out
}

// ── Correctness Test Case ───────────────────────────────────────────────────

#[test]
#[serial]
fn test_gpu_svd_quant_correctness() {
    let caps = detect().expect("GPU should be detected");
    let device = GpuDevice::init(caps.device_id).expect("Failed to init GPU");

    // Matrix dims: out_dim (rows) x in_dim (cols)
    let out_dim = 128;
    let in_dim = 256;
    let rank = 8;

    // 1. Generate synthetic row-major weights with random patterns and huge outliers
    let mut w_f32 = vec![0.0f32; out_dim * in_dim];
    for r in 0..out_dim {
        for c in 0..in_dim {
            let phase = (r as f32) * 0.05 + (c as f32) * 0.03;
            w_f32[r * in_dim + c] = phase.sin() * 0.2 + (phase * 2.0).cos() * 0.08;
        }
    }
    // Inject massive outlier channels at index 7 and 142
    for r in 0..out_dim {
        w_f32[r * in_dim + 7] += 32.0;
        w_f32[r * in_dim + 142] -= 48.0;
    }

    // 2. Perform CPU SVD-Quant decomposition
    let (u, sigma, vt) = svd_thin(&w_f32, out_dim, in_dim);
    let min_mn = out_dim.min(in_dim);
    let k = rank;

    let mut u_k = vec![0.0f32; out_dim * k];
    for row in 0..out_dim {
        for col in 0..k {
            u_k[row * k + col] = u[row * min_mn + col] * sigma[col];
        }
    }

    let mut vt_k = vec![0.0f32; k * in_dim];
    for row in 0..k {
        for col in 0..in_dim {
            vt_k[row * in_dim + col] = vt[row * in_dim + col];
        }
    }

    let low_rank_approx = matmul(&u_k, &vt_k, out_dim, k, in_dim);

    let mut residual = vec![0.0f32; out_dim * in_dim];
    for i in 0..out_dim * in_dim {
        residual[i] = w_f32[i] - low_rank_approx[i];
    }

    // Quantize residual using CPU quantizer
    let q_residual = quantize_matrix_q4_0(&residual);

    // Dequantize residual on CPU to serve as base weights reference
    let dequant_residual = dequantize_q4_0_to_f32(&q_residual, out_dim * in_dim);

    // 3. Generate random input vector x
    let mut x_f32 = vec![0.0f32; in_dim];
    for (c, value) in x_f32.iter_mut().enumerate().take(in_dim) {
        let phase = (c as f32) * 0.11;
        *value = phase.sin() * 0.4 - phase.cos() * 0.15;
    }

    // 4. Compute golden CPU reference output
    // y_ref = W_residual_quantized * x + U * (V * x)
    let y_base = matmul(&dequant_residual, &x_f32, out_dim, in_dim, 1);
    let vt_x = matmul(&vt_k, &x_f32, k, in_dim, 1);
    let u_vt_x = matmul(&u_k, &vt_x, out_dim, k, 1);

    let mut y_ref = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        y_ref[i] = y_base[i] + u_vt_x[i];
    }

    // 5. Upload base weights and SVD sub-tensors to GPU
    // Standard RFM Q4Split layout splits the quantized residual
    let num_gguf_blocks = q_residual.len() / 18;
    let rfm_blocks = num_gguf_blocks / 8;
    let mut split_residual = Vec::new();
    let mut split_scales = Vec::new();
    let split_zps = vec![0u8; rfm_blocks * 16];
    let mut split_nibbles = Vec::new();

    for b in 0..rfm_blocks {
        let base_idx = b * 8;
        for i in 0..8 {
            let g_block = &q_residual[(base_idx + i) * 18..(base_idx + i + 1) * 18];
            split_scales.push(g_block[0]);
            split_scales.push(g_block[1]);
            split_nibbles.extend_from_slice(&g_block[2..18]);
        }
    }
    split_residual.extend_from_slice(&split_scales);
    split_residual.extend_from_slice(&split_zps);
    split_residual.extend_from_slice(&split_nibbles);

    let mut d_base = GpuBuffer::alloc(split_residual.len()).expect("GPU operation should succeed");
    d_base
        .copy_from_host(&split_residual)
        .expect("GPU operation should succeed");

    // Unpack base weights from RFM Q4Split layout to standard GGUF Q4_0 layout
    let num_blocks = num_gguf_blocks;
    let d_base_unpacked = GpuBuffer::alloc(num_blocks * 18).expect("GPU operation should succeed");
    rocmforge::gpu::kernels::quant::gpu_unpack_q4_split(
        d_base.as_ptr() as *const u8,
        d_base_unpacked.as_ptr(),
        num_blocks,
        device.stream(),
    )
    .expect("Unpack split base weights");

    let d_u = upload_f32(&u_k).expect("GPU operation should succeed");
    let d_v = upload_f32(&vt_k).expect("GPU operation should succeed");
    let d_input = upload_f32(&x_f32).expect("GPU operation should succeed");

    let zero_out = vec![0.0f32; out_dim];
    let d_output = upload_f32(&zero_out).expect("GPU operation should succeed");
    let zero_temp = vec![0.0f32; 32];
    let d_temp = upload_f32(&zero_temp).expect("GPU operation should succeed");

    // 6. Setup metadata and launch SVD-Quant GEMV on stream
    let meta = WeightMeta {
        wtype: GgmlType::Q4_0,
        dims: vec![in_dim as u64, out_dim as u64],
        needs_transpose: false,
        role: TensorRole::Generic,
        svd_k: Some(k as u32),
    };

    let svd = SvdCorrection {
        u: d_u,
        v: d_v,
        k: k as u32,
    };

    let d_output_base = upload_f32(&zero_out).expect("GPU operation should succeed");

    // Launch base gemv (no SVD)
    gpu_dispatch_gemv_svd_on_stream(
        &device,
        &d_base_unpacked,
        &meta,
        None,
        d_input.as_ptr() as *const f32,
        d_output_base.as_ptr() as *mut f32,
        out_dim,
        in_dim,
        d_temp.as_ptr() as *mut f32,
        device.stream(),
    )
    .expect("Launch base GPU gemv");

    // Launch SVD gemv
    gpu_dispatch_gemv_svd_on_stream(
        &device,
        &d_base_unpacked,
        &meta,
        Some(&svd),
        d_input.as_ptr() as *const f32,
        d_output.as_ptr() as *mut f32,
        out_dim,
        in_dim,
        d_temp.as_ptr() as *mut f32,
        device.stream(),
    )
    .expect("Launch SVD GPU gemv");

    device.synchronize().expect("GPU synchronize");

    // 7. Verify result
    let y_gpu = download_f32(&d_output, out_dim).expect("GPU operation should succeed");
    let y_gpu_base = download_f32(&d_output_base, out_dim).expect("GPU operation should succeed");

    println!("Base Output comparison:");
    for i in 0..10 {
        println!(
            "  [{}] CPU base: {:.6} | GPU base: {:.6}",
            i, y_base[i], y_gpu_base[i]
        );
    }
    println!(
        "  [28] CPU base: {:.6} | GPU base: {:.6}",
        y_base[28], y_gpu_base[28]
    );

    println!("SVD Output comparison:");
    for i in 0..10 {
        println!(
            "  [{}] CPU SVD: {:.6} | GPU SVD: {:.6}",
            i, y_ref[i], y_gpu[i]
        );
    }
    println!(
        "  [28] CPU SVD: {:.6} | GPU SVD: {:.6}",
        y_ref[28], y_gpu[28]
    );

    for i in 0..out_dim {
        let diff = (y_ref[i] - y_gpu[i]).abs();
        assert!(
            diff < 5e-4,
            "Accuracy check failed at index {}: CPU reference was {:.6}, GPU output was {:.6} (diff={:.6})",
            i, y_ref[i], y_gpu[i], diff
        );
    }

    println!("SVD-Quant outlier acceleration correctness verified successfully on GPU!");
}
