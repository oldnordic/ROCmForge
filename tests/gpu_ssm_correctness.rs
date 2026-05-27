#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use rocmforge::gpu::kernels::{
    dispatch_conv1d_silu, dispatch_fused_qk_l2_norm_scale, dispatch_fused_sigmoid_alpha_gate,
    dispatch_gated_delta_net, dispatch_gated_norm, dispatch_repeat_interleave_qk,
};
use rocmforge::gpu::{GpuBuffer, GpuDevice};
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

#[test]
#[serial]
fn test_fused_sigmoid_alpha_gate_correctness() {
    let device = GpuDevice::init(0).expect("GPU should initialize");
    let n = 32usize;
    let batch_size = 1usize;

    // Generate random mock values
    let mut beta = vec![0.0f32; n];
    let mut alpha = vec![0.0f32; n];
    let mut dt_bias = vec![0.0f32; n];
    let mut a_log = vec![0.0f32; n];

    for i in 0..n {
        beta[i] = (i as f32) * 0.1 - 1.5;
        alpha[i] = (i as f32) * 0.2 - 2.0;
        dt_bias[i] = 0.5;
        a_log[i] = -0.5;
    }

    // CPU Reference
    let mut ref_beta = beta.clone();
    let mut ref_alpha = alpha.clone();
    for i in 0..n {
        ref_beta[i] = 1.0f32 / (1.0f32 + (-ref_beta[i]).exp());

        let biased = ref_alpha[i] + dt_bias[i];
        let sp = if biased > 20.0 {
            biased
        } else if biased < -20.0 {
            biased.exp()
        } else {
            (1.0 + biased.exp()).ln()
        };
        ref_alpha[i] = sp * (-(-0.5f32).exp());
    }

    // GPU Execution
    let gpu_beta = upload_f32(&beta).expect("GPU operation should succeed");
    let gpu_alpha = upload_f32(&alpha).expect("GPU operation should succeed");
    let gpu_dt_bias = upload_f32(&dt_bias).expect("GPU operation should succeed");
    let gpu_a_log = upload_f32(&a_log).expect("GPU operation should succeed");

    dispatch_fused_sigmoid_alpha_gate(
        gpu_beta.as_ptr() as *mut f32,
        gpu_alpha.as_ptr() as *mut f32,
        gpu_dt_bias.as_ptr() as *const f32,
        gpu_a_log.as_ptr() as *const f32,
        n,
        batch_size,
        device.stream(),
    )
    .expect("GPU operation should succeed");

    device.synchronize().expect("GPU operation should succeed");

    let out_beta = download_f32(&gpu_beta, n).expect("GPU operation should succeed");
    let out_alpha = download_f32(&gpu_alpha, n).expect("GPU operation should succeed");

    for i in 0..n {
        assert!(
            (out_beta[i] - ref_beta[i]).abs() < 1e-5,
            "beta mismatch at {}: actual={}, ref={}",
            i,
            out_beta[i],
            ref_beta[i]
        );
        assert!(
            (out_alpha[i] - ref_alpha[i]).abs() < 1e-5,
            "alpha mismatch at {}: actual={}, ref={}",
            i,
            out_alpha[i],
            ref_alpha[i]
        );
    }
}

#[test]
#[serial]
fn test_conv1d_silu_correctness() {
    let device = GpuDevice::init(0).expect("GPU should initialize");
    let n_channels = 16usize;

    let mut input = vec![0.0f32; n_channels];
    let mut weight = vec![0.0f32; n_channels * 4];
    let mut state = vec![0.0f32; n_channels * 3];

    for i in 0..n_channels {
        input[i] = 1.0;
        weight[i * 4] = 0.1;
        weight[i * 4 + 1] = 0.2;
        weight[i * 4 + 2] = 0.3;
        weight[i * 4 + 3] = 0.4;
        state[i * 3] = 2.0; // s0
        state[i * 3 + 1] = 3.0; // s1
        state[i * 3 + 2] = 4.0; // s2
    }

    // CPU Reference
    let mut ref_output = vec![0.0f32; n_channels];
    let mut ref_state = state.clone();
    for i in 0..n_channels {
        let x = input[i];
        let s0 = ref_state[i * 3];
        let s1 = ref_state[i * 3 + 1];
        let s2 = ref_state[i * 3 + 2];
        let y = weight[i * 4 + 3] * x
            + weight[i * 4 + 2] * s0
            + weight[i * 4 + 1] * s1
            + weight[i * 4] * s2;
        ref_output[i] = y / (1.0 + (-y).exp());

        ref_state[i * 3 + 2] = s1;
        ref_state[i * 3 + 1] = s0;
        ref_state[i * 3] = x;
    }

    // GPU Execution
    let gpu_output = upload_f32(&vec![0.0f32; n_channels]).expect("GPU operation should succeed");
    let gpu_input = upload_f32(&input).expect("GPU operation should succeed");
    let gpu_weight = upload_f32(&weight).expect("GPU operation should succeed");
    let gpu_state = upload_f32(&state).expect("GPU operation should succeed");

    dispatch_conv1d_silu(
        gpu_output.as_ptr() as *mut f32,
        gpu_input.as_ptr() as *const f32,
        gpu_weight.as_ptr() as *const f32,
        gpu_state.as_ptr() as *mut f32,
        n_channels,
        device.stream(),
    )
    .expect("GPU operation should succeed");

    device.synchronize().expect("GPU operation should succeed");

    let out_val = download_f32(&gpu_output, n_channels).expect("GPU operation should succeed");
    let out_state = download_f32(&gpu_state, n_channels * 3).expect("GPU operation should succeed");

    for i in 0..n_channels {
        assert!(
            (out_val[i] - ref_output[i]).abs() < 1e-5,
            "conv output mismatch at {}",
            i
        );
        assert!(
            (out_state[i * 3] - ref_state[i * 3]).abs() < 1e-5,
            "conv state s0 mismatch at {}",
            i
        );
        assert!(
            (out_state[i * 3 + 1] - ref_state[i * 3 + 1]).abs() < 1e-5,
            "conv state s1 mismatch at {}",
            i
        );
        assert!(
            (out_state[i * 3 + 2] - ref_state[i * 3 + 2]).abs() < 1e-5,
            "conv state s2 mismatch at {}",
            i
        );
    }
}

#[test]
#[serial]
fn test_fused_qk_l2_norm_scale_correctness() {
    let device = GpuDevice::init(0).expect("GPU should initialize");
    let n_heads = 4usize;
    let head_dim = 128usize;

    let mut q = vec![0.0f32; n_heads * head_dim];
    let mut k = vec![0.0f32; n_heads * head_dim];
    for i in 0..(n_heads * head_dim) {
        q[i] = (i as f32) * 0.01;
        k[i] = (i as f32) * 0.02;
    }

    // CPU Reference
    let mut ref_q = q.clone();
    let mut ref_k = k.clone();
    let eps = 1e-6f32;
    let q_scale = 1.0f32 / (head_dim as f32).sqrt();

    for h in 0..n_heads {
        let mut q_sq = 0.0f32;
        let mut k_sq = 0.0f32;
        for i in 0..head_dim {
            let qv = ref_q[h * head_dim + i];
            let kv = ref_k[h * head_dim + i];
            q_sq += qv * qv;
            k_sq += kv * kv;
        }

        let q_inv_norm = 1.0 / (q_sq + eps).sqrt();
        let k_inv_norm = 1.0 / (k_sq + eps).sqrt();

        for i in 0..head_dim {
            ref_q[h * head_dim + i] *= q_inv_norm;
            ref_q[h * head_dim + i] *= q_scale;
            ref_k[h * head_dim + i] *= k_inv_norm;
        }
    }

    // GPU Execution
    let gpu_q = upload_f32(&q).expect("GPU operation should succeed");
    let gpu_k = upload_f32(&k).expect("GPU operation should succeed");

    dispatch_fused_qk_l2_norm_scale(
        gpu_q.as_ptr() as *mut f32,
        gpu_k.as_ptr() as *mut f32,
        n_heads,
        head_dim,
        1,
        q_scale,
        eps,
        device.stream(),
    )
    .expect("GPU operation should succeed");

    device.synchronize().expect("GPU operation should succeed");

    let out_q = download_f32(&gpu_q, n_heads * head_dim).expect("GPU operation should succeed");
    let out_k = download_f32(&gpu_k, n_heads * head_dim).expect("GPU operation should succeed");

    for i in 0..(n_heads * head_dim) {
        assert!((out_q[i] - ref_q[i]).abs() < 1e-5, "Q mismatch at {}", i);
        assert!((out_k[i] - ref_k[i]).abs() < 1e-5, "K mismatch at {}", i);
    }
}

#[test]
#[serial]
fn test_repeat_interleave_qk_correctness() {
    let device = GpuDevice::init(0).expect("GPU should initialize");
    let n_key_heads = 4usize;
    let ratio = 2usize;
    let head_dim = 128usize;

    let src_len = n_key_heads * head_dim;
    let dst_len = n_key_heads * ratio * head_dim;

    let mut q_src = vec![0.0f32; src_len];
    let mut k_src = vec![0.0f32; src_len];
    for i in 0..src_len {
        q_src[i] = i as f32 * 0.01;
        k_src[i] = i as f32 * -0.02;
    }

    // CPU Reference
    let mut ref_q_dst = vec![0.0f32; dst_len];
    let mut ref_k_dst = vec![0.0f32; dst_len];
    for idx in 0..dst_len {
        let d = idx % head_dim;
        let kh_r = idx / head_dim;
        let kh = kh_r / ratio;
        let src_off = kh * head_dim + d;
        ref_q_dst[idx] = q_src[src_off];
        ref_k_dst[idx] = k_src[src_off];
    }

    // GPU Execution
    let gpu_q_src = upload_f32(&q_src).expect("GPU operation should succeed");
    let gpu_k_src = upload_f32(&k_src).expect("GPU operation should succeed");
    let gpu_q_dst = upload_f32(&vec![0.0f32; dst_len]).expect("GPU operation should succeed");
    let gpu_k_dst = upload_f32(&vec![0.0f32; dst_len]).expect("GPU operation should succeed");

    dispatch_repeat_interleave_qk(
        gpu_q_src.as_ptr() as *const f32,
        gpu_k_src.as_ptr() as *const f32,
        gpu_q_dst.as_ptr() as *mut f32,
        gpu_k_dst.as_ptr() as *mut f32,
        n_key_heads,
        ratio,
        head_dim,
        device.stream(),
    )
    .expect("GPU operation should succeed");

    device.synchronize().expect("GPU operation should succeed");

    let out_q_dst = download_f32(&gpu_q_dst, dst_len).expect("GPU operation should succeed");
    let out_k_dst = download_f32(&gpu_k_dst, dst_len).expect("GPU operation should succeed");

    for i in 0..dst_len {
        assert!(
            (out_q_dst[i] - ref_q_dst[i]).abs() < 1e-5,
            "q_dst mismatch at {}",
            i
        );
        assert!(
            (out_k_dst[i] - ref_k_dst[i]).abs() < 1e-5,
            "k_dst mismatch at {}",
            i
        );
    }
}

#[test]
#[serial]
fn test_gated_norm_correctness() {
    let device = GpuDevice::init(0).expect("GPU should initialize");
    let n_heads = 4usize;
    let head_dim = 128usize;
    let batch_size = 2usize;
    let eps = 1e-5f32;

    let len = n_heads * head_dim * batch_size;
    let mut x = vec![0.0f32; len];
    let mut z = vec![0.0f32; len];
    let mut weight = vec![0.0f32; head_dim];

    for i in 0..len {
        x[i] = ((i % 17) as f32) * 0.1 - 0.8;
        z[i] = ((i % 13) as f32) * 0.15 - 0.9;
    }
    for (i, value) in weight.iter_mut().enumerate().take(head_dim) {
        *value = 1.0f32 + (i as f32) * 0.005;
    }

    // CPU Reference
    let mut ref_out = vec![0.0f32; len];
    for b in 0..batch_size {
        for h in 0..n_heads {
            let offset = b * n_heads * head_dim + h * head_dim;
            let mut sq_sum = 0.0f32;
            for i in 0..head_dim {
                let v = x[offset + i];
                sq_sum += v * v;
            }
            let inv_rms = 1.0f32 / ((sq_sum / head_dim as f32) + eps).sqrt();
            for i in 0..head_dim {
                let normed = x[offset + i] * inv_rms * weight[i];
                let z_val = z[offset + i];
                let silu_z = z_val / (1.0f32 + (-z_val).exp());
                ref_out[offset + i] = normed * silu_z;
            }
        }
    }

    // GPU Execution
    let gpu_x = upload_f32(&x).expect("GPU operation should succeed");
    let gpu_z = upload_f32(&z).expect("GPU operation should succeed");
    let gpu_weight = upload_f32(&weight).expect("GPU operation should succeed");
    let gpu_out = upload_f32(&vec![0.0f32; len]).expect("GPU operation should succeed");

    dispatch_gated_norm(
        gpu_x.as_ptr() as *const f32,
        gpu_z.as_ptr() as *const f32,
        gpu_weight.as_ptr() as *const f32,
        gpu_out.as_ptr() as *mut f32,
        n_heads,
        head_dim,
        batch_size,
        eps,
        device.stream(),
    )
    .expect("GPU operation should succeed");

    device.synchronize().expect("GPU operation should succeed");

    let out_val = download_f32(&gpu_out, len).expect("GPU operation should succeed");

    for i in 0..len {
        assert!(
            (out_val[i] - ref_out[i]).abs() < 1e-4,
            "gated_norm mismatch at {}: actual={}, ref={}",
            i,
            out_val[i],
            ref_out[i]
        );
    }
}

#[test]
#[serial]
fn test_gated_delta_net_correctness() {
    let device = GpuDevice::init(0).expect("GPU should initialize");
    let n_tokens = 4usize;
    let n_heads = 2usize;
    let head_dim = 128usize;

    let len = n_tokens * n_heads * head_dim;
    let state_len = n_heads * head_dim * head_dim;

    let mut q = vec![0.0f32; len];
    let mut k = vec![0.0f32; len];
    let mut v = vec![0.0f32; len];
    let mut gate = vec![0.0f32; n_tokens * n_heads];
    let mut beta = vec![0.0f32; n_tokens * n_heads];
    let mut state = vec![0.0f32; state_len];

    // Initialize with mock values
    for i in 0..len {
        q[i] = ((i % 11) as f32) * 0.05 - 0.25;
        k[i] = ((i % 13) as f32) * 0.04 - 0.24;
        v[i] = ((i % 17) as f32) * 0.03 - 0.23;
    }
    for i in 0..(n_tokens * n_heads) {
        gate[i] = -0.1 - (i as f32) * 0.02; // alpha will be exp(gate)
        beta[i] = 0.5 + (i as f32) * 0.05;
    }
    for (i, value) in state.iter_mut().enumerate().take(state_len) {
        *value = ((i % 7) as f32) * 0.08 - 0.24;
    }

    // CPU Reference
    let mut ref_state = state.clone();
    let mut ref_out = vec![0.0f32; len];
    let stride = n_heads * head_dim;

    for t in 0..n_tokens {
        for h in 0..n_heads {
            let alpha = (gate[t * n_heads + h]).exp();
            let beta_v = beta[t * n_heads + h];

            // Pointers for current step
            let q_t = &q[t * stride + h * head_dim..t * stride + (h + 1) * head_dim];
            let k_t = &k[t * stride + h * head_dim..t * stride + (h + 1) * head_dim];
            let v_t = &v[t * stride + h * head_dim..t * stride + (h + 1) * head_dim];

            let state_offset = h * head_dim * head_dim;

            // For each row r of state
            for r in 0..head_dim {
                // kv = dot(S[r, :], k_t)
                let mut kv = 0.0f32;
                for c in 0..head_dim {
                    kv += ref_state[state_offset + r * head_dim + c] * k_t[c];
                }

                // delta = (v_t[r] - alpha * kv) * beta_v
                let delta = (v_t[r] - alpha * kv) * beta_v;

                // Update row and calculate output: S_new[r, c] = alpha * S[r, c] + k_t[c] * delta
                let mut out_v = 0.0f32;
                for c in 0..head_dim {
                    let old_s = ref_state[state_offset + r * head_dim + c];
                    let new_s = alpha * old_s + k_t[c] * delta;
                    ref_state[state_offset + r * head_dim + c] = new_s;
                    out_v += new_s * q_t[c];
                }
                ref_out[t * stride + h * head_dim + r] = out_v;
            }
        }
    }

    // GPU Execution
    let gpu_q = upload_f32(&q).expect("GPU operation should succeed");
    let gpu_k = upload_f32(&k).expect("GPU operation should succeed");
    let gpu_v = upload_f32(&v).expect("GPU operation should succeed");
    let gpu_gate = upload_f32(&gate).expect("GPU operation should succeed");
    let gpu_beta = upload_f32(&beta).expect("GPU operation should succeed");
    let gpu_state = upload_f32(&state).expect("GPU operation should succeed");
    let gpu_out = upload_f32(&vec![0.0f32; len]).expect("GPU operation should succeed");

    dispatch_gated_delta_net(
        gpu_q.as_ptr() as *const f32,
        gpu_k.as_ptr() as *const f32,
        gpu_v.as_ptr() as *const f32,
        gpu_gate.as_ptr() as *const f32,
        gpu_beta.as_ptr() as *const f32,
        gpu_state.as_ptr() as *mut f32,
        gpu_out.as_ptr() as *mut f32,
        n_tokens,
        n_heads,
        head_dim,
        device.stream(),
    )
    .expect("GPU operation should succeed");

    device.synchronize().expect("GPU operation should succeed");

    let out_val = download_f32(&gpu_out, len).expect("GPU operation should succeed");
    let out_state = download_f32(&gpu_state, state_len).expect("GPU operation should succeed");

    for i in 0..len {
        assert!(
            (out_val[i] - ref_out[i]).abs() < 1e-4,
            "gated_delta_net output mismatch at {}: actual={}, ref={}",
            i,
            out_val[i],
            ref_out[i]
        );
    }
    for i in 0..state_len {
        assert!(
            (out_state[i] - ref_state[i]).abs() < 1e-4,
            "gated_delta_net state mismatch at {}: actual={}, ref={}",
            i,
            out_state[i],
            ref_state[i]
        );
    }
}
