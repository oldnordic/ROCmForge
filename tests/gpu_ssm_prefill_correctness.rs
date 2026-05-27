#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use rocmforge::gpu::kernels::{
    dispatch_batched_conv1d_silu, dispatch_batched_fused_sigmoid_alpha_gate,
    dispatch_batched_gated_delta_net, dispatch_batched_gated_norm,
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

fn init_device() -> GpuDevice {
    GpuDevice::init(0).expect("GPU should initialize")
}

fn expect<T>(r: rocmforge::gpu::GpuResult<T>) -> T {
    r.expect("GPU operation should succeed")
}

#[test]
#[serial]
fn test_batched_fused_sigmoid_alpha_gate_correctness() {
    let device = init_device();
    let n = 32usize;
    let batch_size = 4usize;

    let mut beta = vec![0.0f32; n * batch_size];
    let mut alpha = vec![0.0f32; n * batch_size];
    let dt_bias = vec![0.5f32; n];
    let a_log = vec![-0.5f32; n];

    for b in 0..batch_size {
        for i in 0..n {
            beta[b * n + i] = (i as f32) * 0.1 - 1.5 + (b as f32) * 0.01;
            alpha[b * n + i] = (i as f32) * 0.2 - 2.0 + (b as f32) * 0.02;
        }
    }

    // CPU Reference
    let mut ref_beta = beta.clone();
    let mut ref_alpha = alpha.clone();
    for b in 0..batch_size {
        for i in 0..n {
            ref_beta[b * n + i] = 1.0f32 / (1.0f32 + (-ref_beta[b * n + i]).exp());

            let biased = ref_alpha[b * n + i] + dt_bias[i];
            let sp = if biased > 20.0 {
                biased
            } else if biased < -20.0 {
                biased.exp()
            } else {
                (1.0 + biased.exp()).ln()
            };
            ref_alpha[b * n + i] = sp * (-a_log[i].exp());
        }
    }

    // GPU Execution
    let gpu_beta = expect(upload_f32(&beta));
    let gpu_alpha = expect(upload_f32(&alpha));
    let gpu_dt_bias = expect(upload_f32(&dt_bias));
    let gpu_a_log = expect(upload_f32(&a_log));

    expect(dispatch_batched_fused_sigmoid_alpha_gate(
        gpu_beta.as_ptr() as *mut f32,
        gpu_alpha.as_ptr() as *mut f32,
        gpu_dt_bias.as_ptr() as *const f32,
        gpu_a_log.as_ptr() as *const f32,
        n,
        batch_size,
        device.stream(),
    ));

    expect(device.synchronize());

    let out_beta = expect(download_f32(&gpu_beta, n * batch_size));
    let out_alpha = expect(download_f32(&gpu_alpha, n * batch_size));

    for b in 0..batch_size {
        for i in 0..n {
            assert!(
                (out_beta[b * n + i] - ref_beta[b * n + i]).abs() < 1e-5,
                "beta mismatch at batch {} idx {}: actual={}, ref={}",
                b,
                i,
                out_beta[b * n + i],
                ref_beta[b * n + i]
            );
            assert!(
                (out_alpha[b * n + i] - ref_alpha[b * n + i]).abs() < 1e-5,
                "alpha mismatch at batch {} idx {}: actual={}, ref={}",
                b,
                i,
                out_alpha[b * n + i],
                ref_alpha[b * n + i]
            );
        }
    }
}

#[test]
#[serial]
fn test_batched_conv1d_silu_correctness() {
    let device = init_device();
    let n_channels = 16usize;
    let batch_size = 4usize;

    let mut input = vec![0.0f32; n_channels * batch_size];
    let mut weight = vec![0.0f32; n_channels * 4];
    let mut state = vec![0.0f32; n_channels * 3 * batch_size];

    for b in 0..batch_size {
        for i in 0..n_channels {
            input[b * n_channels + i] = 1.0 + (b as f32) * 0.1;
            state[b * n_channels * 3 + i * 3] = 2.0;
            state[b * n_channels * 3 + i * 3 + 1] = 3.0;
            state[b * n_channels * 3 + i * 3 + 2] = 4.0;
        }
    }
    for i in 0..n_channels {
        weight[i * 4] = 0.1;
        weight[i * 4 + 1] = 0.2;
        weight[i * 4 + 2] = 0.3;
        weight[i * 4 + 3] = 0.4;
    }

    // CPU Reference
    let mut ref_output = vec![0.0f32; n_channels * batch_size];
    let mut ref_state = state.clone();
    for b in 0..batch_size {
        for i in 0..n_channels {
            let x = input[b * n_channels + i];
            let s0 = ref_state[b * n_channels * 3 + i * 3];
            let s1 = ref_state[b * n_channels * 3 + i * 3 + 1];
            let s2 = ref_state[b * n_channels * 3 + i * 3 + 2];
            let y = weight[i * 4 + 3] * x
                + weight[i * 4 + 2] * s0
                + weight[i * 4 + 1] * s1
                + weight[i * 4] * s2;
            ref_output[b * n_channels + i] = y / (1.0 + (-y).exp());

            ref_state[b * n_channels * 3 + i * 3 + 2] = s1;
            ref_state[b * n_channels * 3 + i * 3 + 1] = s0;
            ref_state[b * n_channels * 3 + i * 3] = x;
        }
    }

    // GPU Execution
    let gpu_output = expect(upload_f32(&vec![0.0f32; n_channels * batch_size]));
    let gpu_input = expect(upload_f32(&input));
    let gpu_weight = expect(upload_f32(&weight));
    let gpu_state = expect(upload_f32(&state));

    expect(dispatch_batched_conv1d_silu(
        gpu_output.as_ptr() as *mut f32,
        gpu_input.as_ptr() as *const f32,
        gpu_weight.as_ptr() as *const f32,
        gpu_state.as_ptr() as *mut f32,
        n_channels,
        batch_size,
        device.stream(),
    ));

    expect(device.synchronize());

    let out_val = expect(download_f32(&gpu_output, n_channels * batch_size));
    let out_state = expect(download_f32(&gpu_state, n_channels * 3 * batch_size));

    for b in 0..batch_size {
        for i in 0..n_channels {
            assert!(
                (out_val[b * n_channels + i] - ref_output[b * n_channels + i]).abs() < 1e-5,
                "conv output mismatch at batch {} idx {}",
                b,
                i
            );
            assert!(
                (out_state[b * n_channels * 3 + i * 3] - ref_state[b * n_channels * 3 + i * 3])
                    .abs()
                    < 1e-5,
                "conv state s0 mismatch at batch {} idx {}",
                b,
                i
            );
        }
    }
}

#[test]
#[serial]
fn test_batched_gated_delta_net_correctness() {
    let device = init_device();
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

    for i in 0..len {
        q[i] = ((i % 11) as f32) * 0.05 - 0.25;
        k[i] = ((i % 13) as f32) * 0.04 - 0.24;
        v[i] = ((i % 17) as f32) * 0.03 - 0.23;
    }
    for i in 0..(n_tokens * n_heads) {
        gate[i] = -0.1 - (i as f32) * 0.02;
        beta[i] = 0.5 + (i as f32) * 0.05;
    }
    for (i, s) in state.iter_mut().enumerate().take(state_len) {
        *s = ((i % 7) as f32) * 0.08 - 0.24;
    }

    // CPU Reference
    let mut ref_state = state.clone();
    let mut ref_out = vec![0.0f32; len];
    let stride = n_heads * head_dim;

    for t in 0..n_tokens {
        for h in 0..n_heads {
            let alpha = (gate[t * n_heads + h]).exp();
            let beta_v = beta[t * n_heads + h];

            let q_t = &q[t * stride + h * head_dim..t * stride + (h + 1) * head_dim];
            let k_t = &k[t * stride + h * head_dim..t * stride + (h + 1) * head_dim];
            let v_t = &v[t * stride + h * head_dim..t * stride + (h + 1) * head_dim];

            let state_offset = h * head_dim * head_dim;

            for r in 0..head_dim {
                let mut kv = 0.0f32;
                for c in 0..head_dim {
                    kv += ref_state[state_offset + r * head_dim + c] * k_t[c];
                }

                let delta = (v_t[r] - alpha * kv) * beta_v;

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
    let gpu_q = expect(upload_f32(&q));
    let gpu_k = expect(upload_f32(&k));
    let gpu_v = expect(upload_f32(&v));
    let gpu_gate = expect(upload_f32(&gate));
    let gpu_beta = expect(upload_f32(&beta));
    let gpu_state = expect(upload_f32(&state));
    let gpu_out = expect(upload_f32(&vec![0.0f32; len]));

    expect(dispatch_batched_gated_delta_net(
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
    ));

    expect(device.synchronize());

    let out_val = expect(download_f32(&gpu_out, len));
    let out_state = expect(download_f32(&gpu_state, state_len));

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

#[test]
#[serial]
fn test_batched_gated_norm_correctness() {
    let device = init_device();
    let n_heads = 4usize;
    let head_dim = 128usize;
    let batch_size = 4usize;
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
    let gpu_x = expect(upload_f32(&x));
    let gpu_z = expect(upload_f32(&z));
    let gpu_weight = expect(upload_f32(&weight));
    let gpu_out = expect(upload_f32(&vec![0.0f32; len]));

    expect(dispatch_batched_gated_norm(
        gpu_x.as_ptr() as *const f32,
        gpu_z.as_ptr() as *const f32,
        gpu_weight.as_ptr() as *const f32,
        gpu_out.as_ptr() as *mut f32,
        n_heads,
        head_dim,
        batch_size,
        eps,
        device.stream(),
    ));

    expect(device.synchronize());

    let out_val = expect(download_f32(&gpu_out, len));

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
