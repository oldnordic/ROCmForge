use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::ffi::{hipError_t, hipStream_t};
use std::os::raw::c_int;

extern "C" {
    fn gpu_fused_sigmoid_alpha_gate_f32(
        beta: *mut f32,
        alpha: *mut f32,
        dt_bias: *const f32,
        a_log: *const f32,
        n: c_int,
        batch_size: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gpu_conv1d_silu_f32(
        output: *mut f32,
        input: *const f32,
        weight: *const f32,
        state: *mut f32,
        n_channels: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gpu_fused_qk_l2_norm_scale_f32(
        q: *mut f32,
        k: *mut f32,
        n_heads: c_int,
        head_dim: c_int,
        batch_size: c_int,
        q_scale: f32,
        eps: f32,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gpu_repeat_interleave_qk_f32(
        q_src: *const f32,
        k_src: *const f32,
        q_dst: *mut f32,
        k_dst: *mut f32,
        n_key_heads: c_int,
        ratio: c_int,
        head_dim: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gpu_gated_delta_net_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        gate: *const f32,
        beta: *const f32,
        state: *mut f32,
        output: *mut f32,
        n_tokens: c_int,
        n_heads: c_int,
        head_dim: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gpu_gated_norm_f32(
        x: *const f32,
        z: *const f32,
        weight: *const f32,
        out: *mut f32,
        n_heads: c_int,
        head_dim: c_int,
        batch_size: c_int,
        eps: f32,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gpu_batched_fused_sigmoid_alpha_gate_f32(
        beta: *mut f32,
        alpha: *mut f32,
        dt_bias: *const f32,
        a_log: *const f32,
        n: c_int,
        batch_size: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gpu_batched_conv1d_silu_f32(
        output: *mut f32,
        input: *const f32,
        weight: *const f32,
        state: *mut f32,
        n_channels: c_int,
        batch_size: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gpu_batched_fused_qk_l2_norm_scale_f32(
        q: *mut f32,
        k: *mut f32,
        n_heads: c_int,
        head_dim: c_int,
        batch_size: c_int,
        q_scale: f32,
        eps: f32,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gpu_batched_gated_delta_net_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        gate: *const f32,
        beta: *const f32,
        state: *mut f32,
        output: *mut f32,
        n_tokens: c_int,
        n_heads: c_int,
        head_dim: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gpu_batched_gated_norm_f32(
        x: *const f32,
        z: *const f32,
        weight: *const f32,
        out: *mut f32,
        n_heads: c_int,
        head_dim: c_int,
        batch_size: c_int,
        eps: f32,
        stream: hipStream_t,
    ) -> hipError_t;
}

pub fn dispatch_fused_sigmoid_alpha_gate(
    beta: *mut f32,
    alpha: *mut f32,
    dt_bias: *const f32,
    a_log: *const f32,
    n: usize,
    batch_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if beta.is_null() || alpha.is_null() || dt_bias.is_null() || a_log.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Null pointer in dispatch_fused_sigmoid_alpha_gate".to_string(),
        });
    }
    let res = unsafe {
        gpu_fused_sigmoid_alpha_gate_f32(
            beta,
            alpha,
            dt_bias,
            a_log,
            n as c_int,
            batch_size as c_int,
            stream,
        )
    };
    if res != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: res as i32,
            description: "fused_sigmoid_alpha_gate failed".to_string(),
        });
    }
    Ok(())
}

pub fn dispatch_conv1d_silu(
    output: *mut f32,
    input: *const f32,
    weight: *const f32,
    state: *mut f32,
    n_channels: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if output.is_null() || input.is_null() || weight.is_null() || state.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Null pointer in dispatch_conv1d_silu".to_string(),
        });
    }
    let res =
        unsafe { gpu_conv1d_silu_f32(output, input, weight, state, n_channels as c_int, stream) };
    if res != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: res as i32,
            description: "conv1d_silu failed".to_string(),
        });
    }
    Ok(())
}

pub fn dispatch_fused_qk_l2_norm_scale(
    q: *mut f32,
    k: *mut f32,
    n_heads: usize,
    head_dim: usize,
    batch_size: usize,
    q_scale: f32,
    eps: f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    if q.is_null() || k.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Null pointer in dispatch_fused_qk_l2_norm_scale".to_string(),
        });
    }
    let res = unsafe {
        gpu_fused_qk_l2_norm_scale_f32(
            q,
            k,
            n_heads as c_int,
            head_dim as c_int,
            batch_size as c_int,
            q_scale,
            eps,
            stream,
        )
    };
    if res != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: res as i32,
            description: "fused_qk_l2_norm_scale failed".to_string(),
        });
    }
    Ok(())
}

pub fn dispatch_repeat_interleave_qk(
    q_src: *const f32,
    k_src: *const f32,
    q_dst: *mut f32,
    k_dst: *mut f32,
    n_key_heads: usize,
    ratio: usize,
    head_dim: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if q_src.is_null() || k_src.is_null() || q_dst.is_null() || k_dst.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Null pointer in dispatch_repeat_interleave_qk".to_string(),
        });
    }
    let res = unsafe {
        gpu_repeat_interleave_qk_f32(
            q_src,
            k_src,
            q_dst,
            k_dst,
            n_key_heads as c_int,
            ratio as c_int,
            head_dim as c_int,
            stream,
        )
    };
    if res != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: res as i32,
            description: "repeat_interleave_qk failed".to_string(),
        });
    }
    Ok(())
}

pub fn dispatch_gated_delta_net(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    gate: *const f32,
    beta: *const f32,
    state: *mut f32,
    output: *mut f32,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if q.is_null()
        || k.is_null()
        || v.is_null()
        || gate.is_null()
        || beta.is_null()
        || state.is_null()
        || output.is_null()
    {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Null pointer in dispatch_gated_delta_net".to_string(),
        });
    }
    let res = unsafe {
        gpu_gated_delta_net_f32(
            q,
            k,
            v,
            gate,
            beta,
            state,
            output,
            n_tokens as c_int,
            n_heads as c_int,
            head_dim as c_int,
            stream,
        )
    };
    if res != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: res as i32,
            description: "gated_delta_net failed".to_string(),
        });
    }
    Ok(())
}

pub fn dispatch_gated_norm(
    x: *const f32,
    z: *const f32,
    weight: *const f32,
    out: *mut f32,
    n_heads: usize,
    head_dim: usize,
    batch_size: usize,
    eps: f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    if x.is_null() || z.is_null() || weight.is_null() || out.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Null pointer in dispatch_gated_norm".to_string(),
        });
    }
    let res = unsafe {
        gpu_gated_norm_f32(
            x,
            z,
            weight,
            out,
            n_heads as c_int,
            head_dim as c_int,
            batch_size as c_int,
            eps,
            stream,
        )
    };
    if res != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: res as i32,
            description: "gated_norm failed".to_string(),
        });
    }
    Ok(())
}

// ── Batched Prefill Variants ─────────────────────────────────────────────────────

pub fn dispatch_batched_fused_sigmoid_alpha_gate(
    beta: *mut f32,
    alpha: *mut f32,
    dt_bias: *const f32,
    a_log: *const f32,
    n: usize,
    batch_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if beta.is_null() || alpha.is_null() || dt_bias.is_null() || a_log.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Null pointer in dispatch_batched_fused_sigmoid_alpha_gate".to_string(),
        });
    }
    let res = unsafe {
        gpu_batched_fused_sigmoid_alpha_gate_f32(
            beta,
            alpha,
            dt_bias,
            a_log,
            n as c_int,
            batch_size as c_int,
            stream,
        )
    };
    if res != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: res as i32,
            description: "batched_fused_sigmoid_alpha_gate failed".to_string(),
        });
    }
    Ok(())
}

pub fn dispatch_batched_conv1d_silu(
    output: *mut f32,
    input: *const f32,
    weight: *const f32,
    state: *mut f32,
    n_channels: usize,
    batch_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if output.is_null() || input.is_null() || weight.is_null() || state.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Null pointer in dispatch_batched_conv1d_silu".to_string(),
        });
    }
    let res = unsafe {
        gpu_batched_conv1d_silu_f32(
            output,
            input,
            weight,
            state,
            n_channels as c_int,
            batch_size as c_int,
            stream,
        )
    };
    if res != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: res as i32,
            description: "batched_conv1d_silu failed".to_string(),
        });
    }
    Ok(())
}

pub fn dispatch_batched_fused_qk_l2_norm_scale(
    q: *mut f32,
    k: *mut f32,
    n_heads: usize,
    head_dim: usize,
    batch_size: usize,
    q_scale: f32,
    eps: f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    if q.is_null() || k.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Null pointer in dispatch_batched_fused_qk_l2_norm_scale".to_string(),
        });
    }
    let res = unsafe {
        gpu_batched_fused_qk_l2_norm_scale_f32(
            q,
            k,
            n_heads as c_int,
            head_dim as c_int,
            batch_size as c_int,
            q_scale,
            eps,
            stream,
        )
    };
    if res != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: res as i32,
            description: "batched_fused_qk_l2_norm_scale failed".to_string(),
        });
    }
    Ok(())
}

pub fn dispatch_batched_gated_delta_net(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    gate: *const f32,
    beta: *const f32,
    state: *mut f32,
    output: *mut f32,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if q.is_null()
        || k.is_null()
        || v.is_null()
        || gate.is_null()
        || beta.is_null()
        || state.is_null()
        || output.is_null()
    {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Null pointer in dispatch_batched_gated_delta_net".to_string(),
        });
    }
    let res = unsafe {
        gpu_batched_gated_delta_net_f32(
            q,
            k,
            v,
            gate,
            beta,
            state,
            output,
            n_tokens as c_int,
            n_heads as c_int,
            head_dim as c_int,
            stream,
        )
    };
    if res != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: res as i32,
            description: "batched_gated_delta_net failed".to_string(),
        });
    }
    Ok(())
}

pub fn dispatch_batched_gated_norm(
    x: *const f32,
    z: *const f32,
    weight: *const f32,
    out: *mut f32,
    n_heads: usize,
    head_dim: usize,
    batch_size: usize,
    eps: f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    if x.is_null() || z.is_null() || weight.is_null() || out.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Null pointer in dispatch_batched_gated_norm".to_string(),
        });
    }
    let res = unsafe {
        gpu_batched_gated_norm_f32(
            x,
            z,
            weight,
            out,
            n_heads as c_int,
            head_dim as c_int,
            batch_size as c_int,
            eps,
            stream,
        )
    };
    if res != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: res as i32,
            description: "batched_gated_norm failed".to_string(),
        });
    }
    Ok(())
}
