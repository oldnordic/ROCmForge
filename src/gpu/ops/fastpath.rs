use super::super::device::GpuDevice;
use super::super::error::{GpuError, GpuResult};
use super::super::ffi::hipStream_t;
use super::super::kernels::{
    gemv_gate_up_q4_0_f32_on_stream, gemv_gate_up_q4_0_q8_0_on_stream,
    gemv_gate_up_swiglu_q4_0_f32_on_stream,
    gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_on_stream,
    gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_tile4_on_stream,
    gemv_gate_up_swiglu_q4_0_f32_q8_inline_on_stream_variant,
    gemv_gate_up_swiglu_q4_0_f32_single_row_on_stream, gemv_q4_0_f32_q8_inline_residual_on_stream,
    gemv_q4_0_q8_0_on_stream, gemv_q4_0_q8_0_residual_on_stream, q8_0_workspace_bytes,
    quantize_q8_0_on_stream,
};
use super::super::safety::disable_q8_activation_fastpath_runtime;
use super::super::weights::GpuBuffer;

pub(super) fn quantize_input_q8_workspace(
    device: &GpuDevice,
    input: *const f32,
    n_rows: usize,
    stream: hipStream_t,
) -> GpuResult<*mut u8> {
    let workspace = device.q8_workspace_ptr(q8_0_workspace_bytes(n_rows))?;
    quantize_q8_0_on_stream(input, workspace, n_rows, stream)?;
    Ok(workspace)
}

pub(super) fn q8_fastpath_ok(context: &str, fastpath_result: GpuResult<()>) -> bool {
    match fastpath_result {
        Ok(()) => true,
        Err(err) => {
            disable_q8_activation_fastpath_runtime(&format!("{context}: {err}"));
            false
        }
    }
}

pub(super) fn try_q4_0_q8_0_fastpath(
    device: &GpuDevice,
    weights: &GpuBuffer,
    input: *const f32,
    output: *mut f32,
    in_dim: usize,
    out_dim: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    let workspace = quantize_input_q8_workspace(device, input, in_dim, stream)?;

    gemv_q4_0_q8_0_on_stream(
        weights.as_ptr() as *const u8,
        workspace as *const u8,
        output,
        in_dim,
        out_dim,
        stream,
    )
}

pub(super) fn try_q4_0_q8_0_residual_fastpath(
    weights: &GpuBuffer,
    input: *const f32,
    residual: *const f32,
    output: *mut f32,
    in_dim: usize,
    out_dim: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    gemv_q4_0_f32_q8_inline_residual_on_stream(
        weights.as_ptr() as *const u8,
        input,
        residual,
        output,
        in_dim,
        out_dim,
        stream,
    )
}

pub(super) fn try_q4_0_q8_0_residual_fastpath_prequantized(
    device: &GpuDevice,
    weights: &GpuBuffer,
    input: *const f32,
    residual: *const f32,
    output: *mut f32,
    in_dim: usize,
    out_dim: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    let workspace = quantize_input_q8_workspace(device, input, in_dim, stream)?;
    gemv_q4_0_q8_0_residual_on_stream(
        weights.as_ptr() as *const u8,
        workspace as *const u8,
        residual,
        output,
        in_dim,
        out_dim,
        stream,
    )
}

pub(super) fn try_q4_0_q8_0_gate_up_fastpath(
    device: &GpuDevice,
    w_gate: &GpuBuffer,
    w_up: &GpuBuffer,
    input: *const f32,
    gate_output: *mut f32,
    up_output: *mut f32,
    h: usize,
    ff_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    let workspace = quantize_input_q8_workspace(device, input, h, stream)?;
    gemv_gate_up_q4_0_q8_0_on_stream(
        w_gate.as_ptr() as *const u8,
        w_up.as_ptr() as *const u8,
        workspace as *const u8,
        gate_output,
        up_output,
        h,
        ff_size,
        stream,
    )
}

pub(super) fn try_q4_0_q8_0_fused_gate_up_fastpath(
    w_gate: &GpuBuffer,
    w_up: &GpuBuffer,
    input: *const f32,
    output: *mut f32,
    h: usize,
    ff_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    try_q4_0_q8_0_fused_gate_up_fastpath_variant(w_gate, w_up, input, output, h, ff_size, 0, stream)
}

pub(super) fn try_q4_0_q8_0_fused_gate_up_fastpath_prequantized(
    device: &GpuDevice,
    w_gate: &GpuBuffer,
    w_up: &GpuBuffer,
    input: *const f32,
    output: *mut f32,
    h: usize,
    ff_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    let workspace = quantize_input_q8_workspace(device, input, h, stream)?;
    super::super::kernels::gemv_gate_up_swiglu_q4_0_q8_0_on_stream(
        w_gate.as_ptr() as *const u8,
        w_up.as_ptr() as *const u8,
        workspace as *const u8,
        output,
        h,
        ff_size,
        stream,
    )
}

pub(super) fn try_q4_0_q8_0_fused_gate_up_fastpath_variant(
    w_gate: &GpuBuffer,
    w_up: &GpuBuffer,
    input: *const f32,
    output: *mut f32,
    h: usize,
    ff_size: usize,
    variant: i32,
    stream: hipStream_t,
) -> GpuResult<()> {
    gemv_gate_up_swiglu_q4_0_f32_q8_inline_on_stream_variant(
        w_gate.as_ptr() as *const u8,
        w_up.as_ptr() as *const u8,
        input,
        output,
        h,
        ff_size,
        variant,
        stream,
    )
}

pub(super) fn try_q4_0_q8_0_fused_gate_up_single_row_fastpath(
    w_gate: &GpuBuffer,
    w_up: &GpuBuffer,
    input: *const f32,
    output: *mut f32,
    h: usize,
    ff_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    gemv_gate_up_swiglu_q4_0_f32_single_row_on_stream(
        w_gate.as_ptr() as *const u8,
        w_up.as_ptr() as *const u8,
        input,
        output,
        h,
        ff_size,
        stream,
    )
}

pub(super) fn try_q4_0_q8_0_fused_gate_up_interleaved_fastpath(
    w_gate_up_interleaved: &GpuBuffer,
    input: *const f32,
    output: *mut f32,
    h: usize,
    ff_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_on_stream(
        w_gate_up_interleaved.as_ptr() as *const u8,
        input,
        output,
        h,
        ff_size,
        stream,
    )
}

pub(super) fn try_q4_0_q8_0_fused_gate_up_interleaved_tile4_fastpath(
    w_gate_up_interleaved_tile4: &GpuBuffer,
    input: *const f32,
    output: *mut f32,
    h: usize,
    ff_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_tile4_on_stream(
        w_gate_up_interleaved_tile4.as_ptr() as *const u8,
        input,
        output,
        h,
        ff_size,
        stream,
    )
}
