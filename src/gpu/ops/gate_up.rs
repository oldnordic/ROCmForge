use super::super::device::GpuDevice;
use super::super::error::{GpuError, GpuResult};
use super::super::ffi::{hipStreamCaptureStatus, hipStream_t, hip_stream_is_capturing};
use super::super::kernels::{
    gemv_gate_up_q4_0_f32_on_stream, gemv_gate_up_swiglu_q4_0_f32_on_stream, mul_on_stream,
    silu_on_stream,
};
use super::super::safety::{experimental_q8_activation_fastpath_enabled, q4_0_q8_dp4a_enabled};
use super::super::weights::{GpuBuffer, WeightMeta};
use crate::loader::GgmlType;

use super::fastpath::{
    q8_fastpath_ok, try_q4_0_q8_0_fused_gate_up_fastpath,
    try_q4_0_q8_0_fused_gate_up_fastpath_prequantized,
    try_q4_0_q8_0_fused_gate_up_interleaved_fastpath,
    try_q4_0_q8_0_fused_gate_up_interleaved_tile4_fastpath,
    try_q4_0_q8_0_fused_gate_up_single_row_fastpath, try_q4_0_q8_0_gate_up_fastpath,
};
use super::gemv::gpu_dispatch_gemv_on_stream;
use super::validate_gemv_layout;

pub(crate) fn gpu_dispatch_gate_up_raw_on_stream(
    device: &GpuDevice,
    w_gate: &GpuBuffer,
    gate_meta: &WeightMeta,
    w_up: &GpuBuffer,
    up_meta: &WeightMeta,
    input: *const f32,
    gate_output: *mut f32,
    up_output: *mut f32,
    ff_size: usize,
    h: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    validate_gemv_layout(gate_meta, ff_size, h)?;
    validate_gemv_layout(up_meta, ff_size, h)?;

    if gate_output.is_null() || up_output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gpu_dispatch_gate_up_raw: output pointers must be non-null".to_string(),
        });
    }
    if gate_output == up_output {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gpu_dispatch_gate_up_raw: gate and up outputs must be distinct buffers"
                .to_string(),
        });
    }

    if gate_meta.wtype == GgmlType::Q4_0 && up_meta.wtype == GgmlType::Q4_0 {
        if experimental_q8_activation_fastpath_enabled()
            && q8_fastpath_ok(
                "gemv_gate_up_q4_0_q8_0",
                try_q4_0_q8_0_gate_up_fastpath(
                    device,
                    w_gate,
                    w_up,
                    input,
                    gate_output,
                    up_output,
                    h,
                    ff_size,
                    stream,
                ),
            )
        {
            return Ok(());
        }

        gemv_gate_up_q4_0_f32_on_stream(
            w_gate.as_ptr() as *const u8,
            w_up.as_ptr() as *const u8,
            input,
            gate_output,
            up_output,
            h,
            ff_size,
            stream,
        )?;
        return Ok(());
    }

    gpu_dispatch_gemv_on_stream(
        device,
        w_gate,
        gate_meta,
        input,
        gate_output,
        ff_size,
        h,
        stream,
    )?;
    gpu_dispatch_gemv_on_stream(device, w_up, up_meta, input, up_output, ff_size, h, stream)?;
    Ok(())
}

pub(crate) fn gpu_dispatch_fused_gate_up_with_scratch_on_stream(
    device: &GpuDevice,
    w_gate: &GpuBuffer,
    gate_meta: &WeightMeta,
    w_up: &GpuBuffer,
    up_meta: &WeightMeta,
    w_gate_up_interleaved: Option<&GpuBuffer>,
    w_gate_up_interleaved_tile4: Option<&GpuBuffer>,
    input: *const f32,
    gate_scratch: *mut f32,
    output: *mut f32,
    ff_size: usize,
    h: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    validate_gemv_layout(gate_meta, ff_size, h)?;
    validate_gemv_layout(up_meta, ff_size, h)?;

    if gate_meta.wtype == GgmlType::Q4_0 && up_meta.wtype == GgmlType::Q4_0 {
        if experimental_q8_activation_fastpath_enabled() {
            // DP4A fused gate-up SwiGLU path. This is the highest-throughput
            // variant on RDNA2/RDNA3 when dequant dominates decode time.
            if q4_0_q8_dp4a_enabled()
                && q8_fastpath_ok(
                    "gemv_gate_up_swiglu_q4_0_q8_0",
                    try_q4_0_q8_0_fused_gate_up_fastpath_prequantized(
                        device, w_gate, w_up, input, output, h, ff_size, stream,
                    ),
                )
            {
                return Ok(());
            }

            // Hipfire-derived single-row high-occupancy path. Env-gated while we
            // validate it against the existing tile4/interleaved fastpaths.
            if std::env::var("ROCMFORGE_GATE_UP_SINGLE_ROW").is_ok() {
                if q8_fastpath_ok(
                    "gemv_gate_up_swiglu_q4_0_f32_single_row",
                    try_q4_0_q8_0_fused_gate_up_single_row_fastpath(
                        w_gate, w_up, input, output, h, ff_size, stream,
                    ),
                ) {
                    return Ok(());
                }
            }

            let capture_active = matches!(
                hip_stream_is_capturing(stream),
                Err(_)
                    | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusActive)
                    | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusInvalidated)
            );

            if let Some(w_tile4) = w_gate_up_interleaved_tile4 {
                if q8_fastpath_ok(
                    "gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_tile4",
                    try_q4_0_q8_0_fused_gate_up_interleaved_tile4_fastpath(
                        w_tile4, input, output, h, ff_size, stream,
                    ),
                ) {
                    return Ok(());
                }
            }

            if let Some(w_interleaved) = w_gate_up_interleaved {
                if q8_fastpath_ok(
                    "gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved",
                    try_q4_0_q8_0_fused_gate_up_interleaved_fastpath(
                        w_interleaved,
                        input,
                        output,
                        h,
                        ff_size,
                        stream,
                    ),
                ) {
                    return Ok(());
                }
            }

            if q8_fastpath_ok(
                "gemv_gate_up_swiglu_q4_0_f32_q8_inline",
                try_q4_0_q8_0_fused_gate_up_fastpath(
                    w_gate, w_up, input, output, h, ff_size, stream,
                ),
            ) {
                return Ok(());
            }
        }

        unsafe {
            gemv_gate_up_swiglu_q4_0_f32_on_stream(
                w_gate.as_ptr() as *const u8,
                w_up.as_ptr() as *const u8,
                input,
                output,
                h,
                ff_size,
                stream,
            )?;
        }
        return Ok(());
    }

    if gate_scratch.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gpu_dispatch_fused_gate_up: gate scratch pointer must be non-null"
                .to_string(),
        });
    }
    if gate_scratch == output {
        return Err(GpuError::HipApiError {
            code: -1,
            description:
                "gpu_dispatch_fused_gate_up: gate scratch and output must be distinct buffers"
                    .to_string(),
        });
    }

    gpu_dispatch_gate_up_raw_on_stream(
        device,
        w_gate,
        gate_meta,
        w_up,
        up_meta,
        input,
        gate_scratch,
        output,
        ff_size,
        h,
        stream,
    )?;
    silu_on_stream(gate_scratch, gate_scratch, ff_size, stream)?;
    mul_on_stream(gate_scratch, output, output, ff_size, stream)?;
    Ok(())
}

pub fn gpu_dispatch_fused_gate_up_on_stream(
    device: &GpuDevice,
    w_gate: &GpuBuffer,
    gate_meta: &WeightMeta,
    w_up: &GpuBuffer,
    up_meta: &WeightMeta,
    w_gate_up_interleaved: Option<&GpuBuffer>,
    w_gate_up_interleaved_tile4: Option<&GpuBuffer>,
    input: *const f32,
    gate_scratch: *mut f32,
    output: *mut f32,
    ff_size: usize,
    h: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    gpu_dispatch_fused_gate_up_with_scratch_on_stream(
        device,
        w_gate,
        gate_meta,
        w_up,
        up_meta,
        w_gate_up_interleaved,
        w_gate_up_interleaved_tile4,
        input,
        gate_scratch,
        output,
        ff_size,
        h,
        stream,
    )
}

pub fn gpu_dispatch_fused_gate_up(
    device: &GpuDevice,
    w_gate: &GpuBuffer,
    gate_meta: &WeightMeta,
    w_up: &GpuBuffer,
    up_meta: &WeightMeta,
    w_gate_up_interleaved: Option<&GpuBuffer>,
    w_gate_up_interleaved_tile4: Option<&GpuBuffer>,
    input: *const f32,
    output: *mut f32,
    ff_size: usize,
    h: usize,
) -> GpuResult<()> {
    let mut scratch = GpuBuffer::alloc(ff_size * std::mem::size_of::<f32>())?;
    gpu_dispatch_fused_gate_up_on_stream(
        device,
        w_gate,
        gate_meta,
        w_up,
        up_meta,
        w_gate_up_interleaved,
        w_gate_up_interleaved_tile4,
        input,
        scratch.as_ptr() as *mut f32,
        output,
        ff_size,
        h,
        hipStream_t::null(),
    )
}
